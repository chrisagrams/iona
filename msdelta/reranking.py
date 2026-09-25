"""Map a peptide sequence into the spectrum encoder's embedding space.

A database search hands you a spectrum and a list of candidate sequences. To rerank that
list by the model, spectra and sequences have to live in ONE space where distance means
something. This trains that: the spectrum encoder is frozen and acts as a teacher, and a
fresh peptide encoder learns to reproduce its embedding for the sequence that produced
the spectrum.

**Why freeze the spectrum side.** Training both would let them drift into a shared space
of their own invention -- a legitimate objective, but a different one. The spectrum
embedding would stop meaning what pretraining made it mean, and every measurement taken
against it would stop applying. Freezing holds the space fixed and asks only whether
sequences can be mapped into it.

**Why L2 on normalized vectors.** Retrieval is cosine, and for unit vectors
`||a-b||^2 = 2 - 2 cos(a,b)`, so L2 here IS cosine alignment -- the requested loss and the
geometry the space is actually read with, at once. Left unnormalized the encoder can lower
the loss by shrinking every output toward the mean target, which improves no ranking.

**Why watch the ranking, not the loss.** That collapse is exactly what L2 rewards, so the
loss can fall while every ordering is destroyed. `cross_modal_metrics` is the number that
says whether the alignment is useful.

**Why modification mass goes through Fourier features.** The spectrum side encodes m/z
that way. A modification is a mass; giving both sides the same representation avoids
asking them to invent separate notions of the same physical quantity.
"""

from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Sampler

from msdelta.fourier import FourierFeatures

# 20 standard residues plus `n`, which this corpus uses as an N-terminal marker.
RESIDUES = "ACDEFGHIKLMNPQRSTVWYn"
PAD, UNK = 0, 1
RESIDUE_TO_ID = {residue: index + 2 for index, residue in enumerate(RESIDUES)}
VOCAB_SIZE = len(RESIDUE_TO_ID) + 2
_MOD = re.compile(r"\[([+-]?[0-9.]+)\]")



# ---------------------------------------------------------------------------- probes

# Off unless MSDELTA_PROBES is set, so a real training run pays one boolean per call site.
# Turn them on for debug-queue runs: MSDELTA_PROBES=1.
PROBES = os.environ.get("MSDELTA_PROBES", "") not in ("", "0", "false", "False")

# Force scaled_dot_product_attention onto the unfused MATH backend when set.
#
# oneDNN's fused SDPA kernel is a documented source of GPU page faults on Intel GPUs --
# pytorch/pytorch#195319 reports exactly this fault class from an out-of-bounds access in
# the fused kernel's second-tile handling, triggered by a rank-4 attention mask, with
# "force the MATH backend" as the first workaround. nn.TransformerEncoder builds a rank-4
# mask internally whenever a key_padding_mask is supplied, which this model always does.
#
# Not on by default: MATH materialises the full attention matrix and is slower. This is
# a diagnostic switch, and a fallback if it turns out to be the fix.
SDPA_MATH = os.environ.get("MSDELTA_SDPA_MATH", "") not in ("", "0", "false", "False")


def _sdpa_context():
    """MATH-only SDPA when MSDELTA_SDPA_MATH is set, otherwise a no-op."""
    if not SDPA_MATH:
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH)


def probe(where: str, *, sync: Tensor | None = None, **tensors) -> None:
    """Check tensors at a named point, and optionally make the device catch up first.

    `sync` is the important argument. XPU kernels are asynchronous, so a GPU page fault
    is reported wherever the host happens to be when the driver notices, which can be
    dozens of steps past whatever caused it -- job 8840356 faulted "at step 56" and job
    8840238 "at step 73", and neither number means anything without a synchronise. Pass
    any tensor on the device and this blocks until the queue drains, so the fault is
    attributed to the stage that actually caused it.

    Everything else checked here is cheap and has already bitten once: dtype mismatches
    (8840257, 8840336), indices past the end of an embedding table, and non-finite values.
    """
    if not PROBES:
        return
    if sync is not None and sync.device.type == "xpu":
        torch.xpu.synchronize()
    for name, value in tensors.items():
        if not isinstance(value, Tensor):
            continue
        if value.dtype.is_floating_point and not torch.isfinite(value).all():
            n = int((~torch.isfinite(value)).sum())
            raise RuntimeError(f"[probe {where}] {name}: {n} non-finite of {value.numel()}")


def probe_index(where: str, name: str, index: Tensor, limit: int) -> None:
    """An out-of-range embedding index reads unmapped memory on GPU rather than raising.

    On CPU nn.Embedding raises IndexError. On GPU the gather is unchecked, so the read
    lands wherever the arithmetic points and surfaces as `type: 0 (NotPresent),
    access: 0 (Read)` -- which is exactly the signature of FT9. Checking explicitly turns
    a page fault into a sentence naming the tensor and the offending value.
    """
    if not PROBES or index.numel() == 0:
        return
    low, high = int(index.min()), int(index.max())
    if low < 0 or high >= limit:
        raise RuntimeError(
            f"[probe {where}] {name} out of range for a table of {limit}: "
            f"min {low}, max {high}"
        )


# "weighted" variants scale each peak's contribution before averaging. A mass spectrum
# is mostly noise -- the denoise corpus is 53% noise by peak count -- so an unweighted
# mean over 512 peaks is dominated by peaks that carry no identity, and the max is taken
# over those same dimensions. Weights can be intensity (free, and intense fragments are
# what identify a peptide) or a denoiser's P(signal), which is the same information
# learned rather than assumed.
POOLING_MODES = ("mean", "mean+max", "weighted_mean", "weighted_mean+max")


def pool_sequence(tokens: Tensor, mask: Tensor, mode: str = "mean+max",
                  weights: Tensor | None = None) -> Tensor:
    """Reduce variable-length token embeddings to one vector.

    `mean` is the field's default -- sentence_transformers uses it for every encoder
    model, and SBERT's comparison of CLS/mean/max put mean ahead. `mean+max` is what this
    repo's SpectrumRetrievalHead and embedding.pool_tokens already do, so it is kept as
    the default here to stay consistent with numbers measured against that space; it also
    doubles the width, which is a real cost on the peptide side where a sequence is ~15
    residues and a max over 15 tokens is closer to noise than to a feature.

    Both towers must use the same mode. The alignment target is whatever the teacher
    emits, so a mismatch is not a modelling choice, it is a shape error waiting to happen.
    """
    if mode not in POOLING_MODES:
        raise ValueError(f"pooling must be one of {POOLING_MODES}, got {mode!r}")
    mask = mask.bool().unsqueeze(-1)
    if mode.startswith("weighted"):
        if weights is None:
            raise ValueError(f"pooling {mode!r} needs per-peak weights")
        # Normalised so the result stays on the same scale as the unweighted mean;
        # otherwise every downstream threshold and the L2 target space shift with it.
        w = (weights.unsqueeze(-1) * mask).clamp_min(0)
        mean = (tokens * w).sum(1) / w.sum(1).clamp_min(1e-9)
    else:
        mean = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
    if mode in ("mean", "weighted_mean"):
        return mean
    maximum = torch.nan_to_num(tokens.masked_fill(~mask, float("-inf")).max(1).values,
                               neginf=0.0)
    return torch.cat([mean, maximum], dim=-1)


def pooled_width(hidden_size: int, mode: str) -> int:
    """Width `pool_sequence` produces, so the projection can be sized without a forward."""
    # Keyed on whether a max is concatenated, not on an exact name, so the weighted
    # variants size correctly too.
    return 2 * hidden_size if mode.endswith("mean+max") else hidden_size


def parse_peptide(peptide: str) -> tuple[list[int], list[float]]:
    """Split a modified peptide into residue ids and per-residue modification masses.

    Peptides arrive as `SAC[57.0215]GVC[57.0215]PGR`. A bracket binds to the residue
    before it, which is this corpus's convention. The mass is kept as a continuous
    per-residue feature rather than folded into a token: a mass is a number, two
    different masses on one residue are two different chemical species, and a token
    vocabulary would either collapse them or explode.
    """
    ids: list[int] = []
    masses: list[float] = []
    index = 0
    while index < len(peptide):
        char = peptide[index]
        if char == "[":
            end = peptide.find("]", index)
            if end < 0:
                break
            try:
                mass = float(peptide[index + 1 : end])
            except ValueError:
                mass = 0.0
            if masses:
                masses[-1] += mass
            else:
                # A leading bracket has no preceding residue; attach it to an N-terminal
                # marker rather than discarding it.
                ids.append(RESIDUE_TO_ID["n"])
                masses.append(mass)
            index = end + 1
            continue
        ids.append(RESIDUE_TO_ID.get(char, UNK))
        masses.append(0.0)
        index += 1
    return ids, masses


def peptide_key(peptide: str, charge: int, by_charge: bool = True) -> str:
    """Identity used to decide whether two spectra are "the same sequence".

    Charge-aware by default: one peptide at charge 2 and the same peptide at charge 3
    fragment differently enough that their spectra are not really replicates, and merging
    them would make the task look easier than it is.
    """
    return f"{peptide}_{charge}" if by_charge else peptide


@dataclass
class PeptideCollator:
    """Pad parsed peptides into a batch."""

    max_length: int = 64
    # Pad every batch to max_length instead of to the longest peptide in it, so the shape
    # is identical on every step. See AlignmentCollator for why that is worth the few
    # wasted positions.
    pad_to_max: bool = False

    def __call__(self, peptides: list[str], charges: list[int]) -> dict[str, Tensor]:
        parsed = [parse_peptide(p) for p in peptides]
        width = (self.max_length if self.pad_to_max else
                 max(min(max((len(i) for i, _ in parsed), default=1), self.max_length), 1))
        batch = len(parsed)
        residues = torch.zeros(batch, width, dtype=torch.long)
        modifications = torch.zeros(batch, width, dtype=torch.float32)
        sequence_mask = torch.zeros(batch, width, dtype=torch.long)
        for row, (ids, masses) in enumerate(parsed):
            length = min(len(ids), width)
            if length == 0:
                continue
            residues[row, :length] = torch.tensor(ids[:length], dtype=torch.long)
            modifications[row, :length] = torch.tensor(masses[:length], dtype=torch.float32)
            sequence_mask[row, :length] = 1
        return {
            "residues": residues,
            "modifications": modifications,
            "sequence_mask": sequence_mask,
            "charge": torch.tensor(charges, dtype=torch.long),
        }


class PeptideEncoder(nn.Module):
    """Encode a modified peptide plus its charge into a fixed-size embedding."""

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_length: int = 64,
        n_charges: int = 8,
        mod_n_freqs: int = 16,
        dropout: float = 0.1,
        pooling: str = "mean+max",
        readout: str = "pool",
    ):
        super().__init__()
        self.pooling = pooling
        # How the residue tokens become ONE vector. "pool" (default, every model before
        # A3): `pooling` over the tokens -- mean+max is nearly order-blind, which is why
        # the A1 student ranks an adjacent-residue swap above the truth 30% of the time.
        # "cls": a learned token prepended to the sequence, its output is the embedding.
        # "attn": a learned query attending over the tokens. Both keep order information
        # the transformer already computed (positions are embedded). PLAN.md A3.
        if readout not in ("pool", "cls", "attn"):
            raise ValueError(f"readout must be pool, cls or attn, not {readout!r}")
        self.readout = readout
        if readout == "cls":
            self.cls = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        if readout == "attn":
            self.attn_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
            self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout,
                                              batch_first=True)
        self.residue = nn.Embedding(VOCAB_SIZE, hidden_size, padding_idx=PAD)
        self.position = nn.Embedding(max_length, hidden_size)
        self.charge = nn.Embedding(n_charges, hidden_size)
        # 1e-2..1e3 spans a whole modification down to fine isotopic structure.
        self.mod_features = FourierFeatures(mod_n_freqs, 1e-2, 1e3)
        self.mod_projection = nn.Linear(self.mod_features.out_dim, hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=4 * hidden_size,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        # Skip the NestedTensor conversion. The padding here is a few residues out of 64,
        # so it buys nothing, and it is a second fast path to reason about. See forward()
        # for the one that actually bites.
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        width = pooled_width(hidden_size, pooling) if readout == "pool" else hidden_size
        self.projection = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, embedding_size),
        )

    def forward(self, residues, modifications, sequence_mask, charge) -> Tensor:
        probe_index("student.in", "residues", residues, self.residue.num_embeddings)
        probe_index("student.in", "charge", charge, self.charge.num_embeddings)
        probe("student.in", modifications=modifications)
        length = residues.shape[1]
        position = torch.arange(length, device=residues.device).clamp_max(
            self.position.num_embeddings - 1
        )
        probe_index("student.pos", "position", position, self.position.num_embeddings)
        hidden = self.residue(residues) + self.position(position)[None]
        # Only modified residues get a mass contribution; an unmodified residue must not
        # be handed the Fourier encoding of zero as though it were a real modification.
        modified = (modifications.abs() > 1e-6).unsqueeze(-1)
        mod = self.mod_projection(self.mod_features(modifications).to(hidden.dtype))
        hidden = hidden + mod * modified
        hidden = hidden + self.charge(charge.clamp(0, self.charge.num_embeddings - 1))[:, None]
        if self.readout == "cls":
            hidden = torch.cat([self.cls.expand(hidden.shape[0], -1, -1).to(hidden.dtype),
                                hidden], dim=1)
            sequence_mask = torch.cat([torch.ones_like(sequence_mask[:, :1]),
                                       sequence_mask], dim=1)
        # Run the stack with autocast off, in whatever dtype the weights are.
        #
        # torch's TransformerEncoderLayer has a fused fast path
        # (torch._transformer_encoder_layer_fwd) that it takes only when grad is disabled
        # -- eval, never training -- and that kernel does not honour autocast. It is meant
        # to be guarded by
        #
        #     elif torch.is_autocast_enabled():   # transformer.py:869
        #
        # but the no-argument form of that call reports CUDA's autocast state, so under
        # torch.autocast("xpu") it returns False and the guard never fires. Job 8840257
        # trained 200 steps and died on its first evaluation: "expected scalar type
        # BFloat16 but found Float".
        #
        # What the kernel cannot tolerate is a MISMATCH, not bf16. Forcing fp32 fixed the
        # single-tile case and then broke DeepSpeed, which holds the parameters in bf16 --
        # job 8840336 died at step 0 with the same error inverted, "expected scalar type
        # Float but found BFloat16". Matching the activations to the parameters is right
        # in both: fp32 against fp32 weights on one tile, bf16 against bf16 under ZeRO-2.
        # It also keeps training and evaluation numerically identical, which is the other
        # reason not to leave this to autocast.
        param_dtype = next(self.encoder.parameters()).dtype
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.to(param_dtype)
            with _sdpa_context():
                hidden = self.norm(
                    self.encoder(hidden, src_key_padding_mask=~sequence_mask.bool())
                )
        probe("student.encoded", sync=hidden, hidden=hidden)
        if self.readout == "cls":
            pooled = hidden[:, 0]
        elif self.readout == "attn":
            query = self.attn_query.expand(hidden.shape[0], -1, -1).to(hidden.dtype)
            pooled = self.attn(query, hidden, hidden,
                               key_padding_mask=~sequence_mask.bool(),
                               need_weights=False)[0][:, 0]
        else:
            pooled = pool_sequence(hidden, sequence_mask, self.pooling)
        out = F.normalize(self.projection(pooled).float(), dim=-1)
        probe("student.out", sync=out, out=out)
        return out



def lit_contrastive_loss(target, predicted, group, negatives=None, neg_valid=None,
                         temperature: float = 0.05) -> Tensor:
    """Cross-modal SupCon against frozen targets. Anchor = spectrum target (row i);
    candidates = every peptide embedding in the batch (positives: same `group`) plus
    that row's hard negatives. Unit vectors in, mean over anchors and positives out."""
    t = F.normalize(target.float(), dim=-1)
    p = F.normalize(predicted.float(), dim=-1)
    logits = t @ p.T / temperature                                  # (B, B)
    positive = (group[:, None] == group[None, :]).float()
    all_logits = logits
    if negatives is not None:
        n = F.normalize(negatives.float(), dim=-1)                  # (B, K, D)
        hard = torch.einsum("bd,bkd->bk", t, n) / temperature
        hard = hard.masked_fill(~neg_valid, float("-inf"))
        all_logits = torch.cat([logits, hard], dim=1)
    log_prob = logits - torch.logsumexp(all_logits, dim=1, keepdim=True)
    per_anchor = -(log_prob * positive).sum(1) / positive.sum(1)
    return per_anchor.mean()


def student_readout(state: dict, prefix: str = "sequence_encoder.") -> str:
    """Which PeptideEncoder readout a saved student used, read off its weight names, so
    loaders need no extra config (every pre-A3 student has neither key -> "pool")."""
    keys = {k[len(prefix):] if k.startswith(prefix) else k for k in state}
    return "cls" if "cls" in keys else "attn" if "attn_query" in keys else "pool"


class SequenceAlignmentModel(nn.Module):
    """Frozen spectrum teacher, trainable peptide student, L2 between them."""

    def __init__(self, spectrum_model: nn.Module | None, sequence_encoder: PeptideEncoder,
                 pooling: str = "mean+max", loss: str = "mse", temperature: float = 0.05,
                 mse_weight: float = 0.0):
        """`spectrum_model=None` trains against PRECOMPUTED targets.

        The teacher is frozen, so its embedding for a given spectrum is identical in every
        epoch. Running it inside the training step spends ~92% of the parameters and all
        of the 512-peak attention regenerating a constant -- and it is the one structural
        difference between this model and the denoise model, which is the only one that
        survives twelve tiles (FT9). Precomputing the targets removes it from the graph
        entirely, so the wrapped module is the 4.11M student alone.
        """
        super().__init__()
        self.spectrum_model = spectrum_model
        self.sequence_encoder = sequence_encoder
        # The teacher's pooling defines the target space, so it must match the student's.
        if pooling != sequence_encoder.pooling:
            raise ValueError(
                f"teacher pooling {pooling!r} != student pooling {sequence_encoder.pooling!r}"
            )
        self.pooling = pooling
        # "mse" (A1): regress onto the teacher embedding. "lit" (A4): LiT-style
        # cross-modal contrastive against the FROZEN teacher (Zhai et al., CVPR 2022),
        # SupCon-style multi-positive (every spectrum of the batch's same peptide is a
        # positive), negatives = the batch's other peptides + synthetic hard negatives;
        # plus mse_weight x the A1 term to stay in the teacher's space.
        if loss not in ("mse", "lit"):
            raise ValueError(f"loss must be mse or lit, not {loss!r}")
        self.loss, self.temperature, self.mse_weight = loss, temperature, mse_weight
        # Frozen AND in eval mode. requires_grad_(False) alone leaves dropout active, so
        # the teacher would emit a different target for the same spectrum every epoch and
        # the student would be chasing noise.
        if self.spectrum_model is not None:
            self.spectrum_model.requires_grad_(False)
            self.spectrum_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.spectrum_model is not None:
            self.spectrum_model.eval()
        return self

    def _lit_loss(self, predicted, target, group, neg_residues, neg_modifications,
                  neg_sequence_mask, neg_charge, neg_valid) -> Tensor:
        if group is None:        # no collator groups: each row is its own peptide
            group = torch.arange(len(predicted), device=predicted.device)
        negatives = None
        if neg_residues is not None:
            b, k = neg_valid.shape
            negatives = self.sequence_encoder(neg_residues, neg_modifications,
                                              neg_sequence_mask, neg_charge).view(b, k, -1)
        return lit_contrastive_loss(target, predicted, group, negatives, neg_valid,
                                    self.temperature)

    @torch.no_grad()
    def embed_spectrum(self, mz, log_intensity, attention_mask) -> Tensor:
        encoder = getattr(self.spectrum_model, "msdelta", self.spectrum_model)
        hidden = encoder(mz=mz, log_intensity=log_intensity,
                         attention_mask=attention_mask).last_hidden_state
        return F.normalize(pool_sequence(hidden, attention_mask, self.pooling).float(), dim=-1)

    def forward(self, residues, modifications, sequence_mask, charge,
                mz=None, log_intensity=None, attention_mask=None, target=None,
                return_dict: bool = True, return_loss: bool = True,
                neg_residues=None, neg_modifications=None, neg_sequence_mask=None,
                neg_charge=None, neg_valid=None, peptide_group=None):
        # return_loss is not read here; it exists so transformers' can_return_loss() finds
        # it in the signature (utils/generic.py looks for exactly this name defaulting to
        # True). This task is self-supervised against a frozen teacher, so there is no
        # `labels` argument for find_labels() to latch onto either, and without one of the
        # two the Trainer decides evaluation cannot produce a loss: job 8840277 evaluated
        # fine and then died on `metric_for_best_model='eval_loss'` not existing, with only
        # eval_runtime and friends in the metrics.
        if target is None:
            if self.spectrum_model is None:
                raise ValueError("no teacher and no precomputed target in the batch")
            target = self.embed_spectrum(mz, log_intensity, attention_mask)
            probe("teacher.out", sync=target, target=target)
        else:
            # Cached targets are stored normalised; renormalise anyway, since the loss
            # below is only equal to 2-2cos on unit vectors and a silent drift there
            # would change what is being optimised without changing anything visible.
            target = F.normalize(target.float(), dim=-1)
        predicted = self.sequence_encoder(residues, modifications, sequence_mask, charge)
        probe("loss.in", sync=predicted, predicted=predicted, target=target)
        # Mean squared L2. On unit vectors this equals 2 - 2*cos, so it is simultaneously
        # the requested L2 loss and alignment in the geometry the space is searched with.
        mse = ((predicted - target) ** 2).sum(dim=-1).mean()
        if self.loss == "mse":
            loss = mse
        else:
            loss = self._lit_loss(predicted, target, peptide_group, neg_residues,
                                  neg_modifications, neg_sequence_mask, neg_charge,
                                  neg_valid) + self.mse_weight * mse
        if not return_dict:
            return (loss, predicted, target)
        return {"loss": loss, "embeddings": predicted, "target": target}



@torch.no_grad()
def attach_teacher_embeddings(datasets: dict, spectrum_model: nn.Module, pooling: str,
                              batch_size: int = 16, max_peptide_length: int = 64,
                              device: str | torch.device | None = None,
                              pad_spectra_to: int = 0) -> dict:
    """Run the frozen teacher once and store its embedding as a `target` column.

    The teacher never learns, so its output for a spectrum is the same in epoch 10 as in
    epoch 1. Computing it inside the training step therefore spends ~92% of the model's
    parameters, and all of the 512-peak attention, reproducing a constant -- and it is
    the only structural difference between this model and the denoise model, which is the
    one that survives twelve tiles (FT9).

    Call this on ONE process before the Trainer is built, and let the other ranks pick the
    result up from the datasets cache. The teacher is then not part of the wrapped module
    at all, so neither DDP nor ZeRO-2 ever sees it.
    """
    model = SequenceAlignmentModel(spectrum_model, PeptideEncoder(
        embedding_size=1, hidden_size=8, num_layers=1, num_heads=1, pooling=pooling),
        pooling=pooling)
    device = device or ("xpu" if torch.xpu.is_available() else "cpu")
    model.spectrum_model.to(device).eval()
    # pad_spectra_to=max_peaks makes every batch the same width. Twelve of these on one
    # node with batch-max padding took GPU page faults (8861039: 5 of 12 tiles, a Write
    # at 0xff00....) -- the FT16 mechanism, where memory moves with the widest spectrum.
    collator = AlignmentCollator(max_peptide_length=max_peptide_length,
                                 pad_spectra_to=pad_spectra_to)

    def embed(batch: dict) -> dict:
        rows = [{"mz": mz, "log_intensity": li, "peptide": pep, "charge": ch}
                for mz, li, pep, ch in zip(batch["mz"], batch["log_intensity"],
                                           batch["peptide"], batch["charge"])]
        inputs = collator(rows)
        target = model.embed_spectrum(
            inputs["mz"].to(device), inputs["log_intensity"].to(device),
            inputs["attention_mask"].to(device))
        # Probed like any other stage. This path had none, and job 8840378 faulted inside
        # it with nothing to say where -- the teacher forward is as capable of faulting as
        # the training step, and running it outside the Trainer does not make it safe.
        probe("precompute.target", sync=target, target=target)
        return {"target": target.float().cpu().tolist()}

    return {name: split.map(embed, batched=True, batch_size=batch_size,
                            desc=f"teacher embeddings ({name})")
            for name, split in datasets.items()}



@torch.no_grad()
def group_separation_metrics(embeddings: Tensor, groups: np.ndarray,
                             prefix: str = "sep") -> dict[str, float]:
    """Do replicates of one peptide sit closer together than to other peptides?

    hit@1 answers "is the right candidate first", which is what retrieval needs, but it
    says nothing about WHY a space fails. This asks the underlying geometric question:
    for every group of spectra sharing a peptide and charge, how far apart are they, and
    how far from everything else.

    The one number to read first is `clean`, the fraction of groups whose farthest
    in-group neighbour is nearer than its closest out-group one. That is separation with
    no overlap at all, and it is the property a distance-ranked reranker actually needs.
    `margin` is the softer version: mean out-group distance minus mean in-group distance,
    which stays positive long after `clean` has collapsed to zero.

    Distances are squared euclidean on unit vectors, so d = 2 - 2cos and the scale is
    [0, 4] regardless of dimension.
    """
    embeddings = F.normalize(embeddings.float(), dim=-1)
    distances = torch.cdist(embeddings, embeddings).pow(2)
    groups = np.asarray(groups)
    same = torch.as_tensor(groups[:, None] == groups[None, :])
    self_mask = torch.eye(len(embeddings), dtype=torch.bool)

    in_pairs = same & ~self_mask
    out_pairs = ~same
    if not in_pairs.any() or not out_pairs.any():
        return {f"{prefix}/groups": float(len(set(groups.tolist())))}

    inside, outside = distances[in_pairs], distances[out_pairs]
    # Per-group extremes, then averaged, rather than the global extremes: one pathological
    # group should not be able to hide behind 1,000 well-behaved ones, and the global min
    # over all pairs is dominated by whichever two spectra happen to be near-duplicates.
    clean, worst_in, best_out = 0, [], []
    for group in np.unique(groups):
        rows = torch.as_tensor(groups == group)
        if rows.sum() < 2:
            continue
        block = distances[rows][:, rows]
        far = block[~torch.eye(int(rows.sum()), dtype=torch.bool)].max()
        near = distances[rows][:, ~rows].min()
        worst_in.append(float(far))
        best_out.append(float(near))
        clean += int(far < near)

    return {
        f"{prefix}/in_mean": float(inside.mean()),
        f"{prefix}/in_min": float(inside.min()),
        f"{prefix}/in_max": float(inside.max()),
        f"{prefix}/out_mean": float(outside.mean()),
        f"{prefix}/out_min": float(outside.min()),
        f"{prefix}/out_max": float(outside.max()),
        # Positive means replicates are closer to each other than to other peptides.
        f"{prefix}/margin": float(outside.mean() - inside.mean()),
        # Rank arms by THIS, not by margin. Margin is a difference, so it rises when a
        # model simply inflates the whole space, and the contrastive sweep produced
        # exactly that trap: lr1e4_kl0_t02 took the best margin (+0.278) with a ratio of
        # 2.01 while lr1e4_kl0_t007 clustered far better at 2.95 for a margin 0.0008
        # lower. The ratio is scale-invariant and cannot be bought by expansion.
        f"{prefix}/ratio": float(outside.mean() / inside.mean().clamp_min(1e-9)),
        # 1.0 would mean every group is perfectly separated from every other.
        f"{prefix}/clean": clean / max(len(worst_in), 1),
        f"{prefix}/worst_in_mean": float(np.mean(worst_in)) if worst_in else 0.0,
        f"{prefix}/best_out_mean": float(np.mean(best_out)) if best_out else 0.0,
        f"{prefix}/groups": float(len(worst_in)),
    }


@torch.no_grad()
def cross_modal_metrics(sequence_embeddings, spectrum_embeddings, spectrum_groups,
                        sequence_groups=None) -> dict[str, float]:
    """Rank candidate sequences against each spectrum -- the reranking use case.

    The two sides are indexed independently. `spectrum_groups` says which candidate is
    correct for each spectrum; `sequence_groups` says what each candidate IS. They differ
    whenever duplicate peptides are collapsed to one candidate, which is the realistic
    case for a replicate corpus -- sharing one array between them silently indexes spectra
    by candidate position as soon as the counts diverge.
    """
    sequence = F.normalize(torch.as_tensor(sequence_embeddings).float(), dim=-1)
    spectrum = F.normalize(torch.as_tensor(spectrum_embeddings).float(), dim=-1)
    if sequence.shape[0] < 2 or spectrum.shape[0] < 1:
        return {}
    spectrum_groups = torch.as_tensor(np.asarray(spectrum_groups))
    sequence_groups = (spectrum_groups if sequence_groups is None
                       else torch.as_tensor(np.asarray(sequence_groups)))
    if len(sequence_groups) != sequence.shape[0] or len(spectrum_groups) != spectrum.shape[0]:
        raise ValueError("group arrays must match their own side's row count")

    order = (spectrum @ sequence.T).argsort(dim=-1, descending=True)
    relevant = sequence_groups[order] == spectrum_groups[:, None]
    first = relevant.float().argmax(dim=-1)
    hit = relevant.any(dim=-1)
    reciprocal = torch.where(hit, 1.0 / (first + 1).float(), torch.zeros(len(first)))
    metrics = {
        "crossmodal/hit@1": float(relevant[:, 0].float().mean()),
        "crossmodal/hit@5": float(relevant[:, :5].any(dim=-1).float().mean()),
        "crossmodal/mrr": float(reciprocal.mean()),
        "crossmodal/n_spectra": float(spectrum.shape[0]),
        "crossmodal/n_candidates": float(sequence.shape[0]),
    }
    if sequence.shape[0] == spectrum.shape[0]:
        metrics["crossmodal/paired_cosine"] = float(
            F.cosine_similarity(spectrum, sequence, dim=-1).mean()
        )
    return metrics


REPLICATE_REPO = "chrisagrams/ms2-peptide-replicate-retrieval"


def build_alignment_datasets(repo_id, processor, num_proc=None, validation_fraction=0.1,
                             seed=0):
    """Spectrum/sequence pairs, split BY PEPTIDE.

    The replicate corpus ships one split, so it is divided here. Splitting by row would
    put replicates of one peptide on both sides, letting the encoder memorise a sequence
    it is then evaluated on -- the validation number would measure recall rather than
    generalisation. Peptides are held out whole.
    """
    from datasets import load_dataset

    raw = load_dataset(repo_id)
    split = "train" if "train" in raw else list(raw)[0]

    # Spectra above max_peaks are DROPPED, not truncated, matching build_denoising_datasets.
    # Truncating would hand the teacher a spectrum it never saw in pretraining -- the same
    # label attached to a different object -- so the alignment target would be an embedding
    # of something that does not exist, and the student would learn to predict it.
    #
    # Dropping is not free either: 3,030 of 15,649 (19.4%) go, and not at random, because
    # peak count tracks precursor charge and peptide length. That is a real limit on what
    # this corpus can say, so it is COUNTED AND PRINTED rather than silently swallowed by
    # the except below, which is how it went unnoticed in the first place.
    max_peaks = getattr(processor, "max_peaks", None)

    def prepare(example):
        if max_peaks is not None and len(example["mz"]) > max_peaks:
            return {"mz": [], "log_intensity": [], "peptide": "", "charge": 0, "precursor": 0.0}
        try:
            values = processor(
                torch.as_tensor(example["mz"], dtype=torch.float32),
                torch.as_tensor(example["intensity"], dtype=torch.float32),
                padding=False,
            )
        except (ValueError, KeyError, TypeError):
            return {"mz": [], "log_intensity": [], "peptide": "", "charge": 0, "precursor": 0.0}
        return {
            "mz": values["mz"][0] if values["mz"] and isinstance(values["mz"][0], list)
                  else values["mz"],
            "log_intensity": (values["log_intensity"][0]
                              if values["log_intensity"]
                              and isinstance(values["log_intensity"][0], list)
                              else values["log_intensity"]),
            "peptide": example.get("peptide") or "",
            "charge": int(example.get("charge") or 0),
            # The MEASURED precursor m/z. The rescorer compares each candidate's mass to
            # it; without it the rescorer rebuilt the precursor from the true peptide,
            # which set the truth's mass error to exactly 0 and leaked the label.
            "precursor": float(example.get("precursor") or 0.0),
        }

    rows = raw[split].map(prepare, remove_columns=raw[split].column_names,
                          num_proc=num_proc, desc="preprocess alignment pairs")
    before = len(rows)
    rows = rows.filter(lambda e: len(e["mz"]) > 0 and bool(e["peptide"]),
                       num_proc=num_proc, desc="drop empty pairs")
    dropped = before - len(rows)
    if dropped:
        oversized = sum(1 for n in raw[split]["mz"] if max_peaks and len(n) > max_peaks)
        print(f"[alignment] dropped {dropped:,} of {before:,} pairs "
              f"({100 * dropped / before:.1f}%); {oversized:,} were over max_peaks="
              f"{max_peaks}. Peak count tracks charge and peptide length, so this is a "
              f"biased loss, not a random one.", flush=True)

    peptides = sorted(set(rows["peptide"]))
    generator = np.random.default_rng(seed)
    held_out = set(generator.choice(
        peptides, size=max(1, int(len(peptides) * validation_fraction)), replace=False
    ).tolist())
    return {
        "train": rows.filter(lambda e: e["peptide"] not in held_out, desc="train split"),
        "validation": rows.filter(lambda e: e["peptide"] in held_out, desc="validation split"),
    }



def hard_negatives(peptide: str, rng, k: int, min_delta: float = 0.05,
                   windows=(3, 4)) -> list[str]:
    """Up to k spectrally DISTINGUISHABLE rearrangements of a peptide (PLAN.md A4).

    Adjacent swaps and local shuffles only -- never reversals, which is how the FDR
    decoys are made, so a student trained on these cannot learn "decoy-looking".
    A rearrangement changes the masses of the b-ions (and matching y-ions) that end
    inside it; it is kept only if at least one of those shifts by more than min_delta Da
    (the fragment tolerance: 0.05 Da for ion-trap MS2), so I<->L, K<->Q (0.036 Da) and
    identical residues never become negatives. The C-terminal residue never moves;
    modifications stay attached to their residue.
    """
    from msdelta.chemistry import RESIDUE_MASSES
    from msdelta.rescoring import split_peptide
    residues, mods = split_peptide(peptide)
    n = len(residues)
    if n < 3:
        return []
    mass = [RESIDUE_MASSES.get(r, 0.0) + m for r, m in zip(residues, mods)]

    def fmt(order):
        return "".join(residues[i] + (f"[{mods[i]:.4f}]" if abs(mods[i]) > 1e-6 else "")
                       for i in order)

    def distinguishable(order):
        prefix_old = prefix_new = 0.0
        for j in range(n - 1):
            prefix_old += mass[j]; prefix_new += mass[order[j]]
            if abs(prefix_new - prefix_old) > min_delta:
                return True
        return False

    pool = set()
    base = list(range(n))
    for i in range(n - 2):                                  # adjacent swaps, C-term fixed
        o = base.copy(); o[i], o[i + 1] = o[i + 1], o[i]
        if distinguishable(o):
            pool.add(tuple(o))
    for _ in range(4 * k):                                  # local shuffles
        w = int(rng.choice(windows))
        if n - 1 < w:
            continue
        i = int(rng.integers(0, n - w))                     # window within [0, n-1)
        seg = [int(x) for x in rng.permutation(base[i:i + w])]
        o = base[:i] + seg + base[i + w:]
        if o != base and distinguishable(o):
            pool.add(tuple(o))
    own = fmt(base)
    cands = sorted({fmt(o) for o in pool} - {own})
    if len(cands) > k:
        cands = [cands[i] for i in rng.choice(len(cands), k, replace=False)]
    return cands


# ------------------------------------------------------------------ A8: mass-aware training
# Deployed pipelines filter candidates by precursor mass first, so the student is used to
# separate peptides of (nearly) the SAME mass. Random batch negatives are almost never that.

def peptide_neutral_mass(peptide: str) -> float:
    """Monoisotopic neutral mass of a peptide in our notation (`C[57.0215]`, `[42.0106]P`)."""
    from msdelta.chemistry import RESIDUE_MASSES
    mods = sum(float(x) for x in re.findall(r"\[([-+]?\d+\.?\d*)\]", peptide))
    residues = re.sub(r"\[[^\]]*\]", "", peptide)
    return sum(RESIDUE_MASSES[r] for r in residues) + mods + 18.010565


def _il(peptide: str) -> str:
    """I/L-collapsed, modification-stripped sequence: peptides equal here give the same
    fragment masses up to their modifications, so they are never negatives of each other."""
    return re.sub(r"\[[^\]]*\]", "", peptide).replace("I", "L")


class MassNegativePool:
    """Training peptides sorted by neutral mass, for same-mass hard negatives."""

    def __init__(self, peptides):
        uniq = sorted(set(peptides))
        m = np.array([peptide_neutral_mass(p) for p in uniq])
        order = np.argsort(m, kind="stable")
        self.masses = m[order]
        self.peptides = [uniq[i] for i in order]

    def negatives(self, peptide: str, rng, k: int, ppm: float = 20.0) -> list[str]:
        """Up to k OTHER peptides within +-ppm of this one's mass (I/L-equivalents excluded)."""
        mass = peptide_neutral_mass(peptide)
        tol = mass * ppm * 1e-6
        lo = np.searchsorted(self.masses, mass - tol, "left")
        hi = np.searchsorted(self.masses, mass + tol, "right")
        own = _il(peptide)
        cands = [self.peptides[i] for i in range(lo, hi) if _il(self.peptides[i]) != own]
        if len(cands) <= k:
            return cands
        return [cands[i] for i in rng.choice(len(cands), k, replace=False)]


class MassBatchSampler(Sampler):
    """Batches of rows that are NEIGHBOURS IN MASS: each epoch, masses + uniform jitter
    (+-jitter Da) are sorted and cut into consecutive batches, whose order is shuffled. So
    in-batch negatives are near-same-mass competitors, like the candidates left after a
    precursor window. Reshuffles itself every epoch (the epoch counter advances inside
    __iter__; HF Trainer never calls set_epoch on a custom batch_sampler -- FT14)."""

    def __init__(self, masses, batch_size: int, jitter: float = 0.5, seed: int = 0,
                 drop_last: bool = False):
        self.masses = np.asarray(masses, dtype=np.float64)
        self.batch_size, self.jitter, self.seed, self.drop_last = batch_size, jitter, seed, drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        n = len(self.masses) // self.batch_size
        return n if self.drop_last or len(self.masses) % self.batch_size == 0 else n + 1

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        self.epoch += 1
        key = self.masses + rng.uniform(-self.jitter, self.jitter, len(self.masses))
        order = np.argsort(key, kind="stable")
        batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        for j in rng.permutation(len(batches)):
            yield batches[j].tolist()

@dataclass
class AlignmentCollator:
    """Pad spectra and their peptides into one batch."""

    max_peptide_length: int = 64
    # A4: per row, this many distinguishable rearrangements of its peptide (0 = none,
    # the A1 behaviour), plus a within-batch peptide group id for multi-positive loss.
    hard_negatives: int = 0
    neg_min_delta: float = 0.05
    neg_seed: int = 0
    # A8: "swap" = distinguishable rearrangements (A4); "mass" = other TRAINING peptides
    # within +-neg_ppm of the peptide's mass (needs neg_pool, a MassNegativePool).
    neg_source: str = "swap"
    neg_ppm: float = 20.0
    neg_pool: object = None

    # Only meaningful on the cached-target path, where it makes every batch the same
    # shape; with a live teacher the spectra dominate and vary anyway.
    fixed_shapes: bool = True

    # 0 keeps the historical behaviour of padding to the batch maximum. Set it to
    # max_peaks to make activation memory deterministic; see __call__.
    pad_spectra_to: int = 0

    def __post_init__(self):
        self.peptides = PeptideCollator(max_length=self.max_peptide_length)
        self.padded = PeptideCollator(max_length=self.max_peptide_length, pad_to_max=True)
        self._rng = np.random.default_rng(self.neg_seed)

    def _negatives(self, features) -> dict[str, Tensor]:
        k = self.hard_negatives
        peps = [f["peptide"] for f in features]
        charges = [int(f.get("charge", 0)) for f in features]
        flat, valid = [], []
        for pep in peps:
            if self.neg_source == "mass":
                neg = self.neg_pool.negatives(pep, self._rng, k, self.neg_ppm)
            else:
                neg = hard_negatives(pep, self._rng, k, self.neg_min_delta)
            valid.append([True] * len(neg) + [False] * (k - len(neg)))
            flat += neg + [pep] * (k - len(neg))            # padding rows are masked out
        tok = self.padded(flat, [c for c in charges for _ in range(k)])
        group = {p: i for i, p in enumerate(dict.fromkeys(peps))}
        return {**{f"neg_{name}": v for name, v in tok.items()},
                "neg_valid": torch.tensor(valid, dtype=torch.bool),
                "peptide_group": torch.tensor([group[p] for p in peps], dtype=torch.long)}

    def __call__(self, features: list[dict]) -> dict[str, Tensor]:
        if not features:
            raise ValueError("features must not be empty")
        lengths = [len(f["mz"]) for f in features]
        # Fixed width when asked for, otherwise the longest spectrum in the batch.
        #
        # Padding to the batch maximum makes MEMORY DATA-DEPENDENT, and DeltaMZBias is
        # O(batch * width^2), so one wide spectrum quadruples a batch's cost against a
        # narrow one. Which spectra land together is decided by the sampler's seed, so
        # the same model and config reserved 38.75 GB at seed 0 and 67.14 GB at seed 3
        # -- and at 200m and 400m the wide draws exceeded the tile and took a GPU page
        # fault. Every seed-0 arm survived and every other seed died, at every scale.
        #
        # A fixed width costs the padding on narrow batches and buys a memory figure
        # that can be measured once and trusted. FT9 saw the same effect from the other
        # side: fixed-shape batches moved its fault from step 22 to step 120.
        width = self.pad_spectra_to or max(max(lengths), 1)
        batch = len(features)
        mz = torch.zeros(batch, width, dtype=torch.float32)
        log_intensity = torch.zeros_like(mz)
        attention_mask = torch.zeros(batch, width, dtype=torch.long)
        for row, (feature, length) in enumerate(zip(features, lengths)):
            if length == 0:
                continue
            mz[row, :length] = torch.as_tensor(feature["mz"], dtype=torch.float32)
            log_intensity[row, :length] = torch.as_tensor(feature["log_intensity"],
                                                          dtype=torch.float32)
            attention_mask[row, :length] = 1
        peptide_batch = self.peptides([f["peptide"] for f in features],
                                      [int(f.get("charge", 0)) for f in features])
        if features[0].get("target") is not None:
            # Precomputed target: the spectrum columns are DROPPED, not merely unused.
            #
            # The model does not read them once a target is supplied, but the Trainer
            # moves every tensor in the batch to the device regardless, so leaving them
            # in ships a (batch x up-to-512) float tensor per step that nothing touches.
            # Worse than the waste, it is padded to the widest spectrum in the batch, so
            # the shape changes from step to step -- and variable-shape device
            # allocations are exactly the churn that made the denoise runs unstable
            # before fixed batching. Evaluation does not need them either: it reads the
            # target from the model's own output, which comes from this cached column.
            # Fixed width too, so EVERY tensor in the batch has the same shape on every
            # step. The bisect (job 8840444) ran the full student on twelve tiles for 20
            # steps without a fault using fixed-shape input, while the real job faults at
            # step 0 with variable-shape input -- so shape variation is the difference
            # worth removing, and padding to 64 costs a few unused positions on a tensor
            # that is already tiny.
            padded = (self.padded if self.fixed_shapes else self.peptides)(
                [f["peptide"] for f in features],
                [int(f.get("charge", 0)) for f in features])
            out = {"target": torch.as_tensor([f["target"] for f in features],
                                             dtype=torch.float32), **padded}
            if self.hard_negatives:
                out |= self._negatives(features)
            return out
        return {"mz": mz, "log_intensity": log_intensity,
                "attention_mask": attention_mask, **peptide_batch}


@torch.no_grad()
def teacher_embedding_size(spectrum_model, pooling: str, collator=None) -> int:
    """Width of the frozen teacher's output, measured rather than assumed.

    The student's projection has to match it exactly, and the width depends on both the
    encoder's hidden size and the pooling mode, so guessing is easy to get wrong.
    """
    collator = collator or AlignmentCollator()
    was_training = spectrum_model.training
    spectrum_model.eval()
    batch = collator([{"mz": [100.0, 200.0], "log_intensity": [1.0, 1.0],
                       "peptide": "AK", "charge": 0}])
    device = next(spectrum_model.parameters()).device
    encoder = getattr(spectrum_model, "msdelta", spectrum_model)
    try:
        hidden = encoder(mz=batch["mz"].to(device),
                         log_intensity=batch["log_intensity"].to(device),
                         attention_mask=batch["attention_mask"].to(device)).last_hidden_state
        return int(pool_sequence(hidden, batch["attention_mask"].to(device), pooling).shape[-1])
    finally:
        spectrum_model.train(was_training)


def build_alignment_model(spectrum_model, pooling="mean+max", hidden_size=256,
                          num_layers=4, num_heads=8, dropout=0.1, max_peptide_length=64):
    """Attach a peptide encoder sized to whatever teacher was supplied."""
    embedding_size = teacher_embedding_size(spectrum_model, pooling)
    encoder = PeptideEncoder(embedding_size=embedding_size, hidden_size=hidden_size,
                             num_layers=num_layers, num_heads=num_heads,
                             max_length=max_peptide_length, dropout=dropout,
                             pooling=pooling)
    return SequenceAlignmentModel(spectrum_model, encoder, pooling=pooling)
