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

import os
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

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


POOLING_MODES = ("mean", "mean+max")


def pool_sequence(tokens: Tensor, mask: Tensor, mode: str = "mean+max") -> Tensor:
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
    mean = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
    if mode == "mean":
        return mean
    maximum = torch.nan_to_num(tokens.masked_fill(~mask, float("-inf")).max(1).values,
                               neginf=0.0)
    return torch.cat([mean, maximum], dim=-1)


def pooled_width(hidden_size: int, mode: str) -> int:
    """Width `pool_sequence` produces, so the projection can be sized without a forward."""
    return hidden_size if mode == "mean" else 2 * hidden_size


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

    def __call__(self, peptides: list[str], charges: list[int]) -> dict[str, Tensor]:
        parsed = [parse_peptide(p) for p in peptides]
        width = max(min(max((len(i) for i, _ in parsed), default=1), self.max_length), 1)
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
    ):
        super().__init__()
        self.pooling = pooling
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
        width = pooled_width(hidden_size, pooling)
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
            hidden = self.norm(
                self.encoder(hidden, src_key_padding_mask=~sequence_mask.bool())
            )
        probe("student.encoded", sync=hidden, hidden=hidden)
        pooled = pool_sequence(hidden, sequence_mask, self.pooling)
        out = F.normalize(self.projection(pooled).float(), dim=-1)
        probe("student.out", sync=out, out=out)
        return out


class SequenceAlignmentModel(nn.Module):
    """Frozen spectrum teacher, trainable peptide student, L2 between them."""

    def __init__(self, spectrum_model: nn.Module | None, sequence_encoder: PeptideEncoder,
                 pooling: str = "mean+max"):
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

    @torch.no_grad()
    def embed_spectrum(self, mz, log_intensity, attention_mask) -> Tensor:
        encoder = getattr(self.spectrum_model, "msdelta", self.spectrum_model)
        hidden = encoder(mz=mz, log_intensity=log_intensity,
                         attention_mask=attention_mask).last_hidden_state
        return F.normalize(pool_sequence(hidden, attention_mask, self.pooling).float(), dim=-1)

    def forward(self, residues, modifications, sequence_mask, charge,
                mz=None, log_intensity=None, attention_mask=None, target=None,
                return_dict: bool = True, return_loss: bool = True):
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
        loss = ((predicted - target) ** 2).sum(dim=-1).mean()
        if not return_dict:
            return (loss, predicted, target)
        return {"loss": loss, "embeddings": predicted, "target": target}



@torch.no_grad()
def attach_teacher_embeddings(datasets: dict, spectrum_model: nn.Module, pooling: str,
                              batch_size: int = 16, max_peptide_length: int = 64,
                              device: str | torch.device | None = None) -> dict:
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
    collator = AlignmentCollator(max_peptide_length=max_peptide_length)

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
            return {"mz": [], "log_intensity": [], "peptide": "", "charge": 0}
        try:
            values = processor(
                torch.as_tensor(example["mz"], dtype=torch.float32),
                torch.as_tensor(example["intensity"], dtype=torch.float32),
                padding=False,
            )
        except (ValueError, KeyError, TypeError):
            return {"mz": [], "log_intensity": [], "peptide": "", "charge": 0}
        return {
            "mz": values["mz"][0] if values["mz"] and isinstance(values["mz"][0], list)
                  else values["mz"],
            "log_intensity": (values["log_intensity"][0]
                              if values["log_intensity"]
                              and isinstance(values["log_intensity"][0], list)
                              else values["log_intensity"]),
            "peptide": example.get("peptide") or "",
            "charge": int(example.get("charge") or 0),
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


@dataclass
class AlignmentCollator:
    """Pad spectra and their peptides into one batch."""

    max_peptide_length: int = 64

    def __post_init__(self):
        self.peptides = PeptideCollator(max_length=self.max_peptide_length)

    def __call__(self, features: list[dict]) -> dict[str, Tensor]:
        if not features:
            raise ValueError("features must not be empty")
        lengths = [len(f["mz"]) for f in features]
        width = max(max(lengths), 1)
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
        out = {"mz": mz, "log_intensity": log_intensity,
               "attention_mask": attention_mask, **peptide_batch}
        # A precomputed teacher embedding, if attach_teacher_embeddings has been run. The
        # spectrum columns are still emitted: they cost little and evaluation reuses them.
        if features[0].get("target") is not None:
            out["target"] = torch.as_tensor(
                [f["target"] for f in features], dtype=torch.float32)
        return out


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
