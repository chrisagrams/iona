"""Map a peptide sequence into the spectrum encoder's embedding space.

A peptide encoder is trained against a frozen spectrum encoder (the teacher), with L2 on
unit vectors so the loss matches cosine retrieval.
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

# Debug checks, enabled with MSDELTA_PROBES=1.
PROBES = os.environ.get("MSDELTA_PROBES", "") not in ("", "0", "false", "False")

# Force SDPA onto the unfused MATH backend (workaround for fused-kernel page faults on XPU).
SDPA_MATH = os.environ.get("MSDELTA_SDPA_MATH", "") not in ("", "0", "false", "False")


def _sdpa_context():
    """MATH-only SDPA when MSDELTA_SDPA_MATH is set, otherwise a no-op."""
    if not SDPA_MATH:
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH)


def probe(where: str, *, sync: Tensor | None = None, **tensors) -> None:
    """Check tensors for non-finite values; `sync` blocks until the device queue drains."""
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
    """Raise on out-of-range embedding indices, which fault on GPU instead of raising."""
    if not PROBES or index.numel() == 0:
        return
    low, high = int(index.min()), int(index.max())
    if low < 0 or high >= limit:
        raise RuntimeError(
            f"[probe {where}] {name} out of range for a table of {limit}: "
            f"min {low}, max {high}"
        )


# "weighted" variants weight each token (e.g. by intensity) before averaging.
POOLING_MODES = ("mean", "mean+max", "weighted_mean", "weighted_mean+max")


def pool_sequence(tokens: Tensor, mask: Tensor, mode: str = "mean+max",
                  weights: Tensor | None = None) -> Tensor:
    """Reduce variable-length token embeddings to one vector. Both towers must use the same mode."""
    if mode not in POOLING_MODES:
        raise ValueError(f"pooling must be one of {POOLING_MODES}, got {mode!r}")
    mask = mask.bool().unsqueeze(-1)
    if mode.startswith("weighted"):
        if weights is None:
            raise ValueError(f"pooling {mode!r} needs per-peak weights")
        # Normalised to stay on the same scale as the unweighted mean.
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
    return 2 * hidden_size if mode.endswith("mean+max") else hidden_size


def parse_peptide(peptide: str) -> tuple[list[int], list[float]]:
    """Split a modified peptide into residue ids and per-residue modification masses.

    A bracketed mass binds to the preceding residue, e.g. `SAC[57.0215]GVC[57.0215]PGR`.
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
                # A leading bracket attaches to the N-terminal marker.
                ids.append(RESIDUE_TO_ID["n"])
                masses.append(mass)
            index = end + 1
            continue
        ids.append(RESIDUE_TO_ID.get(char, UNK))
        masses.append(0.0)
        index += 1
    return ids, masses


def peptide_key(peptide: str, charge: int, by_charge: bool = True) -> str:
    """Identity used to decide whether two spectra are the same sequence (charge-aware by default)."""
    return f"{peptide}_{charge}" if by_charge else peptide


@dataclass
class PeptideCollator:
    """Pad parsed peptides into a batch."""

    max_length: int = 64
    # Pad to max_length rather than the longest peptide, for fixed shapes.
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
        # "pool": `pooling` over tokens; "cls": a learned prepended token; "attn": a learned
        # query attending over tokens.
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
        # Only modified residues get a mass contribution.
        modified = (modifications.abs() > 1e-6).unsqueeze(-1)
        mod = self.mod_projection(self.mod_features(modifications).to(hidden.dtype))
        hidden = hidden + mod * modified
        hidden = hidden + self.charge(charge.clamp(0, self.charge.num_embeddings - 1))[:, None]
        if self.readout == "cls":
            hidden = torch.cat([self.cls.expand(hidden.shape[0], -1, -1).to(hidden.dtype),
                                hidden], dim=1)
            sequence_mask = torch.cat([torch.ones_like(sequence_mask[:, :1]),
                                       sequence_mask], dim=1)
        # Autocast off, activations cast to the weights' dtype: the fused eval fast path
        # ignores autocast on XPU and fails on a dtype mismatch.
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
    """Cross-modal SupCon: spectrum targets as anchors, in-batch peptides plus hard negatives as candidates."""
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
    """Which PeptideEncoder readout a saved student used, inferred from its weight names."""
    keys = {k[len(prefix):] if k.startswith(prefix) else k for k in state}
    return "cls" if "cls" in keys else "attn" if "attn_query" in keys else "pool"


class SequenceAlignmentModel(nn.Module):
    """Frozen spectrum teacher, trainable peptide student, L2 between them."""

    def __init__(self, spectrum_model: nn.Module | None, sequence_encoder: PeptideEncoder,
                 pooling: str = "mean+max", loss: str = "mse", temperature: float = 0.05,
                 mse_weight: float = 0.0):
        """`spectrum_model=None` trains against precomputed targets."""
        super().__init__()
        self.spectrum_model = spectrum_model
        self.sequence_encoder = sequence_encoder
        if pooling != sequence_encoder.pooling:
            raise ValueError(
                f"teacher pooling {pooling!r} != student pooling {sequence_encoder.pooling!r}"
            )
        self.pooling = pooling
        # "mse": regress onto the teacher embedding. "lit": cross-modal contrastive against
        # the frozen teacher, plus mse_weight x the MSE term.
        if loss not in ("mse", "lit"):
            raise ValueError(f"loss must be mse or lit, not {loss!r}")
        self.loss, self.temperature, self.mse_weight = loss, temperature, mse_weight
        # Frozen and in eval mode, so dropout does not vary the targets.
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
        # return_loss is unused; its presence lets Trainer's can_return_loss() report an eval loss.
        if target is None:
            if self.spectrum_model is None:
                raise ValueError("no teacher and no precomputed target in the batch")
            target = self.embed_spectrum(mz, log_intensity, attention_mask)
            probe("teacher.out", sync=target, target=target)
        else:
            target = F.normalize(target.float(), dim=-1)
        predicted = self.sequence_encoder(residues, modifications, sequence_mask, charge)
        probe("loss.in", sync=predicted, predicted=predicted, target=target)
        # Mean squared L2; equals 2 - 2cos on unit vectors.
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

    Call on one process before the Trainer is built; the teacher then stays out of the wrapped module.
    """
    model = SequenceAlignmentModel(spectrum_model, PeptideEncoder(
        embedding_size=1, hidden_size=8, num_layers=1, num_heads=1, pooling=pooling),
        pooling=pooling)
    device = device or ("xpu" if torch.xpu.is_available() else "cpu")
    model.spectrum_model.to(device).eval()
    # pad_spectra_to=max_peaks gives fixed-width batches and deterministic memory.
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
        probe("precompute.target", sync=target, target=target)
        return {"target": target.float().cpu().tolist()}

    return {name: split.map(embed, batched=True, batch_size=batch_size,
                            desc=f"teacher embeddings ({name})")
            for name, split in datasets.items()}



@torch.no_grad()
def group_separation_metrics(embeddings: Tensor, groups: np.ndarray,
                             prefix: str = "sep") -> dict[str, float]:
    """Do replicates of one peptide sit closer together than to other peptides?

    Squared euclidean on unit vectors (range [0, 4]). `clean` is the fraction of groups fully separated.
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
        f"{prefix}/margin": float(outside.mean() - inside.mean()),
        # Scale-invariant, unlike margin.
        f"{prefix}/ratio": float(outside.mean() / inside.mean().clamp_min(1e-9)),
        f"{prefix}/clean": clean / max(len(worst_in), 1),
        f"{prefix}/worst_in_mean": float(np.mean(worst_in)) if worst_in else 0.0,
        f"{prefix}/best_out_mean": float(np.mean(best_out)) if best_out else 0.0,
        f"{prefix}/groups": float(len(worst_in)),
    }


@torch.no_grad()
def cross_modal_metrics(sequence_embeddings, spectrum_embeddings, spectrum_groups,
                        sequence_groups=None) -> dict[str, float]:
    """Rank candidate sequences against each spectrum.

    `spectrum_groups` gives each spectrum's correct candidate; `sequence_groups` gives each candidate's identity.
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
    """Spectrum/sequence pairs, split by peptide."""
    from datasets import load_dataset

    raw = load_dataset(repo_id)
    split = "train" if "train" in raw else list(raw)[0]

    # Spectra above max_peaks are dropped, not truncated, and the count is printed.
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
            # Measured precursor m/z, used by the rescorer.
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
    """Up to k spectrally distinguishable rearrangements of a peptide.

    Adjacent swaps and local shuffles (never reversals); the C-terminal residue stays fixed.
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


# ------------------------------------------------------------------ mass-aware training

def peptide_neutral_mass(peptide: str) -> float:
    """Monoisotopic neutral mass of a peptide in our notation (`C[57.0215]`, `[42.0106]P`)."""
    from msdelta.chemistry import RESIDUE_MASSES
    mods = sum(float(x) for x in re.findall(r"\[([-+]?\d+\.?\d*)\]", peptide))
    residues = re.sub(r"\[[^\]]*\]", "", peptide)
    return sum(RESIDUE_MASSES[r] for r in residues) + mods + 18.010565


def _il(peptide: str) -> str:
    """I/L-collapsed, modification-stripped sequence."""
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
    """Batches of rows that are neighbours in mass (with jitter), reshuffled every epoch.

    The epoch advances inside __iter__, since the Trainer never calls set_epoch on a custom batch_sampler.
    """

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
    # Hard negatives per row (0 = none).
    hard_negatives: int = 0
    neg_min_delta: float = 0.05
    neg_seed: int = 0
    # "swap": rearrangements; "mass": training peptides within neg_ppm (needs neg_pool).
    neg_source: str = "swap"
    neg_ppm: float = 20.0
    neg_pool: object = None

    # Fixed peptide shapes on the cached-target path.
    fixed_shapes: bool = True

    # 0 pads spectra to the batch maximum; max_peaks gives a fixed width.
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
            # Precomputed target: drop the spectrum columns.
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
    """Width of the frozen teacher's output, measured with a dummy forward."""
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
