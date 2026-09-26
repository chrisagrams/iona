"""Map a peptide sequence into the spectrum encoder's embedding space.

A peptide encoder is trained against a frozen spectrum encoder (the teacher), with L2 on
unit vectors so the loss matches cosine retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor, nn

from msdelta.fourier import FourierFeatures

# 20 standard residues plus `n`, which this corpus uses as an N-terminal marker.
RESIDUES = "ACDEFGHIKLMNPQRSTVWYn"
PAD, UNK = 0, 1
RESIDUE_TO_ID = {residue: index + 2 for index, residue in enumerate(RESIDUES)}
VOCAB_SIZE = len(RESIDUE_TO_ID) + 2


POOLING_MODES = ("mean", "mean+max")


def pool_sequence(tokens: Tensor, mask: Tensor, mode: str = "mean+max") -> Tensor:
    """Reduce variable-length token embeddings to one vector. Both towers must use the same mode."""
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
    return 2 * hidden_size if mode == "mean+max" else hidden_size


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
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        width = pooled_width(hidden_size, pooling)
        self.projection = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, embedding_size),
        )

    def forward(self, residues, modifications, sequence_mask, charge) -> Tensor:
        length = residues.shape[1]
        position = torch.arange(length, device=residues.device).clamp_max(
            self.position.num_embeddings - 1
        )
        hidden = self.residue(residues) + self.position(position)[None]
        # Only modified residues get a mass contribution.
        modified = (modifications.abs() > 1e-6).unsqueeze(-1)
        mod = self.mod_projection(self.mod_features(modifications).to(hidden.dtype))
        hidden = hidden + mod * modified
        hidden = hidden + self.charge(charge.clamp(0, self.charge.num_embeddings - 1))[:, None]
        # Autocast off, activations cast to the weights' dtype: the fused eval fast path
        # ignores autocast on XPU and fails on a dtype mismatch.
        param_dtype = next(self.encoder.parameters()).dtype
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.to(param_dtype)
            hidden = self.norm(self.encoder(hidden, src_key_padding_mask=~sequence_mask.bool()))
        pooled = pool_sequence(hidden, sequence_mask, self.pooling)
        return F.normalize(self.projection(pooled).float(), dim=-1)


@torch.no_grad()
def embed_spectrum(spectrum_model, mz, log_intensity, attention_mask, pooling: str) -> Tensor:
    """Unit-length pooled embedding of a batch of spectra from a frozen MSDeltaForPreTraining."""
    hidden = spectrum_model.msdelta(mz=mz, log_intensity=log_intensity,
                                    attention_mask=attention_mask).last_hidden_state
    return F.normalize(pool_sequence(hidden, attention_mask, pooling).float(), dim=-1)


class SequenceAlignmentModel(nn.Module):
    """Peptide student trained onto precomputed teacher targets with L2 on unit vectors."""

    def __init__(self, sequence_encoder: PeptideEncoder):
        super().__init__()
        self.sequence_encoder = sequence_encoder

    def forward(self, residues, modifications, sequence_mask, charge, target):
        target = F.normalize(target.float(), dim=-1)
        predicted = self.sequence_encoder(residues, modifications, sequence_mask, charge)
        # Mean squared L2; equals 2 - 2cos on unit vectors.
        loss = ((predicted - target) ** 2).sum(dim=-1).mean()
        return {"loss": loss, "embeddings": predicted, "target": target}


@torch.no_grad()
def attach_teacher_embeddings(datasets: dict, spectrum_model: nn.Module, pooling: str,
                              batch_size: int = 16, max_peptide_length: int = 64,
                              device: str | torch.device | None = None,
                              pad_spectra_to: int = 0) -> dict:
    """Run the frozen teacher once and store its embedding as a `target` column."""
    device = device or ("xpu" if torch.xpu.is_available() else "cpu")
    spectrum_model.to(device).eval()
    collator = AlignmentCollator(max_peptide_length=max_peptide_length,
                                 pad_spectra_to=pad_spectra_to)

    def embed(batch: dict) -> dict:
        rows = [{"mz": mz, "log_intensity": li, "peptide": pep, "charge": ch}
                for mz, li, pep, ch in zip(batch["mz"], batch["log_intensity"],
                                           batch["peptide"], batch["charge"])]
        inputs = collator(rows)
        target = embed_spectrum(spectrum_model, inputs["mz"].to(device),
                                inputs["log_intensity"].to(device),
                                inputs["attention_mask"].to(device), pooling)
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


def build_alignment_datasets(repo_id, processor, num_proc=None, validation_fraction=0.1,
                             seed=0):
    """Spectrum/sequence pairs, split by peptide."""
    raw = load_dataset(repo_id)
    split = "train" if "train" in raw else list(raw)[0]

    # Spectra above max_peaks are dropped, not truncated, and the count is printed.
    max_peaks = processor.max_peaks

    def prepare(example):
        if len(example["mz"]) > max_peaks:
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
        oversized = sum(1 for n in raw[split]["mz"] if len(n) > max_peaks)
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
    """Pad spectra and their peptides into one batch; rows with a precomputed `target` skip the spectra."""

    max_peptide_length: int = 64
    # 0 pads spectra to the batch maximum; max_peaks gives a fixed width.
    pad_spectra_to: int = 0

    def __post_init__(self):
        self.peptides = PeptideCollator(max_length=self.max_peptide_length)
        self.padded = PeptideCollator(max_length=self.max_peptide_length, pad_to_max=True)

    def __call__(self, features: list[dict]) -> dict[str, Tensor]:
        if not features:
            raise ValueError("features must not be empty")
        peptides = [f["peptide"] for f in features]
        charges = [int(f.get("charge", 0)) for f in features]
        if features[0].get("target") is not None:
            target = torch.as_tensor([f["target"] for f in features], dtype=torch.float32)
            return {"target": target, **self.padded(peptides, charges)}
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
        return {"mz": mz, "log_intensity": log_intensity,
                "attention_mask": attention_mask, **self.peptides(peptides, charges)}
