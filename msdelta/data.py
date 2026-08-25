"""Parquet dataset, per-spectrum preprocessing, and MPM collate.

Everything goes through `build_pretraining_datasets` / `build_preprocessed_dataset`,
which load the parquet as a Hugging Face `datasets.Dataset` and precompute
`preprocess_spectrum` with `.map()` (cached to Arrow) — so no per-spectrum
transform runs during training (only the cheap random masking in
`MaskIntensityCollator`) or distributed diagnostics (which read the same
preprocessed rows via `EmbeddingEvalCollator`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch


@dataclass
class PreprocessConfig:
    intensity_threshold_frac: float = 0.01
    top_n: int = 150


N_CHARGES = 8  # embedding rows; index 0 = unknown/default, 1..N-1 = charge z (clamped)

# Monoisotopic residue masses (Da); for computing precursor m/z from the label.
_RESIDUE_MASS = {
    "G": 57.02146, "A": 71.03711, "S": 87.03203, "P": 97.05276, "V": 99.06841,
    "T": 101.04768, "C": 103.00919, "L": 113.08406, "I": 113.08406, "N": 114.04293,
    "D": 115.02694, "Q": 128.05858, "K": 128.09496, "E": 129.04259, "M": 131.04049,
    "H": 137.05891, "F": 147.06841, "R": 156.10111, "Y": 163.06333, "W": 186.07931,
}
_WATER = 18.0105646
_PROTON = 1.0072765
_MOD_RE = re.compile(r"\[([+-]?[0-9.]+)\]")  # sign-tolerant: [15.995] and [+15.995]/[-17.027]


def charge_index(peptide_charge: str | None) -> int:
    """Parse charge from 'PEPTIDE_z' → index in [0, N_CHARGES). 0 = unknown."""
    if not peptide_charge:
        return 0
    parts = str(peptide_charge).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return min(int(parts[1]), N_CHARGES - 1)
    return 0


def precursor_mz(peptide_charge: str | None) -> float:
    """Compute precursor m/z from 'PEPTIDE_z' (mod-aware: sums [x] bracket
    masses + residues + water, /z). Returns 0.0 if unparseable.

    Precursor m/z is an *observed* input at deployment (instrument PEPMASS);
    here we source it from the label for the consensus set, where the observed
    value isn't stored. Same physical quantity — not leakage.
    """
    if not peptide_charge:
        return 0.0
    parts = str(peptide_charge).rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return 0.0
    pep, z = parts[0], int(parts[1])
    if z < 1:
        return 0.0
    mods = sum(float(x) for x in _MOD_RE.findall(pep))
    seq = _MOD_RE.sub("", pep)
    mass = _WATER + mods
    for a in seq:
        m = _RESIDUE_MASS.get(a)
        if m is None:
            return 0.0
        mass += m
    return (mass + z * _PROTON) / z


def preprocess_spectrum(
    mz: torch.Tensor,
    intensity: torch.Tensor,
    cfg: PreprocessConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop low-intensity peaks, take top-N, produce two intensity views.

    Returns (mz, log_int, intensity_prob), all float32 of length K ≤ cfg.top_n:
      - log_int        log1p(intensity) divided by its per-spectrum max — the
                       input feature embedded by `PeakEmbed` (concentrated in
                       [~0.5, 1.0] by construction, which is fine for the input).
      - intensity_prob raw intensity normalised to sum-to-1 across the kept
                       peaks — the KL target for `IntensityHead.loss`. Treating
                       the spectrum as a probability distribution over m/z is
                       what makes isotope/residue intensity ratios (e.g. M+1 ≈
                       20% M+0 for ¹³C) first-class in the loss.
    """
    mz = mz.to(torch.float32)
    intensity = intensity.to(torch.float32)

    if intensity.numel() == 0:
        return mz, intensity, intensity

    base = float(intensity.max())
    if base <= 0:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty, empty

    keep = intensity >= cfg.intensity_threshold_frac * base
    mz = mz[keep]
    intensity = intensity[keep]

    if mz.numel() > cfg.top_n:
        topk = torch.topk(intensity, cfg.top_n, sorted=False)
        mz = mz[topk.indices]
        intensity = topk.values

    log_int = torch.log1p(intensity)
    log_int = log_int / log_int.max().clamp_min(1e-8)

    intensity_prob = intensity / intensity.sum().clamp_min(1e-12)

    return mz.contiguous(), log_int.contiguous(), intensity_prob.contiguous()


def split_paths(root: str | Path, n_val: int) -> tuple[list[Path], list[Path]]:
    """Deterministic file-level split: last n_val shards (sorted) become val."""
    root = Path(root)
    shards = sorted(root.glob("*.parquet"))
    if len(shards) <= n_val:
        raise ValueError(f"only {len(shards)} shards; need > {n_val} for a split")
    return shards[:-n_val], shards[-n_val:]


def hf_split_paths(
    repo_id: str,
    train_split: str = "train",
    val_split: str = "val",
) -> tuple[list[Path], list[Path]]:
    """Resolve (train, val) parquet shard paths from a Hugging Face dataset.

    The dataset is expected to lay its shards out under per-split subdirectories
    (``train/*.parquet``, ``val/*.parquet``) with the same column schema as the
    local consensus parquets (``m/z``, ``int``, ``peptide_charge``) — so the
    downloaded shards drop straight into `build_preprocessed_dataset`.

    Only the two requested splits are pulled (``allow_patterns``); the snapshot is
    served from the HF cache on subsequent calls, so this is cheap to call more
    than once (e.g. for both the loaders and the inline probe). Auth for private
    repos comes from a cached `huggingface-cli login` or the HF_TOKEN env var.
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id,
        repo_type="dataset",
        allow_patterns=[f"{train_split}/*.parquet", f"{val_split}/*.parquet"],
    )
    root = Path(local_dir)
    train_paths = sorted((root / train_split).glob("*.parquet"))
    val_paths = sorted((root / val_split).glob("*.parquet"))
    if not train_paths:
        raise ValueError(f"no parquet shards under '{train_split}/' in {repo_id}")
    if not val_paths:
        raise ValueError(f"no parquet shards under '{val_split}/' in {repo_id}")
    return train_paths, val_paths


def resolve_dataset_paths(dcfg: dict) -> tuple[list[Path], list[Path]]:
    """Return (train_paths, val_paths) from either a local parquet root or a
    Hugging Face dataset, selected by config.

    HF source  — set ``data.hf_repo`` (plus optional ``data.hf_train_split`` /
                 ``data.hf_val_split``, default ``train`` / ``val``).
    Local source — set ``data.root`` and ``data.n_val_files`` (last N sorted
                 shards become validation). This is the original behaviour and
                 the default when ``hf_repo`` is absent.
    """
    if dcfg.get("hf_repo"):
        return hf_split_paths(
            dcfg["hf_repo"],
            train_split=dcfg.get("hf_train_split", "train"),
            val_split=dcfg.get("hf_val_split", "val"),
        )
    return split_paths(dcfg["root"], dcfg["n_val_files"])


# ---------- Hugging Face dataset path (precomputed preprocessing) ----------

def _preprocess_example(example: dict, pp: PreprocessConfig) -> dict:
    """Row transform for `datasets.Dataset.map`: raw {m/z, int, peptide_charge}
    → the preprocessed peak lists + the scalars every consumer needs. This is
    the work that used to run per-spectrum inside the training loop (and inside
    each probe); `.map` runs it once and caches the result to Arrow.

    The row is a superset so the same preprocessed dataset feeds both training
    and the inline probes/retrieval: `log_tic` (log of the raw pre-threshold
    TIC, discarded by preprocessing) is a probe target, and `peptide_charge` is
    kept so the probes can recompute their own labels (strict precursor m/z,
    charge) from it.
    """
    mz = torch.tensor(example["m/z"], dtype=torch.float32)
    inten = torch.tensor(example["int"], dtype=torch.float32)
    mz_p, log_int, intensity_prob = preprocess_spectrum(mz, inten, pp)
    pc = example.get("peptide_charge")
    return {
        "mz": mz_p.tolist(),
        "log_int": log_int.tolist(),
        "intensity_prob": intensity_prob.tolist(),
        "charge": charge_index(pc),
        "precursor_mz": precursor_mz(pc),          # anchor input (mod-aware)
        "log_tic": float(torch.log1p(inten.sum())) if inten.numel() else 0.0,
        "peptide_charge": pc if pc is not None else "",
    }


@dataclass
class MaskIntensityCollator:
    """HF `data_collator`: pad a batch of preprocessed rows and draw fresh
    masked-intensity positions (v9/v13). Masking is per-batch so it varies each
    epoch — the only per-step data work now that preprocessing is precomputed.
    Mirrors `mask_intensity_collate` but consumes `datasets` dict rows.
    """
    mask_ratio: float = 0.15
    min_masked: int = 1

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        B = len(features)
        Ks = [len(f["mz"]) for f in features]
        K_max = max(max(Ks) if Ks else 1, 1)
        mz = torch.zeros(B, K_max, dtype=torch.float32)
        log_int = torch.zeros(B, K_max, dtype=torch.float32)
        intensity_prob = torch.zeros(B, K_max, dtype=torch.float32)
        key_padding_mask = torch.ones(B, K_max, dtype=torch.bool)
        mask_positions = torch.zeros(B, K_max, dtype=torch.bool)
        for b, (f, K) in enumerate(zip(features, Ks)):
            if K == 0:
                continue
            mz[b, :K] = torch.as_tensor(f["mz"], dtype=torch.float32)
            log_int[b, :K] = torch.as_tensor(f["log_int"], dtype=torch.float32)
            intensity_prob[b, :K] = torch.as_tensor(f["intensity_prob"], dtype=torch.float32)
            key_padding_mask[b, :K] = False
            n_mask = min(K, max(self.min_masked, int(round(K * self.mask_ratio))))
            idx = torch.randperm(K)[:n_mask]
            mask_positions[b, idx] = True
        return {
            "mz": mz,
            "log_int": log_int,
            "intensity_prob": intensity_prob,
            "key_padding_mask": key_padding_mask,
            "mask_positions": mask_positions,
        }


def collate_preprocessed(features: list[dict]) -> dict[str, torch.Tensor]:
    """Pad preprocessed rows into an encoder batch, no masking — for the probe /
    attention diagnostics that run the encoder without the masked task."""
    B = len(features)
    Ks = [len(f["mz"]) for f in features]
    K_max = max(max(Ks) if Ks else 1, 1)
    mz = torch.zeros(B, K_max, dtype=torch.float32)
    log_int = torch.zeros(B, K_max, dtype=torch.float32)
    intensity_prob = torch.zeros(B, K_max, dtype=torch.float32)
    key_padding_mask = torch.ones(B, K_max, dtype=torch.bool)
    for b, (f, K) in enumerate(zip(features, Ks)):
        if K == 0:
            continue
        mz[b, :K] = torch.as_tensor(f["mz"], dtype=torch.float32)
        log_int[b, :K] = torch.as_tensor(f["log_int"], dtype=torch.float32)
        intensity_prob[b, :K] = torch.as_tensor(f["intensity_prob"], dtype=torch.float32)
        key_padding_mask[b, :K] = False
    return {"mz": mz, "log_int": log_int, "intensity_prob": intensity_prob,
            "key_padding_mask": key_padding_mask}


@dataclass
class EmbeddingEvalCollator:
    """Fixed-shape, unmasked batches for compiled distributed diagnostics."""

    max_peaks: int

    def __call__(self, features: list[dict]):
        B = len(features)
        mz = torch.zeros(B, self.max_peaks, dtype=torch.float32)
        log_int = torch.zeros(B, self.max_peaks, dtype=torch.float32)
        key_padding_mask = torch.ones(B, self.max_peaks, dtype=torch.bool)
        peak_count = torch.zeros(B, dtype=torch.long)
        log_tic = torch.zeros(B, dtype=torch.float32)
        row_id = torch.zeros(B, dtype=torch.long)
        peptide_charge: list[str] = []

        for b, feature in enumerate(features):
            K = min(len(feature["mz"]), self.max_peaks)
            if K:
                mz[b, :K] = torch.as_tensor(feature["mz"][:K], dtype=torch.float32)
                log_int[b, :K] = torch.as_tensor(
                    feature["log_int"][:K], dtype=torch.float32
                )
                key_padding_mask[b, :K] = False
            peak_count[b] = K
            log_tic[b] = float(feature.get("log_tic", 0.0))
            row_id[b] = int(feature["_eval_id"])
            peptide_charge.append(str(feature.get("peptide_charge", "")))

        return {
            "mz": mz,
            "log_int": log_int,
            "key_padding_mask": key_padding_mask,
            "peak_count": peak_count,
            "log_tic": log_tic,
            "row_id": row_id,
            "peptide_charge": peptide_charge,
        }


def build_preprocessed_dataset(paths: list[Path], pp: PreprocessConfig,
                               num_proc: int | None = None):
    """Load parquet shards as a HF `datasets.Dataset` and precompute
    `preprocess_spectrum` with `.map` (cached to Arrow); drop empty spectra."""
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=[str(p) for p in paths], split="train")
    ds = ds.map(partial(_preprocess_example, pp=pp), remove_columns=ds.column_names,
                num_proc=num_proc, desc="preprocess spectra")
    return ds.filter(lambda ex: len(ex["mz"]) > 0, num_proc=num_proc,
                     desc="drop empty spectra")


def build_pretraining_datasets(
    train_paths: list[Path],
    val_paths: list[Path],
    pp: PreprocessConfig,
    num_proc: int | None = None,
):
    """Preprocessed (train, val) HF datasets — map-style, so `Trainer` handles
    sampling/sharding/finite eval natively. The same preprocessed rows feed the
    distributed diagnostics, so no per-spectrum transform runs anywhere in
    training or evaluation."""
    train = build_preprocessed_dataset(train_paths, pp, num_proc=num_proc)
    val = build_preprocessed_dataset(val_paths, pp, num_proc=num_proc)
    return train, val
