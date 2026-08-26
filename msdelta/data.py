"""Load, preprocess, and collate mass spectrum data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download


@dataclass
class PreprocessConfig:
    intensity_threshold_frac: float = 0.01
    top_n: int = 150


N_CHARGES = 8

_RESIDUE_MASS = {
    "G": 57.02146,
    "A": 71.03711,
    "S": 87.03203,
    "P": 97.05276,
    "V": 99.06841,
    "T": 101.04768,
    "C": 103.00919,
    "L": 113.08406,
    "I": 113.08406,
    "N": 114.04293,
    "D": 115.02694,
    "Q": 128.05858,
    "K": 128.09496,
    "E": 129.04259,
    "M": 131.04049,
    "H": 137.05891,
    "F": 147.06841,
    "R": 156.10111,
    "Y": 163.06333,
    "W": 186.07931,
}
_WATER = 18.0105646
_PROTON = 1.0072765
_MOD_RE = re.compile(r"\[([+-]?[0-9.]+)\]")


def charge_index(peptide_charge: str | None) -> int:
    """Return the charge index from a peptide-charge label."""
    if not peptide_charge:
        return 0
    parts = str(peptide_charge).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return min(int(parts[1]), N_CHARGES - 1)
    return 0


def precursor_mz(peptide_charge: str | None) -> float:
    """Calculate precursor m/z from a peptide-charge label."""
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
    """Filter peaks and return m/z, log intensity, and intensity probability."""
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
    """Use the last sorted shards as validation data."""
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
    """Get training and validation Parquet paths from Hugging Face."""
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
    """Get dataset paths from a local directory or Hugging Face."""
    if dcfg.get("hf_repo"):
        return hf_split_paths(
            dcfg["hf_repo"],
            train_split=dcfg.get("hf_train_split", "train"),
            val_split=dcfg.get("hf_val_split", "val"),
        )
    return split_paths(dcfg["root"], dcfg["n_val_files"])


def _preprocess_example(example: dict, pp: PreprocessConfig) -> dict:
    """Convert one raw dataset row to preprocessed values."""
    mz = torch.tensor(example["m/z"], dtype=torch.float32)
    inten = torch.tensor(example["int"], dtype=torch.float32)
    mz_p, log_int, intensity_prob = preprocess_spectrum(mz, inten, pp)
    pc = example.get("peptide_charge")
    return {
        "mz": mz_p.tolist(),
        "log_int": log_int.tolist(),
        "intensity_prob": intensity_prob.tolist(),
        "charge": charge_index(pc),
        "precursor_mz": precursor_mz(pc),
        "log_tic": float(torch.log1p(inten.sum())) if inten.numel() else 0.0,
        "peptide_charge": pc if pc is not None else "",
    }


@dataclass
class MaskIntensityCollator:
    """Pad rows and select new masked positions for each batch."""

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
    """Pad preprocessed rows without masking."""
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
    return {
        "mz": mz,
        "log_int": log_int,
        "intensity_prob": intensity_prob,
        "key_padding_mask": key_padding_mask,
    }


def build_preprocessed_dataset(
    paths: list[Path], pp: PreprocessConfig, num_proc: int | None = None
):
    """Load and preprocess Parquet shards."""
    ds = load_dataset("parquet", data_files=[str(p) for p in paths], split="train")
    ds = ds.map(
        partial(_preprocess_example, pp=pp),
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="preprocess spectra",
    )
    return ds.filter(lambda ex: len(ex["mz"]) > 0, num_proc=num_proc, desc="drop empty spectra")


def build_pretraining_datasets(
    train_paths: list[Path],
    val_paths: list[Path],
    pp: PreprocessConfig,
    num_proc: int | None = None,
):
    """Build the training and validation datasets."""
    train = build_preprocessed_dataset(train_paths, pp, num_proc=num_proc)
    val = build_preprocessed_dataset(val_paths, pp, num_proc=num_proc)
    return train, val
