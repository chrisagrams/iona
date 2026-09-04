"""Load, preprocess, and collate mass spectrum data."""

from __future__ import annotations

import re
from functools import partial
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download

from msdelta.data.chemistry import PROTON_MASS, RESIDUE_MASSES, WATER_MASS
from msdelta.data.processing import MSDeltaProcessor

N_CHARGES = 8

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
    mass = WATER_MASS + mods
    for a in seq:
        m = RESIDUE_MASSES.get(a)
        if m is None:
            return 0.0
        mass += m
    return (mass + z * PROTON_MASS) / z


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


def resolve_dataset_paths(
    *,
    root: str | None,
    repo_id: str | None,
    train_split: str,
    validation_split: str,
    num_validation_files: int,
) -> tuple[list[Path], list[Path]]:
    """Get dataset paths from a local directory or Hugging Face."""
    if repo_id:
        return hf_split_paths(
            repo_id,
            train_split=train_split,
            val_split=validation_split,
        )
    if root is None:
        raise ValueError("root is required when repo_id is not set")
    return split_paths(root, num_validation_files)


def _preprocess_example(example: dict, processor: MSDeltaProcessor) -> dict:
    """Convert one raw dataset row to preprocessed values."""
    intensity = torch.tensor(example["int"], dtype=torch.float32)
    try:
        values = processor(example["m/z"], example["int"], padding=False, return_labels=True)
    except ValueError:
        values = {"mz": [], "log_intensity": [], "labels": []}
    pc = example.get("peptide_charge")
    return {
        "mz": values["mz"],
        "log_intensity": values["log_intensity"],
        "labels": values["labels"],
        "charge": charge_index(pc),
        "precursor_mz": precursor_mz(pc),
        "log_tic": float(torch.log1p(intensity.sum())) if intensity.numel() else 0.0,
        "peptide_charge": pc if pc is not None else "",
    }


def collate_preprocessed(features: list[dict]) -> dict[str, torch.Tensor]:
    """Pad preprocessed rows without masking."""
    B = len(features)
    Ks = [len(f["mz"]) for f in features]
    K_max = max(max(Ks) if Ks else 1, 1)
    mz = torch.zeros(B, K_max, dtype=torch.float32)
    log_intensity = torch.zeros(B, K_max, dtype=torch.float32)
    labels = torch.zeros(B, K_max, dtype=torch.float32)
    attention_mask = torch.zeros(B, K_max, dtype=torch.long)
    for b, (f, K) in enumerate(zip(features, Ks)):
        if K == 0:
            continue
        mz[b, :K] = torch.as_tensor(f["mz"], dtype=torch.float32)
        log_intensity[b, :K] = torch.as_tensor(f["log_intensity"], dtype=torch.float32)
        labels[b, :K] = torch.as_tensor(f["labels"], dtype=torch.float32)
        attention_mask[b, :K] = 1
    return {
        "mz": mz,
        "log_intensity": log_intensity,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def build_preprocessed_dataset(
    paths: list[Path], processor: MSDeltaProcessor, num_proc: int | None = None
):
    """Load and preprocess Parquet shards."""
    ds = load_dataset("parquet", data_files=[str(p) for p in paths], split="train")
    ds = ds.map(
        partial(_preprocess_example, processor=processor),
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="preprocess spectra",
    )
    return ds.filter(lambda ex: len(ex["mz"]) > 0, num_proc=num_proc, desc="drop empty spectra")


def build_pretraining_datasets(
    train_paths: list[Path],
    val_paths: list[Path],
    processor: MSDeltaProcessor,
    num_proc: int | None = None,
):
    """Build the training and validation datasets."""
    train = build_preprocessed_dataset(train_paths, processor, num_proc=num_proc)
    val = build_preprocessed_dataset(val_paths, processor, num_proc=num_proc)
    return train, val


def build_denoising_datasets(
    repo_id: str,
    processor: MSDeltaProcessor,
    num_proc: int | None = None,
) -> DatasetDict:
    """Load and preprocess the labeled signal/noise dataset."""
    datasets = load_dataset(repo_id)
    return datasets.map(
        lambda example: processor.process_denoising_example(
            example["mz"], example["intensity"], example["noise"]
        ),
        remove_columns=datasets["train"].column_names,
        num_proc=num_proc,
        desc="preprocess denoising spectra",
    )
