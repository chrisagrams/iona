"""Load, preprocess, and collate mass spectrum data."""

from __future__ import annotations

import re
from functools import partial

import torch
from datasets import Dataset, DatasetDict, load_dataset

from msdelta.chemistry import PROTON_MASS, RESIDUE_MASSES, WATER_MASS
from msdelta.processing_msdelta import MSDeltaProcessor

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


def _preprocess_example(example: dict, processor: MSDeltaProcessor) -> dict:
    """Convert one raw dataset row to preprocessed values."""
    intensity = torch.tensor(example["intensity"], dtype=torch.float32)
    try:
        values = processor(
            example["mz"], example["intensity"], padding=False, return_labels=True
        )
    except ValueError:
        values = {"mz": [], "log_intensity": [], "labels": []}
    pc = f'{example["peptide"]}_{example["charge"]}'
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
    dataset: Dataset, processor: MSDeltaProcessor, num_proc: int | None = None
) -> Dataset:
    """Preprocess a mass-spectrum dataset."""
    dataset = dataset.map(
        partial(_preprocess_example, processor=processor),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="preprocess spectra",
    )
    return dataset.filter(
        lambda ex: len(ex["mz"]) > 0, num_proc=num_proc, desc="drop empty spectra"
    )


def build_pretraining_datasets(
    repo_id: str,
    processor: MSDeltaProcessor,
    train_split: str = "train",
    validation_split: str = "validation",
    num_proc: int | None = None,
) -> tuple[Dataset, Dataset]:
    """Load and preprocess training and validation splits from Hugging Face."""
    train, validation = load_dataset(repo_id, split=[train_split, validation_split])
    train = build_preprocessed_dataset(train, processor, num_proc=num_proc)
    validation = build_preprocessed_dataset(validation, processor, num_proc=num_proc)
    return train, validation


def build_denoising_datasets(
    repo_id: str,
    processor: MSDeltaProcessor,
    num_proc: int | None = None,
) -> DatasetDict:
    """Load and preprocess the labeled signal/noise dataset."""
    datasets = load_dataset(repo_id)
    datasets = datasets.filter(
        lambda example: 0 < len(example["mz"]) <= processor.max_peaks,
        num_proc=num_proc,
        desc="drop invalid or oversized denoising spectra",
    )
    return datasets.map(
        lambda example: processor.process_denoising_example(
            example["mz"], example["intensity"], example["noise"]
        ),
        remove_columns=datasets["train"].column_names,
        num_proc=num_proc,
        desc="preprocess denoising spectra",
    )


def build_retrieval_datasets(
    repo_id: str,
    processor: MSDeltaProcessor,
    num_proc: int | None = None,
) -> DatasetDict:
    """Load and preprocess grouped consensus/experimental retrieval spectra."""
    datasets = load_dataset(repo_id)
    datasets = DatasetDict({split: datasets[split] for split in ("train", "validation")})
    datasets = datasets.filter(
        lambda example: all(
            0 < len(spectrum["mz"]) <= processor.max_peaks
            for spectrum in (example["consensus"], *example["experimental"])
        ),
        num_proc=num_proc,
        desc="drop invalid or oversized retrieval groups",
    )

    processed = datasets.map(
        lambda example: processor.process_retrieval_example(
            example["consensus"], example["experimental"]
        ),
        remove_columns=datasets["train"].column_names,
        num_proc=num_proc,
        desc="preprocess retrieval spectra",
    )
    return processed.filter(
        lambda example: len(example["mz"]) == 4 and all(example["mz"]),
        num_proc=num_proc,
        desc="drop invalid retrieval groups",
    )


def build_retrieval_evaluation_datasets(
    validation_dataset: Dataset,
    processor: MSDeltaProcessor,
    *,
    max_analytes: int = 1000,
    replicate_repo_id: str | None = None,
    num_proc: int | None = None,
) -> dict[str, Dataset]:
    """Prepare named retrieval galleries with one spectrum and global label per row."""
    subset = validation_dataset.select(range(min(len(validation_dataset), max_analytes)))
    evaluation_datasets = {
        "retrieval": Dataset.from_list(
            [
                {"mz": mz, "log_intensity": intensity, "retrieval_labels": group_id}
                for group_id, row in enumerate(subset)
                for mz, intensity in zip(row["mz"], row["log_intensity"])
            ]
        ),
    }
    if replicate_repo_id:
        dataset = load_dataset(replicate_repo_id, split="test")
        dataset = dataset.filter(
            lambda row: 0 < len(row["mz"]) <= processor.max_peaks,
            num_proc=num_proc,
            desc="drop invalid or oversized replicate spectra",
        )
        analytes = {}
        labels = [
            analytes.setdefault((peptide, charge), len(analytes))
            for peptide, charge in zip(dataset["peptide"], dataset["charge"])
        ]
        processed = dataset.map(
            lambda row: processor(row["mz"], row["intensity"], padding=False),
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            desc="preprocess replicate retrieval spectra",
        )
        evaluation_datasets["replicate_retrieval"] = processed.add_column(
            "retrieval_labels",
            labels,
        )
    return evaluation_datasets
