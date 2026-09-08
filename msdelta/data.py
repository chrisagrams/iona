"""Load, preprocess, and collate mass spectrum data."""

from __future__ import annotations

import re
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import snapshot_download

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


def build_retrieval_datasets(
    repo_id: str,
    processor: MSDeltaProcessor,
    num_proc: int | None = None,
) -> DatasetDict:
    """Load and preprocess grouped consensus/experimental retrieval spectra."""
    datasets = load_dataset(repo_id)
    datasets = DatasetDict({split: datasets[split] for split in ("train", "validation")})

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


def build_reranking_datasets(
    repo_id: str,
    processor,
    num_proc: int | None = None,
    *,
    revision: str | None = None,
    include_consensus: bool = False,
    include_test: bool = False,
) -> DatasetDict:
    """Preserve modified-peptide identities across charges and distributed batches."""
    splits = ["train", "validation"] + (["test"] if include_test else [])
    raw = DatasetDict(
        {split: load_dataset(repo_id, split=split, revision=revision) for split in splits}
    )
    identities = sorted({peptide for dataset in raw.values() for peptide in dataset["peptide"]})
    peptide_ids = {peptide: index for index, peptide in enumerate(identities)}
    # The supplied splits are disjoint by modified peptide, though not by bare sequence.
    seen = set()
    for split, dataset in raw.items():
        peptides = set(dataset["peptide"])
        if seen & peptides:
            raise ValueError(f"modified peptides overlap between {split} and earlier splits")
        seen.update(peptides)

    def process(row):
        if row["analyte_id"] != f"{row['peptide']}_{row['charge']}":
            raise ValueError("analyte_id must agree with peptide and charge")
        if len(row["experimental"]) != 3:
            raise ValueError("reranking examples must contain three experimental spectra")
        spectra = row["experimental"]
        if include_consensus:
            spectra = [*spectra, row["consensus"]]
        values = [processor(s["mz"], s["intensity"], padding=False) for s in spectra]
        return {
            "mz": [v["mz"] for v in values],
            "log_intensity": [v["log_intensity"] for v in values],
            "peptide_input_ids": processor.tokenize_peptide(row["peptide"]),
            "peptide_id": peptide_ids[row["peptide"]],
        }

    return raw.map(
        process,
        remove_columns=raw["train"].column_names,
        num_proc=num_proc,
        desc="preprocess reranking PSMs",
    )


def build_reranking_evaluation_datasets(
    dataset: Dataset,
    *,
    max_analytes: int | None = 1000,
) -> dict[str, Dataset]:
    """Pair experimental spectrum queries with a deduplicated modified-peptide gallery."""
    if max_analytes is not None:
        if max_analytes <= 0:
            raise ValueError("max_analytes must be positive or None")
        dataset = dataset.select(range(min(len(dataset), max_analytes)))
    peptides = {}
    spectra = []
    for row in dataset:
        label = row["peptide_id"]
        peptides.setdefault(
            label, {"peptide_input_ids": row["peptide_input_ids"], "evaluation_labels": label}
        )
        # Consensus, when requested for training, is appended after the three experiments.
        spectra.extend(
            {"mz": mz, "log_intensity": intensity, "evaluation_labels": label}
            for mz, intensity in zip(row["mz"][:3], row["log_intensity"][:3])
        )
    if not spectra:
        raise ValueError("reranking evaluation requires at least one analyte")
    return {
        "spectra": Dataset.from_list(spectra),
        "peptides": Dataset.from_list(list(peptides.values())),
    }
