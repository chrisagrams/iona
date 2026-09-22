"""Casanovo-compatible preprocessing, batching, and collation for denoising."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from datasets import DatasetDict, Features, Value, load_dataset
from datasets import Sequence as SequenceFeature
from torch.utils.data import BatchSampler

DEFAULT_DATASET_REPO = "chrisagrams/ms-denoise-100k"
MIN_MZ = 50.0
MAX_MZ = 2500.0
PRECURSOR_TOLERANCE_DA = 2.0
MIN_RELATIVE_INTENSITY = 0.01
MAX_NUM_PEAKS = 150
MIN_NUM_PEAKS = 20
LABEL_PAD_VALUE = -100.0

PROCESSED_FEATURES = Features(
    {
        "mz": SequenceFeature(Value("float32")),
        "intensity": SequenceFeature(Value("float32")),
        "labels": SequenceFeature(Value("float32")),
        "num_peaks": Value("int32"),
    }
)


def preprocess_spectrum(
    mz: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    noise: Sequence[bool] | np.ndarray,
    precursor_mz: float | None = None,
    *,
    remove_precursor: bool = True,
    min_mz: float = MIN_MZ,
    max_mz: float = MAX_MZ,
    precursor_tolerance: float = PRECURSOR_TOLERANCE_DA,
    min_relative_intensity: float = MIN_RELATIVE_INTENSITY,
    max_num_peaks: int = MAX_NUM_PEAKS,
    min_num_peaks: int = MIN_NUM_PEAKS,
) -> dict[str, np.ndarray] | None:
    """Apply Casanovo's peak preprocessing and carry noise labels along.

    Returns ``None`` when fewer than ``min_num_peaks`` peaks survive.
    """
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    noise = np.asarray(noise)
    if mz.ndim != 1 or intensity.ndim != 1 or noise.ndim != 1:
        raise ValueError("mz, intensity, and noise must be one-dimensional")
    if not (mz.shape == intensity.shape == noise.shape):
        raise ValueError(
            f"mz, intensity, and noise must have equal shapes, got "
            f"{mz.shape}, {intensity.shape}, {noise.shape}"
        )
    for name, values in (("mz", mz), ("intensity", intensity)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
        if np.any(values < 0):
            raise ValueError(f"{name} must be nonnegative")

    keep = (mz >= min_mz) & (mz <= max_mz)
    if remove_precursor and precursor_mz is not None and np.isfinite(precursor_mz):
        keep &= np.abs(mz - precursor_mz) > precursor_tolerance
    indices = np.flatnonzero(keep)

    if indices.size:
        base_peak = intensity[indices].max()
        indices = indices[intensity[indices] >= min_relative_intensity * base_peak]
    if indices.size > max_num_peaks:
        top = np.argsort(-intensity[indices], kind="stable")[:max_num_peaks]
        indices = np.sort(indices[top])
    if indices.size < min_num_peaks:
        return None

    scaled = np.sqrt(intensity[indices])
    total = scaled.sum()
    if total <= 0:
        return None
    return {
        "mz": mz[indices].astype(np.float32),
        "intensity": (scaled / total).astype(np.float32),
        "labels": noise[indices].astype(np.float32),
    }


def _process_example(example: dict, remove_precursor: bool) -> dict:
    processed = preprocess_spectrum(
        example["mz"],
        example["intensity"],
        example["noise"],
        example.get("precursor_mz"),
        remove_precursor=remove_precursor,
    )
    if processed is None:
        return {"mz": [], "intensity": [], "labels": [], "num_peaks": 0}
    return {**processed, "num_peaks": len(processed["mz"])}


def build_denoising_datasets(
    repo_id: str = DEFAULT_DATASET_REPO,
    *,
    train_split: str = "train",
    validation_split: str = "validation",
    cache_dir: str | None = None,
    num_proc: int | None = None,
    remove_precursor: bool = True,
) -> DatasetDict:
    """Load the labeled dataset and apply Casanovo preprocessing per spectrum."""
    raw = load_dataset(repo_id, cache_dir=cache_dir)
    raw = DatasetDict({"train": raw[train_split], "validation": raw[validation_split]})
    processed = raw.map(
        _process_example,
        fn_kwargs={"remove_precursor": remove_precursor},
        remove_columns=raw["train"].column_names,
        features=PROCESSED_FEATURES,
        num_proc=num_proc,
        desc="Casanovo preprocessing",
    )
    return processed.filter(
        lambda num_peaks: num_peaks >= MIN_NUM_PEAKS,
        input_columns="num_peaks",
        num_proc=num_proc,
        desc=f"drop spectra with fewer than {MIN_NUM_PEAKS} peaks",
    )


def collate_denoising(examples: list[dict]) -> dict[str, torch.Tensor]:
    """Right-pad a batch: m/z and intensity with 0, labels with -100."""
    longest = max(len(example["mz"]) for example in examples)
    mz = torch.zeros(len(examples), longest, dtype=torch.float32)
    intensity = torch.zeros(len(examples), longest, dtype=torch.float32)
    labels = torch.full((len(examples), longest), LABEL_PAD_VALUE, dtype=torch.float32)
    for row, example in enumerate(examples):
        length = len(example["mz"])
        mz[row, :length] = torch.as_tensor(np.asarray(example["mz"], dtype=np.float32))
        intensity[row, :length] = torch.as_tensor(
            np.asarray(example["intensity"], dtype=np.float32)
        )
        labels[row, :length] = torch.as_tensor(np.asarray(example["labels"], dtype=np.float32))
    return {"mz": mz, "intensity": intensity, "labels": labels}


class PeakBudgetBatchSampler(BatchSampler):
    """Build batches bounded by their padded pairwise-attention size.

    Mirrors ``msdelta.denoising.PeakBudgetBatchSampler`` for a single process.
    """

    batch_size = None  # type: ignore[assignment]
    drop_last = False

    def __init__(self, lengths: Sequence[int], peak_pair_budget: int, seed: int, shuffle: bool = True):
        self.lengths = list(lengths)
        self.peak_pair_budget = peak_pair_budget
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        indices = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        batches: list[list[int]] = []
        batch: list[int] = []
        longest = 0
        for index in indices:
            candidate_longest = max(longest, self.lengths[index])
            attention_size = (len(batch) + 1) * candidate_longest**2
            if batch and attention_size > self.peak_pair_budget:
                batches.append(batch)
                batch = []
                longest = 0
            batch.append(index)
            longest = max(longest, self.lengths[index])
        if batch:
            batches.append(batch)

        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
