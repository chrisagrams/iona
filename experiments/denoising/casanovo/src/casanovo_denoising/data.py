"""Casanovo preprocessing, batching, and collation for denoising."""

from __future__ import annotations

import functools
import tempfile
from collections.abc import Callable, Sequence

import numpy as np
import torch
from casanovo.denovo.dataloaders import DeNovoDataModule
from datasets import DatasetDict, Features, Value, load_dataset
from datasets import Sequence as SequenceFeature
from depthcharge.primitives import MassSpectrum
from torch.utils.data import BatchSampler

DEFAULT_DATASET_REPO = "chrisagrams/ms-denoise-100k"
LABEL_PAD_VALUE = -100.0

PROCESSED_FEATURES = Features(
    {
        "mz": SequenceFeature(Value("float32")),
        "intensity": SequenceFeature(Value("float32")),
        "labels": SequenceFeature(Value("float32")),
        "num_peaks": Value("int32"),
    }
)


@functools.cache
def casanovo_preprocessing() -> tuple[list[Callable[[MassSpectrum], MassSpectrum]], np.ndarray]:
    """Return Casanovo's own ``preprocessing_fn`` list and valid charges.

    Taken from a default ``DeNovoDataModule`` so the steps and their
    parameters (m/z range, precursor removal, root scaling, intensity filter,
    minimum peaks, unit norm) are exactly what Casanovo uses. The data module
    only reads ``lance_dir`` in ``setup()``, which is never called here.
    """
    module = DeNovoDataModule(lance_dir=tempfile.gettempdir())
    return module.preprocessing_fn, module.valid_charge


def preprocess_spectrum(
    mz: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    noise: Sequence[bool] | np.ndarray,
    precursor_mz: float,
    precursor_charge: int,
) -> dict[str, np.ndarray] | None:
    """Run Casanovo's preprocessing and carry the noise labels along.

    Casanovo's steps only sort peaks by m/z and drop peaks; they never alter
    m/z values. Each retained peak is therefore matched back to its original
    index by exact m/z to select its label. Returns ``None`` when Casanovo
    would skip the spectrum (it catches the same exceptions in depthcharge's
    parser).
    """
    mz = np.asarray(mz)
    intensity = np.asarray(intensity)
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
    order = np.argsort(mz, kind="stable")
    sorted_mz = mz[order].astype(np.float64)
    if np.any(np.diff(sorted_mz) == 0):
        raise ValueError("duplicate m/z values make label alignment ambiguous")

    preprocessing_fn, valid_charge = casanovo_preprocessing()
    if precursor_charge not in valid_charge:
        return None
    spectrum = MassSpectrum(
        filename="",
        scan_id="",
        mz=mz,
        intensity=intensity,
        precursor_mz=precursor_mz,
        precursor_charge=precursor_charge,
    )
    try:
        for processor in preprocessing_fn:
            spectrum = processor(spectrum)
    except (IndexError, KeyError, ValueError):
        return None

    kept_mz = np.asarray(spectrum.mz, dtype=np.float64)
    positions = np.searchsorted(sorted_mz, kept_mz)
    positions = np.minimum(positions, len(sorted_mz) - 1)
    if not np.array_equal(sorted_mz[positions], kept_mz):
        raise RuntimeError("Casanovo preprocessing changed m/z values; cannot align labels")
    indices = order[positions]
    return {
        "mz": np.asarray(spectrum.mz, dtype=np.float32),
        "intensity": np.asarray(spectrum.intensity, dtype=np.float32),
        "labels": noise[indices].astype(np.float32),
    }


def _process_example(example: dict) -> dict:
    processed = preprocess_spectrum(
        example["mz"],
        example["intensity"],
        example["noise"],
        precursor_mz=float(example["precursor"]),
        precursor_charge=int(example["charge"]),
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
) -> DatasetDict:
    """Load the labeled dataset and apply Casanovo preprocessing per spectrum."""
    raw = load_dataset(repo_id, cache_dir=cache_dir)
    raw = DatasetDict({"train": raw[train_split], "validation": raw[validation_split]})
    processed = raw.map(
        _process_example,
        remove_columns=raw["train"].column_names,
        features=PROCESSED_FEATURES,
        num_proc=num_proc,
        desc="Casanovo preprocessing",
    )
    return processed.filter(
        lambda num_peaks: num_peaks > 0,
        input_columns="num_peaks",
        num_proc=num_proc,
        desc="drop spectra Casanovo would skip",
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

    Mirrors ``msdelta.denoising.PeakBudgetBatchSampler``. Each process yields
    every ``num_processes``-th batch starting at ``process_index``. With
    ``pad=True`` the batch list is padded with repeated batches so every
    process performs the same number of steps (needed for DDP training); use
    ``pad=False`` for evaluation so no spectrum is counted twice.
    """

    batch_size = None  # type: ignore[assignment]
    drop_last = False

    def __init__(
        self,
        lengths: Sequence[int],
        peak_pair_budget: int,
        seed: int,
        shuffle: bool = True,
        *,
        process_index: int = 0,
        num_processes: int = 1,
        pad: bool = True,
    ):
        self.lengths = list(lengths)
        self.peak_pair_budget = peak_pair_budget
        self.seed = seed
        self.shuffle = shuffle
        self.process_index = process_index
        self.num_processes = num_processes
        self.pad = pad
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
        if self.pad and batches:
            padding = (-len(batches)) % self.num_processes
            batches.extend([list(batches[index % len(batches)]) for index in range(padding)])
        return batches[self.process_index :: self.num_processes]

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
