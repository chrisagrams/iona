"""Run encoder inference through ``Trainer.predict`` in length-sorted batches."""

from __future__ import annotations

import random
import tempfile
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainingArguments


def spectrum_lengths(dataset: Dataset) -> np.ndarray:
    """Peak count of every row, from the ``length`` column when preprocessing wrote one."""
    if "length" in dataset.column_names:
        return dataset.select_columns(["length"]).with_format("numpy")[:]["length"]
    return np.concatenate([
        np.zeros(0, dtype=np.int64),
        *(pc.list_value_length(b["mz"]).to_numpy(zero_copy_only=False)  # pyright: ignore
          for b in dataset.with_format("arrow").iter(65_536)),
    ])


def collate_spectra(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Pad the peak lists of a batch and build its attention mask."""
    mz = [torch.as_tensor(r["mz"], dtype=torch.float32) for r in rows]
    log_intensity = [torch.as_tensor(r["log_intensity"], dtype=torch.float32) for r in rows]
    lengths = torch.tensor([m.numel() for m in mz])
    padded = pad_sequence(mz, batch_first=True)
    return {
        "mz": padded,
        "log_intensity": pad_sequence(log_intensity, batch_first=True),
        "attention_mask": torch.arange(padded.shape[1])[None, :] < lengths[:, None],
    }


def _concat_batches(batches):
    """Concatenate per-batch predictions on rows, padding a per-peak axis with -100."""
    first = batches[0]
    if isinstance(first, Mapping):
        return {k: _concat_batches([b[k] for b in batches]) for k in first}
    if isinstance(first, (list, tuple)):
        return type(first)(_concat_batches(list(parts)) for parts in zip(*batches))
    if first.ndim < 2:
        return np.concatenate(batches)
    width = max(b.shape[1] for b in batches)
    return np.concatenate([
        np.pad(b, [(0, 0), (0, width - b.shape[1])] + [(0, 0)] * (b.ndim - 2),
               constant_values=-100)
        for b in batches
    ])


def _take_rows(predictions, index: np.ndarray):
    """Index the row axis of every array in nested predictions."""
    if isinstance(predictions, Mapping):
        return {k: _take_rows(v, index) for k, v in predictions.items()}
    if isinstance(predictions, (list, tuple)):
        return type(predictions)(_take_rows(v, index) for v in predictions)
    return predictions[index]


class SortedPredictionTrainer(Trainer):
    """Trainer that predicts over rows sorted longest first and returns them in input order."""

    def _get_eval_sampler(self, eval_dataset):
        # predict_sorted orders the rows itself; group_by_length would shuffle them.
        if eval_dataset is None:
            return None
        if self.args.world_size > 1:
            return None
        return SequentialSampler(eval_dataset)  # pyright: ignore[reportArgumentType]

    def predict_sorted(self, dataset: Dataset, **kwargs) -> Any:
        """Return ``predict(dataset, **kwargs).predictions`` in input order, longest rows first.

        Sorting by peak count keeps padding, and its quadratic attention cost, small. Rows run
        longest first so an out-of-memory error surfaces on the first batch. Outputs with a
        per-peak axis are padded with -100 to the longest spectrum.
        """
        # select(argsort) is what Dataset.sort does, minus add_column's flatten of any indices mapping.
        order = np.argsort(-spectrum_lengths(dataset), kind="stable")
        sorted_rows = dataset.select(order)
        predictions = self.predict(sorted_rows, **kwargs).predictions  # pyright: ignore[reportArgumentType]
        if isinstance(predictions, list):
            predictions = _concat_batches(predictions)
        return _take_rows(predictions, np.argsort(order))


@contextmanager
def _preserved_rng() -> Generator[None]:
    """Restore the Python, NumPy and torch RNG streams on exit."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device_type = "xpu" if torch.xpu.is_available() else "cuda"
    module = getattr(torch, device_type)
    devices = list(range(module.device_count())) if module.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices, device_type=device_type):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class PredictionTrainer(SortedPredictionTrainer):
    """Prediction-only Trainer that runs ``predict_fn(model, inputs)`` in place of the forward pass.

    Every process of a distributed run must call ``predict_sorted``: rows are sharded across
    ranks and the predictions gathered back to all of them. The model runs on the device
    accelerate picks for this process unless ``device`` is the CPU.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        predict_fn: Callable[[Any, dict], Any],
        *,
        data_collator: Callable[[list[dict]], dict],
        batch_size: int,
        device: str | torch.device | None = None,
    ):
        self.predict_fn = predict_fn
        args = TrainingArguments(
            output_dir=str(Path(tempfile.gettempdir()) / "iona-predict"),
            per_device_eval_batch_size=batch_size,
            # Move each batch to the CPU and keep it whole; concatenating as it goes is quadratic.
            eval_accumulation_steps=1,
            eval_do_concat_batches=False,
            remove_unused_columns=False,
            report_to=[],
            use_cpu=device is not None and torch.device(device).type == "cpu",
        )
        # Trainer reseeds the global RNGs on construction; keep a live training run's streams.
        with _preserved_rng():
            super().__init__(model=model, args=args, data_collator=data_collator)

    def create_accelerator_and_postprocess(self) -> None:
        super().create_accelerator_and_postprocess()
        # Inside a DeepSpeed run the shared accelerator state hands this Trainer the run's plugin,
        # but prediction needs no engine and DeepSpeed refuses inference below ZeRO-3.
        self.is_deepspeed_enabled = False

    def prediction_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, model, inputs, prediction_loss_only, ignore_keys=None
    ):
        with torch.no_grad():
            return None, self.predict_fn(model, self._prepare_inputs(inputs)), None
