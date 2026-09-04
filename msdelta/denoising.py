"""Hugging Face Trainer support for frozen-encoder denoising probes."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import cast

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import BatchSampler, DataLoader
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    ProgressCallback,
    Trainer,
    TrainingArguments,
)

from msdelta.configuration_msdelta import MSDeltaConfig, MSDeltaDenoisingConfig
from msdelta.modeling_msdelta import MSDeltaForDenoising, MSDeltaForPreTraining


class PeakBudgetBatchSampler(BatchSampler):
    """Build batches bounded by their padded pairwise-attention size."""

    batch_size = None  # type: ignore[assignment]
    drop_last = False

    def __init__(self, lengths, peak_pair_budget: int, seed: int):
        self.lengths = lengths
        self.peak_pair_budget = peak_pair_budget
        self.seed = seed
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

        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(batches), generator=generator).tolist()
        return [batches[index] for index in order]

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())


class DenoisingProgressCallback(ProgressCallback):
    """Label the nested Trainer's training and evaluation progress bars."""

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if self.training_bar is not None:
            self.training_bar.set_description("Denoising train")

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        super().on_prediction_step(
            args, state, control, eval_dataloader=eval_dataloader, **kwargs
        )
        if self.prediction_bar is not None:
            self.prediction_bar.set_description("Denoising eval")


class DenoisingTrainer(Trainer):
    """Use attention-aware variable-sized batches for head training."""

    def __init__(self, *args, peak_pair_budget: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_pair_budget = peak_pair_budget

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = cast(Dataset, self.train_dataset)
        batch_sampler = PeakBudgetBatchSampler(
            lengths=[len(mz) for mz in train_dataset["mz"]],
            peak_pair_budget=self.peak_pair_budget,
            seed=self.args.data_seed or self.args.seed,
        )
        dataloader = DataLoader(
            train_dataset,  # pyright: ignore[reportArgumentType]
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return self.accelerator.prepare(dataloader)


def denoising_metrics(prediction: EvalPrediction) -> dict[str, float]:
    """Compute peak-level metrics with noise as the positive class."""
    logits = np.asarray(prediction.predictions).reshape(-1)
    labels = np.asarray(prediction.label_ids).reshape(-1)
    valid = labels != -100
    logits = logits[valid]
    labels = labels[valid].astype(np.int64)
    predicted = logits >= 0
    pr_precision, pr_recall, _ = precision_recall_curve(labels, logits)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(labels, logits)),
        "auprc": float(auc(pr_recall, pr_precision)),
    }


def run_denoising_probe(
    module: MSDeltaForPreTraining,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    output_dir: Path,
    processor,
    peak_pair_budget: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    hidden_size: int,
    dropout: float,
    num_workers: int,
    seed: int,
    bf16: bool,
    fp16: bool,
) -> dict[str, float]:
    """Post-train a fresh denoising head using Hugging Face Trainer."""
    config = MSDeltaDenoisingConfig(
        encoder=copy.deepcopy(cast(MSDeltaConfig, module.config)),
        head_hidden_size=hidden_size,
        head_dropout=dropout,
    )
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    device = next(module.parameters()).device
    cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []

    try:
        with torch.random.fork_rng(devices=cuda_devices):
            probe_encoder = copy.deepcopy(module.msdelta)
            model = MSDeltaForDenoising(
                config,
                encoder=probe_encoder,
                freeze_encoder=True,
            )
            args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=epochs,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                eval_strategy="no",
                save_strategy="no",
                logging_strategy="no",
                remove_unused_columns=False,
                label_names=["labels"],
                dataloader_num_workers=num_workers,
                bf16=bf16,
                fp16=fp16,
                seed=seed,
                data_seed=seed,
                report_to=[],
                ddp_find_unused_parameters=False,
            )
            trainer = DenoisingTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=DataCollatorWithPadding(
                    tokenizer=processor,
                    padding=True,
                    return_tensors="pt",
                ),
                processing_class=processor,
                compute_metrics=denoising_metrics,
                peak_pair_budget=peak_pair_budget,
            )
            trainer.remove_callback(ProgressCallback)
            trainer.add_callback(DenoisingProgressCallback())
            trainer.train()
            evaluated = trainer.evaluate()
            metrics = {
                "denoise/loss": evaluated["eval_loss"],
                "denoise/accuracy": evaluated["eval_accuracy"],
                "denoise/balanced_accuracy": evaluated["eval_balanced_accuracy"],
                "denoise/precision": evaluated["eval_precision"],
                "denoise/recall": evaluated["eval_recall"],
                "denoise/f1": evaluated["eval_f1"],
                "denoise/auroc": evaluated["eval_auroc"],
                "denoise/auprc": evaluated["eval_auprc"],
            }
            trainer.save_model()
            trainer.save_metrics("denoise", metrics)
            return metrics
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
