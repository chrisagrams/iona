"""Hugging Face Trainer support for frozen-encoder denoising probes."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import EvalPrediction, Trainer, TrainingArguments

from msdelta.modeling_msdelta import MSDeltaForDenoising
from msdelta.processing_msdelta import MSDeltaDataCollatorForDenoising


def denoising_metrics(prediction: EvalPrediction) -> dict[str, float]:
    """Compute peak-level metrics with noise as the positive class."""
    logits = np.asarray(prediction.predictions).reshape(-1)
    labels = np.asarray(prediction.label_ids).reshape(-1)
    valid = labels != -100
    logits = logits[valid]
    labels = labels[valid].astype(np.int64)
    predicted = logits >= 0
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(labels, logits)),
        "average_precision": float(average_precision_score(labels, logits)),
    }


def run_denoising_probe(
    module,
    train_dataset,
    validation_dataset,
    *,
    output_dir: Path,
    processor,
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
    config = copy.deepcopy(module.config)
    config.denoising_head_hidden_size = hidden_size
    config.denoising_head_dropout = dropout
    requires_grad = [parameter.requires_grad for parameter in module.msdelta.parameters()]
    was_training = module.training
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    device = next(module.parameters()).device
    cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []

    try:
        with torch.random.fork_rng(devices=cuda_devices):
            model = MSDeltaForDenoising(config, encoder=module.msdelta)
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
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=MSDeltaDataCollatorForDenoising(),
                processing_class=processor,
                compute_metrics=denoising_metrics,
            )
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
                "denoise/average_precision": evaluated["eval_average_precision"],
            }
            trainer.save_model()
            trainer.save_metrics("denoise", metrics)
            return metrics
    finally:
        for parameter, trainable in zip(module.msdelta.parameters(), requires_grad):
            parameter.requires_grad_(trainable)
        module.train(was_training)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
