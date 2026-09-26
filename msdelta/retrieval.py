"""Evaluate trained MSDelta retrieval embeddings."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import cast

import faiss
import numpy as np
import torch
from accelerate.utils import broadcast_object_list
from datasets import Dataset
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import FaissKNN
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import ProgressCallback, Trainer, TrainingArguments

from msdelta.configuration_msdelta import MSDeltaConfig, MSDeltaRetrievalConfig
from msdelta.modeling_msdelta import (
    MSDeltaForPreTraining,
    MSDeltaForRetrieval,
)
from msdelta.processing_msdelta import MSDeltaDataCollatorForRetrieval


class RetrievalEvaluationCollator:
    """Pad individual spectra while preserving their global retrieval labels."""

    def __call__(self, features):
        mzs = [torch.tensor(row["mz"], dtype=torch.float32) for row in features]
        intensities = [torch.tensor(row["log_intensity"], dtype=torch.float32) for row in features]
        mz = pad_sequence(mzs, batch_first=True)
        lengths = torch.tensor([mass.numel() for mass in mzs])
        return {
            "mz": mz,
            "log_intensity": pad_sequence(intensities, batch_first=True),
            "attention_mask": torch.arange(mz.shape[1])[None, :] < lengths[:, None],
            "retrieval_labels": torch.tensor([row["retrieval_labels"] for row in features]),
        }


class RetrievalTrainer(Trainer):
    """Gather spectrum embeddings and labels without evaluating a batch-local loss."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if "retrieval_labels" not in inputs:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        inputs = dict(inputs)
        labels = self._prepare_input(inputs.pop("retrieval_labels"))
        loss, embeddings, _ = super().prediction_step(model, inputs, False, ignore_keys)
        return loss, embeddings, labels


class RetrievalAccuracyCalculator(AccuracyCalculator):
    """Add recall at five to FAISS-backed accuracy calculation."""

    def requires_knn(self):
        return super().requires_knn() + ["recall_at_5"]

    def calculate_recall_at_5(
        self,
        knn_labels,
        query_labels,
        reference_labels,
        ref_includes_query,
        not_lone_query_mask,
        **kwargs,
    ):
        classes, counts = reference_labels.unique(return_counts=True)
        positives = counts[torch.searchsorted(classes, query_labels)] - int(ref_includes_query)
        hits = (knn_labels[:, :5] == query_labels[:, None]).sum(dim=1)
        recall = hits.float() / positives.clamp_min(1)
        return recall[not_lone_query_mask].mean().item()


@torch.no_grad()
def retrieval_metrics(embeddings, labels, device, *, gpus=None):
    """Search with FAISS and average metrics over queries with another matching spectrum.

    MAP@100 uses all relevant gallery items as its per-query denominator, including
    positives outside the retrieved top 100. FAISS excludes each query's own entry.
    """
    vectors = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    targets = torch.as_tensor(labels, dtype=torch.long, device=device)
    calculator = RetrievalAccuracyCalculator(
        include=("precision_at_1", "mean_average_precision", "recall_at_5",
                 "mean_average_precision_at_r", "r_precision"),
        k=min(100, len(vectors) - 1),
        device=device,
        knn_func=FaissKNN(index_init_fn=faiss.IndexFlatIP, gpus=gpus),
    )
    scores = calculator.get_accuracy(vectors, targets)
    return {
        "Hit@1": scores["precision_at_1"],
        "MAP@100": scores["mean_average_precision"],
        "R@5": scores["recall_at_5"],
        "MAP@R": scores["mean_average_precision_at_r"],
        "R-Precision": scores["r_precision"],
    }


class RetrievalProgressCallback(ProgressCallback):
    """Label the nested Trainer's progress bars."""

    evaluation_description = "Retrieval loss"

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if self.training_bar is not None:
            self.training_bar.set_description("Retrieval train")

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        super().on_prediction_step(args, state, control, eval_dataloader=eval_dataloader, **kwargs)
        if self.prediction_bar is not None:
            self.prediction_bar.set_description(self.evaluation_description)


def run_retrieval_probe(
    module: MSDeltaForPreTraining,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    output_dir: Path,
    processor,
    evaluation_datasets: dict[str, Dataset],
    projection_hidden_size: int,
    embedding_size: int,
    dropout: float,
    temperature: float,
    training_args: TrainingArguments,
) -> dict[str, float]:
    """Post-train a fresh retrieval head on an isolated, frozen encoder copy."""
    config = MSDeltaRetrievalConfig(
        encoder=copy.deepcopy(cast(MSDeltaConfig, module.config)),
        projection_hidden_size=projection_hidden_size,
        embedding_size=embedding_size,
        head_dropout=dropout,
        temperature=temperature,
    )
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    device = next(module.parameters()).device
    accelerator_type = device.type if device.type in {"cuda", "xpu"} else "cuda"
    accelerator_devices = [device.index] if device.type in {"cuda", "xpu"} else []

    try:
        with torch.random.fork_rng(devices=accelerator_devices, device_type=accelerator_type):
            probe_encoder = copy.deepcopy(module.msdelta)
            model = MSDeltaForRetrieval(
                config,
                encoder=probe_encoder,
                freeze_encoder=True,
            )
            args = copy.deepcopy(training_args)
            args.output_dir = str(output_dir)
            trainer = RetrievalTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=MSDeltaDataCollatorForRetrieval(),
                processing_class=processor,
            )
            trainer.remove_callback(ProgressCallback)
            progress = RetrievalProgressCallback()
            trainer.add_callback(progress)
            trainer.train()
            evaluated = trainer.evaluate()
            metrics = {"retrieval/loss": evaluated["eval_loss"]}
            # Use the same distributed evaluation loop for both spectrum datasets.
            trainer.data_collator = RetrievalEvaluationCollator()
            trainer.args.prediction_loss_only = False
            trainer.args.dataloader_drop_last = False
            trainer.eval_dataset = evaluation_datasets
            # Rank 0 owns the FAISS indexes on this node's allocated GPUs.
            gpus = (
                list(range(int(os.environ.get("LOCAL_WORLD_SIZE", "1"))))
                if device.type == "cuda"
                else None
            )

            def compute_metrics(prediction):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "xpu":
                    torch.xpu.empty_cache()
                trainer.accelerator.wait_for_everyone()
                result = {}
                if trainer.is_world_process_zero():
                    with tqdm(
                        total=1,
                        desc=f"{name} FAISS ranking",
                        unit="search",
                        disable=bool(args.disable_tqdm),
                        leave=False,
                    ) as bar:
                        result = retrieval_metrics(
                            prediction.predictions,
                            prediction.label_ids,
                            device,
                            gpus=gpus,
                        )
                        bar.update(1)
                return broadcast_object_list([result])[0]

            trainer.compute_metrics = compute_metrics
            for name in evaluation_datasets:
                progress.evaluation_description = f"{name} embeddings"
                evaluated = trainer.evaluate(name, metric_key_prefix=name)
                for metric in ("Hit@1", "MAP@100", "R@5"):
                    metrics[f"{name}/{metric}"] = evaluated[f"{name}_{metric}"]
            trainer.save_model()
            trainer.save_metrics("retrieval", metrics)
            return metrics
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
