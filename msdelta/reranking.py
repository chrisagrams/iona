"""Train and evaluate peptide/spectrum rerankers with a frozen MSDelta backbone."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import faiss
import numpy as np
import torch
from accelerate.utils import broadcast_object_list
from pytorch_metric_learning.utils.inference import FaissKNN
from transformers import ProgressCallback, Trainer, TrainingArguments

from msdelta.configuration_msdelta import MSDeltaRerankingConfig
from msdelta.modeling_msdelta import MSDeltaForPreTraining, MSDeltaForReranking
from msdelta.processing_msdelta import MSDeltaDataCollatorForReranking, MSDeltaRerankingProcessor


class RerankingTrainer(Trainer):
    """Gather each modality independently, preserving global peptide labels."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if "evaluation_labels" not in inputs:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        inputs = dict(inputs)
        labels = self._prepare_input(inputs.pop("evaluation_labels"))
        loss, embeddings, _ = super().prediction_step(model, inputs, False, ignore_keys)
        return loss, embeddings, labels


class RerankingProgressCallback(ProgressCallback):
    """Label the nested Trainer's training and evaluation progress bars."""

    evaluation_description = "Reranking loss"

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if self.training_bar is not None:
            self.training_bar.set_description("Reranking train")

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        super().on_prediction_step(args, state, control, eval_dataloader=eval_dataloader, **kwargs)
        if self.prediction_bar is not None:
            self.prediction_bar.set_description(self.evaluation_description)


@torch.no_grad()
def reranking_metrics(
    spectrum_embeddings,
    spectrum_labels,
    peptide_embeddings,
    peptide_labels,
    device,
    *,
    gpus=None,
    query_batch_size=256,
    k=None,
) -> dict[str, float]:
    """Exact full-gallery MRR, or explicitly truncated MRR@k with zero for misses."""
    spectra = torch.nn.functional.normalize(
        torch.as_tensor(spectrum_embeddings, dtype=torch.float32, device=device), dim=-1
    )
    peptides = torch.nn.functional.normalize(
        torch.as_tensor(peptide_embeddings, dtype=torch.float32, device=device), dim=-1
    )
    queries = torch.as_tensor(spectrum_labels, dtype=torch.long, device=device)
    gallery = torch.as_tensor(peptide_labels, dtype=torch.long, device=device)
    if not len(queries) or not len(gallery) or query_batch_size <= 0:
        raise ValueError("evaluation requires nonempty queries/gallery and positive batch size")
    if spectra.ndim != 2 or peptides.ndim != 2 or spectra.shape[1] != peptides.shape[1]:
        raise ValueError("spectrum and peptide embeddings must share their embedding dimension")
    if len(spectra) != len(queries) or len(peptides) != len(gallery):
        raise ValueError("evaluation labels must align with embeddings")
    if len(gallery.unique()) != len(gallery):
        raise ValueError("gallery must contain one entry per modified peptide")
    if not torch.isin(queries, gallery).all():
        raise ValueError("every spectrum must have a matching peptide in the gallery")
    if not torch.isfinite(spectra).all() or not torch.isfinite(peptides).all():
        raise ValueError("embeddings must be finite")
    if k is not None and k <= 0:
        raise ValueError("k must be positive")
    search_k = len(gallery) if k is None else min(k, len(gallery))
    # Reuse the existing FAISS wrapper; it falls back to CPU for large full-gallery k.
    knn = FaissKNN(
        index_init_fn=faiss.IndexFlatIP, gpus=gpus, reset_before=False, reset_after=False
    )
    knn.train(peptides)
    reciprocal_sum = hit1 = hit5 = 0.0
    try:
        for start in range(0, len(queries), query_batch_size):
            # Always retrieve at least five entries to keep Hit@5 independent of MRR cutoff.
            _, indices = knn(
                spectra[start : start + query_batch_size], min(len(gallery), max(5, search_k))
            )
            matches = gallery[indices] == queries[start : start + query_batch_size, None]
            first = matches[:, :search_k].float().argmax(1) + 1
            found = matches[:, :search_k].any(1)
            reciprocal_sum += (found.float() / first).sum().item()
            hit1 += matches[:, 0].sum().item()
            hit5 += matches[:, :5].any(1).sum().item()
    finally:
        knn.reset()
    metric = "MRR" if k is None else f"MRR@{k}"
    return {
        metric: reciprocal_sum / len(queries),
        "Hit@1": hit1 / len(queries),
        "Hit@5": hit5 / len(queries),
    }


def run_reranking_probe(
    module: MSDeltaForPreTraining,
    train_dataset,
    validation_dataset,
    *,
    output_dir: Path,
    processor: MSDeltaRerankingProcessor,
    config: MSDeltaRerankingConfig,
    training_args: TrainingArguments,
    evaluation_datasets: dict,
) -> dict[str, float]:
    """Train a fresh dual encoder without modifying the caller's encoder or RNG state."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    device = next(module.parameters()).device
    cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []
    if config.peptide_vocab != processor.peptide_vocab or (
        config.peptide_max_length != processor.peptide_max_length
    ):
        raise ValueError("model and processor peptide settings must agree")
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            # Initialize fresh heads deterministically at every probe step.
            torch.manual_seed(training_args.seed)
            model = MSDeltaForReranking(
                copy.deepcopy(config), encoder=copy.deepcopy(module.msdelta)
            )
            args = copy.deepcopy(training_args)
            args.output_dir = str(output_dir)
            trainer = RerankingTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=MSDeltaDataCollatorForReranking(),
                processing_class=processor,
            )
            trainer.remove_callback(ProgressCallback)
            progress = RerankingProgressCallback()
            trainer.add_callback(progress)
            trainer.train()
            metrics = {"reranking/loss": trainer.evaluate()["eval_loss"]}
            trainer.args.prediction_loss_only = False
            trainer.args.dataloader_drop_last = False
            gpus = (
                list(range(int(os.environ.get("LOCAL_WORLD_SIZE", "1"))))
                if (device.type == "cuda")
                else None
            )
            for name, datasets in evaluation_datasets.items():
                progress.evaluation_description = f"{name} spectra"
                spectra = trainer.predict(datasets["spectra"])
                progress.evaluation_description = f"{name} peptides"
                peptides = trainer.predict(datasets["peptides"])
                trainer.accelerator.wait_for_everyone()
                result = {}
                if trainer.is_world_process_zero():
                    result = reranking_metrics(
                        spectra.predictions,
                        spectra.label_ids,
                        peptides.predictions,
                        peptides.label_ids,
                        trainer.args.device,
                        gpus=gpus,
                    )
                result = broadcast_object_list([result])[0]
                metrics.update({f"{name}/{key}": value for key, value in result.items()})
            trainer.save_model()
            trainer.save_metrics("reranking", metrics)
            return metrics
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
