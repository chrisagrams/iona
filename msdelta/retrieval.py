"""Evaluate trained MSDelta retrieval embeddings."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import cast

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP, RetrievalRecall
from transformers import ProgressCallback, Trainer, TrainingArguments

from msdelta.configuration_msdelta import MSDeltaConfig, MSDeltaRetrievalConfig
from msdelta.modeling_msdelta import (
    MSDeltaForPreTraining,
    MSDeltaForRetrieval,
)
from msdelta.processing_msdelta import MSDeltaDataCollatorForRetrieval


def load_benchmark(repo_id: str, split: str = "test"):
    """Load the replicate benchmark and create its label vector."""
    dataset = load_dataset(repo_id, split=split)
    labels = _label_index(
        [f"{peptide}/{charge}" for peptide, charge in zip(dataset["peptide"], dataset["charge"])]
    )
    return dataset, labels


def _label_index(labels) -> np.ndarray:
    """Create dense indices in first-seen order."""
    seen: dict[str, int] = {}
    return np.fromiter(
        (seen.setdefault(label, len(seen)) for label in labels),
        dtype=np.int64,
        count=len(labels),
    )


def _collect_benchmark(dataset, processor, *, batch_size: int = 512):
    """Preprocess each benchmark spectrum in bounded batches."""
    spectra = []
    for start in range(0, len(dataset), batch_size):
        rows = dataset[start : min(start + batch_size, len(dataset))]
        for mz, intensity in zip(rows["mz"], rows["intensity"]):
            processed = processor(mz, intensity, padding=False)
            spectra.append(
                (
                    torch.tensor(processed["mz"], dtype=torch.float32),
                    torch.tensor(processed["log_intensity"], dtype=torch.float32),
                )
            )
    return spectra


@torch.no_grad()
def embed_retrieval_spectra(
    model: MSDeltaForRetrieval,
    spectra,
    device: torch.device,
    *,
    batch_size: int = 128,
) -> np.ndarray:
    """Encode preprocessed spectra with the trained retrieval head."""
    was_training = model.training
    model.to(device).eval()
    chunks = []
    try:
        for start in range(0, len(spectra), batch_size):
            batch = spectra[start : start + batch_size]
            mzs = [mass for mass, _ in batch]
            intensities = [intensity for _, intensity in batch]
            mz = pad_sequence(mzs, batch_first=True).to(device)
            log_intensity = pad_sequence(intensities, batch_first=True).to(device)
            lengths = torch.tensor([mass.numel() for mass in mzs], device=device)
            attention_mask = torch.arange(mz.shape[1], device=device)[None, :] < lengths[:, None]
            output = model(
                mz=mz,
                log_intensity=log_intensity,
                attention_mask=attention_mask,
            )
            chunks.append(output.embeddings.cpu())
        if not chunks:
            raise RuntimeError("no spectra were available for retrieval evaluation")
        return torch.cat(chunks).numpy().astype(np.float32)
    finally:
        model.train(was_training)


@torch.no_grad()
def retrieval_metrics(embeddings, labels, device, *, ks=(5,), chunk: int = 1024):
    """Calculate leave-one-out metrics from normalized retrieval embeddings."""
    vectors = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    targets = torch.as_tensor(labels, dtype=torch.long, device=device)
    count = vectors.shape[0]
    hit_rate = RetrievalHitRate(top_k=1, empty_target_action="neg", sync_on_compute=False)
    mean_ap = RetrievalMAP(empty_target_action="neg", sync_on_compute=False)
    recalls = {
        k: RetrievalRecall(top_k=k, empty_target_action="neg", sync_on_compute=False) for k in ks
    }
    transpose = vectors.t().contiguous()
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        rows = torch.arange(end - start, device=device)
        columns = torch.arange(start, end, device=device)
        similarities = vectors[start:end] @ transpose
        relevant = targets[start:end, None] == targets[None, :]
        relevant[rows, columns] = False
        similarities[rows, columns] = float("-inf")
        indexes = columns[:, None].expand(-1, count)
        predictions = similarities.reshape(-1).cpu()
        flattened_targets = relevant.reshape(-1).cpu()
        flattened_indexes = indexes.reshape(-1).cpu()
        hit_rate.update(predictions, flattened_targets, flattened_indexes)
        mean_ap.update(predictions, flattened_targets, flattened_indexes)
        for metric in recalls.values():
            metric.update(predictions, flattened_targets, flattened_indexes)
    result = {"Hit@1": hit_rate.compute().item(), "MAP": mean_ap.compute().item()}
    result.update({f"R@{k}": metric.compute().item() for k, metric in recalls.items()})
    return result


@torch.no_grad()
def retrieval_dataset_metrics(
    model: MSDeltaForRetrieval,
    dataset,
    device: torch.device,
    *,
    max_analytes: int = 1000,
    batch_size: int = 128,
) -> dict[str, float]:
    """Evaluate grouped validation spectra with leave-one-out retrieval."""
    spectra = []
    labels = []
    subset = dataset.select(range(min(len(dataset), max_analytes)))
    for group_id, row in enumerate(subset):
        for mz, log_intensity in zip(row["mz"], row["log_intensity"]):
            spectra.append(
                (
                    torch.tensor(mz, dtype=torch.float32),
                    torch.tensor(log_intensity, dtype=torch.float32),
                )
            )
            labels.append(group_id)
    embeddings = embed_retrieval_spectra(model, spectra, device, batch_size=batch_size)
    metrics = retrieval_metrics(embeddings, labels, device, ks=(5,))
    return {f"retrieval/{name}": value for name, value in metrics.items()}


@torch.no_grad()
def replicate_retrieval_inline_metrics(
    model: MSDeltaForRetrieval,
    repo_id: str | None,
    device: torch.device,
    processor,
    *,
    split: str = "test",
    batch_size: int = 128,
) -> dict[str, float]:
    """Evaluate trained retrieval embeddings on the external replicate benchmark."""
    if not repo_id:
        return {}
    dataset, labels = load_benchmark(repo_id, split=split)
    spectra = _collect_benchmark(dataset, processor)
    embeddings = embed_retrieval_spectra(model, spectra, device, batch_size=batch_size)
    metrics = retrieval_metrics(embeddings, labels, device, ks=(5,))
    return {f"replicate_retrieval/{name}": value for name, value in metrics.items()}


class RetrievalProgressCallback(ProgressCallback):
    """Label the nested Trainer's progress bars."""

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if self.training_bar is not None:
            self.training_bar.set_description("Retrieval train")

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        super().on_prediction_step(args, state, control, eval_dataloader=eval_dataloader, **kwargs)
        if self.prediction_bar is not None:
            self.prediction_bar.set_description("Retrieval eval")


def run_retrieval_probe(
    module: MSDeltaForPreTraining,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    output_dir: Path,
    processor,
    replicate_repo_id: str | None,
    projection_hidden_size: int,
    embedding_size: int,
    dropout: float,
    temperature: float,
    validation_analytes: int,
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
    cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []

    try:
        with torch.random.fork_rng(devices=cuda_devices):
            probe_encoder = copy.deepcopy(module.msdelta)
            model = MSDeltaForRetrieval(
                config,
                encoder=probe_encoder,
                freeze_encoder=True,
            )
            args = copy.deepcopy(training_args)
            args.output_dir = str(output_dir)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=MSDeltaDataCollatorForRetrieval(),
                processing_class=processor,
            )
            trainer.remove_callback(ProgressCallback)
            trainer.add_callback(RetrievalProgressCallback())
            trainer.train()
            evaluated = trainer.evaluate()
            metrics = {"retrieval/loss": evaluated["eval_loss"]}
            if trainer.is_world_process_zero():
                metrics.update(
                    retrieval_dataset_metrics(
                        model,
                        validation_dataset,
                        device,
                        max_analytes=validation_analytes,
                    )
                )
                metrics.update(
                    replicate_retrieval_inline_metrics(
                        model,
                        replicate_repo_id,
                        device,
                        processor,
                    )
                )
            trainer.save_model()
            trainer.save_metrics("retrieval", metrics)
            return metrics
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
