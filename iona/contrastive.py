"""Training and evaluation utilities for contrastive encoder fine-tuning (see IonaForRetrieval)."""

from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset
from torch import Tensor, nn
from torch.utils.data import Sampler

from iona.data import peptide_key
from iona.inference import PredictionTrainer
from iona.reranking import group_separation_metrics
from iona.retrieval import retrieval_metrics


class GroupBatchSampler(Sampler[list[int]]):
    """P groups x K replicates per batch, so every batch has positives.

    Reshuffles each epoch on its own; HF Trainer never calls set_epoch on a custom batch_sampler.
    """

    def __init__(self, groups, groups_per_batch: int = 12, replicates: int = 4,
                 seed: int = 0, drop_last: bool = True):
        if replicates < 2:
            raise ValueError("replicates must be >= 2 or there are no positive pairs")
        if groups_per_batch < 2:
            raise ValueError(
                "groups_per_batch must be >= 2 or a batch holds one peptide and has no "
                "NEGATIVES: every off-diagonal entry is a positive, the loss collapses "
                "to the constant log(replicates - 1), and nothing is learned")
        self.groups_per_batch = groups_per_batch
        self.replicates = replicates
        self.seed = seed
        self.drop_last = drop_last
        groups = np.asarray(groups)
        order = np.argsort(groups, kind="stable")
        keys, starts = np.unique(groups[order], return_index=True)
        self.members: dict[int, np.ndarray] = {
            int(k): members for k, members in zip(keys, np.split(order, starts[1:]))}
        self.epoch = 0

    def __len__(self) -> int:
        return max(len(self.members) // self.groups_per_batch, 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        # Seed as a pair: seed+epoch would collide across seeds.
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(list(self.members))
        for start in range(0, len(order) - self.groups_per_batch + 1,
                           self.groups_per_batch):
            batch: list[int] = []
            for group in order[start : start + self.groups_per_batch]:
                pool = self.members[int(group)]
                take = rng.choice(pool, size=self.replicates,
                                  replace=len(pool) < self.replicates)
                batch.extend(int(i) for i in take)
            yield batch
        self.epoch += 1


def encoder_layer_states(encoder: nn.Module, mz: Tensor, log_intensity: Tensor,
                         attention_mask: Tensor) -> tuple[list[Tensor], Tensor]:
    """Run the encoder and capture every block's output (via forward hooks) plus the input embedding."""
    captured: dict[int, Tensor] = {}
    handles = [encoder.embed.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__(0, out))]
    for index, block in enumerate(encoder.blocks, start=1):
        handles.append(block.register_forward_hook(
            lambda _m, _i, out, index=index: captured.__setitem__(index, out)))
    try:
        final = encoder(mz=mz, log_intensity=log_intensity,
                        attention_mask=attention_mask).last_hidden_state
    finally:
        for handle in handles:
            handle.remove()
    expected = len(encoder.blocks) + 1
    if len(captured) != expected:
        raise RuntimeError(f"captured {len(captured)} of {expected} layer states")
    return [captured[i] for i in range(expected)], final


@torch.no_grad()
def group_separation_summary(model, dataset, collator, device, max_rows: int = 2000,
                             batch_size: int = 16) -> dict[str, float]:
    """Embed a validation split and report how well replicates cluster."""
    embeddings, groups = embed_dataset(model, dataset, collator, device,
                                       max_rows=max_rows, batch_size=batch_size)
    if embeddings is None:
        return {}
    return group_separation_metrics(embeddings, groups, "sep_spectrum")


def embed_dataset(model, dataset, collator, device, max_rows: int = 2000,
                  batch_size: int = 16):
    """Embed rows and return (embeddings, group ids).

    Every process of a distributed run must call this: encoding is sharded across ranks.
    """
    if not isinstance(dataset, Dataset):
        dataset = Dataset.from_list(list(dataset))
    rows = dataset.select(range(min(len(dataset), max_rows)))
    if not len(rows):
        return None, None
    was_training = model.training
    trainer = PredictionTrainer(
        model,
        lambda m, x: m.embed(x["mz"], x["log_intensity"], x["attention_mask"]),
        data_collator=collator, batch_size=batch_size, device=device,
    )
    try:
        embeddings = trainer.predict_sorted(rows)
    finally:
        model.train(was_training)
    charges = rows["charge"] if "charge" in rows.column_names else [0] * len(rows)
    groups = np.unique(
        np.array([peptide_key(p, int(c)) for p, c in zip(rows["peptide"], charges)]),
        return_inverse=True)[1]
    return torch.from_numpy(embeddings), groups


def retrieval_summary(model, dataset, collator, device, max_rows: int = 2000,
                      batch_size: int = 16) -> dict[str, float]:
    """Retrieval metrics on a validation split, prefixed with retrieval/."""
    embeddings, groups = embed_dataset(model, dataset, collator, device,
                                       max_rows=max_rows, batch_size=batch_size)
    if embeddings is None or len(embeddings) < 3:
        return {}
    counts = np.bincount(groups)
    if (counts > 1).sum() < 2:
        return {}
    scores = retrieval_metrics(embeddings, groups, device)
    if not scores:
        return {}
    return {f"retrieval/{k}": v for k, v in scores.items()} | {
        "retrieval/queries": float(len(embeddings)),
        "retrieval/groups": float(len(counts)),
        "retrieval/scorable_groups": float((counts > 1).sum()),
    }


def subset_by_group(dataset, max_samples: int, group_key, min_members: int = 4):
    """Cap a split by whole groups, not by row, so replicate structure is kept."""
    if max_samples <= 0:
        return dataset
    keys = np.array([group_key(row) for row in dataset])
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    keep, total = set(), 0
    for key, size in sorted(counts.items(), key=lambda kv: -kv[1]):
        if size < min_members:
            continue
        if total + size > max_samples and keep:
            break
        keep.add(key)
        total += size
    indices = [i for i, key in enumerate(keys) if key in keep]
    return dataset.select(indices)
