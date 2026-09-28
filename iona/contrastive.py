"""Training and evaluation utilities for contrastive encoder fine-tuning (see IonaForRetrieval)."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.utils.data import Sampler

from iona.data import peptide_key
from iona.modeling_iona import head_kl
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
    """Embed rows and return (embeddings, group ids)."""
    was_training = model.training
    model.eval()
    rows = list(dataset)[:max_rows]
    embeddings = []
    try:
        # no_grad is required: otherwise the graph is kept alive through .cpu().
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                batch = {k: v.to(device) for k, v in collator(chunk).items()}
                pooled, _ = model.embed(batch["mz"], batch["log_intensity"],
                                        batch["attention_mask"])
                embeddings.append(pooled.cpu())
    finally:
        model.train(was_training)
    if not embeddings:
        return None, None
    groups = np.unique(
        np.array([peptide_key(r["peptide"], int(r.get("charge", 0))) for r in rows]),
        return_inverse=True)[1]
    return torch.cat(embeddings), groups


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


def gradcache_step(model, batch, chunk_size: int, accelerator=None) -> dict[str, Tensor]:
    """One GradCache step (Gao et al., 2021): a contrastive batch larger than memory allows.

    Embed chunks without grad, take dL/dembedding over the whole batch, then re-embed each
    chunk with grad and backprop the cached gradient. Exact, at the cost of two forwards.
    """
    keys = ("mz", "log_intensity", "attention_mask")
    total = len(batch["group_ids"])
    chunks = [{k: batch[k][i:i + chunk_size] for k in keys}
              for i in range(0, total, chunk_size)]

    # 1. embeddings only; save RNG state so step 3 replays the same dropout masks.
    states = []
    cached = []
    with torch.no_grad():
        for chunk in chunks:
            states.append((torch.get_rng_state(), *get_device_states(chunk["mz"])))
            cached.append(model.embed(chunk["mz"], chunk["log_intensity"],
                                      chunk["attention_mask"])[0])

    # 2. loss over the whole batch, gradient w.r.t. the embeddings only.
    leaves = [e.detach().requires_grad_(True) for e in cached]
    contrastive = model.contrastive_loss(torch.cat(leaves), batch["group_ids"])
    contrastive.backward()
    grads = [leaf.grad for leaf in leaves]

    # 3. re-embed with grad, push the cached gradient through, add the KL term.
    kl_total = torch.zeros((), device=contrastive.device)
    for chunk, grad, (cpu_state, devices, device_states) in zip(chunks, grads, states):
        torch.set_rng_state(cpu_state)
        set_device_states(devices, device_states, device_type=chunk["mz"].device.type)
        embeddings, hidden = model.embed(chunk["mz"], chunk["log_intensity"],
                                         chunk["attention_mask"])
        surrogate = (embeddings * grad).sum()
        if model.config.kl_weight > 0 and model.reference is not None:
            logits = model.intensity_head(hidden)
            reference_logits = model.reference_logits(chunk["mz"], chunk["log_intensity"],
                                                      chunk["attention_mask"])
            # Scaled by the chunk's share so the total matches a full-batch step.
            share = len(chunk["mz"]) / total
            kl = head_kl(logits, reference_logits, chunk["attention_mask"])
            kl_total = kl_total + kl.detach() * share
            surrogate = surrogate + model.config.kl_weight * kl * share
        if accelerator is not None:
            accelerator.backward(surrogate)
        else:
            surrogate.backward()

    return {"loss": (contrastive.detach() + model.config.kl_weight * kl_total),
            "contrastive": contrastive.detach(), "kl": kl_total}
