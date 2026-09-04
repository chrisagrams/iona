"""Measure spectrum retrieval with learned and binned embeddings."""

from __future__ import annotations

import itertools

import numpy as np
import torch
from datasets import load_dataset
from torchmetrics.classification import BinaryAveragePrecision
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP, RetrievalRecall

from msdelta.eval.embedding import embed_spectra


def load_benchmark(repo_id: str, split: str = "test"):
    """Load the benchmark and create its label vector."""
    ds = load_dataset(repo_id, split=split)
    y = _label_index([f"{p}/{c}" for p, c in zip(ds["peptide"], ds["charge"])])
    return ds, y


def _prepped_from_dataset(dataset, max_rows):
    """Yield preprocessed spectrum rows."""
    for row in itertools.islice(dataset, max_rows):
        yield (
            row["peptide_charge"],
            torch.tensor(row["mz"], dtype=torch.float32),
            torch.tensor(row["log_intensity"], dtype=torch.float32),
        )


def _label_index(labels) -> np.ndarray:
    """Create dense indices in first-seen order."""
    seen: dict[str, int] = {}
    return np.fromiter(
        (seen.setdefault(label, len(seen)) for label in labels),
        dtype=np.int64,
        count=len(labels),
    )


def _collect_capped(prepped, max_peptides, per_peptide, *, bin_width=1.0, mz_max=2000.0):
    """Collect capped replicate groups and binned vectors."""
    n_bins = int(mz_max / bin_width)
    buckets: dict[str, list] = {}
    binned_b: dict[str, list] = {}
    for pc, mp, lp in prepped:
        existing = buckets.get(pc)
        if existing is not None:
            if len(existing) >= per_peptide:
                continue
        elif len(buckets) >= max_peptides:
            continue
        if mp.numel() == 0:
            continue
        buckets.setdefault(pc, []).append((mp, lp))
        b = np.zeros(n_bins, dtype=np.float32)
        np.add.at(b, np.clip((mp.numpy() / bin_width).astype(int), 0, n_bins - 1), lp.numpy())
        binned_b.setdefault(pc, []).append(b)
        if len(buckets) >= max_peptides and all(len(v) >= per_peptide for v in buckets.values()):
            break

    specs, binned, labels = [], [], []
    for pc, items in buckets.items():
        if len(items) < 2:
            continue
        specs.extend(items)
        binned.extend(binned_b[pc])
        labels.extend([pc] * len(items))
    return specs, labels, np.array(binned, dtype=np.float32)


def _collect_benchmark(ds, pp, *, batch_size=512):
    """Preprocess each benchmark spectrum in bounded batches."""
    specs = []
    n = len(ds)
    for s in range(0, n, batch_size):
        rows = ds[s : min(s + batch_size, n)]
        for mz, it in zip(rows["mz"], rows["intensity"]):
            processed = pp(mz, it, padding=False)
            mp = torch.tensor(processed["mz"], dtype=torch.float32)
            lp = torch.tensor(processed["log_intensity"], dtype=torch.float32)
            specs.append((mp, lp))
    return specs


def all_but_top(X: np.ndarray, k: int) -> np.ndarray:
    """Center vectors and remove the top principal directions."""
    if k <= 0:
        return X
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:k]
    return Xc - (Xc @ Vk.T) @ Vk


def _l2norm(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


@torch.no_grad()
def retrieval_metrics_tm(X, y, device, *, ks=(5,), pairwise=False, chunk=1024):
    """Calculate leave-one-out retrieval metrics."""
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    n = Xt.shape[0]
    p1 = RetrievalHitRate(top_k=1, empty_target_action="neg", sync_on_compute=False)
    mAP = RetrievalMAP(empty_target_action="neg", sync_on_compute=False)
    rec = {
        k: RetrievalRecall(top_k=k, empty_target_action="neg", sync_on_compute=False) for k in ks
    }
    aucpr = BinaryAveragePrecision(sync_on_compute=False) if pairwise else None
    XT = Xt.t().contiguous()
    cols_all = torch.arange(n, device=device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        rows = torch.arange(e - s, device=device)
        cols = torch.arange(s, e, device=device)
        sims = Xt[s:e] @ XT
        sims = sims + 1.0
        tgt = yt[s:e, None] == yt[None, :]
        tgt[rows, cols] = False
        sims[rows, cols] = float("-inf")
        idx = cols[:, None].expand(-1, n)
        preds_c, tgt_c, idx_c = sims.reshape(-1).cpu(), tgt.reshape(-1).cpu(), idx.reshape(-1).cpu()
        p1.update(preds_c, tgt_c, idx_c)
        mAP.update(preds_c, tgt_c, idx_c)
        for m in rec.values():
            m.update(preds_c, tgt_c, idx_c)
        if aucpr is not None:
            um = cols_all[None, :] > cols[:, None]
            aucpr.update(sims[um].cpu(), tgt[um].long().cpu())
    out = {"P@1": p1.compute().item(), "mAP": mAP.compute().item()}
    for k, m in rec.items():
        out[f"R@{k}"] = m.compute().item()
    if aucpr is not None:
        out["AUC-PR"] = aucpr.compute().item()
    return out


def _evaluate(emb, y, device, *, whiten=0, ks=(5,), pairwise=False):
    """Normalize embeddings and calculate retrieval metrics."""
    X = all_but_top(emb, whiten) if whiten > 0 else emb
    return retrieval_metrics_tm(_l2norm(X), y, device, ks=ks, pairwise=pairwise)


@torch.no_grad()
def retrieval_inline_metrics(
    enc,
    dataset,
    device,
    *,
    max_peptides: int = 100,
    per_peptide: int = 20,
    max_scan: int = 150_000,
    whiten: int = 0,
) -> dict[str, float]:
    """Calculate retrieval metrics for validation data."""
    was_training = enc.training
    try:
        specs, labels, binned = _collect_capped(
            _prepped_from_dataset(dataset, max_scan), max_peptides, per_peptide
        )
        if len(labels) < 2 or len(set(labels)) < 2:
            return {}
        y = _label_index(labels)
        emb = embed_spectra(enc, specs, device)
        m = _evaluate(emb, y, device, whiten=whiten, ks=(5,), pairwise=True)
        bm = _evaluate(binned, y, device, ks=(5,))
        return {
            "retrieval/mAP": m["mAP"],
            "retrieval/P@1": m["P@1"],
            "retrieval/R@5": m["R@5"],
            "retrieval/AUC_PR": m["AUC-PR"],
            "retrieval/binned_mAP": bm["mAP"],
            "retrieval/gap_vs_binned": m["mAP"] - bm["mAP"],
        }
    finally:
        if was_training:
            enc.train()


@torch.no_grad()
def replicate_retrieval_inline_metrics(
    enc,
    repo_id,
    device,
    pp,
    *,
    split="test",
    whiten=16,
    batch_size=128,
) -> dict[str, float]:
    """Calculate retrieval metrics for the external benchmark."""
    if not repo_id:
        return {}
    ds, y = load_benchmark(repo_id, split=split)
    was_training = enc.training
    try:
        specs = _collect_benchmark(ds, pp)
        emb = embed_spectra(enc, specs, device, batch_size=batch_size)
        raw = _evaluate(emb, y, device, ks=(5,))
        out = {
            "replicate_retrieval/Hit@1": raw["P@1"],
            "replicate_retrieval/MAP": raw["mAP"],
            "replicate_retrieval/R@5": raw["R@5"],
        }
        if whiten > 0:
            wh = _evaluate(emb, y, device, whiten=whiten, ks=(5,))
            out.update(
                {
                    "replicate_retrieval/Hit@1_w": wh["P@1"],
                    "replicate_retrieval/MAP_w": wh["mAP"],
                    "replicate_retrieval/R@5_w": wh["R@5"],
                }
            )
        return out
    finally:
        if was_training:
            enc.train()
