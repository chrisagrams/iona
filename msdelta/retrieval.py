"""Spectrum-retrieval eval — the model as an embedding model.

Pools the frozen encoder to one vector per spectrum and measures how well
same-peptide replicate spectra retrieve each other (leave-one-out cosine).
Ground truth = peptide+charge. One eval, two datasets:

- **internal** (`retrieval_inline_metrics`): the preprocessed HF val dataset
  (`peptide_charge` label, already-preprocessed `mz`/`log_int`). Caps to a few
  thousand spectra and also scores a 1-Da binned-cosine baseline, so we can watch
  the learned embedding close the gap on binned-cosine as training proceeds.
  wandb prefix `retrieval/`.
- **external** (`replicate_retrieval_inline_metrics`): the
  `chrisagrams/ms2-peptide-replicate-retrieval` benchmark (raw `mz`/`intensity`
  + `charge`/`precursor` columns, preprocessed on the fly). Encodes every
  spectrum and reports the metrics raw and after the all-but-top-K anisotropy
  fix. wandb prefix `replicate_retrieval/`.

Metrics come from **torchmetrics** (`RetrievalHitRate`/`RetrievalMAP`/
`RetrievalRecall` + `BinaryAveragePrecision`), so there is one exact,
well-tested definition for both datasets. All are strict leave-one-out: the
query itself is excluded (never a positive, ranked last), every spectrum counts
toward the denominator, and singletons score 0 (`empty_target_action="neg"`).

Run inline: `RetrievalCallback` / `ReplicateRetrievalCallback` call the two
`*_inline_metrics` fns at probe cadence and log to wandb.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
from datasets import load_dataset
from torchmetrics.classification import BinaryAveragePrecision
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP, RetrievalRecall

from .data import preprocess_spectrum
from .embedding import embed_spectra


# ---------- data adapters ----------

def load_benchmark(repo_id: str, split: str = "test"):
    """Open the external benchmark HF dataset and build its label vector.

    `load_dataset` serves the prepared Arrow tables from the local HF cache, so
    calling this once per probe step is cheap (set HF_HUB_OFFLINE=1 to skip the
    hub freshness check). Returns (ds, y): the Arrow-backed `Dataset` and the
    dense peptide+charge label index for every spectrum (first-seen order; the
    metrics are label-permutation invariant)."""
    ds = load_dataset(repo_id, split=split)
    y = _label_index([f"{p}/{c}" for p, c in zip(ds["peptide"], ds["charge"])])
    return ds, y


def _prepped_from_dataset(dataset, max_rows):
    """(peptide_charge, mz_p, log_int) rows from a preprocessed HF dataset —
    the internal path, no per-spectrum transform."""
    for row in itertools.islice(dataset, max_rows):
        yield (row["peptide_charge"],
               torch.tensor(row["mz"], dtype=torch.float32),
               torch.tensor(row["log_int"], dtype=torch.float32))


def _label_index(labels) -> np.ndarray:
    """Dense first-seen label index for a sequence of label strings."""
    seen: dict[str, int] = {}
    return np.fromiter((seen.setdefault(l, len(seen)) for l in labels),
                       dtype=np.int64, count=len(labels))


# ---------- collectors (dataset-specific input assembly) ----------

def _collect_capped(prepped, max_peptides, per_peptide, *, bin_width=1.0, mz_max=2000.0):
    """Bucket already-preprocessed spectra by label (cap `per_peptide` each, up
    to `max_peptides`; keep only peptides with >=2 reps), and build the 1-Da
    binned-cosine baseline vectors alongside. Returns
    (specs, labels, binned)."""
    n_bins = int(mz_max / bin_width)
    buckets: dict[str, list] = {}     # plain dict — do NOT auto-create entries
    binned_b: dict[str, list] = {}
    for pc, mp, lp in prepped:
        existing = buckets.get(pc)
        if existing is not None:
            if len(existing) >= per_peptide:
                continue
        elif len(buckets) >= max_peptides:
            continue  # hit the peptide cap; skip new peptides (bounds memory)
        if mp.numel() == 0:
            continue
        buckets.setdefault(pc, []).append((mp, lp))
        b = np.zeros(n_bins, dtype=np.float32)
        np.add.at(b, np.clip((mp.numpy() / bin_width).astype(int), 0, n_bins - 1), lp.numpy())
        binned_b.setdefault(pc, []).append(b)
        # Early exit once every bucket is full — avoids scanning the whole file.
        if len(buckets) >= max_peptides and all(len(v) >= per_peptide for v in buckets.values()):
            break

    specs, binned, labels = [], [], []
    for pc, items in buckets.items():
        if len(items) < 2:                     # need >=2 reps to have a positive
            continue
        specs.extend(items)
        binned.extend(binned_b[pc])
        labels.extend([pc] * len(items))
    return specs, labels, np.array(binned, dtype=np.float32)


def _collect_benchmark(ds, pp, *, batch_size=512):
    """Preprocess every benchmark spectrum on the fly (raw mz/intensity), reading
    the Arrow map in slices so the raw lists never all resident at once. Returns
    `specs` aligned 1:1 with `ds` rows — empty spectra keep an empty tensor so
    indices stay aligned with `load_benchmark`'s `y` (they encode to a zero
    vector and can never be a real neighbour, i.e. a miss)."""
    specs = []
    n = len(ds)
    for s in range(0, n, batch_size):
        rows = ds[s:min(s + batch_size, n)]
        for mz, it in zip(rows["mz"], rows["intensity"]):
            mp, lp = preprocess_spectrum(
                torch.tensor(mz, dtype=torch.float32),
                torch.tensor(it, dtype=torch.float32), pp)[:2]
            specs.append((mp, lp))
    return specs


# ---------- metrics ----------

def all_but_top(X: np.ndarray, k: int) -> np.ndarray:
    """Anisotropy fix (Mu & Viswanath): center, remove the top-k principal
    directions. Transformer embeddings are dominated by a few common
    directions that drown the discriminative signal; removing them recovers
    most of the retrieval gap with zero training."""
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
    """Leave-one-out retrieval metrics via torchmetrics. `X` (n,d) is
    L2-normalised so inner product = cosine; `y` (n,) int labels. Self is
    excluded (never a positive, ranked last); every spectrum counts toward the
    denominator and singletons score 0 (`empty_target_action="neg"`).

    Returns {"P@1", "mAP", "R@k"...} and, if `pairwise`, "AUC-PR" (threshold-free
    average precision over all i<j spectrum pairs).

    Metric state accumulates on CPU while the similarity chunks are computed on
    `device`, so GPU memory stays O(chunk*n); only the host holds the O(n^2)
    torchmetrics pair state.
    """
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    n = Xt.shape[0]
    # sync_on_compute=False: this runs only on rank 0 (the inline callbacks are
    # world-process-zero gated) over its own sampled spectra, and the metric
    # state lives on CPU. Left at the default, .compute() would all_gather that
    # CPU state across the NCCL group — which has no CPU backend and no other
    # rank waiting — raising "No backend type associated with device type cpu".
    p1 = RetrievalHitRate(top_k=1, empty_target_action="neg", sync_on_compute=False)
    mAP = RetrievalMAP(empty_target_action="neg", sync_on_compute=False)
    rec = {k: RetrievalRecall(top_k=k, empty_target_action="neg", sync_on_compute=False)
           for k in ks}
    aucpr = BinaryAveragePrecision(sync_on_compute=False) if pairwise else None
    XT = Xt.t().contiguous()
    cols_all = torch.arange(n, device=device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        rows = torch.arange(e - s, device=device)
        cols = torch.arange(s, e, device=device)
        sims = Xt[s:e] @ XT                              # (c, n) cosine in [-1,1]
        sims = sims + 1.0                                # -> [0,2]: torchmetrics'
        #   RetrievalMAP treats pred<=0 as non-relevant, so raw cosine would
        #   silently drop true replicates at negative cosine. A +1 shift is
        #   rank-preserving and keeps every real candidate strictly positive.
        tgt = (yt[s:e, None] == yt[None, :])             # (c, n) bool
        tgt[rows, cols] = False                          # self is NOT a positive
        sims[rows, cols] = float("-inf")                 # and ranks last
        idx = cols[:, None].expand(-1, n)
        preds_c, tgt_c, idx_c = sims.reshape(-1).cpu(), tgt.reshape(-1).cpu(), idx.reshape(-1).cpu()
        p1.update(preds_c, tgt_c, idx_c)
        mAP.update(preds_c, tgt_c, idx_c)
        for m in rec.values():
            m.update(preds_c, tgt_c, idx_c)
        if aucpr is not None:
            um = cols_all[None, :] > cols[:, None]        # i<j pairs, each once
            aucpr.update(sims[um].cpu(), tgt[um].long().cpu())
    out = {"P@1": p1.compute().item(), "mAP": mAP.compute().item()}
    for k, m in rec.items():
        out[f"R@{k}"] = m.compute().item()
    if aucpr is not None:
        out["AUC-PR"] = aucpr.compute().item()
    return out


def _evaluate(emb, y, device, *, whiten=0, ks=(5,), pairwise=False):
    """L2-normalise (optionally all-but-top-`whiten` first) and score."""
    X = all_but_top(emb, whiten) if whiten > 0 else emb
    return retrieval_metrics_tm(_l2norm(X), y, device, ks=ks, pairwise=pairwise)


# ---------- inline training probes ----------

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
    """Flat wandb dict from internal spectrum retrieval — the model as an
    embedding model, vs a 1-Da binned-cosine baseline.

    `dataset` is a preprocessed HF dataset (`build_preprocessed_dataset`). Caps
    keep it cheap enough to run inline at probe cadence; `_collect_capped`
    early-exits once buckets fill, so it rarely scans all `max_scan` rows. Logs
    the learned mAP/P@1/R@5/AUC-PR, the baseline mAP, and the gap. Restores the
    encoder's train/eval mode on exit.
    """
    was_training = enc.training
    try:
        specs, labels, binned = _collect_capped(
            _prepped_from_dataset(dataset, max_scan), max_peptides, per_peptide)
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
    enc, repo_id, device, pp, *, split="test", whiten=16, batch_size=128,
) -> dict[str, float]:
    """Flat wandb dict for the external MS2 peptide-replicate-retrieval
    benchmark, run inline at probe cadence.

    Encodes the whole benchmark set with the *current* weights and reports the
    metrics raw and after the all-but-top-`whiten` anisotropy fix. Returns {} if
    `repo_id` is unset. Restores the encoder's train/eval mode on exit.

    Keys: replicate_retrieval/{Hit@1,MAP,R@5} and their whitened `_w` variants.
    """
    if not repo_id:
        return {}
    ds, y = load_benchmark(repo_id, split=split)
    was_training = enc.training
    try:
        specs = _collect_benchmark(ds, pp)
        emb = embed_spectra(enc, specs, device, batch_size=batch_size)
        raw = _evaluate(emb, y, device, ks=(5,))
        out = {"replicate_retrieval/Hit@1": raw["P@1"],
               "replicate_retrieval/MAP": raw["mAP"],
               "replicate_retrieval/R@5": raw["R@5"]}
        if whiten > 0:
            wh = _evaluate(emb, y, device, whiten=whiten, ks=(5,))
            out.update({"replicate_retrieval/Hit@1_w": wh["P@1"],
                        "replicate_retrieval/MAP_w": wh["mAP"],
                        "replicate_retrieval/R@5_w": wh["R@5"]})
        return out
    finally:
        if was_training:
            enc.train()
