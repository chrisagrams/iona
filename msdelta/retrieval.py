"""Spectrum-retrieval eval — the model as an embedding model.

Pools the frozen encoder to one vector per spectrum and measures how well
same-peptide replicate spectra retrieve each other (leave-one-out cosine).
Ground truth = peptide+charge. One eval, two datasets:

- **internal**: the preprocessed HF val dataset
  (`peptide_charge` label, already-preprocessed `mz`/`log_int`). Caps to a few
  thousand spectra and also scores a 1-Da binned-cosine baseline, so we can watch
  the learned embedding close the gap on binned-cosine as training proceeds.
  wandb prefix `retrieval/`.
- **external**: the
  `chrisagrams/ms2-peptide-replicate-retrieval` benchmark (raw `mz`/`intensity`
  + `charge`/`precursor` columns, preprocessed on the fly). Encodes every
  spectrum and reports raw plus PCA-whitened metrics. wandb prefix
  `replicate_retrieval/`.

Metrics come from **torchmetrics** (`RetrievalHitRate`/`RetrievalMAP`/
`RetrievalRecall` + `BinaryAveragePrecision`), so there is one exact,
well-tested definition for both datasets. All are strict leave-one-out: the
query itself is excluded (never a positive, ranked last), every spectrum counts
toward the denominator, and singletons score 0 (`empty_target_action="neg"`).

Distributed extraction is orchestrated by ``MSDeltaTrainer.evaluate``.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torchmetrics.classification import BinaryAveragePrecision
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP, RetrievalRecall

from .data import preprocess_spectrum


# ---------- data adapters ----------

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


def build_internal_retrieval_dataset(
    dataset,
    *,
    max_peptides: int = 100,
    per_peptide: int = 20,
    max_scan: int = 150_000,
) -> tuple[Dataset, np.ndarray]:
    """Create the deterministic internal retrieval corpus and baseline."""
    specs, labels, binned = _collect_capped(
        _prepped_from_dataset(dataset, max_scan), max_peptides, per_peptide
    )
    rows = {
        "mz": [mz.tolist() for mz, _ in specs],
        "log_int": [li.tolist() for _, li in specs],
        "peptide_charge": labels,
        "log_tic": [0.0] * len(specs),
        "_eval_id": list(range(len(specs))),
    }
    return Dataset.from_dict(rows), binned


def build_external_retrieval_dataset(repo_id: str, pp, split: str = "test"):
    """Preprocess the external benchmark once into the common eval schema."""
    raw = load_dataset(repo_id, split=split)

    def convert(example, index):
        intensity = torch.tensor(example["intensity"], dtype=torch.float32)
        mz, log_int, _ = preprocess_spectrum(
            torch.tensor(example["mz"], dtype=torch.float32), intensity, pp
        )
        return {
            "mz": mz.tolist(),
            "log_int": log_int.tolist(),
            "peptide_charge": f"{example['peptide']}_{example['charge']}",
            "log_tic": float(torch.log1p(intensity.sum())) if intensity.numel() else 0.0,
            "_eval_id": index,
        }

    return raw.map(
        convert,
        with_indices=True,
        remove_columns=raw.column_names,
        desc="preprocess retrieval benchmark",
    )


# ---------- metrics ----------

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
    # sync_on_compute=False: distributed extraction has already gathered the
    # complete corpus, and metric computation runs on rank zero. The metric
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
