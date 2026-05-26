"""Spectrum-retrieval precision/recall eval — the model as an embedding model.

Pools the frozen encoder into one vector per spectrum and measures how well
same-peptide replicate spectra retrieve each other (leave-one-out), against a
classic binned-spectral-cosine baseline. Ground truth = peptide_charge.

The data has ~225 genuine replicate spectra per peptide, so positives are
abundant. Reports both retrieval metrics (mAP, P@1, R@k) and the threshold-free
pairwise precision-recall (average precision), for the learned embedding and
the baseline. Also reports train/val peptide overlap (a high overlap means the
result is "retrieve seen peptides", not generalization).

Usage:
    msdelta-retrieval --ckpt runs/<run>/final.pt
"""
from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score

from .data import PreprocessConfig, preprocess_spectrum, split_paths
from .probe import _pool

_BARE = re.compile(r"_\d+$")  # strip the trailing _z charge suffix


def _bare_peptide(pc: str) -> str:
    return _BARE.sub("", pc)


def _iter(paths, max_rows):
    n = 0
    for p in paths:
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["peptide_charge", "m/z", "int"])
            pcs = t.column("peptide_charge").to_pylist()
            mzc, itc = t.column("m/z"), t.column("int")
            for i in range(len(t)):
                yield pcs[i], mzc[i].as_py(), itc[i].as_py()
                n += 1
                if n >= max_rows:
                    return


# ---------- extraction ----------

@torch.no_grad()
def extract(enc, paths, device, pp, *, max_peptides=150, per_peptide=25,
            batch_size=128, bin_width=1.0, mz_max=2000.0, max_rows=120_000):
    """Collect a balanced set: up to `max_peptides` peptides × `per_peptide`
    replicates (data is grouped by peptide, so we bucket and cap rather than
    read sequentially). Returns learned embeddings, binned-cosine baseline
    vectors, and peptide_charge labels."""
    enc.to(device).eval()
    n_bins = int(mz_max / bin_width)
    buckets: dict[str, list] = collections.defaultdict(list)
    binned_b: dict[str, list] = collections.defaultdict(list)

    for pc, mz_list, int_list in _iter(paths, max_rows):
        if len(buckets[pc]) >= per_peptide:
            continue
        if pc not in buckets and len(buckets) >= max_peptides:
            continue
        mzt = torch.tensor(mz_list, dtype=torch.float32)
        itt = torch.tensor(int_list, dtype=torch.float32)
        mp, lp = preprocess_spectrum(mzt, itt, pp)
        if mp.numel() == 0:
            continue
        buckets[pc].append((mp, lp))
        b = np.zeros(n_bins, dtype=np.float32)
        np.add.at(b, np.clip((mp.numpy() / bin_width).astype(int), 0, n_bins - 1), lp.numpy())
        binned_b[pc].append(b)

    # flatten, keep only peptides with >=2 reps (need positives)
    specs, binned, labels = [], [], []
    for pc, items in buckets.items():
        if len(items) < 2:
            continue
        specs.extend(items)
        binned.extend(binned_b[pc])
        labels.extend([pc] * len(items))

    # encode in batches → pooled embedding
    embs = []
    for s in range(0, len(specs), batch_size):
        chunk = specs[s:s + batch_size]
        K = max(m.numel() for m, _ in chunk)
        mz = torch.zeros(len(chunk), K); li = torch.zeros(len(chunk), K)
        mask = torch.zeros(len(chunk), K, dtype=torch.bool)
        for b, (m, l) in enumerate(chunk):
            k = m.numel(); mz[b, :k] = m; li[b, :k] = l; mask[b, :k] = True
        tok = enc(mz.to(device), li.to(device), (~mask).to(device))
        embs.append(_pool(tok, mask.to(device)).cpu().numpy())

    return {
        "emb": np.concatenate(embs) if embs else np.zeros((0, 1)),
        "binned": np.array(binned),
        "labels": np.array(labels),
    }


# ---------- metrics ----------

def _cosine_sim(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return Xn @ Xn.T


def _ap_from_rel(rel: np.ndarray) -> float:
    """Average precision of a relevance vector already sorted by score (desc)."""
    pos = np.where(rel)[0]
    if len(pos) == 0:
        return np.nan
    precs = (np.arange(len(pos)) + 1) / (pos + 1)
    return float(precs.mean())


def retrieval_metrics(sim: np.ndarray, labels: np.ndarray, ks=(1, 5, 10)) -> dict[str, float]:
    """Leave-one-out: every spectrum queries all others."""
    N = len(labels)
    np.fill_diagonal(sim, -np.inf)  # exclude self
    aps, p1, recall = [], [], {k: [] for k in ks}
    same = labels[None, :] == labels[:, None]
    for i in range(N):
        npos = int(same[i].sum())  # excludes self (diagonal of `same` is True but sim=-inf ranks it last)
        if npos == 0:
            continue
        order = np.argsort(-sim[i])
        rel = same[i][order]
        aps.append(_ap_from_rel(rel))
        p1.append(float(rel[0]))
        for k in ks:
            recall[k].append(rel[:k].sum() / npos)
    out = {"mAP": float(np.nanmean(aps)), "P@1": float(np.mean(p1))}
    for k in ks:
        out[f"R@{k}"] = float(np.mean(recall[k]))
    return out


def pairwise_ap(sim: np.ndarray, labels: np.ndarray) -> float:
    """Threshold-free precision-recall (average precision) over all spectrum pairs."""
    iu = np.triu_indices(len(labels), k=1)
    scores = sim[iu]
    y = (labels[iu[0]] == labels[iu[1]]).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, scores))


def _eval_block(name, X, labels):
    sim = _cosine_sim(X)
    m = retrieval_metrics(sim.copy(), labels)
    m["AUC-PR"] = pairwise_ap(sim, labels)
    print(f"  {name:<18} mAP={m['mAP']:.3f}  P@1={m['P@1']:.3f}  "
          f"R@5={m['R@5']:.3f}  R@10={m['R@10']:.3f}  AUC-PR={m['AUC-PR']:.3f}")
    return m


def peptide_overlap(train_paths, val_labels, max_rows=60_000) -> float:
    val_set = set(val_labels)
    train_set = set()
    for pc, _, _ in _iter(train_paths, max_rows):
        train_set.add(pc)
    inter = val_set & train_set
    return len(inter) / max(1, len(val_set))


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    from .analyze import load_encoder

    p = argparse.ArgumentParser(description="Spectrum-retrieval precision/recall eval")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--max-peptides", type=int, default=150)
    p.add_argument("--per-peptide", type=int, default=25)
    p.add_argument("--bin-width", type=float, default=1.0, help="Da, baseline binning")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--check-overlap", action="store_true",
                   help="report train/val peptide overlap (generalization sanity)")
    args = p.parse_args(argv)

    enc, cfg, step = load_encoder(args.ckpt)
    dcfg = cfg["data"]
    pp = PreprocessConfig(intensity_threshold_frac=dcfg["intensity_threshold_frac"], top_n=dcfg["top_n"])
    train_paths, val_paths = split_paths(dcfg["root"], dcfg["n_val_files"])

    print(f"loaded {args.ckpt} (step {step}); extracting embeddings...")
    d = extract(enc, val_paths, torch.device(args.device), pp,
                max_peptides=args.max_peptides, per_peptide=args.per_peptide,
                bin_width=args.bin_width)
    labels = d["labels"]
    n_pep = len(set(labels))
    print(f"{len(labels)} spectra across {n_pep} peptides "
          f"(~{len(labels)//max(1,n_pep)} reps each)\n")

    print("=== strict positives (same peptide_charge) ===")
    _eval_block("learned embed", d["emb"], labels)
    _eval_block("binned cosine", d["binned"], labels)

    bare = np.array([_bare_peptide(x) for x in labels])
    if len(set(bare)) < n_pep:  # only meaningful if charges collapse
        print("\n=== loose positives (same peptide, any charge) ===")
        _eval_block("learned embed", d["emb"], bare)
        _eval_block("binned cosine", d["binned"], bare)

    if args.check_overlap:
        ov = peptide_overlap(train_paths, labels)
        print(f"\ntrain/val peptide_charge overlap: {ov*100:.1f}% "
              f"({'retrieving SEEN peptides — not generalization' if ov > 0.5 else 'mostly novel peptides'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
