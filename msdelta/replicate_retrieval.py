"""MS2 peptide-replicate-retrieval benchmark for the frozen msdelta encoder.

The benchmark dataset (`ms2-peptide-replicate-retrieval`) is a set of real
experimental MS2 spectra, each labelled with its peptide+charge (a PSM), pooled
across runs so every peptide has many replicate acquisitions (~15 spectra per
peptide). It measures whether an embedding pulls same-peptide replicates
together — the core question for a spectrum-embedding model used as a
retrieval/clustering front-end:

  Hit@1   leave-one-out: fraction of spectra whose nearest (cosine) neighbour
          shares the same peptide/charge label.
  MAP     mean average precision of that same leave-one-out ranking.
  PairF1  pairwise F1 of a spherical k-means clustering (k = #labels) against
          the ground-truth labels.

The metric definitions are fixed and self-contained — Hit@1/MAP use exact
leave-one-out denominators (every spectrum counts, singletons score 0) and
PairF1 uses spherical k-means (seed-shuffled init, cosine assignment,
sum-then-L2-normalise centroid update). PairF1 is mildly k-means-init-sensitive,
so `--kmeans-seeds` averages over several inits.

Each parquet carries columns: peptide, charge, mz, intensity, precursor.
Label = f"{peptide}/{charge}"; `precursor` (observed precursor m/z) and `charge`
feed the encoder directly.

Usage:
    msdelta-replicate-retrieval --ckpt runs/v14_cap_XL_spark/last.pt \
        --data-dir data/ms2-peptide-replicate-retrieval [--baseline] [--whiten 16]
"""
from __future__ import annotations

import argparse
import glob
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .data import N_CHARGES, PreprocessConfig, preprocess_spectrum
from .probe import _pool
from .retrieval import all_but_top


# ---------- data ----------

def load_spectra(paths: list[Path]):
    """Load every spectrum from the benchmark parquets.

    Returns mz_list, int_list (python lists of float lists), y (dense int label
    index, labels sorted for a stable mapping), charges (clamped to the
    encoder's charge table), precursors (observed precursor m/z)."""
    mz_list, int_list, labels, charges, precursors = [], [], [], [], []
    for f in paths:
        d = pq.read_table(
            f, columns=["peptide", "charge", "mz", "intensity", "precursor"]
        ).to_pydict()
        for p, c, mz, inten, prec in zip(
            d["peptide"], d["charge"], d["mz"], d["intensity"], d["precursor"]
        ):
            mz_list.append(mz)
            int_list.append(inten)
            labels.append(f"{p}/{c}")
            charges.append(min(int(c), N_CHARGES - 1))
            precursors.append(float(prec) if prec is not None else 0.0)
    uniq = {l: i for i, l in enumerate(sorted(set(labels)))}
    y = np.array([uniq[l] for l in labels], dtype=np.int64)
    return mz_list, int_list, y, np.array(charges), np.array(precursors, dtype=np.float32)


# ---------- embeddings ----------

@torch.no_grad()
def embed_model(enc, mz_list, int_list, charges, precursors, device, pp,
                *, batch_size=128):
    """Encode every spectrum and mean⊕max-pool to one L2-normalised vector.

    Spectra that preprocess to zero peaks get a zero vector (kept so indices
    stay aligned with `y`); they can never be a nearest neighbour of anything
    real and count as misses, matching how an empty spectrum behaves."""
    enc.to(device).eval()
    pre = [preprocess_spectrum(torch.tensor(mz, dtype=torch.float32),
                               torch.tensor(it, dtype=torch.float32), pp)[:2]
           for mz, it in zip(mz_list, int_list)]
    n = len(pre)
    dim_probe = None
    embs = None
    for s in range(0, n, batch_size):
        idx = list(range(s, min(s + batch_size, n)))
        chunk = [pre[i] for i in idx]
        K = max((m.numel() for m, _ in chunk), default=0)
        if K == 0:
            continue
        B = len(chunk)
        mz = torch.zeros(B, K); li = torch.zeros(B, K)
        mask = torch.zeros(B, K, dtype=torch.bool)  # True = real peak
        for b, (m, l) in enumerate(chunk):
            k = m.numel()
            if k:
                mz[b, :k] = m; li[b, :k] = l; mask[b, :k] = True
        chg = torch.tensor([charges[i] for i in idx], dtype=torch.long, device=device)
        pmz = torch.tensor([precursors[i] for i in idx], dtype=torch.float32, device=device)
        tok = enc(mz.to(device), li.to(device), (~mask).to(device),
                  charge=chg, precursor_mz=pmz)
        pooled = _pool(tok, mask.to(device)).cpu().numpy().astype(np.float32)
        if embs is None:
            dim_probe = pooled.shape[1]
            embs = np.zeros((n, dim_probe), dtype=np.float32)
        for j, i in enumerate(idx):
            if mask[j].any():
                embs[i] = pooled[j]
    if embs is None:
        raise RuntimeError("no spectrum produced any peaks after preprocessing")
    return embs


def native_binning(mz_list, int_list, n_bins=65536, mz_min=0.0, mz_max=1000.0):
    """Sparse 65536-bin native-binning baseline:
    log1p intensity ÷ per-spectrum max, m/z → bin, per-bin max, L2-normalise."""
    from scipy import sparse
    span = mz_max - mz_min
    rows, cols, vals = [], [], []
    for r, (mzs, intens) in enumerate(zip(mz_list, int_list)):
        mzs = np.asarray(mzs, dtype=np.float64)
        intens = np.asarray(intens, dtype=np.float64)
        if mzs.size == 0:
            continue
        logi = np.log1p(np.maximum(intens, 0.0))
        mx = logi.max()
        if mx <= 0:
            continue
        v = (logi / mx).astype(np.float32)
        idx = np.floor((mzs - mz_min) / span * n_bins).astype(np.int64)
        keep = (idx >= 0) & (idx < n_bins)
        idx, v = idx[keep], v[keep]
        if idx.size == 0:
            continue
        # per-bin max within this spectrum
        order = np.argsort(idx)
        idx, v = idx[order], v[order]
        ub, inv = np.unique(idx, return_inverse=True)
        bmax = np.zeros(len(ub), dtype=np.float32)
        np.maximum.at(bmax, inv, v)
        nrm = np.sqrt((bmax * bmax).sum()) or 1.0
        rows.append(np.full(len(ub), r)); cols.append(ub); vals.append(bmax / nrm)
    X = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(mz_list), n_bins), dtype=np.float32)
    return X


# ---------- metrics ----------

def _l2norm_dense(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def retrieval_metrics(X, y, *, chunk=256, sparse_in=False):
    """Leave-one-out Hit@1 and MAP. X rows are L2-normalised so cosine = dot.

    Denominators: Hit@1 = hits / N, MAP = Σ AP / N, where every spectrum
    (including singletons, which score 0) counts toward N."""
    n = X.shape[0]
    cnt = Counter(y.tolist())
    hit1 = 0.0
    ap_sum = 0.0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        if sparse_in:
            sims = (X[s:e] @ X.T).toarray().astype(np.float32)
        else:
            sims = X[s:e] @ X.T
        for r in range(e - s):
            qi = s + r
            lq = y[qi]
            rel = cnt[lq] - 1
            sims[r, qi] = -np.inf
            if rel <= 0:
                continue
            order = np.argsort(-sims[r], kind="stable")
            if y[order[0]] == lq:
                hit1 += 1.0
            match = (y[order] == lq)
            ranks = np.flatnonzero(match)[:rel]        # 0-based ranks of relevants
            prec = (np.arange(1, rel + 1)) / (ranks + 1.0)
            ap_sum += prec.sum() / rel
    return hit1 / n, ap_sum / n


def spherical_kmeans(X, k, *, n_iter=25, seed=0):
    """Cosine k-means matching kmeans_pair_f1: random-shuffle init (first k
    points as centroids), assign by max cosine, centroid = L2-normalised sum."""
    rng = np.random.default_rng(seed)
    cen = X[rng.permutation(X.shape[0])[:k]].copy()
    assign = np.full(X.shape[0], -1)
    for _ in range(n_iter):
        new = (X @ cen.T).argmax(1)
        if np.array_equal(new, assign):
            break
        assign = new
        cen = np.zeros((k, X.shape[1]), dtype=np.float32)
        np.add.at(cen, assign, X)
        nrm = np.linalg.norm(cen, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        cen /= nrm
    return assign


def pair_f1(assign, y):
    """Pairwise precision/recall/F1 from the cluster×label contingency table
    (standard pairwise co-membership counting)."""
    pairs = np.stack([assign, y], 1)
    _, cell = np.unique(pairs, axis=0, return_counts=True)
    _, cl = np.unique(assign, return_counts=True)
    _, la = np.unique(y, return_counts=True)
    c2 = lambda a: (a * (a - 1) // 2).sum()
    tp = c2(cell)
    same_cluster = c2(cl)
    same_label = c2(la)
    prec = tp / same_cluster if same_cluster else 0.0
    rec = tp / same_label if same_label else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return float(f1), float(prec), float(rec)


# ---------- GPU-vectorised metrics (same definitions, ~100x faster) ----------

@torch.no_grad()
def retrieval_metrics_torch(X, y, device, *, chunk=1024):
    """Hit@1 / MAP on the GPU. X (n,d) L2-normalised, y (n,) long — both on
    `device`. Fully vectorised per chunk; same denominators as `retrieval_metrics`
    (self is masked to -inf so it ranks last and is never counted)."""
    n = X.shape[0]
    counts = torch.bincount(y, minlength=int(y.max()) + 1)
    rel_total = (counts[y] - 1).float()              # # relevant others per query
    Xt = X.t().contiguous()
    ranks = torch.arange(1, n + 1, device=device, dtype=torch.float32)
    hit1 = 0.0
    ap_sum = 0.0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = X[s:e] @ Xt                            # (c, n)
        rows = torch.arange(e - s, device=device)
        sims[rows, torch.arange(s, e, device=device)] = float("-inf")  # mask self
        order = sims.argsort(dim=1, descending=True)
        match = (y[order] == y[s:e, None])           # (c, n) bool
        match[:, -1] = False                          # drop self (always ranked last)
        found = match.cumsum(1).float()
        prec = found / ranks
        rel = rel_total[s:e]
        valid = rel > 0
        ap = (prec * match).sum(1) / rel.clamp(min=1)
        hit1 += ((match[:, 0]) & valid).sum().item()  # match[:,0] True iff top-1 same label
        ap_sum += (ap * valid).sum().item()
    return hit1 / n, ap_sum / n


@torch.no_grad()
def spherical_kmeans_torch(X, k, device, *, n_iter=25, seed=0):
    """Cosine k-means matching kmeans_pair_f1, on the GPU. Shuffle-init (first k
    points), cosine assignment, centroid = L2-normalised cluster sum."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cen = X[torch.randperm(X.shape[0], generator=g)[:k].to(device)].clone()
    assign = torch.full((X.shape[0],), -1, device=device, dtype=torch.long)
    for _ in range(n_iter):
        new = (X @ cen.t()).argmax(1)
        if torch.equal(new, assign):
            break
        assign = new
        cen = torch.zeros(k, X.shape[1], device=device)
        cen.index_add_(0, assign, X)
        cen = torch.nn.functional.normalize(cen, dim=1, eps=1e-12)
    return assign.cpu().numpy()


def evaluate(X, y, *, sparse_in=False, kmeans_seeds=(0,), whiten=0, name="embed",
             device=None, verbose=True):
    """Score one embedding. Dense inputs run on `device` (GPU-vectorised);
    the sparse native-binning baseline stays on the numpy/CPU path."""
    if whiten > 0 and not sparse_in:
        X = _l2norm_dense(all_but_top(X, whiten))
    elif not sparse_in:
        X = _l2norm_dense(X)
    k = int(y.max()) + 1

    if not sparse_in and device is not None and device.type != "cpu":
        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        yt = torch.as_tensor(y, dtype=torch.long, device=device)
        h1, mAP = retrieval_metrics_torch(Xt, yt, device)
        f1s = [pair_f1(spherical_kmeans_torch(Xt, k, device, seed=sd), y)[0]
               for sd in kmeans_seeds]
    else:
        h1, mAP = retrieval_metrics(X, y, sparse_in=sparse_in)
        Xd = X.toarray().astype(np.float32) if sparse_in else X
        f1s = [pair_f1(spherical_kmeans(Xd, k, seed=sd), y)[0] for sd in kmeans_seeds]
    pf1 = float(np.mean(f1s))
    if verbose:
        print(f"  {name:<24} Hit@1={h1:.4f}  MAP={mAP:.4f}  "
              f"PairF1={pf1:.4f}" + (f"  (±{np.std(f1s):.4f} over {len(f1s)} seeds)"
                                     if len(f1s) > 1 else ""))
    return {"Hit@1": h1, "MAP": mAP, "PairF1": pf1}


# ---------- inline training probe ----------

_SPECTRA_CACHE: dict[str, tuple] = {}


@torch.no_grad()
def replicate_retrieval_inline_metrics(
    enc, data_dir, device, pp, *, whiten=16, kmeans_seeds=1, batch_size=128,
) -> dict[str, float]:
    """Flat wandb dict for the MS2 peptide-replicate-retrieval benchmark, run
    inline at probe cadence during training.

    Encodes the whole benchmark set with the *current* weights and reports the
    three metrics raw and after the all-but-top-`whiten` anisotropy fix. The
    parsed spectra are cached across calls (keyed by `data_dir`) so only the
    encode + metrics recompute each probe step. Returns {} if `data_dir` is
    unset or empty. Restores the encoder's train/eval mode on exit.

    Keys: replicate_retrieval/{Hit@1,MAP,PairF1} and the whitened
    replicate_retrieval/{Hit@1,MAP,PairF1}_w.
    """
    if not data_dir:
        return {}
    key = str(data_dir)
    if key not in _SPECTRA_CACHE:
        paths = [Path(x) for x in sorted(glob.glob(str(Path(data_dir) / "*.parquet")))]
        if not paths:
            return {}
        _SPECTRA_CACHE[key] = load_spectra(paths)
    mz_list, int_list, y, charges, precursors = _SPECTRA_CACHE[key]
    was_training = enc.training
    try:
        emb = embed_model(enc, mz_list, int_list, charges, precursors, device, pp,
                          batch_size=batch_size)
        seeds = tuple(range(kmeans_seeds))
        raw = evaluate(emb, y, kmeans_seeds=seeds, device=device, verbose=False)
        out = {f"replicate_retrieval/{k}": v for k, v in raw.items()}
        if whiten > 0:
            wh = evaluate(emb, y, kmeans_seeds=seeds, whiten=whiten, device=device,
                          verbose=False)
            out.update({f"replicate_retrieval/{k}_w": v for k, v in wh.items()})
        return out
    finally:
        if was_training:
            enc.train()


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    from .analyze import load_encoder

    p = argparse.ArgumentParser(
        description="MS2 peptide-replicate-retrieval benchmark for the msdelta encoder")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--data-dir", type=Path,
                   default=Path("data/ms2-peptide-replicate-retrieval"),
                   help="dir of benchmark *.parquet files")
    p.add_argument("--whiten", type=int, default=0, metavar="K",
                   help="all-but-top-K anisotropy fix on the learned embedding (0=off)")
    p.add_argument("--baseline", action="store_true",
                   help="also score the 65536-bin native-binning baseline (harness check)")
    p.add_argument("--kmeans-seeds", type=int, default=1,
                   help="average PairF1 over this many k-means inits (seeds 0..n-1)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    paths = [Path(x) for x in sorted(glob.glob(str(args.data_dir / "*.parquet")))]
    if not paths:
        raise SystemExit(f"no parquet files under {args.data_dir}")
    print(f"loading {len(paths)} parquet file(s) from {args.data_dir} ...")
    mz_list, int_list, y, charges, precursors = load_spectra(paths)
    print(f"{len(y)} spectra, {int(y.max()) + 1} unique peptide/charge labels\n")
    seeds = tuple(range(args.kmeans_seeds))

    enc, cfg, step = load_encoder(args.ckpt)
    dcfg = cfg["data"]
    pp = PreprocessConfig(intensity_threshold_frac=dcfg["intensity_threshold_frac"],
                          top_n=dcfg["top_n"])
    device = torch.device(args.device)
    print(f"loaded {args.ckpt} (step {step}); encoding ...")
    emb = embed_model(enc, mz_list, int_list, charges, precursors,
                      device, pp, batch_size=args.batch_size)

    print("\n=== MS2 peptide-replicate-retrieval benchmark ===")
    evaluate(emb, y, kmeans_seeds=seeds, name="msdelta (learned)", device=device)
    if args.whiten > 0:
        evaluate(emb, y, kmeans_seeds=seeds, whiten=args.whiten,
                 name=f"msdelta (whiten-{args.whiten})", device=device)
    if args.baseline:
        print("building native-binning baseline (65536 bins) ...")
        Xb = native_binning(mz_list, int_list)
        evaluate(Xb, y, sparse_in=True, kmeans_seeds=seeds, name="native binning 65536")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
