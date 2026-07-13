"""Score-tercile replicate-retrieval eval on the 4G held-out consensus set.

Question: does the frozen msdelta encoder pull same-peptide replicates together
*much better* for high-quality spectra? We stratify the held-out spectra into
top / middle / low thirds by a per-spectrum PSM score (`max_score`, the X!Tandem-
style "high score"; `mean_score` also reported) and run the standard
leave-one-out replicate-retrieval metrics (Hit@1 / MAP / PairF1) inside each
tercile, for one or more checkpoints.

The 4G corpus (`/home/cgrams/datasets/4G_dataset/consensus_wsbin_*.parquet`,
~4.18M spectra, ~50 replicates/peptide) is held out from the training of both
checkpoints, so the whole set is fair game as "validation". Retrieval is O(N^2),
so we draw a deterministic peptide subsample (hash of peptide/charge) across all
shards and cap replicates per peptide — reproducible run to run.

Embedding is computed ONCE per checkpoint over the whole subsample; the tercile
split only slices those vectors, so adding `mean_score` alongside `max_score` is
free. Reuses `embed_model` / `evaluate` from replicate_retrieval (the 4G columns
mz/intensity/charge/precursor match its defaults).

Usage:
    python scripts/tercile_retrieval.py \
        --ckpt runs/consensus_xl_28M_7235042/final.pt \
        --ckpt runs/v14_cap_XL_hf_7232144/final.pt \
        --n-peptides 2500 --max-reps 40 --whiten 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from msdelta.analyze import load_encoder
from msdelta.data import PreprocessConfig
from msdelta.replicate_retrieval import embed_model, evaluate, _l2norm_dense
from msdelta.retrieval import all_but_top

DATA_DIR = "/home/cgrams/datasets/4G_dataset"
SCALAR_COLS = ["peptide", "charge", "max_score", "mean_score"]
ARRAY_COLS = ["mz", "intensity", "precursor"]


def _keep_label(label: str, keep_permille: int) -> bool:
    """Deterministic peptide/charge subsample: stable hash, keep the first
    `keep_permille` per-mille of the hash space. Scales the eval pool without
    coordinating across shards, and is identical run to run."""
    h = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(h, "big") % 1000) < keep_permille


def select_pool(data_dir: str, n_peptides: int, max_reps: int):
    """One scalar-only pass over every shard to choose the eval pool.

    Returns a DataFrame with columns [file_idx, local_row, y, max_score,
    mean_score] for the selected spectra — the (file_idx, local_row) pointers
    drive the second (array) read. Labels are subsampled by hash to ~n_peptides
    and each capped at max_reps replicates (first occurrences in shard order).

    Vectorised: labels are factorised once and only the ~unique labels are
    hashed, so the full 4M-row pass stays a few seconds rather than minutes."""
    files = sorted(Path(data_dir).glob("consensus_wsbin_*.parquet"))
    if not files:
        raise SystemExit(f"no consensus_wsbin_*.parquet under {data_dir}")

    peptide_of, charge_of, mx_of, mn_of, file_of, local_of = [], [], [], [], [], []
    for fi, f in enumerate(files):
        t = pq.ParquetFile(f).read(columns=SCALAR_COLS)
        pep = t.column("peptide").to_pandas()               # object Series
        chg = t.column("charge").to_numpy()
        peptide_of.append(pep)
        charge_of.append(chg)
        mx_of.append(np.asarray(t.column("max_score").to_numpy(), dtype=np.float64))
        mn_of.append(np.asarray(t.column("mean_score").to_numpy(), dtype=np.float64))
        file_of.append(np.full(len(chg), fi, dtype=np.int32))
        local_of.append(np.arange(len(chg), dtype=np.int64))
    peptide = pd.concat(peptide_of, ignore_index=True)
    charge = np.concatenate(charge_of)
    # vectorised label string "peptide/charge", then factorise once
    labels = peptide.str.cat(pd.Series(charge.astype(str)), sep="/")
    codes, uniq = pd.factorize(labels, sort=False)          # uniq ~ #labels

    keep_permille = max(1, round(1000 * n_peptides / len(uniq)))
    keep_uniq = np.fromiter((_keep_label(str(u), keep_permille) for u in uniq),
                            dtype=bool, count=len(uniq))
    row_keep = keep_uniq[codes]
    print(f"  {len(uniq):,} distinct peptide/charge labels; "
          f"keep_permille={keep_permille} -> {int(keep_uniq.sum()):,} selected labels")

    df = pd.DataFrame({
        "file_idx": np.concatenate(file_of)[row_keep],
        "local_row": np.concatenate(local_of)[row_keep],
        "code": codes[row_keep],
        "max_score": np.concatenate(mx_of)[row_keep],
        "mean_score": np.concatenate(mn_of)[row_keep],
    })
    # cap replicates per label (stable: keep first max_reps in shard order)
    df = df.groupby("code", sort=False).head(max_reps).reset_index(drop=True)
    df["y"] = pd.factorize(df["code"])[0]                   # dense contiguous ids
    return df, files


def load_arrays(df: pd.DataFrame, files: list[Path]) -> pd.DataFrame:
    """Second pass: read the peak arrays for the selected rows only, preserving
    the row order of `df` so it stays aligned with y / scores."""
    df = df.reset_index(drop=True)
    mz_col = [None] * len(df)
    int_col = [None] * len(df)
    prec_col = np.zeros(len(df), dtype=np.float32)
    chg_col = np.zeros(len(df), dtype=np.int64)
    for fi in sorted(df["file_idx"].unique()):
        sub = df[df["file_idx"] == fi]
        gi = sub.index.to_numpy()
        lr = pa.array(sub["local_row"].to_numpy())
        t = pq.ParquetFile(files[fi]).read(columns=ARRAY_COLS + ["charge"])
        # take() slices only the selected rows — no full-column Python materialise
        mz = t.column("mz").take(lr).to_pylist()
        it = t.column("intensity").take(lr).to_pylist()
        pr = t.column("precursor").take(lr).to_numpy()
        cg = t.column("charge").take(lr).to_numpy()
        for j, g in enumerate(gi):
            mz_col[g] = np.asarray(mz[j], dtype=np.float32)
            int_col[g] = np.asarray(it[j], dtype=np.float32)
            prec_col[g] = pr[j]
            chg_col[g] = int(cg[j])
    out = pd.DataFrame({
        "mz": mz_col, "intensity": int_col,
        "charge": chg_col, "precursor": prec_col,
        "y": df["y"].to_numpy(),
        "max_score": df["max_score"].to_numpy(),
        "mean_score": df["mean_score"].to_numpy(),
    })
    return out


def tercile_ids(scores: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """0=low, 1=mid, 2=top by the 33.3/66.7 percentiles of `scores`."""
    lo, hi = np.percentile(scores, [100 / 3, 200 / 3])
    t = np.zeros(len(scores), dtype=np.int64)
    t[scores > lo] = 1
    t[scores > hi] = 2
    return t, (float(lo), float(hi))


def _remap_y(y: np.ndarray) -> np.ndarray:
    """Contiguous 0..k-1 label ids for a tercile subset (evaluate needs dense y)."""
    return pd.factorize(y)[0]


def eval_terciles(emb, y, scores, *, score_name, device, whiten, kmeans_seeds):
    """Overall + per-tercile metrics for one embedding under one score column."""
    tids, (lo, hi) = tercile_ids(scores)
    seeds = tuple(range(kmeans_seeds))
    rows = []
    for name, sel in [("all", np.ones(len(y), bool)),
                      ("low", tids == 0), ("mid", tids == 1), ("top", tids == 2)]:
        X = emb[sel]
        yy = _remap_y(y[sel])
        n_lab = len(np.unique(yy))
        n_multi = int((np.bincount(yy) >= 2).sum())
        raw = evaluate(X, yy, kmeans_seeds=seeds, device=device, verbose=False)
        rec = {"tercile": name, "n": int(sel.sum()), "n_labels": n_lab,
               "n_labels_multi": n_multi, **raw}
        if whiten > 0:
            wh = evaluate(X, yy, kmeans_seeds=seeds, whiten=whiten, device=device,
                          verbose=False)
            rec.update({f"{k}_w": v for k, v in wh.items()})
        rows.append(rec)
    return {"score": score_name, "boundaries": [lo, hi], "rows": rows}


@torch.no_grad()
def per_query_metrics_torch(X, y, device, *, chunk=512):
    """Leave-one-out per-query Hit@1 and AP against the FULL pool. X (n,d)
    L2-normalised, y (n,) long, both on `device`. Returns numpy (hit1[n] bool,
    ap[n] float, rel[n] int) — same ranking definitions as
    replicate_retrieval.retrieval_metrics_torch, but not reduced, so queries can
    be regrouped (e.g. by within-peptide score rank) afterwards."""
    n = X.shape[0]
    counts = torch.bincount(y, minlength=int(y.max()) + 1)
    rel_total = (counts[y] - 1).float()
    Xt = X.t().contiguous()
    ranks = torch.arange(1, n + 1, device=device, dtype=torch.float32)
    hit1 = torch.zeros(n, dtype=torch.bool, device=device)
    ap = torch.zeros(n, dtype=torch.float32, device=device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = X[s:e] @ Xt
        rows = torch.arange(e - s, device=device)
        sims[rows, torch.arange(s, e, device=device)] = float("-inf")
        order = sims.argsort(dim=1, descending=True)
        match = (y[order] == y[s:e, None])
        match[:, -1] = False
        found = match.cumsum(1).float()
        prec = found / ranks
        rel = rel_total[s:e]
        ap[s:e] = (prec * match).sum(1) / rel.clamp(min=1)
        hit1[s:e] = match[:, 0]
    return hit1.cpu().numpy(), ap.cpu().numpy(), rel_total.cpu().numpy()


@torch.no_grad()
def per_query_sibling_ranks_torch(X, y, device, *, chunk=512):
    """Sibling of `per_query_metrics_torch`: same full-pool LOO ranking pass,
    but also recovers every (query, sibling-target, rank) triple instead of
    only the reduced Hit@1/AP. `rank` is 1-indexed retrieval rank (1 = nearest,
    excluding self) at which sibling `t_idx[i]` is retrieved by query `q_idx[i]`.
    Every true same-peptide pair appears exactly once (not just top-K), so
    callers can derive top-K target-side findability for any K, or full
    retrieval-rank statistics, without a second O(n^2) pass. Returns
    (hit1[n] bool, ap[n] float, rel[n] int, q_idx, t_idx, rank) as numpy."""
    n = X.shape[0]
    counts = torch.bincount(y, minlength=int(y.max()) + 1)
    rel_total = (counts[y] - 1).float()
    Xt = X.t().contiguous()
    ranks = torch.arange(1, n + 1, device=device, dtype=torch.float32)
    hit1 = torch.zeros(n, dtype=torch.bool, device=device)
    ap = torch.zeros(n, dtype=torch.float32, device=device)
    q_list, t_list, r_list = [], [], []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = X[s:e] @ Xt
        rows = torch.arange(e - s, device=device)
        sims[rows, torch.arange(s, e, device=device)] = float("-inf")
        order = sims.argsort(dim=1, descending=True)
        match = (y[order] == y[s:e, None])
        match[:, -1] = False
        found = match.cumsum(1).float()
        prec = found / ranks
        rel = rel_total[s:e]
        ap[s:e] = (prec * match).sum(1) / rel.clamp(min=1)
        hit1[s:e] = match[:, 0]
        rr, cc = match.nonzero(as_tuple=True)
        q_list.append(rr + s)
        t_list.append(order[rr, cc])
        r_list.append(cc + 1)
    q_idx = torch.cat(q_list).cpu().numpy()
    t_idx = torch.cat(t_list).cpu().numpy()
    rank = torch.cat(r_list).cpu().numpy()
    return (hit1.cpu().numpy(), ap.cpu().numpy(), rel_total.cpu().numpy(),
            q_idx, t_idx, rank)


def within_peptide_bins(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Rank each label's own members by score and bin into 0/1/2 (low/mid/high)
    thirds *within the peptide*. Controls for peptide identity: every multi-
    replicate peptide contributes to all three bins, so a bin difference isolates
    spectrum quality rather than which peptides fall where."""
    order = np.lexsort((scores, y))          # sort by label, then score asc
    bins = np.empty(len(y), dtype=np.int64)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and y[order[j]] == y[order[i]]:
            j += 1
        m = j - i
        rank = np.arange(m)
        frac = rank / (m - 1) if m > 1 else np.full(m, 0.5)
        b = np.digitize(frac, [1 / 3, 2 / 3])   # 0,1,2
        bins[order[i:j]] = b
        i = j
    return bins


def within_peptide_control(emb, y, scores, *, score_name, device, whiten):
    """The identity-controlled test: does a peptide's HIGH-score replicate
    retrieve its siblings better than its LOW-score replicate? Per-query LOO
    metrics vs the full pool, grouped by within-peptide score bin."""
    yt = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.long, device=device)
    bins = within_peptide_bins(y, scores)
    out = {"score": score_name, "bins": []}
    variants = [("raw", _l2norm_dense(emb))]
    if whiten > 0:
        variants.append(("w", _l2norm_dense(all_but_top(emb, whiten))))
    per = {}
    for tag, X in variants:
        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        per[tag] = per_query_metrics_torch(Xt, yt, device)
    for b, bname in [(0, "low"), (1, "mid"), (2, "high")]:
        sel = (bins == b)
        rec = {"bin": bname, "n": int(sel.sum()),
               "mean_score": float(scores[sel].mean()) if sel.any() else float("nan")}
        for tag, (h1, ap, rel) in per.items():
            valid = sel & (rel > 0)
            suf = "" if tag == "raw" else "_w"
            rec[f"Hit@1{suf}"] = float(h1[valid].mean()) if valid.any() else float("nan")
            rec[f"MAP{suf}"] = float(ap[valid].mean()) if valid.any() else float("nan")
        out["bins"].append(rec)
    return out


def _embedding_variants(emb, y, device, whiten):
    """Raw + (optional) all-but-top-K-whitened L2-normalised variants, matching
    the raw/`_w` convention used elsewhere in this script. Returns
    {tag: per_query_sibling_ranks_torch(...)} computed ONCE per checkpoint —
    the pairwise ranking doesn't depend on the score column, so this is shared
    across max_score / mean_score to avoid a redundant O(n^2) pass each."""
    yt = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.long, device=device)
    variants = [("raw", _l2norm_dense(emb))]
    if whiten > 0:
        variants.append(("w", _l2norm_dense(all_but_top(emb, whiten))))
    pair_data = {}
    for tag, X in variants:
        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        pair_data[tag] = per_query_sibling_ranks_torch(Xt, yt, device)
    return pair_data


def global_target_findability(pair_data, scores, *, score_name, whiten, Ks=(1, 5)):
    """GLOBAL target-side findability: treat every spectrum as a TARGET and ask
    how often its same-peptide sibling QUERIES retrieve it in their top-K
    (full mixed-pool gallery), grouped by the spectrum's GLOBAL score tercile.
    CONFOUNDED: a peptide that happens to score high overall may also just have
    more/easier siblings, which would inflate the top tercile's findability
    even with zero genuine quality preference. Reported alongside the
    base-rate LIFT — observed top-K-hit tercile share divided by the tercile
    share among all *available* siblings — which divides that confound back
    out: lift > 1 in the top tercile means high-score spectra are hit MORE
    than their availability alone would predict."""
    n = len(scores)
    tids, (lo, hi) = tercile_ids(scores)
    out = {"score": score_name, "boundaries": [float(lo), float(hi)], "variants": {}}
    for tag, (hit1, ap, rel, q_idx, t_idx, rank) in pair_data.items():
        if tag == "w" and whiten <= 0:
            continue
        denom = np.bincount(t_idx, minlength=n).astype(np.float64)
        avail_frac = np.bincount(tids[t_idx], minlength=3) / max(len(t_idx), 1)
        v = {"K": {}}
        for K in Ks:
            hit_mask = rank <= K
            hitc = np.bincount(t_idx[hit_mask], minlength=n).astype(np.float64)
            find_k = np.divide(hitc, denom, out=np.full(n, np.nan), where=denom > 0)
            by_tercile = {}
            for tname, tid in [("low", 0), ("mid", 1), ("top", 2)]:
                sel = (tids == tid) & (denom > 0)
                by_tercile[tname] = {
                    "n": int(sel.sum()),
                    "mean_findability": float(find_k[sel].mean()) if sel.any() else float("nan"),
                }
            hit_frac = np.bincount(tids[t_idx[hit_mask]], minlength=3) / max(int(hit_mask.sum()), 1)
            safe_avail = np.where(avail_frac > 0, avail_frac, np.nan)
            lift = (hit_frac / safe_avail).tolist()
            v["K"][K] = {"by_tercile": by_tercile, "avail_frac": avail_frac.tolist(),
                         "hit_frac": hit_frac.tolist(), "lift": lift}
        out["variants"][tag] = v
    return out


def within_peptide_target_findability(pair_data, y, scores, *, score_name, whiten,
                                       Ks=(1, 5), min_sib=3):
    """CONFOUND-FREE target-side test: bin each spectrum into its OWN peptide's
    within-peptide score third (`within_peptide_bins`), then measure the same
    top-K target-side findability plus mean/median retrieval rank among true
    siblings, grouped by that bin. Every multi-replicate peptide contributes to
    all three bins, so a monotonic low->high rise isolates spectrum quality
    from peptide identity. Also reports the per-query Spearman correlation
    between a sibling's score and its retrieval rank (sign-flipped so positive
    = higher-score siblings retrieved earlier / at a lower rank number),
    averaged over queries with >= min_sib siblings."""
    n = len(scores)
    bins = within_peptide_bins(y, scores)
    out = {"score": score_name, "variants": {}}
    for tag, (hit1, ap, rel, q_idx, t_idx, rank) in pair_data.items():
        if tag == "w" and whiten <= 0:
            continue
        denom = np.bincount(t_idx, minlength=n).astype(np.float64)
        rank_f = rank.astype(np.float64)
        mean_rank = np.divide(np.bincount(t_idx, weights=rank_f, minlength=n), denom,
                              out=np.full(n, np.nan), where=denom > 0)
        median_rank = pd.Series(rank_f).groupby(t_idx).median().reindex(range(n)).to_numpy()

        find = {}
        for K in Ks:
            hitc = np.bincount(t_idx[rank <= K], minlength=n).astype(np.float64)
            find[K] = np.divide(hitc, denom, out=np.full(n, np.nan), where=denom > 0)

        by_bin = []
        for b, bname in [(0, "low"), (1, "mid"), (2, "high")]:
            sel = (bins == b) & (denom > 0)
            rec = {"bin": bname, "n": int(sel.sum())}
            for K in Ks:
                rec[f"findability_top{K}"] = float(find[K][sel].mean()) if sel.any() else float("nan")
            rec["mean_rank"] = float(mean_rank[sel].mean()) if sel.any() else float("nan")
            rec["median_rank"] = float(np.nanmedian(median_rank[sel])) if sel.any() else float("nan")
            by_bin.append(rec)

        # per-query Spearman(score, retrieval rank of its siblings), vectorised
        # via rank-transform + Pearson-on-ranks (no python loop over queries)
        pairs = pd.DataFrame({"q": q_idx, "rank": rank_f, "score": scores[t_idx]})
        pairs["rs"] = pairs.groupby("q")["score"].rank(method="average")
        pairs["rr"] = pairs.groupby("q")["rank"].rank(method="average")
        pairs["rs2"] = pairs["rs"] ** 2
        pairs["rr2"] = pairs["rr"] ** 2
        pairs["rsr"] = pairs["rs"] * pairs["rr"]
        agg = pairs.groupby("q").agg(n=("rank", "size"), sx=("rs", "sum"), sy=("rr", "sum"),
                                     sxx=("rs2", "sum"), syy=("rr2", "sum"), sxy=("rsr", "sum"))
        agg = agg[agg["n"] >= min_sib]
        num = agg["n"] * agg["sxy"] - agg["sx"] * agg["sy"]
        den = np.sqrt((agg["n"] * agg["sxx"] - agg["sx"] ** 2) *
                      (agg["n"] * agg["syy"] - agg["sy"] ** 2))
        spearman = -(num / den)  # flip sign: positive = high score -> early (low) rank
        spearman = spearman.replace([np.inf, -np.inf], np.nan).dropna()

        out["variants"][tag] = {
            "by_bin": by_bin,
            "mean_spearman": float(spearman.mean()) if len(spearman) else float("nan"),
            "n_queries_spearman": int(len(spearman)),
        }
    return out


def _fmt_global_findability(res, whiten):
    lines = [f"  GLOBAL target-side findability by {res['score']} (tercile cuts: "
             f"{res['boundaries'][0]:.2f} / {res['boundaries'][1]:.2f}) "
             f"[confounded -- see lift]"]
    for tag, v in res["variants"].items():
        vlabel = "raw" if tag == "raw" else f"whiten{whiten}"
        lines.append(f"    -- {vlabel} --")
        for K, rk in v["K"].items():
            lines.append(f"    {'tercile':<6}{'N':>8}{'find@'+str(K):>10}")
            for tname in ("low", "mid", "top"):
                r = rk["by_tercile"][tname]
                lines.append(f"    {tname:<6}{r['n']:>8}{r['mean_findability']:>10.4f}")
            lo_l, mid_l, top_l = rk["lift"]
            lines.append(f"      base-rate lift (obs/avail) K={K}: "
                         f"low={lo_l:.3f}  mid={mid_l:.3f}  top={top_l:.3f}")
    return "\n".join(lines)


def _fmt_within_findability(res, whiten):
    lines = [f"  WITHIN-PEPTIDE target-side findability by {res['score']} "
             f"(confound-free: own peptide's spectra ranked into thirds)"]
    for tag, v in res["variants"].items():
        vlabel = "raw" if tag == "raw" else f"whiten{whiten}"
        lines.append(f"    -- {vlabel} --  spearman(score,rank)[flipped]="
                     f"{v['mean_spearman']:.4f}  (n_queries={v['n_queries_spearman']})")
        lines.append(f"    {'bin':<6}{'N':>8}{'find@1':>9}{'find@5':>9}"
                     f"{'meanRank':>10}{'medRank':>9}")
        for r in v["by_bin"]:
            lines.append(f"    {r['bin']:<6}{r['n']:>8}{r['findability_top1']:>9.4f}"
                         f"{r['findability_top5']:>9.4f}{r['mean_rank']:>10.2f}"
                         f"{r['median_rank']:>9.1f}")
    return "\n".join(lines)


def _fmt_within(res, whiten):
    lines = [f"  within-peptide control by {res['score']} "
             f"(each peptide's own replicates ranked; gallery = full pool)"]
    hdr = f"    {'bin':<6}{'N':>8}{'meanScore':>11}{'Hit@1':>9}{'MAP':>9}"
    if whiten > 0:
        hdr += f"{'Hit@1_w':>10}{'MAP_w':>9}"
    lines.append(hdr)
    for r in res["bins"]:
        line = (f"    {r['bin']:<6}{r['n']:>8}{r['mean_score']:>11.2f}"
                f"{r['Hit@1']:>9.4f}{r['MAP']:>9.4f}")
        if whiten > 0:
            line += f"{r['Hit@1_w']:>10.4f}{r['MAP_w']:>9.4f}"
        lines.append(line)
    return "\n".join(lines)


def _fmt_block(res, whiten):
    lines = [f"  by {res['score']}  (tercile cuts: "
             f"{res['boundaries'][0]:.2f} / {res['boundaries'][1]:.2f})"]
    hdr = f"    {'tercile':<6}{'N':>8}{'labels':>8}{'Hit@1':>9}{'MAP':>9}{'PairF1':>9}"
    if whiten > 0:
        hdr += f"{'Hit@1_w':>10}{'MAP_w':>9}{'PairF1_w':>10}"
    lines.append(hdr)
    for r in res["rows"]:
        line = (f"    {r['tercile']:<6}{r['n']:>8}{r['n_labels']:>8}"
                f"{r['Hit@1']:>9.4f}{r['MAP']:>9.4f}{r['PairF1']:>9.4f}")
        if whiten > 0:
            line += f"{r['Hit@1_w']:>10.4f}{r['MAP_w']:>9.4f}{r['PairF1_w']:>10.4f}"
        lines.append(line)
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", action="append", required=True, type=Path,
                   help="checkpoint(s) to score (repeatable)")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--n-peptides", type=int, default=2500,
                   help="approx distinct peptide/charge labels in the eval pool")
    p.add_argument("--max-reps", type=int, default=40,
                   help="cap replicates per label")
    p.add_argument("--whiten", type=int, default=16, metavar="K",
                   help="all-but-top-K anisotropy fix (0=off)")
    p.add_argument("--kmeans-seeds", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", type=Path, default=Path("runs/tercile_retrieval.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    device = torch.device(args.device)

    t0 = time.time()
    print(f"selecting eval pool from {args.data_dir} ...")
    pool, files = select_pool(args.data_dir, args.n_peptides, args.max_reps)
    print(f"  pool: {len(pool):,} spectra, {pool['y'].nunique():,} labels "
          f"({time.time() - t0:.0f}s)")
    print("reading peak arrays ...")
    ds = load_arrays(pool, files)
    y = ds["y"].to_numpy()
    max_score = ds["max_score"].to_numpy()
    mean_score = ds["mean_score"].to_numpy()
    print(f"  arrays loaded ({time.time() - t0:.0f}s)")

    report = {"data_dir": args.data_dir, "n_pool": len(ds),
              "n_labels": int(pool["y"].nunique()),
              "n_peptides_req": args.n_peptides, "max_reps": args.max_reps,
              "whiten": args.whiten, "checkpoints": {}}

    # embedding cache keyed by (ckpt, pool signature) so re-runs skip encoding
    cache_dir = Path("runs/emb_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = f"{args.n_peptides}_{args.max_reps}_{len(ds)}"

    for ckpt in args.ckpt:
        enc, cfg, step = load_encoder(str(ckpt))
        dcfg = cfg["data"]
        pp = PreprocessConfig(intensity_threshold_frac=dcfg["intensity_threshold_frac"],
                              top_n=dcfg["top_n"])
        cache = cache_dir / f"{Path(ckpt).parent.name}_{sig}.npy"
        if cache.exists():
            emb = np.load(cache)
            print(f"\n=== {ckpt} (step {step}) — cached embeddings {emb.shape} ===")
        else:
            print(f"\n=== {ckpt} (step {step}) — encoding {len(ds):,} spectra ===")
            te = time.time()
            emb = embed_model(enc, ds, device, pp, batch_size=args.batch_size)
            np.save(cache, emb)
            print(f"  encoded ({time.time() - te:.0f}s)")
        del enc
        torch.cuda.empty_cache() if device.type == "cuda" else None

        # one shared full-pool LOO ranking pass per embedding variant (raw /
        # whitened) -- independent of score column, so both target-side
        # findability measures reuse it for max_score AND mean_score below
        pair_data = _embedding_variants(emb, y, device, args.whiten)

        by = {}
        wp = {}
        gfind = {}
        wfind = {}
        for sname, sc in [("max_score", max_score), ("mean_score", mean_score)]:
            res = eval_terciles(emb, y, sc, score_name=sname, device=device,
                                whiten=args.whiten, kmeans_seeds=args.kmeans_seeds)
            by[sname] = res
            print(_fmt_block(res, args.whiten))
            ctrl = within_peptide_control(emb, y, sc, score_name=sname,
                                          device=device, whiten=args.whiten)
            wp[sname] = ctrl
            print(_fmt_within(ctrl, args.whiten))
            gf = global_target_findability(pair_data, sc, score_name=sname,
                                           whiten=args.whiten)
            gfind[sname] = gf
            print(_fmt_global_findability(gf, args.whiten))
            wf = within_peptide_target_findability(pair_data, y, sc, score_name=sname,
                                                    whiten=args.whiten)
            wfind[sname] = wf
            print(_fmt_within_findability(wf, args.whiten))
        report["checkpoints"][str(ckpt)] = {
            "step": step, "by_score": by, "within_peptide": wp,
            "target_findability_global": gfind, "target_findability_within": wfind,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
