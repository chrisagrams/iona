"""Stage 2 of the PSM-reranking eval: PSMs and peptides at 1% FDR, per method.

    python scripts/rerank_psm_fdr.py --rows DIR_OF_STAGE1_PARQUETS --out OUT.json

Target-decoy competition with the +1 correction: each method picks one candidate per
spectrum, and accepted targets at q <= 0.01 are counted. Methods are MSFragger's e-value,
the embedding cosine, and linear/MLP rescorers over feature sets with and without the
embedding, cross-validated by run.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = ("msfragger_hyperscore", "search_delta_score", "search_neglog10_evalue",
                 "matched_fraction", "num_matched_ions", "abs_massdiff", "num_tol_term",
                 "num_missed_cleavages", "length", "charge", "search_rank")


WS_FEATURES = ("cosine", "cos_delta", "cos_rank", "cos_z", "cos_gap12")


def within_spectrum(spectrum, cosine) -> dict[str, np.ndarray]:
    """Cosine relative to the other candidates of the same spectrum: delta, rank, z, top-2 gap."""
    s = pd.Series(np.asarray(cosine, dtype=np.float64))
    g = s.groupby(np.asarray(spectrum))
    rank = g.rank(ascending=False, method="first").to_numpy()
    best = g.transform("max").to_numpy()
    second = s.where(rank != 1).groupby(np.asarray(spectrum)).transform("max").to_numpy()
    second = np.where(np.isnan(second), best, second)
    best_other = np.where(rank == 1, second, best)
    mean, sd = g.transform("mean").to_numpy(), g.transform("std", ddof=0).to_numpy()
    z = np.divide(s.to_numpy() - mean, sd, out=np.zeros(len(s)), where=sd > 0)
    return {"cos_delta": s.to_numpy() - best_other, "cos_rank": rank, "cos_z": z,
            "cos_gap12": best - second}


def qvalues(scores: np.ndarray, is_decoy: np.ndarray) -> np.ndarray:
    """TDC q-values for one-PSM-per-spectrum lists (higher score = better)."""
    order = np.argsort(-scores, kind="stable")
    d = np.cumsum(is_decoy[order]); t = np.cumsum(~is_decoy[order])
    fdr = (d + 1) / np.maximum(t, 1)
    q = np.minimum.accumulate(fdr[::-1])[::-1]
    out = np.empty_like(q); out[order] = q
    return out


def accepted(scores, is_decoy, peptides, level=0.01) -> dict:
    q = qvalues(scores, is_decoy)
    psms = int(((q <= level) & ~is_decoy).sum())
    # Peptide level: best PSM per peptide, then TDC again.
    best: dict = {}
    for i, p in enumerate(peptides):
        if p not in best or scores[i] > scores[best[p]]:
            best[p] = i
    idx = np.fromiter(best.values(), dtype=np.int64)
    qp = qvalues(scores[idx], is_decoy[idx])
    peps = int(((qp <= level) & ~is_decoy[idx]).sum())
    return {"psms_1pct": psms, "peptides_1pct": peps}


def top_per_spectrum(spectrum_codes, score):
    """Index of the best-scoring candidate within each spectrum."""
    order = np.lexsort((-score, spectrum_codes))
    first = np.r_[True, spectrum_codes[order][1:] != spectrum_codes[order][:-1]]
    return order[first]


def load_lab_features(lab_dir: str, run_ids) -> "tuple":
    """The lab's feature tables for these runs, keyed on `candidate`: (frame, usable columns)."""
    frames = []
    for run in sorted(set(run_ids)):
        hits = glob.glob(str(Path(lab_dir) / "**" / f"{run}.parquet"), recursive=True)
        if len(hits) != 1:
            raise SystemExit(f"lab features for {run}: {len(hits)} files under {lab_dir}")
        names = pq.ParquetFile(hits[0]).schema.names
        cols = ["candidate_id"] + [c for c in names if c.startswith("feat__")]
        frames.append(pq.read_table(hits[0], columns=cols).to_pandas())
    lab = pd.concat(frames, ignore_index=True).rename(columns={"candidate_id": "candidate"})
    feat = [c for c in lab.columns if c.startswith("feat__")]
    for c in feat:
        if "mod_count_" in c:
            lab[c] = lab[c].fillna(0.0)
    keep = [c for c in feat
            if lab[c].notna().any() and lab[c].nunique(dropna=True) > 1]
    return lab[["candidate"] + keep], tuple(keep)


def fit_linear(Xtr, y, w0=None, l2: float = 1e-4, steps: int = 60):
    """Class-balanced L2 logistic regression by L-BFGS, warm-started from w0; returns weights + bias."""
    X = torch.tensor(Xtr, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)
    n_pos = float(Y.sum()); n_neg = float(len(Y) - n_pos)
    sw = torch.where(Y > 0, 0.5 / max(n_pos, 1.0), 0.5 / max(n_neg, 1.0))
    w = torch.zeros(X.shape[1] + 1) if w0 is None else torch.tensor(w0, dtype=torch.float32)
    w.requires_grad_(True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ w[:-1] + w[-1]
        loss = (sw * torch.nn.functional.binary_cross_entropy_with_logits(
            z, Y, reduction="none")).sum() + l2 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy().astype(np.float64)


def calibrate(train_scores, train_decoy, train_spec, test_scores):
    """mokapot calibration: 0 at the training fold's 1% FDR threshold, -1 at its median decoy."""
    top = top_per_spectrum(train_spec, train_scores)
    s, d = train_scores[top], train_decoy[top]
    q = qvalues(s, d)
    passing = s[(q <= 0.01) & ~d]
    thr = passing.min() if len(passing) else np.quantile(s[~d], 0.99)
    med = np.median(s[d]) if d.any() else thr - 1.0
    return (test_scores - thr) / max(thr - med, 1e-9)


def fit_mlp(Xtr, y, Xte, seed: int = 0, epochs: int = 8, hidden: int = 64) -> np.ndarray:
    """Two-layer MLP (torch, CPU), class-balanced BCE; returns logits for Xte."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Dropout(0.1), torch.nn.Linear(hidden, hidden),
                              torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    X = torch.tensor(Xtr, dtype=torch.float32); Y = torch.tensor(y, dtype=torch.float32)
    pos_weight = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1.0), dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for batch in torch.randperm(len(X), generator=g).split(4096):
            opt.zero_grad(); loss_fn(net(X[batch]).squeeze(-1), Y[batch]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.cat([net(chunk).squeeze(-1) for chunk in
                          torch.tensor(Xte, dtype=torch.float32).split(65536)]).numpy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--handfeat", default="", help="dir of stage-1b hand-feature tables")
    ap.add_argument("--max-neg", type=int, default=3_000_000,
                    help="decoys per training fold (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="fold assignment and subsampling")
    cli = ap.parse_args(argv)

    files = sorted(glob.glob(str(Path(cli.rows) / "*.parquet")))
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    df["matched_fraction"] = df["num_matched_ions"] / df["tot_num_ions"].clip(lower=1)
    df["abs_massdiff"] = df["massdiff"].abs()
    hand: tuple = ()
    if cli.handfeat:
        hf = pd.concat([pq.read_table(f).to_pandas() for f in
                        sorted(glob.glob(str(Path(cli.handfeat) / "*.parquet")))],
                       ignore_index=True)
        before = len(df)
        df = df.merge(hf, on="candidate", how="inner")
        if len(df) != before:
            raise SystemExit(f"hand features cover {len(df):,} of {before:,} candidates")
        hand = tuple(c for c in hf.columns if c.startswith("hf_"))
    for c in BASE_FEATURES + hand + ("cosine",):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    spec = df["spectrum_id"].astype("category").cat.codes.to_numpy()
    for k, v in within_spectrum(spec, df["cosine"].to_numpy(float)).items():
        df[k] = v
    decoy = df["is_decoy"].to_numpy(bool)
    peptide = df["peptide"].to_numpy()
    n_spectra = int(spec.max() + 1)
    report = {"files": [Path(f).name for f in files], "spectra": n_spectra,
              "candidates": int(len(df)), "methods": {}}

    subsets = {"all": np.ones(len(df), bool)}
    subsets |= {d: (df["dataset"] == d).to_numpy() for d in sorted(df["dataset"].unique())}

    def evaluate(name, score):
        for sub, mask in subsets.items():
            top = top_per_spectrum(spec[mask], score[mask])
            idx = np.flatnonzero(mask)[top]
            n = len(top)
            r = accepted(score[idx], decoy[idx], peptide[idx])
            r["rate"] = r["psms_1pct"] / max(n, 1); r["spectra"] = n
            report["methods"].setdefault(sub, {})[name] = r
            print(f"  [{sub:6s}] {name:11s} PSMs@1% {r['psms_1pct']:>8,} ({100 * r['rate']:5.2f}% "
                  f"of {n:,})  peptides@1% {r['peptides_1pct']:>7,}", flush=True)

    # MSFragger rank-1 by e-value.
    ms = df["search_neglog10_evalue"].to_numpy(float) - 1e-6 * df["search_rank"].to_numpy(float)
    evaluate("msfragger", ms)
    # Embedding alone.
    evaluate("embedding", df["cosine"].to_numpy(float))
    evaluate("embedding:delta", df["cos_delta"].to_numpy(float))

    # Rescorers, cross-validated by run.
    units = df["run_id"].to_numpy()
    uniq = np.array(sorted(set(units)))
    rng = np.random.default_rng(cli.seed); rng.shuffle(uniq)
    fold_of = {u: i % cli.folds for i, u in enumerate(uniq)}
    fold = np.array([fold_of[u] for u in units])
    report["cv"] = {"group": "run", "folds": cli.folds, "units": len(uniq)}
    print(f"[fdr] CV over {len(uniq)} runs in {cli.folds} folds", flush=True)
    rank1 = df["search_rank"].to_numpy() == 1
    sets = {"ms": BASE_FEATURES}
    if hand:
        sets |= {"hand": hand, "ms+hand": BASE_FEATURES + hand}
    variants = {"": (), "+emb": ("cosine",), "+embws": WS_FEATURES}
    for model in ("linear", "mlp"):
        for set_name, base in sets.items():
            for emb, more in variants.items():
                cols = base + more
                name = f"{model}:{set_name}{emb}"
                score = np.zeros(len(df))
                for f in range(cli.folds):
                    tr, te = fold != f, np.flatnonzero(fold == f)
                    tr_top = np.flatnonzero(tr & rank1)
                    q = qvalues(ms[tr_top], decoy[tr_top])
                    pos = tr_top[(q <= 0.01) & ~decoy[tr_top]]
                    neg = np.flatnonzero(tr & decoy)
                    if cli.max_neg and len(neg) > cli.max_neg:
                        neg = np.sort(np.random.default_rng(f).choice(
                            neg, cli.max_neg, replace=False))
                    if len(pos) == 0:
                        raise SystemExit(f"{name} fold {f}: no MSFragger target passes 1% "
                                         f"FDR in the training runs -- nothing to learn")
                    idx = np.r_[pos, neg]
                    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                    Xall = df[list(cols)].to_numpy(float)[np.r_[idx, te]]
                    scaler = StandardScaler().fit(Xall[:len(idx)])
                    Xtr, Xte = scaler.transform(Xall[:len(idx)]), scaler.transform(Xall[len(idx):])
                    if model == "linear":
                        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                        clf.fit(Xtr, y)
                        score[te] = clf.decision_function(Xte)
                        if f == 0:
                            report.setdefault("weights_fold0", {})[name] = {
                                k: float(v) for k, v in zip(cols, clf.coef_[0].round(4))}
                    else:
                        score[te] = fit_mlp(Xtr, y, Xte, seed=f)
                evaluate(name, score)

    report["seed"] = cli.seed

    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps(report, indent=1))
    print(f"[fdr] wrote {cli.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
