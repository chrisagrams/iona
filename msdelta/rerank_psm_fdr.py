"""Stage 2 of the PSM-reranking eval: PSMs and peptides at 1% FDR, per method.

    python -m msdelta.rerank_psm_fdr --rows DIR_OF_STAGE1_PARQUETS --out OUT.json

Target-decoy competition (TDC), the field's standard:
  1. each method picks ONE candidate per spectrum (its top score) -- target or decoy;
  2. spectra are sorted by that score, best first;
  3. at every cut-off FDR = (decoys + 1) / targets above it (Kall et al. 2008's +1
     correction); the q-value of a spectrum is the lowest FDR at which it is accepted;
  4. PSMs at 1% FDR = targets with q <= 0.01. Peptide level: keep each peptide's best
     PSM (sequence + mods), then the same procedure.
More accepted targets at the same 1% is a better reranker.

Methods:
  msfragger    MSFragger's rank-1 candidate, scored by -log10 e-value (the dataset's own
               baseline: 16.5% of HEK spectra at 1% FDR on its tuning run)
  embedding    the candidate with the highest student-encoder cosine, scored by it
  embedding:delta   the same pick, scored ACROSS spectra by its lead over the best other
               candidate of its own spectrum (deltaCn-style) instead of the raw level
  <model>:<features>[+emb|+embws]   a rescorer, model in {linear, mlp}, features in
               {ms (MSFragger's scores), hand (our 22 fragment features, stage 1b),
                ms+hand}; `+emb` adds the embedding cosine, `+embws` adds it with its
               within-spectrum versions (WS_FEATURES; Percolator's deltaCn/rank idea
               applied to our score). `linear:ms` is the Percolator-style baseline; each
               +emb pair isolates the embedding's effect.

Feature sets: `ms` (MSFragger), `hand` / `ms+hand` (stage 1b), `lab` (--labfeat: the
dataset's own published features/ tables, 380 columns of which the all-NaN and constant
ones are dropped over the loaded runs; mod_count_* columns are per-run dynamic, absent ->
0). `+embvec` (--vectors, R5) adds WS_FEATURES plus the element-wise product
spectrum * peptide projected by PCA fitted on the TRAINING rows of each fold (cosine is
the product's sum, so a linear model on it learns a weighted cosine); `+nullvec` is its
leakage control -- the product with a RANDOM other spectrum; it must add nothing.

Regimes (--regime):
  global   one model over runs, CV by run/series, fixed labels (below). The plug-and-play
           model: it scores a new run without training on it.
  perrun   Percolator's protocol, comparable to MS2Rescore/Percolator/mokapot numbers:
           each run separately, 3-fold CV by SPECTRUM within the run, labels re-derived
           from the current model for `--iters` rounds (positives = targets at q <= 1%
           among each spectrum's current top candidate; negatives = the fold's decoys),
           linear model only; fold scores calibrated as in mokapot,
           (s - s@1%) / (s@1% - median decoy s), so folds and runs pool into one list.

Global rescorers are CROSS-VALIDATED BY RUN (a spectrum is never scored by a model trained on
its own run; HEK runs are correlated MudPIT steps, so rows are not independent). Training
data, as in mokapot/Percolator: positives = targets that MSFragger ranks first AND that
pass 1% FDR on its e-value within the training folds; negatives = every decoy candidate.
Standardisation is fitted on training folds only.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

BASE_FEATURES = ("msfragger_hyperscore", "search_delta_score", "search_neglog10_evalue",
                 "matched_fraction", "num_matched_ions", "abs_massdiff", "num_tol_term",
                 "num_missed_cleavages", "length", "charge", "search_rank")


WS_FEATURES = ("cosine", "cos_delta", "cos_rank", "cos_z", "cos_gap12")


def within_spectrum(spectrum, cosine) -> dict[str, np.ndarray]:
    """The cosine relative to the other candidates of the SAME spectrum (the raw level
    depends on the spectrum, so it ranks badly across spectra):
      cos_delta  minus the best OTHER candidate (the top one: minus the second best)
      cos_rank   1 = highest cosine in the pool
      cos_z      standardised within the pool (0 when the pool has no spread)
      cos_gap12  best - second best, the same for the whole pool
    A single-candidate pool gets delta 0, rank 1, z 0, gap 0. Same definitions as the
    MS2Rescore converter (baselines_wip/ms2rescore/convert.py)."""
    import pandas as pd
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
    # peptide level: best-scoring PSM per peptide, then TDC again
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


def series_of(run_id: str) -> str:
    """Acquisition series of a run: HEK293 MudPIT runs share a prefix and differ in a
    trailing salt-step index (0718-5 -> 0718, HEK-U100ug-V2-3_10 -> HEK-U100ug-V2-3,
    HEK-U100ug-exp33-500c-h10 -> HEK-U100ug-exp33-500c, 0310-9a -> 0310). HCT116 runs are
    fractions of one sample and stay individual groups."""
    import re
    if "HCT116" in run_id:
        return run_id
    return re.sub(r"[-_]h?\d+a?$", "", run_id)


LAB_REVISION = "a6df7948d4500a1f303b0dcf4b6a6a040784d0c6"   # features/ first published


def load_lab_features(lab_dir: str, run_ids) -> "tuple":
    """The dataset's features/<dataset>/<run>.parquet for these runs, one frame keyed on
    `candidate`. Returns (frame, usable feature columns)."""
    import pandas as pd
    import pyarrow.parquet as pq
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
    """Class-balanced L2 logistic regression by full-batch L-BFGS (torch, CPU); returns
    (weights incl. bias). Warm-starts from w0, so Percolator's iterations stay cheap."""
    import torch
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
    """mokapot's cross-fold calibration: 0 at the training fold's 1% FDR threshold, -1 at
    its median decoy (top-per-spectrum)."""
    top = top_per_spectrum(train_spec, train_scores)
    s, d = train_scores[top], train_decoy[top]
    q = qvalues(s, d)
    passing = s[(q <= 0.01) & ~d]
    thr = passing.min() if len(passing) else np.quantile(s[~d], 0.99)
    med = np.median(s[d]) if d.any() else thr - 1.0
    return (test_scores - thr) / max(thr - med, 1e-9)


class ProductPCA:
    """Element-wise product spectrum * peptide for rows, projected on the top-k principal
    directions of the product over `fit_rows` (a training sample). Vectors: unit-norm
    fp16 arrays, spectrum[owner[i]] and peptide[i] for candidate row i."""

    def __init__(self, spectrum, peptide, owner, k: int):
        self.s, self.p, self.owner, self.k = spectrum, peptide, owner, k

    def product(self, rows, owner=None):
        o = self.owner if owner is None else owner
        return self.s[o[rows]].astype(np.float32) * self.p[rows].astype(np.float32)

    def fit(self, fit_rows, max_rows: int = 200_000, seed: int = 0):
        import torch
        if len(fit_rows) > max_rows:
            fit_rows = np.sort(np.random.default_rng(seed).choice(fit_rows, max_rows,
                                                                  replace=False))
        x = torch.from_numpy(self.product(fit_rows))
        self.mean = x.mean(0)
        _, _, v = torch.pca_lowrank(x - self.mean, q=self.k, center=False, niter=4)
        self.components = v[:, :self.k]
        return self

    def transform(self, rows, owner=None, chunk: int = 262_144) -> np.ndarray:
        import torch
        out = [((torch.from_numpy(self.product(rows[s:s + chunk], owner)) - self.mean)
                @ self.components).numpy() for s in range(0, len(rows), chunk)]
        return np.concatenate(out) if out else np.zeros((0, self.k), np.float32)


def load_vectors(vec_dir: str, candidates) -> "tuple":
    """Stage-1 vectors (rerank_psm_embed --vectors-out) for these candidate rows, in their
    order: (spectrum matrix, peptide matrix, owner, null owner) with global indices."""
    import pyarrow.parquet as pq
    specs, peps, owners, nulls, cands = [], [], [], [], []
    offset = 0
    for d in sorted(p for p in Path(vec_dir).iterdir() if p.is_dir()):
        s = np.load(d / "spectrum.npy", mmap_mode="r"); p = np.load(d / "peptide.npy")
        idx = pq.read_table(d / "index.parquet").to_pandas()
        specs.append(np.asarray(s)); peps.append(p)
        owners.append(idx["owner"].to_numpy() + offset)
        nulls.append(idx["null_owner"].to_numpy() + offset)
        cands.append(idx["candidate"].to_numpy()); offset += len(s)
    import pandas as pd
    where = pd.Series(np.arange(sum(len(c) for c in cands)), index=np.concatenate(cands))
    pos = where.reindex(np.asarray(candidates)).to_numpy()
    if np.isnan(pos).any():
        raise SystemExit(f"vectors missing for {int(np.isnan(pos).sum()):,} candidates")
    pos = pos.astype(np.int64)
    return (np.concatenate(specs), np.concatenate(peps)[pos],
            np.concatenate(owners)[pos], np.concatenate(nulls)[pos])


def fit_mlp(Xtr, y, Xte, seed: int = 0, epochs: int = 8, hidden: int = 64) -> np.ndarray:
    """Two-layer MLP (torch, CPU), class-balanced BCE; returns logits for Xte."""
    import torch
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
    ap.add_argument("--models", default="linear,mlp")
    ap.add_argument("--group", default="run", choices=["run", "series"],
                    help="CV unit: single runs, or whole acquisition series (HEK293 MudPIT "
                         "series; each HCT116 fraction is its own group)")
    ap.add_argument("--max-neg", type=int, default=3_000_000,
                    help="decoy candidates subsampled per training fold (0 = all)")
    ap.add_argument("--sets", default="", help="comma list of feature sets to fit "
                    "(default all: ms,hand,ms+hand[,lab])")
    ap.add_argument("--variants", default="", help="comma list of embedding variants "
                    "(default all available: none,emb,embws[,embvec,nullvec])")
    ap.add_argument("--runs", default="", help="only these run ids (comma list); default "
                    "every table under --rows")
    ap.add_argument("--labfeat", default="", help="dir holding the dataset's features/ tables")
    ap.add_argument("--regime", default="global", help="global, perrun, or global,perrun")
    ap.add_argument("--iters", type=int, default=10, help="perrun: label-refinement rounds")
    ap.add_argument("--vectors", default="", help="stage-1 vectors dir (R5 product features)")
    ap.add_argument("--pca-k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="fold assignment and subsampling")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="CONTROL: permute the training labels every fit; any real "
                         "acceptance left over is leakage or an FDR bug")
    cli = ap.parse_args(argv)

    import pyarrow.parquet as pq
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    files = sorted(glob.glob(str(Path(cli.rows) / "*.parquet")))
    if cli.runs:
        want = set(cli.runs.split(","))
        files = [f for f in files if Path(f).stem in want]
        if len(files) != len(want):
            raise SystemExit(f"--runs: found {len(files)} of {len(want)} tables in {cli.rows}")
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
    lab: tuple = ()
    if cli.labfeat:
        lf, lab = load_lab_features(cli.labfeat, df["run_id"].unique())
        before = len(df)
        df = df.merge(lf, on="candidate", how="inner")
        if len(df) != before:
            raise SystemExit(f"lab features cover {len(df):,} of {before:,} candidates")
        print(f"[fdr] lab features: {len(lab)} usable columns", flush=True)
    for c in BASE_FEATURES + hand + lab + ("cosine",):
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

    # Leakage check (A4): can a candidate's cosine to a RANDOM spectrum tell targets from
    # decoys? It must not (~0.5); the real cosine is reported beside it for scale.
    from sklearn.metrics import roc_auc_score
    for col in ("cosine", "cosine_null"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy()
            ok = np.isfinite(v)          # tables made before cosine_null existed lack it
            if ok.sum() == 0:
                continue
            report.setdefault("leakage_auroc_target_vs_decoy", {})[col] = float(
                roc_auc_score(~decoy[ok], v[ok]))
            print(f"  [leak] AUROC target-vs-decoy on {col}: "
                  f"{report['leakage_auroc_target_vs_decoy'][col]:.4f}", flush=True)

    # 1. MSFragger rank-1 by e-value (rank-1 wins ties by construction).
    ms = df["search_neglog10_evalue"].to_numpy(float) - 1e-6 * df["search_rank"].to_numpy(float)
    evaluate("msfragger", ms)
    # 2. embedding alone: the candidate with the highest cosine
    evaluate("embedding", df["cosine"].to_numpy(float))
    # the same pick (delta > 0 exactly for the top cosine), ranked across spectra by lead
    evaluate("embedding:delta", df["cos_delta"].to_numpy(float))

    # 3/4. Percolator-style linear rescorer, cross-validated by run
    units = df["run_id"].map(series_of).to_numpy() if cli.group == "series" \
        else df["run_id"].to_numpy()
    uniq = np.array(sorted(set(units)))
    rng = np.random.default_rng(cli.seed); rng.shuffle(uniq)
    fold_of = {u: i % cli.folds for i, u in enumerate(uniq)}
    fold = np.array([fold_of[u] for u in units])
    report["cv"] = {"group": cli.group, "folds": cli.folds, "units": len(uniq)}
    print(f"[fdr] CV over {len(uniq)} {cli.group} groups in {cli.folds} folds", flush=True)
    rank1 = df["search_rank"].to_numpy() == 1
    sets = {"ms": BASE_FEATURES}
    if hand:
        sets |= {"hand": hand, "ms+hand": BASE_FEATURES + hand}
    if lab:
        sets |= {"lab": lab}
    if cli.sets:
        sets = {k: v for k, v in sets.items() if k in cli.sets.split(",")}
    # variant -> (extra scalar columns, product vectors: None | "real" | "null")
    variants = {"": ((), None), "+emb": (("cosine",), None), "+embws": (WS_FEATURES, None)}
    pca = None
    if cli.vectors:
        spec_v, pep_v, owner, null_owner = load_vectors(cli.vectors, df["candidate"])
        pca = ProductPCA(spec_v, pep_v, owner, cli.pca_k)
        variants |= {"+embvec": (WS_FEATURES, "real"), "+nullvec": ((), "null")}
        print(f"[fdr] vectors: {spec_v.shape[0]:,} spectra x {spec_v.shape[1]}, "
              f"PCA k={cli.pca_k}", flush=True)
    if cli.variants:
        keep = {("" if v == "none" else "+" + v) for v in cli.variants.split(",")}
        variants = {k: v for k, v in variants.items() if k in keep}

    def design(cols, vec, fit_rows, rows):
        """Feature matrix for `rows`; product PCA (if any) fitted on `fit_rows` only."""
        X = df[list(cols)].to_numpy(float)[rows]
        if vec is None:
            return X
        pca.fit(fit_rows)
        extra = pca.transform(rows, owner=null_owner if vec == "null" else None)
        return np.hstack([X, extra.astype(np.float64)])

    regimes = cli.regime.split(",")
    if "global" in regimes:
        for model in cli.models.split(","):
            for set_name, base in sets.items():
                for emb, (more, vec) in variants.items():
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
                        if cli.shuffle_labels:
                            y = np.random.default_rng(cli.seed + f).permutation(y)
                        Xall = design(cols, vec, np.flatnonzero(tr), np.r_[idx, te])
                        scaler = StandardScaler().fit(Xall[:len(idx)])
                        Xtr, Xte = scaler.transform(Xall[:len(idx)]), scaler.transform(Xall[len(idx):])
                        if model == "linear":
                            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                            clf.fit(Xtr, y)
                            score[te] = clf.decision_function(Xte)
                            if f == 0 and vec is None:
                                report.setdefault("weights_fold0", {})[name] = {
                                    k: float(v) for k, v in zip(cols, clf.coef_[0].round(4))}
                        else:
                            score[te] = fit_mlp(Xtr, y, Xte, seed=f)
                    evaluate(name, score)

    if "perrun" in regimes:
        runs = df["run_id"].to_numpy()
        for set_name, base in sets.items():
            for emb, (more, vec) in variants.items():
                cols = base + more
                name = f"perrun/linear:{set_name}{emb}"
                score = np.zeros(len(df))
                for run in sorted(set(runs)):
                    rows = np.flatnonzero(runs == run)
                    codes = np.unique(spec[rows])
                    sfold = dict(zip(codes, np.random.default_rng(cli.seed).permutation(
                        len(codes)) % cli.folds))
                    rfold = np.array([sfold[c] for c in spec[rows]])
                    for f in range(cli.folds):
                        tr, te = rows[rfold != f], rows[rfold == f]
                        Xall = design(cols, vec, tr, np.r_[tr, te])
                        scaler = StandardScaler().fit(Xall[:len(tr)])
                        Xtr, Xte = scaler.transform(Xall[:len(tr)]), scaler.transform(Xall[len(tr):])
                        cur, w = ms[tr], None
                        for _ in range(cli.iters):
                            top = top_per_spectrum(spec[tr], cur)
                            qq = qvalues(cur[top], decoy[tr][top])
                            pos = top[(qq <= 0.01) & ~decoy[tr][top]]
                            neg = np.flatnonzero(decoy[tr])
                            if len(pos) == 0 or len(neg) == 0:
                                break
                            if cli.max_neg and len(neg) > cli.max_neg:
                                neg = np.random.default_rng(f).choice(neg, cli.max_neg,
                                                                      replace=False)
                            sel = np.r_[pos, neg]
                            yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                            if cli.shuffle_labels:
                                yy = np.random.default_rng(cli.seed + f).permutation(yy)
                            w = fit_linear(Xtr[sel], yy, w0=w)
                            cur = Xtr @ w[:-1] + w[-1]
                        if w is None:      # no confident targets at all: fall back to the engine
                            score[te] = calibrate(ms[tr], decoy[tr], spec[tr], ms[te])
                            continue
                        score[te] = calibrate(cur, decoy[tr], spec[tr], Xte @ w[:-1] + w[-1])
                evaluate(name, score)
        report["perrun"] = {"folds": cli.folds, "iters": cli.iters, "model": "linear"}
    report["seed"] = cli.seed; report["shuffle_labels"] = cli.shuffle_labels

    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps(report, indent=1))
    print(f"[fdr] wrote {cli.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
