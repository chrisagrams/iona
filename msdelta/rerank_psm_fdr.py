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
  <model>:<features>[+emb]   a rescorer, model in {linear, mlp}, features in
               {ms (MSFragger's scores), hand (our 22 fragment features, stage 1b),
                ms+hand}; `+emb` adds the embedding cosine. `linear:ms` is the
               Percolator-style baseline; each +emb pair isolates the embedding's effect.

Rescorers are CROSS-VALIDATED BY RUN (a spectrum is never scored by a model trained on
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
                    "(default all: ms,hand,ms+hand)")
    cli = ap.parse_args(argv)

    import pyarrow.parquet as pq
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

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

    # 3/4. Percolator-style linear rescorer, cross-validated by run
    units = df["run_id"].map(series_of).to_numpy() if cli.group == "series" \
        else df["run_id"].to_numpy()
    uniq = np.array(sorted(set(units)))
    rng = np.random.default_rng(0); rng.shuffle(uniq)
    fold_of = {u: i % cli.folds for i, u in enumerate(uniq)}
    fold = np.array([fold_of[u] for u in units])
    report["cv"] = {"group": cli.group, "folds": cli.folds, "units": len(uniq)}
    print(f"[fdr] CV over {len(uniq)} {cli.group} groups in {cli.folds} folds", flush=True)
    rank1 = df["search_rank"].to_numpy() == 1
    sets = {"ms": BASE_FEATURES}
    if hand:
        sets |= {"hand": hand, "ms+hand": BASE_FEATURES + hand}
    if cli.sets:
        sets = {k: v for k, v in sets.items() if k in cli.sets.split(",")}
    for model in cli.models.split(","):
        for set_name, base in sets.items():
            for emb in (False, True):
                cols = base + (("cosine",) if emb else ())
                name = f"{model}:{set_name}" + ("+emb" if emb else "")
                X = df[list(cols)].to_numpy(float)
                score = np.zeros(len(df))
                for f in range(cli.folds):
                    tr, te = fold != f, fold == f
                    tr_top = np.flatnonzero(tr & rank1)
                    q = qvalues(ms[tr_top], decoy[tr_top])
                    pos = tr_top[(q <= 0.01) & ~decoy[tr_top]]
                    neg = np.flatnonzero(tr & decoy)
                    if cli.max_neg and len(neg) > cli.max_neg:
                        neg = np.sort(np.random.default_rng(f).choice(neg, cli.max_neg,
                                                                      replace=False))
                    idx = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                    scaler = StandardScaler().fit(X[idx])
                    Xtr, Xte = scaler.transform(X[idx]), scaler.transform(X[te])
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

    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps(report, indent=1))
    print(f"[fdr] wrote {cli.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
