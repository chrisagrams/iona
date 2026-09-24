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
  linear       Percolator-style linear rescorer on MSFragger's features (no cosine)
  linear+emb   the same with `cosine` added

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=3)
    cli = ap.parse_args(argv)

    import pyarrow.parquet as pq
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    files = sorted(glob.glob(str(Path(cli.rows) / "*.parquet")))
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    df["matched_fraction"] = df["num_matched_ions"] / df["tot_num_ions"].clip(lower=1)
    df["abs_massdiff"] = df["massdiff"].abs()
    for c in BASE_FEATURES + ("cosine",):
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

    # 1. MSFragger rank-1 by e-value (rank-1 wins ties by construction).
    ms = df["search_neglog10_evalue"].to_numpy(float) - 1e-6 * df["search_rank"].to_numpy(float)
    evaluate("msfragger", ms)
    # 2. embedding alone: the candidate with the highest cosine
    evaluate("embedding", df["cosine"].to_numpy(float))

    # 3/4. Percolator-style linear rescorer, cross-validated by run
    runs = df["run_id"].to_numpy()
    uniq = np.array(sorted(set(runs)))
    rng = np.random.default_rng(0); rng.shuffle(uniq)
    fold_of_run = {r: i % cli.folds for i, r in enumerate(uniq)}
    fold = np.array([fold_of_run[r] for r in runs])
    rank1 = df["search_rank"].to_numpy() == 1
    for name, cols in (("linear", BASE_FEATURES), ("linear+emb", BASE_FEATURES + ("cosine",))):
        X = df[list(cols)].to_numpy(float)
        score = np.zeros(len(df))
        for f in range(cli.folds):
            tr, te = fold != f, fold == f
            tr_top = np.flatnonzero(tr & rank1)
            q = qvalues(ms[tr_top], decoy[tr_top])
            pos = tr_top[(q <= 0.01) & ~decoy[tr_top]]
            neg = np.flatnonzero(tr & decoy)
            idx = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            scaler = StandardScaler().fit(X[idx])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(scaler.transform(X[idx]), y)
            score[te] = clf.decision_function(scaler.transform(X[te]))
            if f == 0:
                report.setdefault("weights_fold0", {})[name] = {
                    k: float(v) for k, v in zip(cols, clf.coef_[0].round(4))}
        evaluate(name, score)

    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps(report, indent=1))
    print(f"[fdr] wrote {cli.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
