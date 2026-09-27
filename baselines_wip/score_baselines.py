"""Count PSMs at 1% FDR for external rescorers with OUR TDC (msdelta.rerank_psm_fdr.qvalues).

    python score_baselines.py --ms2rescore DIR_WITH_<run>.tsv [...] --okt OKT_RUN_DIR [...]
                              [--msfragger PARQUET ...] --out OUT.json

One PSM per spectrum = the tool's top-scored candidate (MS2Rescore already outputs rank 1
only). Reported: 'pooled' (all runs' scores in one TDC list, as our numbers are computed;
scores from per-run models are on per-run scales), 'pooled_by_q' (pooled, ranked by the
tool's own per-run q-value then score), and 'sum_per_run' (TDC within each run, summed).
Also the tool's own count (its q-value <= 0.01 on targets).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "rerank_psm_fdr", "/home/khuss/code/msdelta/msdelta/rerank_psm_fdr.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
qvalues = _m.qvalues

DS = lambda run: "HCT116" if run.startswith("Xiaomu_HCT116") else "HEK293"  # noqa: E731


def counts(df, score="score", own_q=None):
    """df: one row per spectrum with run, is_decoy, score (higher better)."""
    out = {}
    for sub, d in [("all", df)] + [(k, g) for k, g in df.groupby(df["run"].map(DS))]:
        s = d[score].to_numpy(float); dec = d["is_decoy"].to_numpy(bool)
        r = {"spectra": int(len(d)),
             "pooled": int(((qvalues(s, dec) <= 0.01) & ~dec).sum())}
        per = 0
        for _, g in d.groupby("run"):
            gs = g[score].to_numpy(float); gd = g["is_decoy"].to_numpy(bool)
            per += int(((qvalues(gs, gd) <= 0.01) & ~gd).sum())
        r["sum_per_run"] = per
        if own_q is not None:
            q = d[own_q].to_numpy(float)
            r["tool_own_q<=0.01"] = int(((q <= 0.01) & ~dec).sum())
            # pooled by the tool's own q (then score): calibrated merge of per-run models
            key = -q + 1e-9 * (s - s.min()) / (np.ptp(s) + 1e-12)
            r["pooled_by_q"] = int(((qvalues(key, dec) <= 0.01) & ~dec).sum())
        out[sub] = r
    return out


def load_ms2rescore(d):
    parts = []
    for f in sorted(glob.glob(str(Path(d) / "*.tsv"))):
        if f.endswith((".intermediate.tsv", ".feature_names.tsv")) or "." in Path(f).stem:
            continue
        t = pd.read_csv(f, sep="\t", usecols=["spectrum_id", "run", "is_decoy", "score",
                                                "qvalue", "rank"], low_memory=False)
        t = t[t["rank"] == 1]
        t["is_decoy"] = t["is_decoy"].astype(str).str.lower().isin(["true", "1"])
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    assert not df.duplicated(["run", "spectrum_id"]).any()
    return df


def load_okt(d, which):
    """Percolator target+decoy PSM outputs of an Oktoberfest run; top per spectrum."""
    d = Path(d) / "results" / "percolator"
    t = pd.read_csv(d / f"{which}.percolator.psms.txt", sep="\t"); t["is_decoy"] = False
    x = pd.read_csv(d / f"{which}.percolator.decoy.psms.txt", sep="\t"); x["is_decoy"] = True
    df = pd.concat([t, x], ignore_index=True)
    # PSMId = RAW_FILE-SCAN_NUMBER-MODIFIED_SEQUENCE-CHARGE-... ; spectrum = run + scan
    df["run"] = df["filename"] if "filename" in df.columns else df["PSMId"].str.rsplit("-", n=4).str[0]
    df["spec"] = df["PSMId"].str.rsplit("-", n=1).str[1]  # ...-CHARGE-SCAN_EVENT(=scan)
    df = df.sort_values("score", ascending=False).drop_duplicates(["run", "spec"])
    return df.rename(columns={"q-value": "qvalue"})


def load_msfragger(paths):
    import pyarrow.parquet as pq
    rows = []
    for p in paths:
        t = pq.read_table(p, columns=["run_id", "spectrum_id", "candidates"]).to_pylist()
        for r in t:
            best = max(r["candidates"], key=lambda c: (c["search_neglog10_evalue"], -c["search_rank"]))
            rows.append((r["run_id"], r["spectrum_id"], best["is_decoy"], best["search_neglog10_evalue"]))
    return pd.DataFrame(rows, columns=["run", "spectrum_id", "is_decoy", "score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms2rescore", nargs="*", default=[])
    ap.add_argument("--okt", nargs="*", default=[])
    ap.add_argument("--msfragger", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {}
    for d in a.ms2rescore:
        df = load_ms2rescore(d)
        rep[f"ms2rescore:{Path(d).name}"] = counts(df, own_q="qvalue")
        rep[f"ms2rescore:{Path(d).name}"]["runs"] = sorted(df["run"].unique().tolist())
    for which in ("original", "rescore"):  # all --okt runs pooled (one Percolator model per run)
        parts = [load_okt(d, which) for d in a.okt]
        if parts:
            df = pd.concat(parts, ignore_index=True)
            rep[f"oktoberfest:{which}"] = counts(df, own_q="qvalue")
            rep[f"oktoberfest:{which}"]["runs"] = sorted(df["run"].unique().tolist())
    if a.msfragger:
        rep["msfragger_rank1"] = counts(load_msfragger(a.msfragger))
    print(json.dumps(rep, indent=1))
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
