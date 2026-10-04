"""K197-C: where binned cosine beats our spectrum encoders, from pbs/diag/k197_binned_edge.py's per-query output.

    .venv/bin/python sweeps/k197_analyze.py [--dir results/raw/diag/k197]

Writes results/summary/k197_binned_edge.md:
  1. MAP@R / Hit@1 per method and late fusion (alpha * encoder + (1 - alpha) * binned cosine) per set
  2. what each method's WRONG top-1 hit is: the same peptide at another charge, the same sequence with other
     modifications, an isobaric peptide (same charge, neutral mass within 20 ppm), or an unrelated peptide
  3. MAP@R by group size R, charge, peak count and replicate similarity (binned cosine of a query to its own
     replicates, i.e. how alike the replicates are as raw spectra)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETS = ["test", "oodval", "mouse", "human", "yeast"]
PROTON = 1.007276


def strip(p):
    return re.sub(r"[^A-Z]", "", re.sub(r"\[[^\]]*\]|\([^)]*\)", "", p))


def load(d, s):
    f = d / f"{s}_summary.json"
    if not f.exists():
        return None
    meta = dict(np.load(d / f"{s}_meta.npz", allow_pickle=False))
    summ = json.loads(f.read_text())
    res = {m: dict(np.load(d / f"{s}_{m}.npz")) for m in summ["methods"]}
    return meta, summ, res


def taxonomy(meta, r):
    """Share of queries whose top-1 is wrong, split by what the wrong hit is."""
    keep = r["n_rel"] > 0
    q = np.flatnonzero(keep & (r["hit1"] == 0))
    t = r["top"][q, 0]
    pep, ch, g = meta["peptide"], meta["charge"], meta["group"]
    mass = (meta["precursor"] - PROTON) * ch
    stripped = np.array([strip(p) for p in pep])
    cats = np.where(pep[t] == pep[q], "same peptide, other charge",
           np.where(stripped[t] == stripped[q], "same sequence, other mods",
           np.where((ch[t] == ch[q]) & (np.abs(mass[t] - mass[q]) / mass[q] * 1e6 <= 20), "isobaric (20 ppm, same z)",
                    "different peptide")))
    n = keep.sum()
    names = ["same peptide, other charge", "same sequence, other mods", "isobaric (20 ppm, same z)", "different peptide"]
    return {c: (cats == c).sum() / n for c in names}, (1 - r["hit1"][keep].mean())


def strata(meta, res, binned_pos):
    keep = res["binned0.1"]["n_rel"] > 0
    R = res["binned0.1"]["n_rel"]
    out = {}
    out["R"] = [("1", R == 1), ("2-4", (R >= 2) & (R <= 4)), ("5-9", (R >= 5) & (R <= 9)), ("10-19", R >= 10)]
    out["charge"] = [(f"{z}", meta["charge"] == z) for z in (2, 3)] + [("4+", meta["charge"] >= 4)]
    qs = np.quantile(meta["n_peaks"][keep], [0.25, 0.5, 0.75])
    npk = meta["n_peaks"]
    out["peaks"] = [(f"<{qs[0]:.0f}", npk < qs[0]), (f"{qs[0]:.0f}-{qs[1]:.0f}", (npk >= qs[0]) & (npk < qs[1])),
                    (f"{qs[1]:.0f}-{qs[2]:.0f}", (npk >= qs[1]) & (npk < qs[2])), (f">={qs[2]:.0f}", npk >= qs[2])]
    out["replicate cos (binned)"] = [(f"{a:.1f}-{b:.1f}", (binned_pos >= a) & (binned_pos < b))
                                     for a, b in ((-1, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 2))]
    return {k: [(name, m & keep) for name, m in v] for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results/raw/diag/k197"))
    d = Path(ap.parse_args().dir)
    L = ["# K197-C: where binned cosine 0.1 Da beats our spectrum encoders", "",
         "Per-query diagnostic (pbs/diag/k197_binned_edge.py): consensus-recipe encoders at 540k, seed 0; experimental "
         "queries vs experimental gallery, open search, peptide+charge groups (the evaluation's definitions).", ""]
    got = {s: load(d, s) for s in SETS}
    got = {s: v for s, v in got.items() if v}
    L += ["## 1. Methods and late fusion (MAP@R)", "",
          "fusion = alpha * encoder cosine + (1 - alpha) * binned cosine; alpha 0 = binned, 1 = encoder.", "",
          "| set | binned | 25m | 400m | best fusion 25m (alpha) | best fusion 400m (alpha) |", "|---|---:|---:|---:|---:|---:|"]
    for s, (meta, summ, res) in got.items():
        m = summ["methods"]
        best = {n: max(f.items(), key=lambda kv: kv[1]["MAP@R"]) for n, f in summ["fusion"].items()}
        L.append(f"| {s} | {m['binned0.1']['MAP@R']:.3f} | {m['25m']['MAP@R']:.3f} | {m['400m']['MAP@R']:.3f} | "
                 + " | ".join(f"{best[n][1]['MAP@R']:.3f} ({best[n][0]})" for n in ("25m", "400m")) + " |")
    L += ["", "## 2. What the wrong top-1 hit is (share of all queries)", "",
          "| set | method | top-1 wrong | same peptide, other charge | same sequence, other mods | isobaric | different peptide |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for s, (meta, summ, res) in got.items():
        for mth in ("binned0.1", "25m", "400m"):
            t, wrong = taxonomy(meta, res[mth])
            L.append(f"| {s} | {mth} | {wrong:.3f} | " + " | ".join(f"{v:.3f}" for v in t.values()) + " |")
    L += ["", "## 3. MAP@R by stratum (share of queries in brackets)", ""]
    for s, (meta, summ, res) in got.items():
        st = strata(meta, res, res["binned0.1"]["pos_mean"])
        L += [f"### {s}", "", "| stratum | bin | share | binned | 25m | 400m | 400m - binned |", "|---|---|---:|---:|---:|---:|---:|"]
        n = (res["binned0.1"]["n_rel"] > 0).sum()
        for k, bins in st.items():
            for name, m in bins:
                if m.sum() == 0:
                    continue
                v = {mth: res[mth]["ap"][m].mean() for mth in ("binned0.1", "25m", "400m")}
                L.append(f"| {k} | {name} | {m.sum() / n:.2f} | {v['binned0.1']:.3f} | {v['25m']:.3f} | {v['400m']:.3f} | "
                         f"{v['400m'] - v['binned0.1']:+.3f} |")
        L.append("")
    out = ROOT / "results/summary/k197_binned_edge.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
