"""Package the R (PSM rescoring) results: the embedding's extra classification power.

    .venv/bin/python sweeps/package_rerank.py

Reads the committed per-run rescoring results (results/rerank/psm/*.json; A2 seed 0 from its log)
and the MS2Rescore baseline (baselines_wip/results_ms2rescore.json). Nothing typed in. Plotting only.

    R_embedding_gain.png / .csv   % more PSMs at 1% FDR from adding our embedding features vs a null control
    R_benchmark.png / .csv        PSMs at 1% FDR: MSFragger, MS2Rescore, ours with and without the embedding

Per-run rescoring = Percolator-style linear model trained within each run (3-fold by spectrum,
iterative labels). "+ embedding" = cosine between the spectrum's and the candidate peptide's
embeddings plus within-spectrum features of it (rank, gap to best, z-score, difference to best);
"+ null" = the same features computed with a random other spectrum's embedding.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
PSM = REPO / "results" / "rerank" / "psm"
FIG = REPO / "results" / "figures" / "SUMMARY"
INK, MUTED, GRIDC = "#1f2937", "#6b7280", "#e5e7eb"
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9ca3af", "axes.linewidth": 0.8,
         "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "figure.dpi": 200}
# embedding -> (label, per-seed sources for the 8-run per-run results)
EMB = {
    "A1": ("Iona embedding (50M)", [PSM / f"a1-rerun_r4_global+perrun_seed{s}.json" for s in (0, 1, 2)]),
    "A2": ("Iona embedding (400M)", [PSM / "a2-400m_r4_seed0.log"] + [PSM / f"a2-400m_r4_global+perrun_seed{s}.json" for s in (1, 2)]),
}
BASES = {"ms": "MSFragger features", "lab": "rich features"}   # rich = the lab's per-candidate feature table
VARIANTS = ["", "+embws", "+nullws"]


def perrun(src: Path) -> dict:
    """{'ms': ..., 'ms+embws': ..., ...} PSMs@1% (all runs) for per-run linear rescoring."""
    out = {}
    if src.suffix == ".log":
        for m in re.finditer(r"\[all\s*\] perrun/linear:(\S+) PSMs@1%\s+([\d,]+)", src.read_text()):
            out[m.group(1)] = int(m.group(2).replace(",", ""))
    else:
        d = json.loads(src.read_text())
        for k, v in d["methods"]["all"].items():
            if k.startswith("perrun/linear:"):
                out[k.split(":", 1)[1]] = v["psms_1pct"]
    return out


def gain_rows():
    rows = []
    for key, (label, srcs) in EMB.items():
        for seed, src in enumerate(srcs):
            p = perrun(src)
            for b in BASES:
                base = p[b]
                for arm, feat in (("embedding", f"{b}+embws"), ("null control", f"{b}+nullws")):
                    rows.append(dict(dataset="8 runs (6 HEK, 2 HCT116)", embedding=label, base=BASES[b], arm=arm,
                                     seed=seed, base_psms=base, psms=p[feat], gain_psms=p[feat] - base,
                                     gain_pct=100 * (p[feat] - base) / base, source=src.name))
    p = perrun(PSM / "a2-400m_r4_hct116_perrun_seed0.json")
    for b in BASES:
        for arm, feat in (("embedding", f"{b}+embws"), ("null control", f"{b}+nullws")):
            rows.append(dict(dataset="HCT116, 18 runs (unseen)", embedding=EMB["A2"][0], base=BASES[b], arm=arm,
                             seed=0, base_psms=p[b], psms=p[feat], gain_psms=p[feat] - p[b],
                             gain_pct=100 * (p[feat] - p[b]) / p[b], source="a2-400m_r4_hct116_perrun_seed0.json"))
    return rows


def fig_gain(rows):
    cols = {"null control": "#9ca3af", "embedding": "#2563eb"}
    panels = [("8 runs (6 HEK, 2 HCT116)", BASES["ms"]), ("8 runs (6 HEK, 2 HCT116)", BASES["lab"]),
              ("HCT116, 18 runs (unseen)", None)]
    titles = ["8 runs\niona-rerank, MSFragger features", "8 runs\niona-rerank, rich features",
              "HCT116, 18 runs (unseen)\niona-rerank, Iona embedding (400M)"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2, 2, 2]})
        for ax, (ds, base), title in zip(axes, panels, titles):
            ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True); ax.axhline(0, color="#9ca3af", lw=0.8)
            groups = [e[0] for e in EMB.values()] if base else list(BASES.values())
            for gi, g in enumerate(groups):
                for k, arm in enumerate(("null control", "embedding")):
                    sel = [r for r in rows if r["dataset"] == ds and r["arm"] == arm
                           and (r["base"] == base and r["embedding"] == g if base else r["base"] == g)]
                    v = np.array([r["gain_pct"] for r in sel]); x = gi + (k - 0.5) * 0.38
                    ax.bar(x, v.mean(), 0.36, color=cols[arm], zorder=2,
                           yerr=v.std(ddof=1) if len(v) > 1 else None, capsize=3,
                           error_kw=dict(elinewidth=1, capthick=1, ecolor=INK))
                    if len(v) > 1:
                        ax.scatter(np.full(len(v), x), v, s=10, color=INK, zorder=3)
                    ax.text(x, max(v.max(), 0) + 0.12, f"{v.mean():+.1f}%", ha="center", fontsize=8)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([g.replace(" (", "\n(") for g in groups], fontsize=8.5)
            ax.set_title(title, loc="left", fontsize=10.5)
        axes[0].set_ylabel("% more PSMs at 1% FDR\n(vs Iona-rerank without it)")
        top = max(a.get_ylim()[1] for a in axes)
        for a in axes:
            a.set_ylim(-0.6, top)
        fig.suptitle("The Iona embedding adds identifications; a random-spectrum control does not", x=0.07, ha="left",
                     fontsize=13, fontweight="bold", y=1.05)
        fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()],
                   labels=["+ null control (random-spectrum embedding)", "+ Iona embedding"], frameon=False,
                   fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0))
        fig.savefig(FIG / "R_embedding_gain.png"); plt.close(fig)


def benchmark_rows():
    ms2r = json.loads((REPO / "baselines_wip" / "results_ms2rescore.json").read_text())
    rows = []
    d0 = perrun(EMB["A2"][1][0])
    src = EMB["A2"][1][1]
    rows.append(dict(method="MSFragger (e-value)", seed="", source=f"{src.name}:methods/all/msfragger",
                     psms=json.loads(src.read_text())["methods"]["all"]["msfragger"]["psms_1pct"]))
    for k, lab in (("ms2rescore:searchonly", "MS2Rescore, MSFragger features"),
                   ("ms2rescore:full", "MS2Rescore, MSFragger features + MS2PIP + DeepLC")):
        rows.append(dict(method=lab, seed="", psms=ms2r[k]["all"]["pooled"], source=f"results_ms2rescore.json:{k}"))
    emb = json.loads((REPO / "baselines_wip" / "results_ms2rescore_emb8_a2.json").read_text())
    for k, lab in (("ms2rescore:full+embedding(400M teacher, cosws)", "MS2Rescore, MSFragger features + MS2PIP + DeepLC + Iona embedding"),):
        rows.append(dict(method=lab, seed="", psms=emb[k]["all"]["pooled"], source=f"results_ms2rescore_emb8_a2.json:{k}"))
    for seed, src in enumerate(EMB["A2"][1]):
        p = perrun(src)
        for feat, lab in (("ms", "Iona-rerank, MSFragger features"), ("ms+embws", "Iona-rerank, MSFragger features + Iona embedding"),
                          ("lab", "Iona-rerank, rich features"), ("lab+embws", "Iona-rerank, rich features + Iona embedding")):
            rows.append(dict(method=lab, seed=seed, psms=p[feat], source=src.name))
    assert d0  # A2 seed 0 parsed from its log
    return rows


def fig_benchmark(rows):
    """Drawn by the paper folder's standalone script, so the two figures are the same code."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("plot_reranking", REPO / "paper" / "experiments" / "reranking" / "plot_reranking.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.benchmark(csv_path=FIG / "R_benchmark.csv", out_path=FIG / "R_benchmark.png")


def write(name, rows):
    with open(FIG / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    g = gain_rows(); write("R_embedding_gain.csv", g); fig_gain(g)
    b = benchmark_rows(); write("R_benchmark.csv", b); fig_benchmark(b)
    for ds in dict.fromkeys(r["dataset"] for r in g):
        for e in dict.fromkeys(r["embedding"] for r in g if r["dataset"] == ds):
            for base in BASES.values():
                for arm in ("embedding", "null control"):
                    v = [r["gain_psms"] for r in g if (r["dataset"], r["embedding"], r["base"], r["arm"]) == (ds, e, base, arm)]
                    if v:
                        print(f"  {ds[:8]:8s} {e:28s} {base[:18]:18s} {arm:12s} {v}")


if __name__ == "__main__":
    main()
