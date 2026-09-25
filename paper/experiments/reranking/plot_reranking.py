"""Regenerate the two figures in this folder from their CSVs.

    python plot_reranking.py      # needs matplotlib + numpy

    R_embedding_gain.csv -> R_embedding_gain.png   % more PSMs at 1% FDR from adding the embedding vs a null control
    R_benchmark.csv      -> R_benchmark.png        PSMs at 1% FDR: MSFragger, MS2Rescore, our per-run classifier
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INK, MUTED, GRIDC = "#1f2937", "#6b7280", "#e5e7eb"
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9ca3af", "axes.linewidth": 0.8,
         "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "figure.dpi": 200}
ERR = dict(elinewidth=1, capthick=1, ecolor=INK)
EIGHT, HCT = "8 runs (6 HEK, 2 HCT116)", "HCT116, 18 runs (unseen)"


def embedding_gain():
    rows = list(csv.DictReader(open(HERE / "R_embedding_gain.csv")))
    for r in rows:   # exact percentage from the counts
        r["pct"] = 100 * (int(r["psms"]) - int(r["base_psms"])) / int(r["base_psms"])
    cols = {"null control": "#9ca3af", "embedding": "#2563eb"}
    teachers = list(dict.fromkeys(r["embedding"] for r in rows if r["dataset"] == EIGHT))
    bases = list(dict.fromkeys(r["base"] for r in rows))
    panels = [(EIGHT, "MSFragger features only", teachers, "8 runs\nMSFragger features only"),
              (EIGHT, "all features", teachers, "8 runs\nall features"),
              (HCT, None, bases, "HCT116, 18 runs (unseen)\n400M-teacher embedding")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
        for ax, (ds, base, groups, title) in zip(axes, panels):
            ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True); ax.axhline(0, color="#9ca3af", lw=0.8)
            for gi, g in enumerate(groups):
                for k, arm in enumerate(("null control", "embedding")):
                    sel = [r for r in rows if r["dataset"] == ds and r["arm"] == arm
                           and ((r["base"] == base and r["embedding"] == g) if base else r["base"] == g)]
                    v = np.array([r["pct"] for r in sel]); x = gi + (k - 0.5) * 0.38
                    ax.bar(x, v.mean(), 0.36, color=cols[arm], zorder=2, yerr=v.std(ddof=1) if len(v) > 1 else None,
                           capsize=3, error_kw=ERR)
                    if len(v) > 1:
                        ax.scatter(np.full(len(v), x), v, s=10, color=INK, zorder=3)
                    ax.text(x, max(v.max(), 0) + 0.12, f"{v.mean():+.1f}%", ha="center", fontsize=8)
            ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=8.5)
            ax.set_title(title, loc="left", fontsize=10.5)
        axes[0].set_ylabel("% more PSMs at 1% FDR\n(vs our per-run classifier without it)")
        top = max(a.get_ylim()[1] for a in axes)
        for a in axes:
            a.set_ylim(-0.6, top)
        fig.suptitle("Our embedding adds identifications; a random-spectrum control does not", x=0.07, ha="left",
                     fontsize=13, fontweight="bold", y=1.05)
        fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()],
                   labels=["+ null control (random-spectrum embedding)", "+ our embedding"], frameon=False,
                   fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0))
        fig.savefig(HERE / "R_embedding_gain.png"); plt.close(fig)


def benchmark():
    rows = list(csv.DictReader(open(HERE / "R_benchmark.csv")))
    order = list(dict.fromkeys(r["method"] for r in rows))
    order = [order[0]] + [m for m in order if "MSFragger features only" in m] + \
            [m for m in order if m.startswith("MS2Rescore")] + \
            [m for m in order if m.startswith("our") and "MSFragger features only" not in m]
    col = lambda m: "#2563eb" if "+ embedding" in m else ("#93c5fd" if m.startswith("our ") else
                                                          ("#f59e0b" if "MS2Rescore" in m else "#9ca3af"))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.xaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
        for i, m in enumerate(order):
            v = np.array([float(r["psms"]) for r in rows if r["method"] == m])
            ax.barh(i, v.mean(), 0.7, color=col(m), xerr=v.std(ddof=1) if len(v) > 1 else None, capsize=3,
                    error_kw=ERR, zorder=2)
            ax.text(v.mean() + 700, i, f"{v.mean():,.0f}", va="center", fontsize=8.5)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.invert_yaxis()
        ax.set_xlim(80000, 138000); ax.set_xlabel("PSMs at 1% FDR (8 runs: 6 HEK, 2 HCT116)")
        ax.set_title("PSM rescoring: identifications at 1% FDR", loc="left", fontsize=13, fontweight="bold", pad=10)
        fig.text(0.01, -0.03, "Our per-run classifier: mean ± sd over 3 seeds; embedding = 400M-teacher peptide embedder.",
                 fontsize=8, color=MUTED)
        fig.savefig(HERE / "R_benchmark.png"); plt.close(fig)


if __name__ == "__main__":
    embedding_gain(); benchmark()
