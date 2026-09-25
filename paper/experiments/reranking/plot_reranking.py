"""Regenerate the two figures in this folder from their CSVs.

    python plot_reranking.py      # needs matplotlib + numpy

    R_embedding_gain.csv -> R_embedding_gain.png   % more PSMs at 1% FDR from adding the embedding vs a null control
    R_benchmark.csv      -> R_benchmark.png        PSMs at 1% FDR: MSFragger, MS2Rescore, iona-rerank
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
    panels = [(EIGHT, "MSFragger features", teachers, "8 runs\niona-rerank, MSFragger features"),
              (EIGHT, "rich features", teachers, "8 runs\niona-rerank, rich features"),
              (HCT, None, bases, "HCT116, 18 runs (unseen)\niona-rerank, iona embedding (400M)")]
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
            ax.set_xticks(range(len(groups))); ax.set_xticklabels([g.replace(" (", "\n(") for g in groups], fontsize=8.5)
            ax.set_title(title, loc="left", fontsize=10.5)
        axes[0].set_ylabel("% more PSMs at 1% FDR\n(vs iona-rerank without it)")
        top = max(a.get_ylim()[1] for a in axes)
        for a in axes:
            a.set_ylim(-0.6, top)
        fig.suptitle("The iona embedding adds identifications; a random-spectrum control does not", x=0.07, ha="left",
                     fontsize=13, fontweight="bold", y=1.05)
        fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()],
                   labels=["+ null control (random-spectrum embedding)", "+ iona embedding"], frameon=False,
                   fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0))
        fig.savefig(HERE / "R_embedding_gain.png"); plt.close(fig)


FEATURE_COLS = [("MSFragger\nfeatures", "msf"), ("rich\nfeatures", "rich"), ("MS2PIP", "ms2pip"),
                ("DeepLC", "deeplc"), ("iona\nembedding", "iona"), ("null\ncontrol", "null")]


def features_of(method):
    """Which feature groups a benchmark row uses, read off its label."""
    f = {"msf"}                                   # every method uses MSFragger's search features
    if "rich features" in method:
        f.add("rich")                             # the lab's table (it also contains MSFragger's scores)
    if "MS2PIP" in method:
        f.add("ms2pip")
    if "DeepLC" in method:
        f.add("deeplc")
    if "iona embedding" in method:
        f.add("iona")
    if "null control" in method:
        f.add("null")
    return f


def benchmark(csv_path=None, out_path=None):
    rows = list(csv.DictReader(open(csv_path or HERE / "R_benchmark.csv")))
    order = list(dict.fromkeys(r["method"] for r in rows))
    order = [order[0]] + [m for m in order if m.startswith("iona-rerank, MSFragger")] + \
            [m for m in order if m.startswith("MS2Rescore")] + \
            [m for m in order if m.startswith("iona-rerank, rich")]
    col = lambda m: ("#b45309" if m.startswith("MS2Rescore") and "+ iona embedding" in m else
                     "#fcd34d" if "null control" in m else
                     "#2563eb" if "+ iona embedding" in m else "#93c5fd" if m.startswith("iona-rerank") else
                     "#f59e0b" if "MS2Rescore" in m else "#9ca3af")
    name = lambda m: m.split(",")[0].replace(" (e-value)", "")
    with plt.rc_context(STYLE):
        fig, (axt, ax) = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True,
                                      gridspec_kw={"width_ratios": [1.7, 2.2], "wspace": 0.04})
        n = len(order)
        for i, m in enumerate(order):
            f = features_of(m)
            for j, (_, key) in enumerate(FEATURE_COLS):
                on = key in f
                axt.scatter(j, i, s=90 if on else 16, marker="o", zorder=3,
                            color=(col(m) if key in ("iona", "null") and on else INK) if on else "#d1d5db")
            if i % 2 == 0:
                axt.axhspan(i - 0.5, i + 0.5, color="#f9fafb", zorder=0); ax.axhspan(i - 0.5, i + 0.5, color="#f9fafb", zorder=0)
            v = np.array([float(r["psms"]) for r in rows if r["method"] == m])
            ax.barh(i, v.mean(), 0.66, color=col(m), xerr=v.std(ddof=1) if len(v) > 1 else None, capsize=3,
                    error_kw=ERR, zorder=2)
            ax.text(v.mean() + 700, i, f"{v.mean():,.0f}", va="center", fontsize=8.5)
        axt.set_xticks(range(len(FEATURE_COLS))); axt.set_xticklabels([c for c, _ in FEATURE_COLS], fontsize=8.5)
        axt.xaxis.tick_top(); axt.tick_params(axis="x", length=0)
        axt.set_xlim(-0.6, len(FEATURE_COLS) - 0.4)
        axt.set_yticks(range(n)); axt.set_yticklabels([name(m) for m in order], fontsize=9.5); axt.invert_yaxis()
        axt.tick_params(axis="y", length=0)
        for s_ in ("top", "right", "bottom", "left"):
            axt.spines[s_].set_visible(False)
        ax.xaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
        ax.set_xlim(80000, 140000); ax.set_xlabel("PSMs at 1% FDR (8 runs: 6 HEK, 2 HCT116)")
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        fig.suptitle("PSM rescoring: identifications at 1% FDR, and the features each method uses", x=0.06, ha="left",
                     fontsize=13, fontweight="bold", y=1.03)
        fig.text(0.06, -0.05, "iona-rerank: our per-run classifier, mean ± sd over 3 seeds; MS2Rescore rows: one run each. "
                 "Rich features: the lab's per-candidate table (scores from 4 search engines incl. MSFragger,\n"
                 "fragment-ion and cross-candidate features). MS2Rescore also computes its own fragment-match features. "
                 "iona embedding: 400M; null control: the same features from a random spectrum's embedding.",
                 fontsize=8, color=MUTED)
        fig.savefig(out_path or HERE / "R_benchmark.png"); plt.close(fig)


if __name__ == "__main__":
    embedding_gain(); benchmark()
