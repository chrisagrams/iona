"""Regenerate C_benchmarks.png from C_benchmarks.csv (this folder).

    python plot_benchmarks.py      # needs matplotlib + numpy

Four benchmarks x four methods. Ours: mean ± sd over seeds. Binned cosine: the row whose note is
tagged "[plotted: best width]" (the better of 1 Da and 0.1 Da bins on that benchmark).
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
BENCH = [("ms-contrastive-100k", "ms-contrastive-100k test\n(in-distribution)"),
         ("hek-lowres", "HEK\n(unseen, low-res MS2)"),
         ("yeast-full", "nine-species yeast\n(unseen, high-res)"),
         ("yeast-20k", "yeast 20k subset\n(unseen, high-res)")]
SERIES = [("ours: C7 400M", "#2563eb"), ("ours: replicate corpus only 400M", "#93c5fd"),
          ("GLEAMS", "#f59e0b"), ("binned cosine", "#9ca3af")]
LEGEND = ["ours: C7 400M", "ours: replicate corpus only 400M", "GLEAMS", "binned cosine (best bin width)"]


def main():
    rows = list(csv.DictReader(open(HERE / "C_benchmarks.csv")))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4.4))
        ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
        w = 0.19
        for i, (name, col) in enumerate(SERIES):
            for j, (b, _) in enumerate(BENCH):
                sel = [r for r in rows if r["benchmark"] == b and r["model"] == name]
                if name == "binned cosine":
                    sel = [r for r in sel if "[plotted: best width]" in r["note"]][:1]
                if not sel:
                    continue
                v = [float(r["map_at_r"]) for r in sel]
                m = float(np.mean(v)); s = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                x = j + (i - 1.5) * w
                ax.bar(x, m, w * 0.92, color=col, yerr=s if len(v) > 1 else None, capsize=3,
                       error_kw=dict(elinewidth=1, capthick=1, ecolor=INK), zorder=2)
                ax.text(x, m + s + 0.012, f"{m:.2f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(BENCH))); ax.set_xticklabels([l for _, l in BENCH])
        ax.set_ylim(0, 1.0); ax.set_ylabel("MAP@R (experimental spectra)")
        ax.set_title("Spectrum retrieval: ours vs GLEAMS and binned cosine", loc="left", fontsize=13,
                     fontweight="bold", pad=12)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c in SERIES], labels=LEGEND,
                  frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
        fig.text(0.01, -0.13, "Ours: mean ± sd over 3 seeds (replicate-only on yeast 20k: 1 seed). "
                 "Binned cosine: better of 1 Da and 0.1 Da bins per benchmark.", fontsize=8, color=MUTED)
        fig.savefig(HERE / "C_benchmarks.png"); plt.close(fig)


if __name__ == "__main__":
    main()
