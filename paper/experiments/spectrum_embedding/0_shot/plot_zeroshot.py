"""Regenerate C_zeroshot.png from C_zeroshot.csv (this folder).

    python plot_zeroshot.py      # needs matplotlib + numpy

Frozen pretrained encoders, MAP@R of the best layer, raw vs after all-but-the-top (ABTT).
Left: every encoder on ms-contrastive-100k test (4 sizes x 6 pretraining checkpoints), one line per size
(solid: best layer + ABTT, dashed: best layer raw). Right: the encoders scored on the unseen yeast 20k
subset, as bars.
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


SIZE_COLS = {"50m": "#93c5fd", "100m": "#3b82f6", "200m": "#1d4ed8", "400m": "#1e3a8a"}
CKPTS = ["10k", "120k", "220k", "330k", "430k", "540k"]


def main():
    from matplotlib.lines import Line2D
    rows = list(csv.DictReader(open(HERE / "C_zeroshot.csv")))
    key = lambda e: (int(e.split("m@")[0]), int(e.split("@")[1][:-1]))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
        ax = axes[0]
        ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
        rs = [r for r in rows if r["benchmark"] == "ms-contrastive-100k"]
        for sc in ["50m", "100m", "200m", "400m"]:
            pts = sorted((CKPTS.index(r["encoder"].split("@")[1]), r) for r in rs if r["encoder"].split("@")[0] == sc)
            x = [p[0] for p in pts]
            ax.plot(x, [float(p[1]["abtt_best"]) for p in pts], "-o", color=SIZE_COLS[sc], lw=2, ms=6,
                    mec="white", mew=1.1, zorder=3)
            ax.plot(x, [float(p[1]["raw_best_layer"]) for p in pts], "--o", color=SIZE_COLS[sc], lw=1.1, ms=3.5,
                    alpha=0.7, zorder=2)
        ax.set_xticks(range(len(CKPTS))); ax.set_xticklabels(CKPTS); ax.set_xlabel("pretraining steps")
        ax.set_title("ms-contrastive-100k test (in-distribution)", loc="left", fontsize=10.5)
        ax.set_ylabel("MAP@R (experimental spectra)")
        h = [Line2D([], [], color=SIZE_COLS[s], lw=2, marker="o", ms=6, mec="white", label=s.upper())
             for s in ["400m", "200m", "100m", "50m"]]
        h += [Line2D([], [], color=MUTED, lw=2, label="best layer + ABTT"),
              Line2D([], [], color=MUTED, lw=1.1, ls="--", label="best layer, raw")]
        ax.legend(handles=h, frameon=False, fontsize=8, loc="upper left", ncol=3)
        ax.set_ylim(0, 0.85)
        ax = axes[1]
        ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
        yr = {r["encoder"]: r for r in rows if r["benchmark"] == "yeast-20k"}
        enc = sorted(yr, key=key); x = np.arange(len(enc))
        ax.bar(x - 0.2, [float(yr[e]["raw_best_layer"]) for e in enc], 0.38, color="#9ca3af",
               label="frozen, best layer", zorder=2)
        ax.bar(x + 0.2, [float(yr[e]["abtt_best"]) for e in enc], 0.38, color="#2563eb",
               label="frozen, best layer + ABTT", zorder=2)
        for i, e in enumerate(enc):
            ax.text(i + 0.2, float(yr[e]["abtt_best"]) + 0.01, f"{float(yr[e]['abtt_best']):.2f}", ha="center", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels([e.upper().replace("K", "k") for e in enc], rotation=30, ha="right")
        ax.set_title("yeast 20k subset (unseen)", loc="left", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        fig.suptitle("Zero-shot retrieval from the pretrained encoder (no contrastive training)", x=0.07, ha="left",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.savefig(HERE / "C_zeroshot.png"); plt.close(fig)


if __name__ == "__main__":
    main()
