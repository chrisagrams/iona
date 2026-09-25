"""Regenerate C_zeroshot.png from C_zeroshot.csv (this folder).

    python plot_zeroshot.py      # needs matplotlib + numpy

Per frozen pretrained encoder (size@pretraining step): MAP@R of the best layer, raw vs after
all-but-the-top (ABTT). Two panels: ms-contrastive-100k test and the unseen yeast 20k subset.
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


def main():
    rows = list(csv.DictReader(open(HERE / "C_zeroshot.csv")))
    key = lambda e: (int(e.split("m@")[0]), int(e.split("@")[1][:-1]))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for ax, bench, title in ((axes[0], "ms-contrastive-100k", "ms-contrastive-100k test (in-distribution)"),
                                 (axes[1], "yeast-20k", "yeast 20k subset (unseen)")):
            ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
            rs = {r["encoder"]: r for r in rows if r["benchmark"] == bench}
            enc = sorted(rs, key=key); x = np.arange(len(enc))
            ax.bar(x - 0.2, [float(rs[e]["raw_best_layer"]) for e in enc], 0.38, color="#9ca3af",
                   label="frozen, best layer", zorder=2)
            ax.bar(x + 0.2, [float(rs[e]["abtt_best"]) for e in enc], 0.38, color="#2563eb",
                   label="frozen, best layer + ABTT", zorder=2)
            for i, e in enumerate(enc):
                ax.text(i + 0.2, float(rs[e]["abtt_best"]) + 0.01, f"{float(rs[e]['abtt_best']):.2f}", ha="center", fontsize=7.5)
            ax.set_xticks(x); ax.set_xticklabels([e.upper().replace("K", "k") for e in enc], rotation=30, ha="right")
            ax.set_title(title, loc="left", fontsize=10.5)
        axes[0].set_ylabel("MAP@R (experimental spectra)")
        axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
        fig.suptitle("Zero-shot retrieval from the pretrained encoder (no contrastive training)", x=0.07, ha="left",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.savefig(HERE / "C_zeroshot.png"); plt.close(fig)


if __name__ == "__main__":
    main()
