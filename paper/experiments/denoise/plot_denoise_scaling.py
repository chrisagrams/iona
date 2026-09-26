"""Regenerate D_denoise.png from denoise_scaling_pretraining.csv (this folder).

    python plot_denoise_scaling.py        # needs matplotlib + numpy

One line per pretraining checkpoint (and from scratch = pretraining_steps 0); x = model size,
y = denoise test AUROC, error bars = sd over seeds (none where a point has one run).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SCALES = ["50m", "100m", "200m", "400m"]
INK, MUTED = "#1f2937", "#6b7280"
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9ca3af", "axes.linewidth": 0.8,
         "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "figure.dpi": 200}


def main():
    cells = {}
    for r in csv.DictReader(open(HERE / "denoise_scaling_pretraining.csv")):
        cells[(r["scale"], int(r["pretraining_steps"]))] = [float(v) for v in r["per_seed_auroc"].split()]
    ckpts = sorted({c for _, c in cells if c > 0})
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    xi = np.arange(len(SCALES))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.yaxis.grid(True, color="#e5e7eb", lw=0.8); ax.set_axisbelow(True)
        cmap = plt.get_cmap("viridis_r")
        cols = {c: cmap(0.25 + 0.7 * i / (len(ckpts) - 1)) for i, c in enumerate(ckpts)}
        series = [(0, "#9ca3af", "from scratch", "s", "--")] + \
                 [(c, cols[c], f"{c // 1000}k steps", "o", "-") for c in ckpts]
        handles = []
        for c, col, lab, mk, ls in series:
            v = [cells[(s, c)] for s in SCALES]
            ax.errorbar(xi, [np.mean(u) for u in v], yerr=[sd(u) for u in v], color=col, lw=2.0, ls=ls,
                        marker=mk, ms=6.5, mfc=col, mec="white", mew=1.2, capsize=3.5, capthick=1.2,
                        elinewidth=1.2, zorder=3)
            handles.append(Line2D([], [], color=col, lw=2.0, ls=ls, marker=mk, ms=6.5, mec="white", mew=1.2, label=lab))
        ax.set_xticks(xi); ax.set_xticklabels([s.upper() for s in SCALES]); ax.set_xlim(-0.3, len(SCALES) - 0.7)
        ax.set_xlabel("model size (parameters)", labelpad=6); ax.set_ylabel("test AUROC", labelpad=6)
        ax.set_title("Denoising performance across checkpoints", loc="left", fontsize=13, fontweight="bold", pad=12)
        leg = ax.legend(handles=handles[::-1], title="pretraining", frameon=False, fontsize=9, title_fontsize=9.5,
                        loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=2.6, labelspacing=0.7)
        leg._legend_box.align = "left"
        fig.text(0.01, -0.02, "Mean ± sd over 3–5 seeds per point; 50M from scratch is a single run.",
                 fontsize=8, color=MUTED)
        fig.savefig(HERE / "D_denoise.png"); plt.close(fig)


if __name__ == "__main__":
    main()
