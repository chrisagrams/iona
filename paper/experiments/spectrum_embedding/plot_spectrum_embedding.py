"""Regenerate the four figures in this folder from their CSVs.

    python plot_spectrum_embedding.py      # needs matplotlib + numpy

    C_transfer.csv             -> C_transfer.png
    C_transfer_ours.csv        -> C_transfer_ours.png
    C_pretraining_scaling.csv  -> C_pretraining_scaling.png
    C_pretraining_ablation.csv -> C_pretraining_ablation.png
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
INK, MUTED, GRIDC = "#1f2937", "#6b7280", "#e5e7eb"
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9ca3af", "axes.linewidth": 0.8,
         "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "figure.dpi": 200}
ERR = dict(elinewidth=1, capthick=1, ecolor=INK)


def read(name):
    return list(csv.DictReader(open(HERE / name)))


def msd(v):
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def transfer_ours():
    rows = read("C_transfer_ours.csv")
    series = [("fine-tuned 400M", "#1e3a8a"), ("fine-tuned 50M", "#93c5fd"),
              ("replicate corpus only 400M", "#0d9488"), ("replicate corpus only 50M", "#5eead4"),
              ("frozen + ABTT (best encoder)", "#7c3aed")]
    panels = [("ms-contrastive-100k", "ms-contrastive-100k test\n(in-distribution)"),
              ("yeast-20k", "yeast 20k subset\n(unseen)")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8), sharey=True)
        for ax, (b, title) in zip(axes, panels):
            ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
            x = 0
            for name, col in series:
                sel = [r for r in rows if r["benchmark"] == b and r["model"] == name]
                if not sel:
                    continue
                m, s = msd([float(r["map_at_r"]) for r in sel])
                ax.bar(x, m, 0.8, color=col, yerr=s if len(sel) > 1 else None, capsize=3, error_kw=ERR, zorder=2)
                ax.text(x, m + s + 0.015, f"{m:.2f}", ha="center", fontsize=8)
                if name.startswith("frozen"):
                    enc = f"{sel[0]['scale'].upper()}@{sel[0]['pretrain_ckpt']}".replace("K", "k")
                    ax.text(x, 0.02, enc, rotation=90, ha="center", va="bottom", fontsize=7.5, color="white")
                x += 1
            ax.set_xticks([]); ax.set_xlim(-0.7, x - 0.3); ax.set_title(title, loc="left", fontsize=10.5)
        axes[0].set_ylim(0, 1.0); axes[0].set_ylabel("MAP@R (experimental spectra)")
        fig.suptitle("Our models in-distribution vs unseen", x=0.07, ha="left", fontsize=13, fontweight="bold", y=1.03)
        fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c in series], labels=[n for n, _ in series],
                   frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.03))
        fig.savefig(HERE / "C_transfer_ours.png"); plt.close(fig)


def transfer():
    """Our models vs GLEAMS and binned cosine (the better bin width, tagged in the CSV)."""
    rows = read("C_transfer.csv")
    series = [("Iona spectrum encoder 400M", "#1e3a8a"), ("Iona spectrum encoder 50M", "#93c5fd"),
              ("Iona spectrum encoder 400M (replicate corpus only)", "#0d9488"),
              ("Iona spectrum encoder 50M (replicate corpus only)", "#5eead4"),
              ("frozen + ABTT (best encoder)", "#7c3aed"), ("GLEAMS", "#f59e0b"), ("binned cosine", "#9ca3af")]
    panels = [("ms-contrastive-100k", "ms-contrastive-100k test\n(in-distribution)"),
              ("yeast-20k", "yeast 20k subset\n(unseen)"), ("mouse-20k", "mouse 20k subset\n(unseen)")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 4.8), sharey=True)
        for ax, (b, title) in zip(axes, panels):
            ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
            x = 0
            for name, col in series:
                sel = [r for r in rows if r["benchmark"] == b and r["model"] == name]
                if name == "binned cosine":
                    sel = [r for r in sel if "[plotted: best width]" in r["note"]]
                if not sel:
                    continue
                if name == "GLEAMS":
                    x += 0.5                                   # gap between ours and the baselines
                m, s = msd([float(r["map_at_r"]) for r in sel])
                ax.bar(x, m, 0.8, color=col, yerr=s if len(sel) > 1 else None, capsize=3, error_kw=ERR, zorder=2)
                ax.text(x, m + s + 0.015, f"{m:.2f}", ha="center", fontsize=8)
                if name.startswith("frozen"):
                    enc = f"{sel[0]['scale'].upper()}@{sel[0]['pretrain_ckpt']}".replace("K", "k")
                    ax.text(x, 0.02, enc, rotation=90, ha="center", va="bottom", fontsize=7.5, color="white")
                x += 1
            ax.set_xticks([]); ax.set_xlim(-0.7, x - 0.3); ax.set_title(title, loc="left", fontsize=10.5)
        axes[0].set_ylim(0, 1.0); axes[0].set_ylabel("MAP@R (experimental spectra)")
        fig.suptitle("Our models vs baselines, in-distribution and on unseen data", x=0.07, ha="left",
                     fontsize=13, fontweight="bold", y=1.03)
        fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c in series], labels=[n for n, _ in series],
                   frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.03))
        fig.savefig(HERE / "C_transfer.png"); plt.close(fig)


def pretraining_scaling():
    rows = read("C_pretraining_scaling.csv")
    cells = defaultdict(list)
    for r in rows:
        cells[(r["scale"], int(r["pretrain_ckpt"].rstrip("k")))].append(float(r["map_at_r"]))
    cks = sorted({c for _, c in cells})
    cols = {"50m": "#93c5fd", "100m": "#3b82f6", "200m": "#1d4ed8", "400m": "#1e3a8a"}
    mk = {"50m": "o", "100m": "o", "200m": "D", "400m": "s"}
    dx = {"50m": 0, "100m": 0, "200m": -14, "400m": 14}
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
        handles = []
        for sc in ["400m", "200m", "100m", "50m"]:
            pts = [(c, msd(cells[(sc, c)])) for c in cks if (sc, c) in cells]
            ax.errorbar([c + dx[sc] for c, _ in pts], [p[0] for _, p in pts], yerr=[p[1] for _, p in pts],
                        color=cols[sc], lw=2.0, marker=mk[sc], ms=7, mec="white", mew=1.2, capsize=3.5,
                        elinewidth=1.2, zorder=4 if len(pts) == 1 else 3)
            handles.append(Line2D([], [], color=cols[sc], lw=2 if len(pts) > 1 else 0, marker=mk[sc], ms=7,
                                  mec="white", mew=1.2, label=sc.upper() + ("" if len(pts) > 1 else "  (220k only)")))
        ax.set_xticks(cks); ax.set_xticklabels([f"{c}k" for c in cks])
        ax.set_xlabel("pretraining steps of the starting checkpoint", labelpad=6)
        ax.set_ylabel("MAP@R, ms-contrastive-100k test", labelpad=6)
        ax.set_title("Pretraining and scale improve spectrum retrieval", loc="left", fontsize=13, fontweight="bold", pad=12)
        ax.legend(handles=handles, title="model size", frameon=False, fontsize=9, title_fontsize=9.5,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.text(0.01, -0.04, "Replicate-corpus recipe (24 epochs), mean ± sd over 3 seeds.", fontsize=8, color=MUTED)
        fig.savefig(HERE / "C_pretraining_scaling.png"); plt.close(fig)


def pretraining_ablation():
    rows = read("C_pretraining_ablation.csv")
    scales = ["50m", "100m", "200m", "400m"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for ax, key, lab in ((axes[0], "map_at_100", "MAP@100"), (axes[1], "hit_at_1", "Hit@1")):
            ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
            x = np.arange(len(scales))
            for k, (kind, col) in enumerate((("random init", "#9ca3af"), ("pretrained", "#2563eb"))):
                vals = [[float(r[key]) for r in rows if r["init"] == kind and r["scale"] == sc] for sc in scales]
                m = [np.mean(v) for v in vals]; e = [np.std(v, ddof=1) for v in vals]
                xx = x + (k - 0.5) * 0.38
                ax.bar(xx, m, 0.36, yerr=e, capsize=3, color=col, label=f"{kind} + contrastive training",
                       error_kw=ERR, zorder=2)
                for xi, mi, ei in zip(xx, m, e):
                    ax.text(xi, mi + ei + 0.012, f"{mi:.2f}", ha="center", fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels([sc.upper() for sc in scales]); ax.set_ylabel(lab)
            ax.set_xlabel("model size")
        axes[0].set_ylim(0, 0.52); axes[1].set_ylim(0, 0.92)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Pretraining is what makes contrastive training work", x=0.07, ha="left", fontsize=13,
                     fontweight="bold", y=1.03)
        fig.savefig(HERE / "C_pretraining_ablation.png"); plt.close(fig)


if __name__ == "__main__":
    transfer(); transfer_ours(); pretraining_scaling(); pretraining_ablation()
