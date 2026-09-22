"""Contrastive: the scale curve, and the pretraining ablation.

    python sweeps/plot_contrastive.py

LEFT   Separation ratio against model size, 6 seeds per scale, at the configuration
       the 96-arm HP grid selected (lr 2e-5, KL 0, temperature 0.2). The faint series
       is the earlier curve at lr 2e-5 / KL 10 / temperature 0.07 under the broken
       sampler, which is what "contrastive saturates at 100m" was read from.

RIGHT  Pretrained against randomly initialised, one point per hyperparameter cell.
       PROVISIONAL: this is the only ablation data that exists, and it is at 50m only
       and at the OLD configuration for both arms. They are matched to each other, so
       the comparison is valid; it is not matched to the left panel. The re-run at the
       corrected configuration and all four scales is job 8848471.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "results" / "figures"
RUNS = "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs"
INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"
BLUE, RED, GREY = "#2563eb", "#dc2626", "#9ca3af"
FLOOR = 1.35

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.7, "figure.facecolor": "white",
})


def ratios(job, pattern="sweep-*"):
    out = {}
    for d in glob.glob(f"{RUNS}/{pattern}-{job}"):
        arm = re.sub(rf"^sweep-|-{job}$", "", os.path.basename(d))
        f = os.path.join(d, "all_results.json")
        if os.path.exists(f):
            try:
                out[arm] = json.load(open(f))["sep_spectrum/ratio"]
            except Exception:
                pass
    return out


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    new, old = ratios("8848049"), ratios("8845057")
    pre50, rand50 = ratios("8842232"), ratios("8842288")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                  gridspec_kw={"width_ratios": [1.3, 1]})

    scales = ["s050m", "s100m", "s200m", "s400m"]
    labels = ["50m", "100m", "200m", "400m"]
    x = np.arange(len(scales))
    for series, colour, alpha, lab in ((old, GREY, 0.75, "KL 10 / t 0.07, pre-FT14"),
                                       (new, BLUE, 1.0, "KL 0 / t 0.2, post-FT14")):
        m, s, xi = [], [], []
        for i, k in enumerate(scales):
            v = [r for a, r in series.items() if a.startswith(k)]
            if len(v) > 1:
                m.append(np.mean(v)); s.append(np.std(v, ddof=1)); xi.append(i)
        if not m:
            continue
        m, s, xi = np.array(m), np.array(s), np.array(xi)
        ax.fill_between(xi, m - s, m + s, color=colour, alpha=0.15 * alpha, lw=0)
        ax.errorbar(xi, m, yerr=s, fmt="o-", ms=6, lw=2.2, color=colour,
                    ecolor=colour, capsize=4, alpha=alpha, label=lab)
        if colour == BLUE:
            for a_, b_, e_ in zip(xi, m, s):
                ax.annotate(f"{b_:.2f}", (a_, b_ + e_), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=8.5, color=INK)
    ax.axhline(FLOOR, color=RED, lw=1.1, ls=":")
    ax.annotate(f"untrained floor {FLOOR}", (3.3, FLOOR), fontsize=8, color=RED,
                ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlim(-0.35, 3.45)
    ax.set_xlabel("model size"); ax.set_ylabel("separation ratio")
    ax.set_title("separation ratio by scale", loc="left", pad=14)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")

    for i, (series, colour, lab) in enumerate(((pre50, BLUE, "pretrained"),
                                               (rand50, RED, "random init"))):
        v = list(series.values())
        if not v:
            continue
        jitter = np.random.default_rng(0).normal(0, 0.05, len(v))
        ax2.scatter(np.full(len(v), i) + jitter, v, s=34, color=colour, alpha=0.85,
                    zorder=3)
        ax2.hlines(np.mean(v), i - 0.22, i + 0.22, color=INK, lw=2, zorder=4)
        ax2.annotate(f"{np.mean(v):.2f}", (i + 0.26, np.mean(v)), fontsize=9,
                     color=INK, va="center")
    ax2.axhline(FLOOR, color=RED, lw=1.1, ls=":")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["pretrained", "random init"])
    ax2.set_xlim(-0.5, 1.6)
    ax2.set_ylabel("separation ratio")
    ax2.set_title("50m, one point per hyperparameter cell", loc="left", pad=14)
    ax2.grid(axis="y"); ax2.set_axisbelow(True)

    fig.suptitle("Left: job 8848049, 6 seeds per scale, band ±1 sd. Right: jobs "
                 "8842232 and 8842288, 12 cells each at the OLD configuration — "
                 "matched to each other, not to the left panel.",
                 x=0.005, ha="left", fontsize=8.3, color=MUTED, y=1.03)
    out = FIGS / "contrastive_scaling_and_ablation.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
