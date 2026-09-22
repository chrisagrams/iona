"""Contrastive: the scale curve, and the pretraining ablation, as separate figures.

    python sweeps/plot_contrastive.py

contrastive_scaling.png   Separation ratio against model size, 6 SEEDS per scale at
                          one configuration, so the band is run-to-run noise. The faint
                          series is the earlier curve at lr 2e-5 / KL 10 / temperature
                          0.07 under the broken sampler, which is what "contrastive
                          saturates at 100m" was read from.

contrastive_ablation.png  Pretrained against randomly initialised, BEST cell each.
                          Selection is max over hyperparameters everywhere in this
                          project, so the spread across cells is not an uncertainty and
                          is not drawn. These two points have no error bars, because
                          neither grid was seed-replicated. PROVISIONAL: 50m only, old
                          configuration. Job 8848471 re-runs it with 6 seeds at all
                          four scales.
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
FIGS = REPO / "results" / "finetune" / "contrastive" / "figures"
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


def ratios(job):
    out = {}
    for d in glob.glob(f"{RUNS}/sweep-*-{job}"):
        arm = re.sub(rf"^sweep-|-{job}$", "", os.path.basename(d))
        f = os.path.join(d, "all_results.json")
        if os.path.exists(f):
            try:
                out[arm] = json.load(open(f))["sep_spectrum/ratio"]
            except Exception:
                pass
    return out


def fig_scaling():
    new, old = ratios("8848049"), ratios("8845057")
    scales = ["s050m", "s100m", "s200m", "s400m"]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for series, colour, alpha, lab in ((old, GREY, 0.8, "KL 10 / t 0.07, pre-FT14"),
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
    ax.annotate(f"untrained {FLOOR}", (3.35, FLOOR), fontsize=8, color=RED,
                ha="right", va="bottom")
    ax.set_xticks(range(4)); ax.set_xticklabels(["50m", "100m", "200m", "400m"])
    ax.set_xlim(-0.35, 3.45)
    ax.set_xlabel("model size"); ax.set_ylabel("separation ratio")
    ax.set_title("separation ratio by scale", loc="left", pad=14)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.4, loc="upper left")
    fig.suptitle("Job 8848049. One configuration, 6 SEEDS per scale — the band is "
                 "run-to-run noise.", x=0.005, ha="left", fontsize=8.5, color=MUTED,
                 y=1.02)
    fig.savefig(FIGS / "contrastive_scaling.png"); plt.close(fig)


def fig_ablation():
    """Best cell only. HP variance is not an error bar.

    Selection here is max over hyperparameters, as everywhere else in this project, so
    showing the spread across cells would plot the sweep rather than an uncertainty and
    invite comparison with the seed bands in the scaling figure. Until the matched
    ablation lands (job 8848471, 4 scales x 6 seeds) these two points carry no
    uncertainty at all, and the figure says so rather than implying one.
    """
    pre, rand = ratios("8842232"), ratios("8842288")
    if not pre or not rand:
        print("  ablation: no data"); return
    best = [max(pre.values()), max(rand.values())]

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.bar([0, 1], best, width=0.5, color=[BLUE, RED], alpha=0.9)
    for i, b in enumerate(best):
        ax.annotate(f"{b:.2f}", (i, b), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=10, color=INK)
    ax.axhline(FLOOR, color=RED, lw=1.1, ls=":")
    ax.annotate(f"untrained {FLOOR}", (-0.42, FLOOR), fontsize=8, color=RED,
                ha="left", va="bottom")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pretrained", "random init"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, max(best) * 1.18)
    ax.set_ylabel("separation ratio")
    ax.set_title("50m, best of 12 hyperparameter cells", loc="left", pad=14)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.suptitle("Jobs 8842232 and 8842288, one seed each — no error bars available.\n"
                 "Old configuration, matched to each other. Seed-replicated re-run at "
                 "all four scales: job 8848471.",
                 x=0.005, ha="left", fontsize=8.5, color=MUTED, y=1.06)
    fig.savefig(FIGS / "contrastive_ablation.png"); plt.close(fig)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_scaling(); print("  wrote contrastive_scaling.png")
    fig_ablation(); print("  wrote contrastive_ablation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
