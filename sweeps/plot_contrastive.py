"""Contrastive: the scale curve, and the pretraining ablation, as separate figures.

    python sweeps/plot_contrastive.py

contrastive_scaling.png   Separation ratio against model size, 6 SEEDS per scale at
                          one configuration, so the band is run-to-run noise. The faint
                          series is the earlier curve at lr 2e-5 / KL 10 / temperature
                          0.07 under the broken sampler, which is what "contrastive
                          saturates at 100m" was read from.

contrastive_ablation.png  Pretrained against randomly initialised, one point per
                          hyperparameter CELL -- twelve settings at a single seed, not
                          twelve runs of one setting. The spread is the sweep, not
                          noise, so a mean over it would average good settings with
                          collapsed ones and estimate nothing; the bar marks the best
                          cell. PROVISIONAL: 50m only, and at the OLD configuration for
                          both arms, so they are matched to each other but not to the
                          scaling figure. Job 8848471 re-runs it properly.
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
    pre, rand = ratios("8842232"), ratios("8842288")
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    rng = np.random.default_rng(0)
    for i, (series, colour) in enumerate(((pre, BLUE), (rand, RED))):
        v = list(series.values())
        if not v:
            continue
        ax.scatter(np.full(len(v), i) + rng.normal(0, 0.05, len(v)), v, s=36,
                   color=colour, alpha=0.85, zorder=3)
        # Range and best, never a mean: these are twelve different hyperparameter
        # cells at one seed, so averaging them mixes good settings with collapsed
        # ones and estimates nothing.
        ax.vlines(i, min(v), max(v), color=colour, lw=1.2, alpha=0.5, zorder=2)
        ax.hlines(max(v), i - 0.18, i + 0.18, color=INK, lw=2, zorder=4)
        ax.annotate(f"best {max(v):.2f}", (i + 0.22, max(v)), fontsize=8.5, color=INK,
                    va="center")
    ax.axhline(FLOOR, color=RED, lw=1.1, ls=":")
    ax.annotate(f"untrained {FLOOR}", (-0.45, FLOOR), fontsize=8, color=RED,
                ha="left", va="bottom")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pretrained", "random init"])
    ax.set_xlim(-0.5, 1.7)
    ax.set_ylabel("separation ratio")
    ax.set_title("50m, 12 hyperparameter cells at one seed", loc="left", pad=14)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.suptitle("Jobs 8842232 and 8842288. The spread is the hyperparameter sweep, "
                 "not noise.\nOld configuration — matched to each other, not to the "
                 "scaling figure. Re-run: job 8848471.",
                 x=0.005, ha="left", fontsize=8.5, color=MUTED, y=1.06)
    fig.savefig(FIGS / "contrastive_ablation.png"); plt.close(fig)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_scaling(); print("  wrote contrastive_scaling.png")
    fig_ablation(); print("  wrote contrastive_ablation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
