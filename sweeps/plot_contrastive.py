"""Contrastive: the scale curve, and the pretraining ablation, as separate figures.

    python sweeps/plot_contrastive.py

contrastive_scaling.png   Separation ratio against model size, 6 SEEDS per scale at
                          one configuration, so the band is run-to-run noise. The faint
                          series is the earlier curve at lr 2e-5 / KL 10 / temperature
                          0.07 under the broken sampler, which is what "contrastive
                          saturates at 100m" was read from.

contrastive_ablation.png  Pretrained against randomly initialised, 4 scales x 6 seeds,
                          grids identical but for --random_init.

extra/contrastive_breadth.png
                          Separation ratio against rows per contrastive step. Filed
                          under extra/ because it is a negative result that closes a
                          question rather than opening one: more negatives measurably
                          HURT here, -2.80 from 4 rows to 64 at t=-4.5, so breadth is
                          not a lever this project will pull. Worth keeping because it
                          contradicts standard contrastive practice and because the
                          earlier verdict on the same question was confounded four
                          ways.
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
FIGS = REPO / "results" / "processed" / "figures" / "C_contrastive" / "superseded"
# Supporting figures: results that closed a question without changing what
# we do. GradCache is here because more negatives measurably hurt, so the
# lever is not one we will pull.
EXTRA = FIGS   # superseded figures are kept flat, no extra/ subfolder
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


def by_scale(job):
    out = {}
    for arm, v in ratios(job).items():
        if arm[:1] == "s" and "_seed" in arm:
            out.setdefault(arm[:5], []).append(v)
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
    """Pretrained against randomly initialised, all four scales, 6 seeds each.

    Both grids are identical apart from --random_init true, so they subtract. The
    earlier version of this figure used the 50m-only control at the old configuration
    and had no error bars; jobs 8848049 and 8848471 replace it with a matched pair.
    """
    pre, rnd = by_scale("8848049"), by_scale("8848471")
    if not pre or not rnd:
        print("  ablation: missing data"); return
    scales = ["s050m", "s100m", "s200m", "s400m"]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for series, colour, lab in ((pre, BLUE, "pretrained"), (rnd, RED, "random init")):
        m, s, xi = [], [], []
        for i, k in enumerate(scales):
            v = series.get(k, [])
            if len(v) > 1:
                m.append(np.mean(v)); s.append(np.std(v, ddof=1)); xi.append(i)
        m, s, xi = np.array(m), np.array(s), np.array(xi)
        ax.fill_between(xi, m - s, m + s, color=colour, alpha=0.16, lw=0)
        ax.errorbar(xi, m, yerr=s, fmt="o-", ms=6, lw=2.2, color=colour, ecolor=colour,
                    capsize=4, label=lab)
        for a_, b_, e_ in zip(xi, m, s):
            ax.annotate(f"{b_:.2f}", (a_, b_ + e_), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8.5, color=INK)
    ax.axhline(FLOOR, color=MUTED, lw=1.1, ls=":")
    ax.annotate(f"untrained {FLOOR}", (3.4, FLOOR), fontsize=8, color=MUTED,
                ha="right", va="bottom")
    ax.set_xticks(range(4)); ax.set_xticklabels(["50m", "100m", "200m", "400m"])
    ax.set_xlim(-0.35, 3.45)
    ax.set_xlabel("model size"); ax.set_ylabel("separation ratio")
    ax.set_title("pretrained against random init", loc="left", pad=14)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.4, loc="upper left")
    fig.suptitle("Jobs 8848049 and 8848471 — identical but for --random_init. "
                 "6 seeds per point, band ±1 sd.",
                 x=0.005, ha="left", fontsize=8.5, color=MUTED, y=1.02)
    fig.savefig(FIGS / "contrastive_ablation.png"); plt.close(fig)


def fig_breadth():
    """Separation ratio against how many rows a contrastive step sees.

    The standard lever in contrastive learning is more negatives. Here it costs.
    """
    gc = ratios("8848463")
    if not gc:
        print("  breadth: no data"); return
    shapes = [("p02k02", 4, "P=2 K=2"), ("p08k02", 16, "P=8 K=2"),
              ("p16k04", 64, "P=16 K=4")]
    xs, m, s, labs = [], [], [], []
    for key, rows, lab in shapes:
        v = [x for a, x in gc.items() if a.startswith(key)]
        if len(v) > 1:
            xs.append(rows); m.append(np.mean(v)); s.append(np.std(v, ddof=1))
            labs.append(f"{lab}\n{rows} rows")
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    x = np.arange(len(xs))
    ax.errorbar(x, m, yerr=s, fmt="o-", ms=7, lw=2.2, color=BLUE, ecolor=BLUE,
                capsize=4)
    for xi, b_, e_ in zip(x, m, s):
        ax.annotate(f"{b_:.2f}", (xi, b_ + e_), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_xlim(-0.35, len(xs) - 0.65)
    ax.set_xlabel("rows per contrastive step")
    ax.set_ylabel("separation ratio")
    ax.set_title("more negatives, via GradCache", loc="left", pad=14)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.suptitle("Job 8848463, 50m, 4 seeds per point, band ±1 sd. Only the batch "
                 "shape varies.", x=0.005, ha="left", fontsize=8.5, color=MUTED, y=1.02)
    fig.savefig(EXTRA / "contrastive_breadth.png"); plt.close(fig)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    EXTRA.mkdir(parents=True, exist_ok=True)
    fig_scaling(); print("  wrote contrastive_scaling.png")
    fig_ablation(); print("  wrote contrastive_ablation.png")
    fig_breadth(); print("  wrote contrastive_breadth.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
