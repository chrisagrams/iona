"""Contrastive recipe (PLAN.md C1) and scale at the best recipe (C2). Run on a compute node.

    python sweeps/plot_c1.py

Combines sweep-conbig (8856460: t 0.01-0.03, width 4-64) and sweep-conneg (8856643 and
8856642: t 0.003-0.01, width 64-512), all at checkpoint-220000, scored on MAP@R.

  c1_width_temperature.png  MAP@R against batch width (P*K, log scale), one line per
                            temperature, one panel per scale. Error bars are seed sem.
  c1_scale.png              each scale's best cell, with its seed sem.

Step-matched controls (arms with an _epNN tag) are drawn as open markers on the 50m
panel, at their width.
"""

from __future__ import annotations

import collections
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
RUNS = "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs"
FIGS = REPO / "results" / "finetune" / "contrastive" / "figures"
JOBS = ("8856460", "8856643", "8856642")
METRIC = "retrieval/MAP@R"
INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"
SCALES = ("050m", "100m", "200m", "400m")
SCALE_COLOUR = {"050m": "#93c5fd", "100m": "#60a5fa", "200m": "#2563eb", "400m": "#1e3a8a"}
TEMP_COLOUR = {0.003: "#7c2d12", 0.005: "#c2410c", 0.01: "#f97316",
               0.02: "#fdba74", 0.03: "#fed7aa"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.7, "figure.facecolor": "white",
})

ARM = re.compile(r"^s(\d{3}m)_t(\d+)_pk(\d+)(?:_ep(\d+))?_seed\d+$")


def temperature(digits: str) -> float:
    # arm tags drop the decimal point: 001 -> 0.01, 002 -> 0.02, 0003 -> 0.003
    return int(digits) / 10 ** (len(digits) - 1)


def collect():
    main = collections.defaultdict(list)       # (scale, t, width) -> values
    ctrl = collections.defaultdict(list)       # (scale, t, width, epochs) -> values
    for job in JOBS:
        for d in glob.glob(f"{RUNS}/sweep-*-{job}"):
            p = os.path.join(d, "all_results.json")
            if not os.path.exists(p):
                continue
            try:
                r = json.load(open(p))
            except Exception:
                continue
            if METRIC not in r:
                continue
            m = ARM.match(re.sub(rf"^sweep-|-{job}$", "", os.path.basename(d)))
            if not m:
                continue
            scale, t, width, ep = m.group(1), temperature(m.group(2)), int(m.group(3)), m.group(4)
            if ep:
                ctrl[(scale, t, width, int(ep))].append(r[METRIC])
            else:
                main[(scale, t, width)].append(r[METRIC])
    return main, ctrl


def stat(v):
    a = np.array(v)
    return a.mean(), (a.std(ddof=1) / len(a) ** 0.5 if len(a) > 1 else 0.0)


def width_temperature(main, ctrl):
    fig, axes = plt.subplots(1, len(SCALES), figsize=(4.2 * len(SCALES), 3.9), sharey=True)
    for ax, s in zip(axes, SCALES):
        temps = sorted({t for (ss, t, _) in main if ss == s})
        for t in temps:
            pts = sorted((w, stat(v)) for (ss, tt, w), v in main.items() if ss == s and tt == t)
            if not pts:
                continue
            x = [p[0] for p in pts]; y = [p[1][0] for p in pts]; e = [p[1][1] for p in pts]
            ax.errorbar(x, y, yerr=e, marker="o", ms=4, lw=1.5, capsize=2,
                        color=TEMP_COLOUR.get(t, MUTED), label=f"t {t:g}")
        for (ss, t, w, ep), v in ctrl.items():
            if ss != s:
                continue
            m, e = stat(v)
            ax.errorbar([w], [m], yerr=[e], marker="o", ms=6, mfc="white", lw=0,
                        elinewidth=1, capsize=2, color=TEMP_COLOUR.get(t, MUTED))
            ax.annotate(f"{ep} ep", (w, m), textcoords="offset points", xytext=(6, -3),
                        fontsize=7.5, color=MUTED)
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 16, 64, 128, 256, 512])
        ax.set_xticklabels(["4", "16", "64", "128", "256", "512"])
        ax.set_title(s.lstrip("0"), loc="left")
        ax.set_xlabel("batch width P×K")
        ax.grid(); ax.set_axisbelow(True)
    axes[0].set_ylabel("MAP@R")
    axes[-1].legend(frameon=False, fontsize=8, title="temperature", title_fontsize=8)
    fig.tight_layout()
    out = FIGS / "c1_width_temperature.png"
    fig.savefig(out); plt.close(fig)
    return out


def scale(main):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    xs, ys, es, labels = [], [], [], []
    for i, s in enumerate(SCALES):
        cells = [(stat(v), t, w) for (ss, t, w), v in main.items() if ss == s and len(v) >= 2]
        if not cells:
            continue
        (m, e), t, w = max(cells, key=lambda c: c[0][0])
        xs.append(i); ys.append(m); es.append(e); labels.append(f"t {t:g}, width {w}")
        ax.errorbar([i], [m], yerr=[e], marker="o", ms=7, capsize=3, color=SCALE_COLOUR[s])
        ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
        ax.annotate(labels[-1], (i, m), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=7, color=MUTED)
    ax.plot(xs, ys, color=MUTED, lw=1, zorder=0)
    ax.set_xticks(range(len(SCALES)))
    ax.set_xticklabels([s.lstrip("0") for s in SCALES])
    ax.set_xlabel("model size"); ax.set_ylabel("best MAP@R")
    ax.set_title("best recipe per scale, checkpoint 220k", loc="left")
    ax.margins(y=0.25); ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.tight_layout()
    out = FIGS / "c1_scale.png"
    fig.savefig(out); plt.close(fig)
    return out


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    main_, ctrl = collect()
    print(f"  cells: {len(main_)} main, {len(ctrl)} control")
    for out in (width_temperature(main_, ctrl), scale(main_)):
        print(f"  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
