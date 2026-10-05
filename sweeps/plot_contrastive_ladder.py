"""Contrastive retrieval across pretraining checkpoint and scale (PLAN.md C2, C4).

    python sweeps/plot_contrastive_ladder.py        # on a compute node

  contrastive_ladder.png
    left   MAP@100 against pretraining grad steps, one line per scale, all at ONE recipe:
           lr 1e-4, KL 10, t 0.07, P*K 4 -- the recipe that ran at every checkpoint.
           Superseded since, so read it for shape, not level.
    right  the current best recipe (t 0.003-0.005, P*K 64) at checkpoint 220k, the only
           checkpoint it has been run at.

MAP@100 because the older runs predate MAP@R; every run here reports MAP@100.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homes  # noqa: E402  (data homes, configs/homes.env)

REPO = Path(__file__).resolve().parent.parent
RUNS = "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs"
FIGS = homes.RESULTS / "contrastive" / "superseded"
METRIC = "retrieval/MAP@100"
INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"
ORDER = ("50m", "100m", "200m", "400m")
COLOUR = {"50m": "#93c5fd", "100m": "#60a5fa", "200m": "#2563eb", "400m": "#1e3a8a"}
# The checkpoint each scale's first runs used, before the canonical ladder existed.
ORIGINAL = {"50m": 133233, "100m": 138073, "200m": 192799, "400m": 181381}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.7, "figure.facecolor": "white",
})


def read(job):
    for d in glob.glob(f"{RUNS}/sweep-*-{job}"):
        p = os.path.join(d, "all_results.json")
        if not os.path.exists(p):
            continue
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if METRIC in r:
            yield re.sub(rf"^sweep-|-{job}$", "", os.path.basename(d)), r[METRIC]


def old_recipe():
    """(scale, checkpoint) -> values, all at lr1e-4 / KL10 / t0.07 / P*K 4."""
    cells = collections.defaultdict(list)
    for job in ("8853557", "8854412"):          # original checkpoints (+ repaired arms)
        for arm, v in read(job):
            m = re.match(r"s0*(\d+m)_seed\d+$", arm)
            if m:
                cells[(m.group(1), ORIGINAL[m.group(1)])].append(v)
    for arm, v in read("8851663"):              # canonical 220k / 330k ladder
        m = re.match(r"(\d+m)_ck(\d+)k_seed\d+$", arm)
        if m:
            cells[(m.group(1), int(m.group(2)) * 1000)].append(v)
    for arm, v in read("8849088"):              # 50m @ 540423, same config in the HP probe
        if re.match(r"lr1e4_kl10_t007_seed\d+$", arm):
            cells[("50m", 540423)].append(v)
    return cells


def new_recipe():
    """Best (t, P*K=64) cell per scale at 220k, from the recipe sweeps."""
    cells = collections.defaultdict(list)
    for job in ("8856643", "8856642", "8856460"):
        for arm, v in read(job):
            m = re.match(r"s0*(\d+m)_t(\d+)_pk0*64_seed\d+$", arm)
            if m:
                cells[(m.group(1), m.group(2))].append(v)
    best = {}
    for (s, t), v in cells.items():
        if len(v) >= 2 and (s not in best or np.mean(v) > np.mean(best[s][1])):
            best[s] = (int(t) / 10 ** (len(t) - 1), v)
    return best


def stat(v):
    a = np.array(v)
    return a.mean(), (a.std(ddof=1) / len(a) ** 0.5 if len(a) > 1 else 0.0)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    old, new = old_recipe(), new_recipe()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                                  gridspec_kw={"width_ratios": [2.2, 1]})
    for s in ORDER:
        pts = sorted((c, stat(v)) for (ss, c), v in old.items() if ss == s)
        if not pts:
            continue
        x = [p[0] for p in pts]; y = [p[1][0] for p in pts]; e = [p[1][1] for p in pts]
        ax.errorbar(x, y, yerr=e, marker="o", ms=5, lw=1.7, capsize=3, color=COLOUR[s],
                    label=s)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color=MUTED)
        print(f"  old recipe {s}: " + ", ".join(f"{c}:{m:.4f}(n={len(old[(s, c)])})"
                                                for c, (m, _) in pts))
    ax.set_xlabel("pretraining grad steps"); ax.set_ylabel("MAP@100")
    ax.set_title("t 0.07, P×K 4", loc="left")
    ax.grid(); ax.set_axisbelow(True); ax.margins(y=0.2)
    ax.legend(frameon=False, fontsize=8.5, title="model size", title_fontsize=8.5)

    for i, s in enumerate(ORDER):
        if s not in new:
            continue
        t, v = new[s]; m, e = stat(v)
        ax2.errorbar([i], [m], yerr=[e], marker="o", ms=7, capsize=3, color=COLOUR[s])
        ax2.annotate(f"{m:.3f}", (i, m), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)
        ax2.annotate(f"t {t:g}", (i, m), textcoords="offset points", xytext=(0, -15),
                     ha="center", fontsize=7, color=MUTED)
        print(f"  new recipe {s}: t{t:g} MAP@100 {m:.4f} (n={len(v)})")
    ax2.set_xticks(range(len(ORDER))); ax2.set_xticklabels(ORDER)
    ax2.set_xlabel("model size"); ax2.set_ylabel("MAP@100")
    ax2.set_title("best recipe, P×K 64, 220k", loc="left")
    ax2.grid(axis="y"); ax2.set_axisbelow(True); ax2.margins(x=0.2, y=0.25)
    fig.tight_layout()
    out = FIGS / "contrastive_ladder.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
