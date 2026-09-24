"""Contrastive figures on the ms-contrastive-100k test split (Stage 2 / 3 of PLAN.md).

    python sweeps/plot_contrastive_100k.py

Reads the per-model JSONs that msdelta.eval_grouped_retrieval wrote under
results/finetune/contrastive/grouped100k-test/ and draws, for experimental-spectrum
MAP@R (the headline) and Hit@1:

  c100k_scale.png        C2: model size at the frozen recipe (220k), replicate corpus
  c100k_checkpoint.png   C4 + C2: pretraining grad steps, one line per scale
  c100k_c7.png           C7: continuing on ms-contrastive-100k, x = steps into its epoch
  c100k_transfer.png     small replicate eval vs this test, every Stage-1 model

Every panel carries the untrained-encoder and binned-cosine references as horizontal
lines. Points are 3-seed means with seed sd as error bars, labelled with the mean.
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
RESULTS = REPO / "results" / "finetune" / "contrastive" / "grouped100k-test"
FIGS = REPO / "results" / "finetune" / "contrastive" / "figures"
RUNS = "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs"
INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"
SCALE_COLOUR = {"50m": "#93c5fd", "100m": "#60a5fa", "200m": "#2563eb", "400m": "#1e3a8a"}
ORDER = ["50m", "100m", "200m", "400m"]
PARAMS = {"50m": 50, "100m": 100, "200m": 200, "400m": 400}
METRICS = (("experimental/MAP@R", "MAP@R"), ("experimental/Hit@1", "Hit@1"))
BASELINE_STYLE = {"binned_w0.1": ("binned cosine 0.1 Da", "#b45309", "--"),
                  "binned_w1.0005": ("binned cosine 1 Da", "#d97706", ":")}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.7, "figure.facecolor": "white",
})


def load() -> dict[str, dict]:
    out = {}
    for f in glob.glob(str(RESULTS / "*.json")):
        d = json.load(open(f))
        out[d["name"]] = d
    return out


def agg(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def cells(res, pattern):
    """regex with named groups -> {groupdict tuple: {metric: [values]}}"""
    c = collections.defaultdict(lambda: collections.defaultdict(list))
    for name, d in res.items():
        m = re.fullmatch(pattern, name)
        if not m:
            continue
        for key, _ in METRICS:
            if key in d["metrics"]:
                c[tuple(m.groupdict().values())][key].append(d["metrics"][key])
    return c


def label(ax, xs, ys, below=False):
    """Mean above the point; below it for the lower of two near-overlapping lines."""
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, -13 if below else 7), ha="center", fontsize=7.2,
                    color=MUTED)


def references(ax, res, key):
    for name, (text, colour, style) in BASELINE_STYLE.items():
        if name in res:
            ax.axhline(res[name]["metrics"][key], color=colour, ls=style, lw=1.2,
                       label=text)
    base = [d["metrics"][key] for n, d in res.items()
            if n.startswith("base_") and key in d["metrics"]]
    if base:
        ax.axhspan(min(base), max(base), color="#9ca3af", alpha=0.18,
                   label="untrained encoders")


def finish(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", pad=12)
    ax.grid(); ax.set_axisbelow(True); ax.margins(x=0.15, y=0.12)
    ax.legend(frameon=False, fontsize=7.8)


def fig_scale(res):
    c = cells(res, r"c2_s0*(?P<scale>\d+m)_ck220k_seed\d")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, name) in zip(axes, METRICS):
        pts = [(s, agg(c[(s,)][key])) for s in ORDER if (s,) in c]
        x = [PARAMS[s] for s, _ in pts]; y = [p[0] for _, p in pts]
        ax.errorbar(x, y, yerr=[p[1] for _, p in pts], color=MUTED, lw=1.4, capsize=3,
                    zorder=2)
        for (s, _), xi, yi in zip(pts, x, y):
            ax.scatter([xi], [yi], s=55, color=SCALE_COLOUR[s], zorder=3, label=s)
        label(ax, x, y)
        references(ax, res, key)
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([s for s, _ in pts])
        finish(ax, "model size", name, f"{name}, checkpoint 220k")
    fig.tight_layout(); out = FIGS / "c100k_scale.png"; fig.savefig(out); plt.close(fig)
    return out


def fig_checkpoint(res):
    c = cells(res, r"c[24]_s0*(?P<scale>\d+m)_ck(?P<ck>\d+)k_seed\d")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, name) in zip(axes, METRICS):
        for s in ORDER:
            pts = sorted((int(ck) * 1000, agg(v[key])) for (sc, ck), v in c.items()
                         if sc == s and key in v)
            if len(pts) < 2:
                continue
            x = [p[0] for p in pts]; y = [p[1][0] for p in pts]
            ax.errorbar(x, y, yerr=[p[1][1] for p in pts], marker="o", ms=5, lw=1.6,
                        capsize=3, color=SCALE_COLOUR[s], label=s)
            label(ax, x, y, below=(s == "50m"))
        references(ax, res, key)
        finish(ax, "pretraining grad steps", name, name)
    fig.tight_layout(); out = FIGS / "c100k_checkpoint.png"; fig.savefig(out); plt.close(fig)
    return out


def fig_c7(res):
    c = cells(res, r"c7b_cont0*(?P<scale>\d+m)_(?P<step>s\d+|final)_seed\d")
    start = {"50m": cells(res, r"s050m_t0002_pk256_ep24_seed\d"),
             "400m": cells(res, r"s400m_t0002_pk256_ep12_seed\d")}
    total = 1062
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, name) in zip(axes, METRICS):
        for s in ("50m", "400m"):
            pts = [(0, agg(v[key])) for v in start[s].values() if key in v]
            pts += sorted((total if st == "final" else int(st[1:]), agg(v[key]))
                          for (sc, st), v in c.items() if sc == s and key in v)
            if len(pts) < 2:
                continue
            x = [p[0] for p in pts]; y = [p[1][0] for p in pts]
            ax.errorbar(x, y, yerr=[p[1][1] for p in pts], marker="o", ms=5, lw=1.6,
                        capsize=3, color=SCALE_COLOUR[s], label=s)
            label(ax, x, y, below=(s == "50m"))
        references(ax, res, key)
        finish(ax, "steps on ms-contrastive-100k (1,062 = one epoch)", name, name)
    fig.tight_layout(); out = FIGS / "c100k_c7.png"; fig.savefig(out); plt.close(fig)
    return out


def fig_transfer(res):
    small, big, colour = [], [], []
    for name, d in res.items():
        path = d.get("path", "")
        if not path.endswith("/final") or "/runs/sweep-" not in path:
            continue
        p = path[: -len("/final")] + "/all_results.json"
        if not os.path.exists(p):
            continue
        a = json.load(open(p))
        if "retrieval/MAP@R" not in a or "c7b_" in name or "c9" in name:
            continue
        m = re.search(r"s0*(\d+m)", name)
        small.append(a["retrieval/MAP@R"]); big.append(d["metrics"]["experimental/MAP@R"])
        colour.append(SCALE_COLOUR.get(m.group(1) if m else "", MUTED))
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.scatter(small, big, s=22, c=colour, alpha=0.85)
    for s in ORDER:
        ax.scatter([], [], s=22, color=SCALE_COLOUR[s], label=s)
    if len(small) > 2:
        from scipy.stats import spearmanr
        rho = spearmanr(small, big)[0]
        ax.set_title(f"n={len(small)}, Spearman {rho:.2f}", loc="left", pad=12)
    finish(ax, "MAP@R, replicate-corpus eval (99 groups)",
           "exp MAP@R, ms-contrastive-100k test", ax.get_title(loc="left"))
    fig.tight_layout(); out = FIGS / "c100k_transfer.png"; fig.savefig(out); plt.close(fig)
    return out


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    res = load()
    if not res:
        raise SystemExit(f"no results under {RESULTS}")
    print(f"  {len(res)} scored models")
    for f in (fig_scale, fig_checkpoint, fig_c7, fig_transfer):
        print(f"  wrote {f(res).relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
