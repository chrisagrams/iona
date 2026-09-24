"""Denoise across pretraining checkpoints: per scale, per checkpoint, and combined.

    python sweeps/plot_ladder.py

Three figures, each for AUROC and F1:

  ladder_by_scale.png       one panel per scale, x = pretraining grad steps.
                            Does a bigger model keep improving as it pretrains?
  ladder_by_checkpoint.png  one panel per checkpoint, x = model size.
                            Does the scale curve change shape as pretraining runs?
  ladder_compute.png        everything on one axis, x = pretraining grad steps,
                            colour = scale. The combined view.

X IS PRETRAINING GRAD STEPS, NOT COMPUTE. A step at 400m costs ~8x a step at 50m, so
this axis understates the large models. Real budgets get substituted later; until then
no cross-scale point on the x axis is a fair comparison, only the shape within a colour.
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
FIGS = REPO / "results" / "finetune" / "denoise" / "figures"
INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"
SCALE_COLOUR = {"50m": "#93c5fd", "100m": "#60a5fa", "200m": "#2563eb", "400m": "#1e3a8a"}
ORDER = ["50m", "100m", "200m", "400m"]
# The point every denoise HP grid and the checkpoint probe independently chose. Matched
# on FIELDS, not a regex over the whole file: a substring match on "--learning_rate 2e-4"
# alone still admits ep2, ep8 and narrow heads, which read 0.88 where this configuration
# scores 0.93 -- a statement about the grid's other arms, not about the checkpoint.
CONFIG = {"--learning_rate": "2e-4", "--encoder_lr_scale": "0.5",
          "--num_train_epochs": "4", "--head_hidden_size": "512",
          "--per_device_train_batch_size": "1"}


# Jobs whose test scores are not comparable. 8840345 was the first 216-arm grid, scored
# on a reduced test set (1,440 spectra, not ~8,600) -- results/finetune/README.md marks
# it "not comparable". One of its arms (0.8597) sat in the 50m@133k cell and dragged
# that point from 0.932 to 0.927.
EXCLUDE_JOBS = {"8840345"}


def matches_config(text: str) -> bool:
    fields = dict(zip(text.split()[::2], text.split()[1::2]))
    return all(fields.get(k) == v for k, v in CONFIG.items())

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.7, "figure.facecolor": "white",
})


# Test-split metrics plotted. Loss is the per-peak BCE on the test split; lower is better.
METRICS = (("test_auroc", "AUROC"), ("test_f1", "F1"), ("test_auprc", "AUPRC"),
           ("test_loss", "Test loss"))

def collect():
    """(scale, checkpoint) -> {metric: [values]}, from every denoise arm on disk."""
    meta = {}
    for g in glob.glob(str(REPO / "configs" / "sweep-*")):
        for arm in os.listdir(g):
            f = os.path.join(g, arm, "training.args")
            if not os.path.exists(f):
                continue
            t = open(f).read()
            m = re.search(r"msdelta-(\d+m)-production-\d+-checkpoint-(\d+)", t)
            # random-init arms have no pretraining budget and belong to the ablation.
            if not m or "--random_init" in t:
                continue
            # ONLY the ladder's own configuration. The 50m cell holds a 216-arm HP grid
            # including frozen encoders and lr 1e-6; averaging those with the tuned arms
            # reads 0.75 where the configuration actually scores 0.93, which is a
            # statement about the grid's worst arms, not about the checkpoint.
            if not matches_config(t):
                continue
            meta.setdefault(arm, (m.group(1), int(m.group(2))))
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in glob.glob(f"{RUNS}/sweep-*"):
        m = re.match(r"sweep-(.+)-(\d+)$", os.path.basename(d))
        if not m or m.group(1) not in meta or m.group(2) in EXCLUDE_JOBS:
            continue
        p = os.path.join(d, "all_results.json")
        if not os.path.exists(p):
            continue
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if "test_auroc" not in r:
            continue
        for k, _ in METRICS:
            if k in r:
                out[meta[m.group(1)]][k].append(r[k])
    return out


def agg(vals):
    a = np.array(vals, dtype=float)
    return a.mean(), (a.std(ddof=1) / len(a) ** 0.5 if len(a) > 1 else 0.0), len(a)


def label_points(ax, xs, ys):
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.2, color=MUTED)


def by_scale(cov, metric, name):
    scales = [s for s in ORDER if any(k[0] == s for k in cov)]
    fig, axes = plt.subplots(1, len(scales), figsize=(4.3 * len(scales), 3.9),
                             squeeze=False)
    for ax, s in zip(axes[0], scales):
        pts = sorted((c, agg(cov[(s, c)][metric])) for c in
                     [k[1] for k in cov if k[0] == s] if metric in cov[(s, c)])
        if not pts:
            continue
        x = [p[0] for p in pts]; y = [p[1][0] for p in pts]; e = [p[1][1] for p in pts]
        ax.errorbar(x, y, yerr=e, marker="o", ms=5, lw=1.6, capsize=3,
                    color=SCALE_COLOUR[s])
        label_points(ax, x, y)
        ax.set_title(f"{s}   ({len(pts)} checkpoint{'s' if len(pts) > 1 else ''})",
                     loc="left", pad=12)
        ax.set_xlabel("pretraining grad steps"); ax.set_ylabel(name)
        ax.grid(); ax.set_axisbelow(True); ax.margins(x=0.18, y=0.22)
    fig.tight_layout()
    out = FIGS / f"ladder_by_scale_{metric.replace('test_', '')}.png"
    fig.savefig(out); plt.close(fig); return out


def by_checkpoint(cov, metric, name):
    cks = sorted({k[1] for k in cov})
    cks = [c for c in cks if sum(1 for k in cov if k[1] == c and metric in cov[k]) > 0]
    fig, axes = plt.subplots(1, len(cks), figsize=(3.6 * len(cks), 3.9), squeeze=False)
    for ax, c in zip(axes[0], cks):
        pts = [(s, agg(cov[(s, c)][metric])) for s in ORDER
               if (s, c) in cov and metric in cov[(s, c)]]
        if not pts:
            continue
        xi = list(range(len(pts)))
        y = [p[1][0] for p in pts]; e = [p[1][1] for p in pts]
        ax.errorbar(xi, y, yerr=e, marker="o", ms=5, lw=1.6, capsize=3, color=MUTED,
                    zorder=2)
        for i, (s, _) in enumerate(pts):
            ax.scatter([i], [y[i]], s=55, color=SCALE_COLOUR[s], zorder=3)
        label_points(ax, xi, y)
        ax.set_xticks(xi); ax.set_xticklabels([p[0] for p in pts])
        ax.set_title(f"step {c:,}", loc="left", pad=12)
        ax.set_xlabel("model size"); ax.set_ylabel(name)
        ax.grid(); ax.set_axisbelow(True); ax.margins(x=0.22, y=0.22)
    fig.tight_layout()
    out = FIGS / f"ladder_by_checkpoint_{metric.replace('test_', '')}.png"
    fig.savefig(out); plt.close(fig); return out


def combined(cov, metric, name, ax):
    for s in ORDER:
        pts = sorted((c, agg(cov[(s, c)][metric])) for c in
                     [k[1] for k in cov if k[0] == s] if metric in cov[(s, c)])
        if not pts:
            continue
        x = [p[0] for p in pts]; y = [p[1][0] for p in pts]; e = [p[1][1] for p in pts]
        ax.errorbar(x, y, yerr=e, marker="o", ms=5, lw=1.7, capsize=3,
                    color=SCALE_COLOUR[s], label=s)
        label_points(ax, x, y)
    ax.set_xlabel("pretraining grad steps"); ax.set_ylabel(name)
    ax.set_title(name, loc="left", pad=12)
    ax.grid(); ax.set_axisbelow(True); ax.margins(x=0.16, y=0.22)
    ax.legend(frameon=False, fontsize=8.5, title="model size", title_fontsize=8.5)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    cov = collect()
    if not cov:
        raise SystemExit("no denoise arms found")
    print(f"  cells: {len(cov)}")
    for (s, c) in sorted(cov, key=lambda k: (ORDER.index(k[0]), k[1])):
        n = len(cov[(s, c)].get("test_auroc", []))
        print(f"    {s:5s} step {c:<7d} n={n}")
    written = []
    for metric, name in METRICS:
        written += [by_scale(cov, metric, name), by_checkpoint(cov, metric, name)]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))
    for ax, (metric, name) in zip(axes.flat, METRICS):
        combined(cov, metric, name, ax)
    fig.tight_layout()
    out = FIGS / "ladder_compute.png"
    fig.savefig(out); plt.close(fig); written.append(out)
    for w in written:
        print(f"  wrote {w.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
