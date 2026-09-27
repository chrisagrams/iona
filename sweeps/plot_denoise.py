"""Figures for the denoise results, written to results/processed/figures/D_denoise/superseded/.

    python sweeps/plot_denoise.py --runs /lus/flare/projects/UIC-HPC/$USER/msdelta/runs

Four figures, each making one point that a table makes badly:

  scaling              denoise against model size, AUROC and F1
  pretrain_ablation    pretrained against a random encoder, AUROC and F1

and under figures/extra/, the three that explain how those were reached:

  top_cluster          most of the 216-arm ranking sits inside the noise band
  encoder_lr_scale     encoder_lr_scale interacts with batch size, at every scale
  probe_heatmap        encoder_lr_scale x epochs at 400m

Everything is drawn from the same run directories as summarise_denoise.py.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "results" / "processed" / "figures" / "D_denoise" / "superseded"
# Supporting figures: the three that explain HOW the headline numbers were
# reached rather than what they are.
EXTRA = FIGS   # superseded figures are kept flat, no extra/ subfolder

INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"
BLUE, RED, AMBER, GREEN = "#2563eb", "#dc2626", "#d97706", "#059669"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.7, "figure.facecolor": "white",
})


def load(runs: Path, job: str) -> dict[str, dict]:
    out = {}
    for d in sorted(runs.glob(f"sweep-*-{job}")):
        f = d / "test_results.json"
        if f.exists():
            try:
                out[re.sub(rf"^sweep-|-{job}$", "", d.name)] = json.loads(f.read_text())
            except Exception:
                pass
    return out


def annotate(ax, text, xy, xytext, color=INK):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=8, color=color,
                ha="center", arrowprops=dict(arrowstyle="-", lw=0.7, color=color,
                                             shrinkA=0, shrinkB=3))


def fig_scaling(runs: Path) -> None:
    """The denoise scale curve on both metrics, FT5: 4 scales x 6 seeds (job 8845262).

    One figure rather than two: AUROC and F1 answer the same question and agree on
    every step, so putting them side by side is the comparison. The per-step table that
    used to occupy a third panel is in results/raw/finetune/denoise/denoise_scale_seeds.txt -- a bar chart of
    three differences restated what the curve already shows.
    """
    recs = load(runs, "8845262")
    scales = ["50m", "100m", "200m", "400m"]
    x = np.arange(len(scales))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, metric, label, colour in ((axes[0], "test_auroc", "AUROC", BLUE),
                                      (axes[1], "test_f1", "F1", "#7c3aed")):
        mean, sd = [], []
        for s in scales:
            v = [r[metric] for a, r in recs.items() if a.startswith(s + "_")]
            mean.append(np.mean(v)); sd.append(np.std(v, ddof=1))
        mean, sd = np.array(mean), np.array(sd)

        ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.16, lw=0)
        ax.plot(x[:3], mean[:3], "-", color=colour, lw=2.2, zorder=3)
        ax.plot(x[2:], mean[2:], "-", color=RED, lw=2.2, zorder=3)
        ax.errorbar(x, mean, yerr=sd, fmt="o", ms=6, color=colour, ecolor=colour,
                    capsize=4, lw=1.4, zorder=4)
        ax.plot(x[3], mean[3], "o", ms=6, color=RED, zorder=5)
        span = (mean + sd).max() - (mean - sd).min()
        for xi, m, e in zip(x, mean, sd):
            ax.annotate(f"{m:.4f}", (xi, m + e), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=INK)
        ax.set_ylim((mean - sd).min() - 0.12 * span, (mean + sd).max() + 0.24 * span)
        ax.set_xticks(x); ax.set_xticklabels(scales)
        ax.set_xlim(-0.35, 3.45)
        ax.set_xlabel("model size"); ax.set_ylabel(f"test {label}")
        ax.set_title(f"test {label}", loc="left", pad=14)
        ax.grid(axis="y"); ax.set_axisbelow(True)

    fig.suptitle("FT5, job 8845262. 6 seeds per scale, band is ±1 sd.",
                 x=0.005, ha="left", fontsize=8.5, color=MUTED, y=1.02)
    fig.savefig(FIGS / "scaling.png")
    plt.close(fig)


def fig_top_cluster(runs: Path) -> None:
    """Two things a table hides: the top cluster is noise, and the floor is lr, not freezing."""
    recs = load(runs, "8840408")
    items = sorted(recs.items(), key=lambda kv: -kv[1]["test_auroc"])
    v = np.array([r["test_auroc"] for _, r in items])
    names = [a for a, _ in items]
    frozen = np.array([bool(re.search(r"_es0(?![0-9])", a)) for a in names])
    lowlr = np.array([a.startswith("lr1e6") for a in names])
    rank = np.arange(1, len(v) + 1)
    best, noise = v[0], 0.0005
    inside = int((v >= best - 2 * noise).sum())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                                  gridspec_kw={"width_ratios": [1.5, 1]})

    ax.plot(rank, v, "-", color=GRID, lw=3, zorder=1)
    ax.scatter(rank[~frozen & ~lowlr], v[~frozen & ~lowlr], s=11, color=BLUE,
               zorder=3, label="encoder trains, lr ≥ 1e-5")
    ax.scatter(rank[frozen], v[frozen], s=13, color=RED, zorder=4,
               label="encoder_lr_scale = 0 (frozen)")
    ax.scatter(rank[lowlr & ~frozen], v[lowlr & ~frozen], s=13, color=AMBER, zorder=4,
               label="lr 1e-6")
    ax.set_xlabel("arm, ranked by test AUROC"); ax.set_ylabel("test AUROC")
    ax.set_title("216 arms ranked by test AUROC", loc="left", pad=16)
    ax.grid(axis="y"); ax.set_axisbelow(True); ax.set_xlim(0, len(v) + 2)
    ax.legend(frameon=False, fontsize=8, loc="lower left", handletextpad=0.3)

    k = 26
    ax2.axhspan(best - 2 * noise, best, color=AMBER, alpha=0.20, lw=0)
    ax2.plot(rank[:k], v[:k], "o-", color=BLUE, lw=1.5, ms=4.5)
    ax2.axhline(best, color=RED, lw=0.9, ls="--")
    ax2.set_xlabel("rank"); ax2.set_ylabel("test AUROC")
    ax2.set_title(f"Top {k}", loc="left", pad=16)
    ax2.grid(axis="y"); ax2.set_axisbelow(True)
    ax2.annotate(f"best {best:.4f}", xy=(1, best), xytext=(7, best + 0.0006),
                 fontsize=8.2, color=RED,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=RED))

    fig.suptitle("50m hyperparameter grid (job 8840408). Band is ±2 sd of the 0.0005 "
                 "seed noise measured in FT5.",
                 x=0.005, ha="left", fontsize=8.5, color=MUTED, y=1.02)
    fig.savefig(EXTRA / "top_cluster.png")
    plt.close(fig)


def fig_encoder_lr(runs: Path) -> None:
    """How hard to push the pretrained encoder, and why the answer depends on batch.

    encoder_lr_scale multiplies the encoder's learning rate relative to the head's: 1.0
    trains it as fast as the freshly initialised head, 0 freezes it.

    The right two panels plot the ACTUAL scores rather than the 0.5-minus-1.0 difference
    an earlier version showed. A difference makes the reader do arithmetic to find out
    which setting won, and it hid that the two grids use different large batches -- 144
    at 50m, 48 elsewhere -- so a shared "batch 48" series silently had no 50m point.

    An earlier version also drew a dotted extrapolation from the probe's last two points
    to show what assuming the trend continued would have predicted. That was there
    because the assumption had been made and was wrong; with es 1.0 now measured at
    every scale it is a picture of a mistake rather than of the data.
    """
    probe = load(runs, "8845252")
    G = {"50m": ("8840408", "lr2e4_es05_ep4_h512", "lr2e4_es10_ep4_h512", 12, 144),
         "100m": ("8841345", "lr2e4_es05", "lr2e4_es10", 12, 48),
         "200m": ("8841992", "lr2e4_es05", "lr2e4_es10", 12, 48),
         "400m": ("8842147", "lr2e4_es05", "lr2e4_es10", 12, 48)}
    grids = {s: load(runs, j) for s, (j, *_ ) in G.items()}
    scales = list(G)

    def val(scale, half, batch_kind):
        job, a05, a10, bs, bl = G[scale]
        b = bs if batch_kind == "small" else bl
        arm = f"{a05 if half == '05' else a10}_b{b}"
        r = grids[scale].get(arm)
        return r["test_auroc"] if r else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})

    ax = axes[0]
    xs = [0.1, 0.25, 0.5]
    ys = [probe[f"es{k}_ep4"]["test_auroc"] for k in ("01", "025", "05")]
    g05, g10 = val("400m", "05", "small"), val("400m", "10", "small")
    ax.plot(xs, ys, "o-", color=BLUE, lw=2.2, ms=7, zorder=4, label="probe, 4 epochs")
    ax.plot([0.5, 1.0], [g05, g10], "s--", color=RED, lw=2.2, ms=7, zorder=4,
            label="400m grid, batch 12")
    ax.set_xlabel("encoder_lr_scale"); ax.set_ylabel("test AUROC")
    ax.set_title("400m, 4 epochs", loc="left", pad=16)
    ax.grid(axis="y"); ax.set_axisbelow(True); ax.set_xlim(0.02, 1.13)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right", borderaxespad=0.5)

    x = np.arange(len(scales))
    for ax, kind, title in (
            (axes[1], "small", "Effective batch 12"),
            (axes[2], "large", "Large batch (144 at 50m, 48 elsewhere)")):
        a = np.array([val(s, "05", kind) for s in scales])
        b = np.array([val(s, "10", kind) for s in scales])
        ax.fill_between(x, a, b, color=GRID, alpha=0.9, lw=0, zorder=1)
        ax.plot(x, a, "o-", color=BLUE, lw=2, ms=6, zorder=3, label="encoder_lr_scale 0.5")
        ax.plot(x, b, "s--", color=AMBER, lw=2, ms=6, zorder=3, label="encoder_lr_scale 1.0")
        for xi, va, vb in zip(x, a, b):
            hi, lo = (va, vb) if va > vb else (vb, va)
            ax.annotate(f"{hi-lo:+.4f}", (xi, (va + vb) / 2), fontsize=7.4,
                        color=INK, ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                  alpha=0.85))
        ax.set_xticks(x); ax.set_xticklabels(scales)
        ax.set_xlabel("model size"); ax.set_ylabel("test AUROC")
        ax.set_title(title, loc="left", pad=16, fontsize=10)
        ax.grid(axis="y"); ax.set_axisbelow(True); ax.set_xlim(-0.3, 3.3)
        ax.legend(frameon=False, fontsize=7.8, loc="lower right")

    fig.suptitle("encoder_lr_scale is the encoder LR as a multiple of the head's. "
                 "Left: the 400m probe. Right: 0.5 against 1.0 at each scale.",
                 x=0.005, ha="left", fontsize=8.5, color=MUTED, y=1.03)
    fig.savefig(EXTRA / "encoder_lr_scale.png")
    plt.close(fig)


def fig_probe_heatmap(runs: Path) -> None:
    """encoder_lr_scale x epochs at 400m.

    The es 1.0 row comes from the 400m HP grid, not the probe, which never sampled it --
    the omission that made the probe look monotone. That grid arm differs from a probe
    arm only in encoder_lr_scale (lr 2e-4, 4 epochs, per_device 1, accumulation 1, the
    same 1,680,125 scored peaks), and where the two jobs overlap they agree to 0.0003,
    inside the 0.0005 seed noise. It is hatched so the borrowed cell stays visible.
    """
    recs = load(runs, "8845252")
    grid = load(runs, "8842147")
    es = ["es01", "es025", "es05", "es10"]
    labels = ["0.1", "0.25", "0.5", "1.0"]
    eps = ["ep2", "ep4", "ep8"]
    M = np.full((len(es), len(eps)), np.nan)
    for i, e in enumerate(es[:3]):
        for j, ep in enumerate(eps):
            r = recs.get(f"{e}_{ep}")
            if r:
                M[i, j] = r["test_auroc"]
    borrowed = grid.get("lr2e4_es10_b12")
    if borrowed:
        M[3, 1] = borrowed["test_auroc"]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto",
                   vmin=np.nanmin(M), vmax=np.nanmax(M))
    for i in range(len(es)):
        for j in range(len(eps)):
            if np.isnan(M[i, j]):
                txt = ("cut at walltime\nresumed from\ncheckpoint" if i < 3
                       else "not run")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.1,
                        color=MUTED, style="italic")
            else:
                rel = (M[i, j] - np.nanmin(M)) / max(np.nanmax(M) - np.nanmin(M), 1e-9)
                ax.text(j, i, f"{M[i,j]:.4f}", ha="center", va="center", fontsize=9.5,
                        color="white" if rel > 0.6 else INK)
    if borrowed:
        ax.add_patch(plt.Rectangle((0.5, 2.5), 1, 1, fill=False, edgecolor=RED,
                                   lw=1.8, hatch="///", alpha=0.85))
    ax.set_xticks(range(len(eps))); ax.set_xticklabels([e[2:] + " epochs" for e in eps])
    ax.set_yticks(range(len(es))); ax.set_yticklabels(labels)
    ax.set_ylabel("encoder_lr_scale"); ax.set_xlabel("fine-tuning budget")
    ax.set_title("400m: encoder_lr_scale × fine-tuning budget", loc="left", pad=16)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="test AUROC")
    fig.suptitle("Job 8845252, one arm per cell. Fixed-seed noise 0.00022. The "
                 "hatched cell is from the 400m HP grid (8842147), not the probe.",
                 x=0.005, ha="left", fontsize=8.5, color=MUTED, y=1.02)
    fig.savefig(EXTRA / "probe_heatmap.png")
    plt.close(fig)


def fig_pretrain_ablation(runs: Path) -> None:
    """What pretraining is worth, on both metrics.

    The grids originally did not sweep the same epoch counts -- pretrained ran 2 and 4,
    scratch ran 4 and 8 -- which left one hole. FT13 (job 8847610) filled it, and the
    ambiguity did not matter much: pretrained gains only +0.0014 AUROC from 4 to 8
    epochs, so matched-at-8 and the old cross-budget figure nearly coincide.

    The gap narrows with budget on both metrics, because the random encoder gains about
    ten times as much from the extra epochs.
    """
    pre = load(runs, "8840408")
    scr = load(runs, "8841984")
    # FT13, job 8847610. NOT 8846027: that one ran without TILES_PER_ARM=12 and so
    # trained at effective batch 1 instead of 12 -- a different experiment, discarded.
    ep8 = load(runs, "8847610")

    def best(recs, ep, metric):
        v = [r[metric] for a, r in recs.items() if re.search(rf"_ep{ep}(?:_|$)", a)]
        return max(v) if v else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, metric, label in ((axes[0], "test_auroc", "AUROC"),
                              (axes[1], "test_f1", "F1")):
        P = {2: best(pre, 2, metric), 4: best(pre, 4, metric),
             8: max((r[metric] for r in ep8.values()), default=np.nan)}
        S = {4: best(scr, 4, metric), 8: best(scr, 8, metric)}
        xs = [e for e in (2, 4, 8) if not np.isnan(P[e])]
        ax.plot(xs, [P[e] for e in xs], "o-", color=BLUE, lw=2.2, ms=7,
                label="pretrained encoder")
        ax.plot([4, 8], [S[4], S[8]], "s-", color=RED, lw=2.2, ms=7,
                label="random encoder")
        top8 = P[8] if not np.isnan(P[8]) else P[4]
        for e in xs:
            ax.annotate(f"{P[e]:.4f}", (e, P[e]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=BLUE)
        for e in (4, 8):
            ax.annotate(f"{S[e]:.4f}", (e, S[e]), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=8, color=RED)
        span = max(P[4], top8) - min(S[4], S[8])
        ax.set_ylim(min(S[4], S[8]) - 0.18 * span, max(P[4], top8) + 0.14 * span)
        ax.set_xticks([2, 4, 8]); ax.set_xticklabels([f"{e} epochs" for e in (2, 4, 8)])
        ax.set_xlim(1.5, 8.9)
        ax.set_xlabel("fine-tuning budget"); ax.set_ylabel(f"test {label}")
        ax.set_title(f"test {label}", loc="left", pad=14)
        ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.4, loc="lower right")
    fig.savefig(FIGS / "pretrain_ablation.png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="/lus/flare/projects/UIC-HPC/khuss/msdelta/runs")
    cli = ap.parse_args()
    runs = Path(cli.runs)
    if not runs.is_dir():
        raise SystemExit(f"no run directory at {runs}")
    FIGS.mkdir(parents=True, exist_ok=True)
    EXTRA.mkdir(parents=True, exist_ok=True)
    for fn in (fig_scaling, fig_top_cluster, fig_encoder_lr,
               fig_probe_heatmap, fig_pretrain_ablation):
        fn(runs)
        print(f"  {fn.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
