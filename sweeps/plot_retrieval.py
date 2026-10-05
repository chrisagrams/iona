"""Does the separation ratio predict retrieval? Three panels, one question.

    python sweeps/plot_retrieval.py $MSDELTA_EVAL/contrastive/retrieval_vs_separation_8848049.json

Every contrastive conclusion in this project -- which hyperparameters win, whether the
metric improves with scale, whether the pair loss is behind -- was decided by
`sep_spectrum/ratio`. Contrastive exists to serve retrieval. If the two do not track,
those are conclusions about a number that does not matter.

Two task metrics, because they disagree in sign and reporting one alone misleads:
Hit@1 asks whether the single nearest neighbour is a replicate (a LOCAL property of the
embedding), MAP@100 scores the whole ranked list (a GLOBAL one). Contrastive training
inflates the distance between group centroids, which can help the second while hurting
the first.

  LEFT    Hit@1 by scale, against the untrained encoder.
  MIDDLE  MAP@100 by scale, against the untrained encoder.
  RIGHT   separation ratio against the task, one point per arm, with Spearman. The
          selection question: was ranking by the ratio the same as ranking by the task?
"""

from __future__ import annotations

import argparse
import json
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
FIGS = homes.RESULTS / "contrastive" / "superseded"
INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"
BLUE, RED, GREEN = "#2563eb", "#dc2626", "#059669"
SCALE_COLOUR = {"s050m": "#93c5fd", "s100m": "#60a5fa",
                "s200m": "#2563eb", "s400m": "#1e3a8a"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.7, "figure.facecolor": "white",
})


def bar_panel(ax, trained, base, scales, metric):
    """One metric by scale, with the untrained encoder as a reference line."""
    means, sds = [], []
    for s in scales:
        v = [trained[k][metric] for k in trained
             if k.startswith(s) and metric in trained[k]]
        means.append(np.mean(v))
        sds.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
    x = np.arange(len(scales))
    ax.bar(x, means, yerr=sds, capsize=4, width=0.6,
           color=[SCALE_COLOUR[s] for s in scales])
    # Above the error bar, not on the bar top: at 6 seeds the cap sits over the label.
    for xi, m, sd in zip(x, means, sds):
        ax.annotate(f"{m:.3f}", (xi, m + sd), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8.5)
    if base and metric in base:
        ax.axhline(base[metric], color=RED, lw=1.4, ls="--")
        ax.annotate(f"untrained {base[metric]:.3f}", (-0.45, base[metric]),
                    fontsize=8.5, color=RED, ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([s[1:].lstrip("0") for s in scales])
    ax.set_xlabel("model size")
    ax.set_ylabel(metric.split("/")[-1])
    ax.set_title(metric.split("/")[-1] + " by scale", loc="left", pad=14)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    # Headroom so the reference line and the tallest label are never clipped.
    top = max(means[i] + sds[i] for i in range(len(means)))
    if base and metric in base:
        top = max(top, base[metric])
    ax.set_ylim(0, top * 1.18)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", help="output of msdelta.eval_retrieval")
    ap.add_argument("--metrics", nargs="+",
                    default=["retrieval/Hit@1", "retrieval/MAP@100"])
    ap.add_argument("--scatter-metric", default="retrieval/Hit@1")
    cli = ap.parse_args()
    data = json.loads(Path(cli.json).read_text())
    arms = data["arms"]
    base = arms.get("BASELINE-untrained")
    trained = {k: v for k, v in arms.items() if not k.startswith("BASELINE")}
    if not trained:
        raise SystemExit("no trained arms in that file")

    FIGS.mkdir(parents=True, exist_ok=True)
    scales = sorted({k[:5] for k in trained if re.match(r"s\d+m", k)})
    n = len(cli.metrics) + 1
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.2))

    for ax, metric in zip(axes, cli.metrics):
        if scales:
            bar_panel(ax, trained, base, scales, metric)

    ax2 = axes[-1]
    m = cli.scatter_metric
    keep = [k for k, v in trained.items()
            if "sep_spectrum/ratio" in v and m in v]
    xs = [trained[k]["sep_spectrum/ratio"] for k in keep]
    ys = [trained[k][m] for k in keep]
    cols = [SCALE_COLOUR.get(k[:5], BLUE) for k in keep]
    ax2.scatter(xs, ys, s=38, c=cols, zorder=3)
    if base and "sep_spectrum/ratio" in base and m in base:
        bx, by = base["sep_spectrum/ratio"], base[m]
        ax2.scatter([bx], [by], s=70, marker="X", color=RED, zorder=4)
        # Label the point itself. A legend keyed to an X reads as a second data point.
        ax2.annotate("untrained", (bx, by), textcoords="offset points",
                     xytext=(9, 0), va="center", fontsize=8.5, color=RED)
    corr = data.get("correlation", {}).get(m, {})
    rho = corr.get("spearman_vs_ratio")
    if rho is not None and len(xs) > 2:
        z = np.polyfit(xs, ys, 1)
        gx = np.linspace(min(xs), max(xs), 50)
        ax2.plot(gx, np.polyval(z, gx), "-", color=MUTED, lw=1.2, zorder=2)
    ax2.set_xlabel("separation ratio (the proxy)")
    ax2.set_ylabel(m.split("/")[-1] + "  (the task)")
    title = "proxy against task"
    if rho is not None:
        p = corr.get("p")
        title += f" — Spearman {rho:+.2f}"
        if p is not None:
            title += f", p={p:.2f}"
    ax2.set_title(title, loc="left", pad=14)
    # Room on the right for the untrained label, which sits at the axis edge.
    ax2.margins(x=0.12)
    ax2.grid()
    ax2.set_axisbelow(True)

    fig.suptitle(f"Job {data['job']}, {len(trained)} saved encoders scored on the same "
                 f"validation rows.", x=0.005, ha="left", fontsize=8.5, color=MUTED,
                 y=1.02)
    fig.tight_layout()
    out = FIGS / "retrieval_vs_separation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
