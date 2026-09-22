"""Does the separation ratio predict retrieval? Two panels, one question.

    python sweeps/plot_retrieval.py results/retrieval_vs_separation_8848049.json

Every contrastive conclusion in this project -- which hyperparameters win, whether the
metric improves with scale, whether the pair loss is behind -- was decided by
`sep_spectrum/ratio`. Contrastive exists to serve retrieval. If the two do not track,
those are conclusions about a number that does not matter.

  LEFT   retrieval against the untrained encoder, per scale. The absolute question:
         does contrastive training move the task at all?
  RIGHT  separation ratio against Hit@1, one point per arm, with Spearman. The
         selection question: was ranking by the ratio the same as ranking by the task?
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "results" / "figures"
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", help="output of msdelta.eval_retrieval")
    ap.add_argument("--metric", default="retrieval/Hit@1")
    cli = ap.parse_args()
    data = json.loads(Path(cli.json).read_text())
    arms = data["arms"]
    base = arms.get("BASELINE-untrained")
    trained = {k: v for k, v in arms.items() if not k.startswith("BASELINE")}
    if not trained:
        raise SystemExit("no trained arms in that file")

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    scales = sorted({k[:5] for k in trained if re.match(r"s\d+m", k)})
    labels = [s.lstrip("s0") or "50m" for s in scales]
    if scales:
        means, sds = [], []
        for s in scales:
            v = [trained[k][cli.metric] for k in trained
                 if k.startswith(s) and cli.metric in trained[k]]
            means.append(np.mean(v)); sds.append(np.std(v, ddof=1) if len(v) > 1 else 0)
        x = np.arange(len(scales))
        ax.bar(x, means, yerr=sds, capsize=4, color=[SCALE_COLOUR[s] for s in scales],
               width=0.6)
        for xi, m in zip(x, means):
            ax.annotate(f"{m:.3f}", (xi, m), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8.5)
        if base and cli.metric in base:
            ax.axhline(base[cli.metric], color=RED, lw=1.4, ls="--")
            ax.annotate(f"untrained {base[cli.metric]:.3f}",
                        (len(scales) - 0.5, base[cli.metric]), fontsize=8.5, color=RED,
                        ha="right", va="bottom")
        ax.set_xticks(x); ax.set_xticklabels([f"{s[1:].lstrip('0')}" for s in scales])
        ax.set_xlabel("model size"); ax.set_ylabel(cli.metric.split("/")[-1])
        ax.set_title(cli.metric.split("/")[-1] + " by scale", loc="left", pad=14)
        ax.grid(axis="y"); ax.set_axisbelow(True)

    xs = [v["sep_spectrum/ratio"] for v in trained.values()
          if "sep_spectrum/ratio" in v and cli.metric in v]
    ys = [v[cli.metric] for v in trained.values()
          if "sep_spectrum/ratio" in v and cli.metric in v]
    cols = [SCALE_COLOUR.get(k[:5], BLUE) for k, v in trained.items()
            if "sep_spectrum/ratio" in v and cli.metric in v]
    ax2.scatter(xs, ys, s=38, c=cols, zorder=3)
    if base:
        ax2.scatter([base.get("sep_spectrum/ratio")], [base.get(cli.metric)], s=70,
                    marker="X", color=RED, zorder=4, label="untrained")
        ax2.legend(frameon=False, fontsize=8.4, loc="lower right")
    corr = data.get("correlation", {}).get(cli.metric, {})
    rho = corr.get("spearman_vs_ratio")
    if rho is not None and len(xs) > 2:
        z = np.polyfit(xs, ys, 1)
        gx = np.linspace(min(xs), max(xs), 50)
        ax2.plot(gx, np.polyval(z, gx), "-", color=MUTED, lw=1.2, zorder=2)
    ax2.set_xlabel("separation ratio (the proxy)")
    ax2.set_ylabel(cli.metric.split("/")[-1] + "  (the task)")
    title = "proxy against task"
    if rho is not None:
        title += f" — Spearman {rho:+.2f}"
    ax2.set_title(title, loc="left", pad=14)
    ax2.grid(); ax2.set_axisbelow(True)

    fig.suptitle(f"Job {data['job']}, {len(trained)} saved encoders scored on the same "
                 f"validation rows.", x=0.005, ha="left", fontsize=8.5, color=MUTED,
                 y=1.02)
    out = FIGS / "retrieval_vs_separation.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
