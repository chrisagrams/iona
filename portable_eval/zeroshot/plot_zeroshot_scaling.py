"""Zero-shot scaling plot + CSV from the per-encoder JSONs of run_zeroshot.sh.

    python plot_zeroshot_scaling.py RESULTS_DIR      # -> RESULTS_DIR/zeroshot_scaling.{png,csv}

Two panels, x = model size, one line per pretraining checkpoint:
  left   frozen encoder, best layer (raw)
  right  frozen encoder, best layer after ABTT (mean + top-D components removed, fitted on
         the TRAIN sample; the layer and D are the best on the test set, i.e. an upper bound)
Metric: MAP@R over experimental spectra.
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALES = ["50m", "100m", "200m", "400m"]
INK, MUTED = "#1f2937", "#6b7280"
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9ca3af", "axes.linewidth": 0.8,
         "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "figure.dpi": 200}


def summarise(path):
    d = json.loads(Path(path).read_text())
    raw = {lay: v["experimental/MAP@R"] for lay, v in d["layers"].items()}
    best_raw = max(raw, key=raw.get)
    tr = d["abtt"]["train"]
    abtt = max(((D, lay, v["experimental/MAP@R"]) for D, L in tr.items() if D != "center"
                for lay, v in L.items()), key=lambda t: t[2])
    return dict(raw_final=raw.get("final"), raw_best=raw[best_raw], raw_best_layer=best_raw,
                abtt_best=abtt[2], abtt_layer=abtt[1], abtt_D=abtt[0])


def main(out_dir):
    out_dir = Path(out_dir)
    rows = []
    for f in sorted(out_dir.glob("zs_*.json")):
        m = re.match(r"zs_0*(\d+m)_ck0*(\d+)k", f.stem)
        if not m:
            continue
        rows.append(dict(scale=m.group(1), ckpt_k=int(m.group(2)), **summarise(f)))
    if not rows:
        raise SystemExit(f"no zs_*.json in {out_dir}")
    with open(out_dir / "zeroshot_scaling.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    ckpts = sorted({r["ckpt_k"] for r in rows})
    cmap = plt.get_cmap("viridis_r")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
        for ax, key, title in ((axes[0], "raw_best", "frozen, best layer"),
                               (axes[1], "abtt_best", "frozen, best layer + ABTT")):
            ax.yaxis.grid(True, color="#e5e7eb", lw=0.8); ax.set_axisbelow(True)
            for i, c in enumerate(ckpts):
                pts = [(SCALES.index(r["scale"]), r[key]) for r in rows if r["ckpt_k"] == c and r["scale"] in SCALES]
                pts.sort()
                col = cmap(0.25 + 0.7 * i / max(len(ckpts) - 1, 1))
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, lw=2, ms=6, mec="white", mew=1.2,
                        label=f"{c}k steps")
            ax.set_xticks(range(len(SCALES))); ax.set_xticklabels([s.upper() for s in SCALES])
            ax.set_xlabel("model size"); ax.set_title(title, loc="left", fontsize=10.5)
        axes[0].set_ylabel("MAP@R, ms-contrastive-100k test")
        h, l = axes[1].get_legend_handles_labels()
        axes[1].legend(h[::-1], l[::-1], title="pretraining", frameon=False, fontsize=8.5,
                       loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.suptitle("Zero-shot spectrum retrieval from the frozen pretrained encoder", x=0.07, ha="left",
                     fontsize=13, fontweight="bold", y=1.03)
        fig.savefig(out_dir / "zeroshot_scaling.png"); plt.close(fig)
    print(f"wrote {out_dir / 'zeroshot_scaling.png'} and .csv ({len(rows)} encoders)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/zeroshot")
