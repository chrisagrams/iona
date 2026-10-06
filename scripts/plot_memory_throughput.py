"""Plot peak memory and throughput vs batch size from bench_memory_throughput.py CSVs.

    python scripts/plot_memory_throughput.py results.csv [more.csv ...] --out figure.pdf

Top row: peak memory over every timed batch; bottom row: spectra/s. Left: inference;
right: training. Solid lines are eager, dashed are torch.compile. Length-sorted variants
(#48) draw as markers only in the memory panels, because sorting keeps the worst batch and
so sits exactly on the unsorted line of the same code.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

STYLE = {
    # Preprint style; Computer Modern via mathtext so no TeX install is needed.
    "font.family": "serif", "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "font.size": 11,
    "figure.dpi": 300, "axes.labelsize": 12, "axes.titlesize": 14, "axes.linewidth": 0.8,
    "axes.prop_cycle": cycler(color=["#440154", "#31688E", "#35B779", "#FDE725"]),
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.5, "lines.markersize": 5, "legend.fontsize": 9,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}
COLORS = STYLE["axes.prop_cycle"].by_key()["color"]
# Fixed order: color follows the variant, never its rank. (CSV variant, label, color, marker)
VARIANTS = [
    ("main", "main", COLORS[0], "o"),
    ("46+48", "#48+#46", COLORS[1], "s"),
    ("46", "#46", COLORS[2], "D"),
]


def load(paths):
    """{mode: {(variant, compiled): {batch: (peak GiB, spectra/s)}}}, plus device capacity."""
    data, capacity = {"infer": {}, "train": {}}, None
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "ok" or not row["spectra_per_s"]:
                    continue
                key = (row["variant"], row["compiled"] == "1")
                data[row["mode"]].setdefault(key, {})[int(row["batch"])] = (
                    float(row["peak_gib"]), float(row["spectra_per_s"]))
                capacity = capacity or float(row["device_total_gib"])
    return data, capacity


def draw(ax, data, metric, markers_only=()):
    for i, (variant, label, color, marker) in enumerate(VARIANTS):
        for compiled in (False, True):
            pts = sorted(data.get((variant, compiled), {}).items())
            if not pts:
                continue
            xs, ys = zip(*((b, v[metric]) for b, v in pts))
            bare = variant in markers_only
            dashed = compiled and not bare
            # Compiled dashes are offset by variant so coincident compiled lines alternate colors.
            ax.plot(xs, ys, ls="none" if bare else (i * 3.5, (3.5, 3.5)) if dashed else "-",
                    color=color, marker=marker, markersize=4.5 if compiled else 5,
                    markeredgecolor="#222222", markeredgewidth=0.5,
                    markerfacecolor="white" if compiled else color,
                    label=f"{label}, {'compiled' if compiled else 'eager'}",
                    zorder=4 if bare else 3 if dashed else 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--out", default="memory_vs_batch.pdf")
    args = parser.parse_args(argv)

    plt.rcParams.update(STYLE)
    data, capacity = load(args.csv)
    batches = [b for mode in data.values() for series in mode.values() for b in series]
    ticks = [2**k for k in range(2, max(batches, default=128).bit_length() + 1)]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharex=True)
    for col, (mode, title) in enumerate((("infer", "Inference"), ("train", "Training"))):
        mem, tput = axes[0, col], axes[1, col]
        draw(mem, data[mode], 0, markers_only=("46+48",))
        draw(tput, data[mode], 1)
        if capacity:
            mem.axhline(capacity, color="#888888", lw=0.8, ls=":", zorder=1)
            mem.text(ticks[0] * 1.05, capacity * 1.015, "Device capacity", fontsize=8,
                     color="#555555", va="bottom")
            mem.set_ylim(0, capacity * 1.08)
        mem.set_title(title)
        tput.set_ylim(bottom=0)
        tput.set_xlabel("Batch size")
        tput.set_xscale("log", base=2)
        tput.set_xticks(ticks)
        tput.set_xticklabels([str(t) for t in ticks])
        tput.set_xlim(ticks[0] * 0.875, ticks[-1] * 1.17)
    axes[0, 0].set_ylabel("Peak memory (GiB)")
    axes[1, 0].set_ylabel("Throughput (spectra/s)")
    for row in axes:
        row[1].sharey(row[0])
        row[1].tick_params(labelleft=False)
    handles = {}
    for ax in (*axes[1], *axes[0]):  # throughput first: its lines carry the canonical styles
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    order = [f"{label}, {m}" for _, label, *_ in VARIANTS for m in ("eager", "compiled")
             if f"{label}, {m}" in handles]
    fig.legend([handles[label] for label in order], order, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.2)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
