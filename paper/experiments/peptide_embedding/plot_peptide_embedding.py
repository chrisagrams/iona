"""Regenerate the figure and table in this folder from their CSVs.

    python plot_peptide_embedding.py      # needs matplotlib + numpy

    A_windows.csv          -> A_windows.png          (Hit@1 vs precursor-mass window, ours vs yHydra)
    A_teacher_student.csv  -> A_teacher_student.md   (student Hit@1 vs teacher quality, table)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
INK, MUTED, GRIDC = "#1f2937", "#6b7280", "#e5e7eb"
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9ca3af", "axes.linewidth": 0.8,
         "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "figure.dpi": 200}
DATASETS = [("yhydra", "ms-contrastive-100k test\n(in-distribution for ours)"),
            ("c11_cap20", "HEK\n(unseen, low-res MS2)"),
            ("nine_yeast", "nine-species yeast\n(unseen, high-res)")]
WINDOWS = ["open search", "±1.1 Da", "20 ppm"]
LINES = [("yHydra", "#f59e0b", "s", "--"), ("ours", "#1e3a8a", "o", "-")]


def msd(v):
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def windows():
    rows = list(csv.DictReader(open(HERE / "A_windows.csv")))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
        for ax, (ds, title) in zip(axes, DATASETS):
            ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
            for lab, col, mk, ls in LINES:
                sel = [r for r in rows if r["dataset"] == ds and r["series"] == lab]
                if not sel:
                    continue
                ms = [msd([float(r["hit_at_1"]) for r in sel if r["window"] == w]) for w in WINDOWS]
                ax.errorbar(range(3), [m for m, _ in ms], yerr=[s for _, s in ms], color=col, marker=mk, ms=7,
                            ls=ls, lw=2, mec="white", mew=1.2, capsize=3, zorder=3)
                for xi, (m, _) in enumerate(ms):
                    ax.annotate(f"{m:.2f}", (xi, m), textcoords="offset points",
                                xytext=(0, 8 if lab != "yHydra" else -14), ha="center", fontsize=8, color=col)
            out = [float(r["true_outside_window"]) for r in rows if r["dataset"] == ds and r["window"] == "20 ppm"
                   and r["series"] == "ours" and r["true_outside_window"] not in ("", None)]
            if out and max(out) > 0:
                ceil = 1 - max(out)
                ax.plot([1.75, 2.25], [ceil, ceil], color=MUTED, lw=1.2, ls=":")
                ax.text(1.7, ceil + 0.03, f"ceiling {ceil:.2f}", ha="right", va="center", fontsize=7.5, color=MUTED)
            ax.set_xticks(range(3)); ax.set_xticklabels(WINDOWS); ax.set_xlim(-0.3, 2.3)
            ax.set_title(title, loc="left", fontsize=10.5)
        axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Hit@1 (spectrum → peptide)")
        axes[1].set_xlabel("candidate peptides restricted to the precursor-mass window", labelpad=8)
        fig.suptitle("Peptide retrieval vs yHydra: open search and precursor-mass windows", x=0.07, ha="left",
                     fontsize=13, fontweight="bold", y=1.03)
        fig.legend(handles=[Line2D([], [], color=c, marker=mk, ls=ls, lw=2, ms=7, mec="white", mew=1.2)
                            for _, c, mk, ls in LINES], labels=[l for l, *_ in LINES], frameon=False,
                   fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.02))
        fig.savefig(HERE / "A_windows.png"); plt.close(fig)


def teacher_student_table():
    rows = list(csv.DictReader(open(HERE / "A_teacher_student.csv")))
    lines = ["| teacher | teacher spectrum MAP@R (test) | student peptide Hit@1 (test, 3 seeds) | Hit@5 | MRR |",
             "|---|---|---|---|---|"]
    for teacher in dict.fromkeys(r["teacher"] for r in rows):
        rs = [r for r in rows if r["teacher"] == teacher]
        m, sd = msd([float(r["hit_at_1"]) for r in rs])
        lines.append(f"| {teacher} | {float(rs[0]['teacher_map_at_r']):.3f} | {m:.3f} ± {sd:.4f} | "
                     f"{np.mean([float(r['hit_at_5']) for r in rs]):.3f} | {np.mean([float(r['mrr']) for r in rs]):.3f} |")
    (HERE / "A_teacher_student.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    windows(); teacher_student_table()
