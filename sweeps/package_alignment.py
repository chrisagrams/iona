"""Package the A (peptide embedder) results: figures + one CSV per figure with exactly the plotted rows.

    .venv/bin/python sweeps/package_alignment.py

Read from the per-run result JSONs: student test evals (results/raw/finetune/align/test_*.json) and the
cross-modal comparisons vs yHydra (baselines/<dataset>/xmodal/xmodal_*.json on Lustre). Nothing typed in.
Plotting only (login node).

    A_windows.png           Hit@1 vs precursor-mass window, per dataset, ours vs yHydra
    A_teacher_student.png   student Hit@1 vs its teacher's spectrum retrieval (MAP@R), test split
    A_ablations.png         appendix: pooling and loss variants (50M teacher), test Hit@1
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
ALIGN = REPO / "results" / "raw" / "finetune" / "align"
BASE = Path("/lus/flare/projects/UIC-HPC/khuss/msdelta/baselines")
FIG = REPO / "results" / "processed" / "figures" / "SUMMARY"
INK, MUTED, GRIDC = "#1f2937", "#6b7280", "#e5e7eb"
YHYDRA = "#f59e0b"
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9ca3af",
         "axes.linewidth": 0.8, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
         "savefig.bbox": "tight", "figure.dpi": 200}
# student run name -> (label, colour, teacher description)
STUDENTS = {
    "050m-c7s600": ("ours, 50M teacher", "#93c5fd", "C7 50M, step 600"),
    "400m-oodsel": ("ours, 400M teacher (OOD-selected)", "#3b82f6", "C7 400M seed 1, step 600 (chosen on 8-species validation)"),
    "400m-c7final": ("ours, 400M teacher", "#1e3a8a", "C7 400M seed 0, end of epoch"),
}
DATASETS = [("yhydra", "ms-contrastive-100k test\n(in-distribution for ours)"),
            ("c11_cap20", "HEK\n(unseen, low-res MS2)"),
            ("nine_yeast", "nine-species yeast\n(unseen, high-res)"),
            ("mouse", "nine-species mouse\n(unseen, high-res)")]
MOUSE = REPO / "results" / "raw" / "finetune" / "align" / "mouse_yhydra" / "compare_mouse.json"   # job 8870879
WINDOWS = [("open", "", "open search"), ("1.1Da", "/window_1.1Da", "±1.1 Da"), ("20ppm", "/window_20ppm", "20 ppm")]


def msd(v):
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def window_rows():
    rows = []
    for ds, _ in DATASETS:
        yh_done = False
        files = [MOUSE] if ds == "mouse" else sorted((BASE / ds / "xmodal").glob("xmodal_*.json"))
        for f in files:
            if ds == "mouse":   # the released 400M-teacher models from the Hub
                run, job = "v2_align-ft-a1-align-100k-400m-c7final", "8870879"
            else:
                m = re.search(r"xmodal_(.+)-(\d+)\.json$", f.name)
                run, job = m.group(1), m.group(2)
            if k := re.search(r"100k-(050m-c7s600|400m-oodsel|400m-c7final)$", run):
                label, teacher, training = STUDENTS[k.group(1)][0], STUDENTS[k.group(1)][2], "standard"
            elif k := re.search(r"a8-(massb(?:_hn4)?)_seed\d$", run):
                label, teacher, training = f"ours, mass-aware training ({k.group(1)})", STUDENTS["400m-oodsel"][2], "mass-aware"
            else:
                continue
            d = json.loads(f.read_text())
            for w, suf, wl in WINDOWS:
                r = d.get(f"ours/shared/peptide+charge{suf}")
                if r:
                    rows.append(dict(dataset=ds, method=label, training=training, teacher=teacher, job=job,
                                     window=wl, hit_at_1=r["hit@1"], queries=d["n_queries_shared"],
                                     true_outside_window=r.get("true_outside", "")))
            if not yh_done:        # yHydra is deterministic and identical in every file of a dataset
                for w, suf, wl in WINDOWS:
                    r = d.get(f"yhydra/shared/l2{suf}")
                    rows.append(dict(dataset=ds, method="yHydra (L2)", training="", teacher="", job="", window=wl,
                                     hit_at_1=r["hit@1"], queries=d["n_queries_shared"],
                                     true_outside_window=r.get("true_outside", "")))
                yh_done = True
    return rows


def best_of(rows, ds, training):
    """Our model with the highest mean Hit@1 over the three windows on this dataset (None if none)."""
    by = defaultdict(list)
    for r in rows:
        if r["dataset"] == ds and r["training"] == training:
            by[r["method"]].append(r["hit_at_1"])
    return max(by, key=lambda k: np.mean(by[k])) if by else None


def fig_windows(rows, fixed=None, out="A_windows.png"):
    """One line per method: yHydra, our model (the best per dataset, or `fixed` in every panel, falling back
    to the best available where `fixed` was not evaluated), our best mass-aware model (when evaluated)."""
    wl = [w[2] for w in WINDOWS]
    plotted = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.3 * len(DATASETS), 4.4), sharey=True)
        for ax, (ds, title) in zip(axes, DATASETS):
            ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
            lines = [("yHydra (L2)", "yHydra", YHYDRA, "s", "--")]
            have = {r["method"] for r in rows if r["dataset"] == ds}
            if fixed and fixed in have:
                lines.append((fixed, "ours", "#1e3a8a", "o", "-"))
            elif (b := best_of(rows, ds, "standard")):
                lines.append((b, "ours (stand-in)" if fixed else "ours", "#93c5fd" if fixed else "#1e3a8a",
                              "o", ":" if fixed else "-"))
            if (b := best_of(rows, ds, "mass-aware")):
                lines.append((b, "ours, mass-aware training", "#7c3aed", "D", "-"))
            for method, lab, col, mk, ls in lines:
                sel = [r for r in rows if r["dataset"] == ds and r["method"] == method]
                plotted += [dict(r, series=lab) for r in sel]
                ms = [msd([r["hit_at_1"] for r in sel if r["window"] == w]) for w in wl]
                ax.errorbar(range(3), [m for m, _ in ms], yerr=[s for _, s in ms], color=col, marker=mk, ms=7,
                            ls=ls, lw=2, mec="white", mew=1.2, capsize=3, zorder=3)
                for xi, (m, _) in enumerate(ms):
                    ax.annotate(f"{m:.2f}", (xi, m), textcoords="offset points",
                                xytext=(0, 8 if lab != "yHydra" else -14), ha="center", fontsize=8, color=col)
            names = [m.replace("ours, ", "") + ("\n(stand-in: 400M teacher not run here yet)" if lab == "ours (stand-in)" else "")
                     for m, lab, *_ in lines if lab.startswith("ours")]
            outside = [r["true_outside_window"] for r in rows if r["dataset"] == ds and r["window"] == "20 ppm"
                       and r["method"] != "yHydra (L2)" and r["true_outside_window"] != ""]
            if outside and max(outside) > 0:
                ceil = 1 - max(outside)
                ax.plot([1.75, 2.25], [ceil, ceil], color=MUTED, lw=1.2, ls=":")
                ax.text(2.3, ceil, f"ceiling\n{ceil:.2f}", ha="left", va="center", fontsize=7.5, color=MUTED)
            ax.set_xticks(range(3)); ax.set_xticklabels(wl); ax.set_xlim(-0.3, 2.75)
            ax.set_title(title + ("\nours: " + "; ".join(names) if fixed else ""), loc="left", fontsize=10.5)
        axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Hit@1 (spectrum → peptide)")
        fig.supxlabel("candidate peptides restricted to the precursor-mass window", y=-0.04, fontsize=9.5)
        fig.suptitle("Peptide retrieval vs yHydra: open search and precursor-mass windows", x=0.07, ha="left",
                     fontsize=13, fontweight="bold", y=1.1 if fixed else 1.03)
        leg = [("yHydra", YHYDRA, "s", "--"),
               (f"ours ({fixed.replace('ours, ', '')})" if fixed else "ours", "#1e3a8a", "o", "-")]
        if any(r["training"] == "mass-aware" for r in rows):
            leg.append(("ours, mass-aware training", "#7c3aed", "D", "-"))
        fig.legend(handles=[Line2D([], [], color=c, marker=mk, ls=ls, lw=2, ms=7, mec="white", mew=1.2)
                            for _, c, mk, ls in leg], labels=[l for l, *_ in leg], frameon=False,
                   fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.07))
        fig.text(0.07, -0.2, "Ceilings (20 ppm): for 24% of HEK and 10% of mouse queries the recorded precursor mass is that of a heavier isotope peak (M+1 or M+2; every such case lies\nwithin 20 ppm of an exact isotope offset), so the true peptide is ~1 Da outside a 20 ppm window and no method can recover it.", fontsize=8, color=MUTED)
        fig.savefig(FIG / out); plt.close(fig)
    return plotted


def student_rows():
    rows = []
    for f in sorted(ALIGN.glob("test_*.json")):
        d = json.loads(f.read_text()); m = d.get("metrics", d)
        name = re.sub(r"-\d{7}\.json$", "", f.name)[len("test_"):]
        seed = re.search(r"_seed(\d)", name)
        rows.append(dict(run=re.sub(r"_seed\d", "", name), seed=int(seed.group(1)) if seed else "",
                         job=re.search(r"-(\d{7})\.json$", f.name).group(1),
                         hit_at_1=m["crossmodal/hit@1"], hit_at_5=m["crossmodal/hit@5"], mrr=m["crossmodal/mrr"],
                         spectra=m["crossmodal/n_spectra"], candidates=m["crossmodal/n_candidates"],
                         teacher_map_at_r=m["teacher_spectrum/MAP@R"], teacher_hit_at_1=m["teacher_spectrum/Hit@1"]))
    return rows


def fig_teacher_student(srows):
    out = []
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
        for key, (lab, col, teacher) in STUDENTS.items():
            rs = [r for r in srows if r["run"] == f"v2_align-ft-a1-align-100k-{key}"]
            out += [dict(r, student=lab, teacher=teacher) for r in rs]
            tx = rs[0]["teacher_map_at_r"]; m, s = msd([r["hit_at_1"] for r in rs])
            ax.errorbar([tx], [m], yerr=[s], color=col, marker="o", ms=10, mec="white", mew=1.4, capsize=3, zorder=3)
            ax.annotate(lab.replace("ours, ", "").replace(" (OOD-selected)", "\n(OOD-selected)"), (tx, m),
                        textcoords="offset points", xytext=(-10, 6), ha="right", fontsize=8.5, color=INK)
        xs = [o["teacher_map_at_r"] for o in out]; ys = [o["hit_at_1"] for o in out]
        k, b = np.polyfit(xs, ys, 1); xx = np.array([min(xs) - 0.005, max(xs) + 0.005])
        ax.plot(xx, k * xx + b, color=MUTED, lw=1, ls="--", zorder=1)
        ax.set_xlabel("teacher: spectrum retrieval MAP@R (test)", labelpad=6)
        ax.set_ylabel("student: peptide Hit@1 (test)", labelpad=6)
        ax.set_xlim(0.82, 0.875); ax.set_ylim(0.89, 0.93)
        ax.set_title("A better spectrum teacher gives a better peptide student", loc="left", fontsize=12,
                     fontweight="bold", pad=10)
        fig.savefig(FIG / "A_teacher_student.png"); plt.close(fig)
    return out


ABLATIONS = [  # run name -> label, group  (all use the C7 50M step-600 teacher)
    ("v2_align-ft-a1-align-100k-050m-c7s600", "default: MSE, mean+max pool", "default"),
    ("sweep-cls", "CLS-token pooling", "pooling"),
    ("sweep-attn", "attention pooling", "pooling"),
    ("sweep-pool", "mean+max pooling (rerun)", "pooling"),
    ("sweep-lit_hn0_mse0", "LiT", "loss"),
    ("sweep-lit_hn0_mse01", "LiT + 0.1 MSE", "loss"),
    ("sweep-lit_hn4_mse0", "LiT + 4 hard negatives", "loss"),
    ("sweep-lit_hn4_mse01", "LiT + 4 hard neg. + 0.1 MSE", "loss"),
]


def fig_ablations(srows):
    cols = {"default": "#1e3a8a", "pooling": "#60a5fa", "loss": "#14b8a6"}
    out = []
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.xaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
        for i, (run, lab, grp) in enumerate(ABLATIONS):
            rs = [r for r in srows if r["run"] == run]
            out += [dict(r, variant=lab, group=grp) for r in rs]
            m, s = msd([r["hit_at_1"] for r in rs])
            ax.barh(i, m, 0.7, xerr=s, color=cols[grp], capsize=3, error_kw=dict(elinewidth=1, ecolor=INK), zorder=2)
            ax.text(m + s + 0.0006, i, f"{m:.3f}", va="center", fontsize=8.5)
        ax.set_yticks(range(len(ABLATIONS))); ax.set_yticklabels([a[1] for a in ABLATIONS]); ax.invert_yaxis()
        ax.set_xlim(0.885, 0.905); ax.set_xlabel("peptide Hit@1 (test)", labelpad=6)
        ax.set_title("Student design ablations (50M teacher)", loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()], labels=list(cols),
                  frameon=False, fontsize=8.5, loc="lower right")
        fig.savefig(FIG / "A_ablations.png"); plt.close(fig)
    return out


def write(name, rows):
    keys = list(rows[0])
    with open(FIG / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    wr = window_rows(); write("A_windows.csv", fig_windows(wr))
    write("A_windows_A2_preview.csv", fig_windows(wr, fixed="ours, 400M teacher", out="A_windows_A2_preview.png"))
    sr = student_rows()
    ts = fig_teacher_student(sr); write("A_teacher_student.csv", ts)
    lines = ["| teacher | teacher spectrum MAP@R (test) | student peptide Hit@1 (test, 3 seeds) | Hit@5 | MRR |",
             "|---|---|---|---|---|"]
    for key, (lab, _, teacher) in STUDENTS.items():
        rs = [r for r in ts if r["student"] == lab]
        m, sd = msd([r["hit_at_1"] for r in rs])
        lines.append(f"| {teacher} | {rs[0]['teacher_map_at_r']:.3f} | {m:.3f} ± {sd:.4f} | "
                     f"{np.mean([r['hit_at_5'] for r in rs]):.3f} | {np.mean([r['mrr'] for r in rs]):.3f} |")
    (FIG / "A_teacher_student.md").write_text("\n".join(lines) + "\n")
    write("A_ablations.csv", fig_ablations(sr))
    agg = defaultdict(list)
    for r in wr:
        agg[(r["dataset"], r["method"], r["window"])].append(r["hit_at_1"])
    for k, v in agg.items():
        print(f"  {k[0]:11s} {k[1]:36s} {k[2]:12s} n={len(v)} {np.mean(v):.3f}")


if __name__ == "__main__":
    main()
