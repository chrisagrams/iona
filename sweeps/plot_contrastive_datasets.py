"""Consensus-recipe spectrum encoders on every evaluation set, against the baselines we have (K163/K188).

    .venv/bin/python sweeps/plot_contrastive_datasets.py

Sets (n experimental queries): validation 25,057 and test 25,137 (ms-contrastive-100k), oodval 20,004 (nine-species,
8 non-yeast species; new since submission), mouse 20,003 and human 20,000 (noble 20k), yeast 86,184 (full
nine-species yeast). Ours: K163 consensus finals (540k, cons-<set>/) and the K188 checkpoint curves
(cons-allck-<set>/), 3 seeds, mean +- sd. Baselines, open search, on the same queries:
  binned cosine 0.1 / 1 Da   <set dir>/binned_w0.1.json, binned_w1.0005.json (validation: k90-refcheck/binned01.json)
  GLEAMS                     gleams/<set>.json (copied from $S/baselines; none for validation / oodval)
  paper's best (C7)          results/processed/figures/SUMMARY/C_benchmarks.csv / C_transfer.csv (400M; 50M where run)
With precursor filters (20 ppm, isotope-tolerant 20 ppm; F = queries failing 20 ppm, Fbar = passing): binned 0.1 Da on
mouse / human from filter-failure/summary.json.

  datasets_scale.png       MAP@R and Hit@1 vs model size at 540k, one panel per set, baselines as lines
  datasets_checkpoint.png  MAP@R vs pretraining steps per set (K188), one line per scale, baselines as lines
  datasets_filters.png     mouse / human: no filter vs 20 ppm vs iso-20 ppm, all / F / Fbar queries, vs binned 0.1 Da
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_contrastive_100k as p  # noqa: E402
import plot_contrastive_cons as pc  # noqa: E402
from plot_contrastive_100k import plt  # noqa: E402

R = pc.R
FIGS = p.FIGS
SETS = [("validation", "validation (ms-contrastive-100k)"), ("test", "test (ms-contrastive-100k)"),
        ("oodval", "oodval (8 species, NEW)"), ("mouse", "mouse 20k"), ("human", "human 20k"),
        ("yeast", "yeast (full, 86k)")]
BINNED_DIR = {"test": "grouped100k-test", "oodval": "oodval20k", "mouse": "mouse20k", "human": "human20k",
              "yeast": "nine_yeast"}
PAPER = {"test": ("C_benchmarks", "ms-contrastive-100k"), "yeast": ("C_benchmarks", "yeast-full"),
         "mouse": ("C_transfer", "mouse-20k"), "human": ("C_transfer", "human-20k")}
LINES = {"binned 0.1 Da": ("#b45309", "--"), "binned 1 Da": ("#d97706", ":"), "GLEAMS": ("#7c3aed", "-.")}
KEYS = {"MAP@R": "experimental/MAP@R", "Hit@1": "experimental/Hit@1"}


def baselines(s):
    """{name: {"MAP@R": x, "Hit@1": y}} for the open-search baselines on set s."""
    out = {}
    files = ({"binned 0.1 Da": R / "k90-refcheck" / "binned01.json"} if s == "validation" else
             {"binned 0.1 Da": R / BINNED_DIR[s] / "binned_w0.1.json", "binned 1 Da": R / BINNED_DIR[s] / "binned_w1.0005.json"})
    for name, f in files.items():
        if f.exists():
            m = json.load(open(f))["metrics"]
            out[name] = {k: m[v] for k, v in KEYS.items() if v in m}
    g = R / "gleams" / f"{s}.json"
    if g.exists():
        e = json.load(open(g))["mp512/experimental"]
        out["GLEAMS"] = {"MAP@R": e["cos_MAP@R"], "Hit@1": e["cos_Hit@1"]}
    return out


def paper_best(s):
    """{scale: (MAP@R mean, sd, Hit@1 mean, sd)} of the paper's C7 models on set s."""
    if s not in PAPER:
        return {}
    f, bench = PAPER[s]
    rows = {}
    for r in csv.DictReader(open(p.REPO / "results/processed/figures/SUMMARY" / f"{f}.csv")):
        m = re.search(r"(\d+)M", r["model"])
        if r["benchmark"] == bench and "C7" in (r["note"] + r["model"]) and m:
            rows.setdefault(f"{m.group(1)}m", []).append((float(r["map_at_r"]), float(r["hit_at_1"])))
    return {sc: (*p.agg([a for a, _ in v]), *p.agg([b for _, b in v])) for sc, v in rows.items()}


def ours(s):
    new = pc.load_new(s)
    return pc.cells(new, pc.NEW, list(KEYS.values()) + [f"experimental/{f}/{sub}/MAP@R" for f in ("open", "20ppm", "iso20ppm")
                                                         for sub in ("full", "F", "Fbar")])


def draw_baselines(ax, base, metric):
    for name, vals in base.items():
        if metric in vals:
            colour, ls = LINES[name]
            ax.axhline(vals[metric], color=colour, ls=ls, lw=1.2, label=f"{name} {vals[metric]:.3f}")


def fig_scale():
    fig, axes = plt.subplots(2, len(SETS), figsize=(4.1 * len(SETS), 8.2))
    for col, (s, title) in enumerate(SETS):
        c, base, best = ours(s), baselines(s), paper_best(s)
        for row, (metric, key) in enumerate(KEYS.items()):
            ax = axes[row][col]
            pts = [(sc, p.agg(c[(sc, "540")][key])) for sc in pc.ORDER if (sc, "540") in c and key in c[(sc, "540")]]
            x = [pc.PARAMS[sc] for sc, _ in pts]; y = [m for _, (m, _) in pts]
            ax.errorbar(x, y, yerr=[sd for _, (_, sd) in pts], color=p.MUTED, lw=1.3, capsize=3, zorder=2)
            for (sc, _), xi, yi in zip(pts, x, y):
                ax.scatter([xi], [yi], s=50, color=pc.COLOUR[sc], zorder=3)
            p.label(ax, x, y)
            for sc, (mm, msd, hm, hsd) in best.items():
                v, sd = (mm, msd) if metric == "MAP@R" else (hm, hsd)
                ax.errorbar([pc.PARAMS[sc]], [v], yerr=[sd], marker="*", ms=12, color=pc.COLOUR[sc], mec=p.INK,
                            mew=0.6, ls="none", capsize=3, zorder=4, label=f"paper C7 {sc} {v:.3f}")
            draw_baselines(ax, base, metric)
            ax.set_xscale("log"); ax.set_xticks([pc.PARAMS[s_] for s_ in pc.ORDER]); ax.set_xticklabels(pc.ORDER)
            ax.minorticks_off()
            p.finish(ax, "model size", f"experimental {metric}", title if row == 0 else "")
            ax.legend(frameon=False, fontsize=6.6, loc="lower right")
    fig.suptitle("Consensus recipe, final pretraining checkpoint (540k), open search (no precursor filter). "
                 "Lines: baselines on the same queries; stars: the paper's best models (C7).",
                 x=0.01, ha="left", fontsize=9, color=p.MUTED)
    fig.tight_layout(); out = FIGS / "datasets_scale.png"; fig.savefig(out); plt.close(fig)
    return out


def fig_checkpoint():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    for ax, (s, title) in zip(axes.flat, SETS):
        c = ours(s)
        cv = pc.curves(c, "experimental/MAP@R")
        pc.draw_curves(ax, cv, labels=False)
        pc.end_labels(ax, cv)
        draw_baselines(ax, baselines(s), "MAP@R")
        p.finish(ax, "pretraining grad steps", "experimental MAP@R", title)
        ax.legend(frameon=False, fontsize=6.6)
        ax.margins(x=0.12)
    fig.suptitle("Consensus recipe on every pretraining checkpoint (K188), open search; dashed/dotted lines = baselines "
                 "on the same queries", x=0.01, ha="left", fontsize=9, color=p.MUTED)
    fig.tight_layout(); out = FIGS / "datasets_checkpoint.png"; fig.savefig(out); plt.close(fig)
    return out


def fig_filters():
    summary = json.load(open(R / "filter-failure" / "summary.json"))["results"]
    filters = (("open", "no filter"), ("20ppm", "20 ppm"), ("iso20ppm", "iso-20 ppm"))
    subsets = (("full", "all queries"), ("F", "F: fail 20 ppm"), ("Fbar", "Fbar: pass 20 ppm"))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    for row, s in enumerate(("mouse", "human")):
        c = ours(s)
        binned = summary[s]["models"]["binned_0.1"]
        for col, (sub, sname) in enumerate(subsets):
            ax = axes[row][col]
            xs = range(len(filters))
            for sc in pc.ORDER:
                vals = [c.get((sc, "540"), {}).get(f"experimental/{f}/{sub}/MAP@R") for f, _ in filters]
                if all(v is not None for v in vals):
                    ms = [p.agg(v) for v in vals]
                    ax.errorbar(xs, [m for m, _ in ms], yerr=[sd for _, sd in ms], marker="o", ms=5, lw=1.4,
                                capsize=3, color=pc.COLOUR[sc], label=f"ours {sc}")
            bv = [binned[f][sub]["MAP@R"] for f, _ in filters]
            ax.plot(xs, bv, marker="s", ms=6, lw=1.4, ls="--", color=LINES["binned 0.1 Da"][0], label="binned 0.1 Da")
            for x, v in zip(xs, bv):
                ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, -13), ha="center",
                            fontsize=7, color=LINES["binned 0.1 Da"][0])
            n = binned["open"][sub]["n"]
            ax.set_xticks(list(xs)); ax.set_xticklabels([f for _, f in filters])
            p.finish(ax, "precursor filter", "experimental MAP@R", f"{s}: {sname} (n={n})")
    fig.suptitle("Consensus recipe at 540k vs binned cosine 0.1 Da, with and without precursor filters. F queries lose "
                 "their true matches under the plain 20 ppm window (mass calibration); the isotope-tolerant window "
                 "recovers them.", x=0.01, ha="left", fontsize=8.5, color=p.MUTED)
    fig.tight_layout(); out = FIGS / "datasets_filters.png"; fig.savefig(out); plt.close(fig)
    return out


def main() -> int:
    for f in (fig_scale, fig_checkpoint, fig_filters):
        print(f())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
