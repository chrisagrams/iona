"""Contrastive figures for the consensus recipe on every pretraining checkpoint (K188-C), ms-contrastive-100k test.

    .venv/bin/python sweeps/plot_contrastive_cons.py

New models: lr 4e-4, P170 x K2, ms-contrastive-100k + consensus spectra (K163/K188), 3 seeds per point.
  results/raw/finetune/contrastive/cons-allck-test/  (job 8901080: every non-final checkpoint, 50m-400m)
  results/raw/finetune/contrastive/cons-test/        (K163 finals at 540k, 25m-400m)
Old (paper) models: C2/C4 replicate-corpus-only fine-tunes and C7, in grouped100k-test/ -- the SAME 25,137
experimental queries (same eval-data dir), so old and new share axes and the paper's reference lines.

  c100k_cons_scale.png       model size at checkpoint 220k (as c100k_scale.png) + the 540k finals
  c100k_cons_checkpoint.png  pretraining grad steps, one line per scale (as c100k_checkpoint.png)
  c100k_cons_filters.png     experimental MAP@R and library-search Hit@1 with no / 20 ppm / iso-20 ppm filter
  c100k_old_vs_new.png       new recipe (solid) vs the paper's replicate-corpus curves (dashed) + C7 (stars)

Points are 3-seed means with seed sd as error bars, labelled with the mean. Reference lines (binned cosine,
frozen encoders' final layer) are the paper's, on the same queries.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_contrastive_100k as p  # noqa: E402  (style, loaders and helpers of the paper figures)
from plot_contrastive_100k import plt  # noqa: E402

import glob  # noqa: E402
import json  # noqa: E402

R = p.REPO / "results" / "raw" / "finetune" / "contrastive"
FIGS = p.FIGS
COLOUR = {"25m": "#bfdbfe", **p.SCALE_COLOUR}
ORDER = ["25m"] + p.ORDER
PARAMS = {"25m": 25, **p.PARAMS}
NEW = r"s0*(?P<scale>\d+m)_ck(?P<ck>\d+)k_lr4e-4_p170k2_cons_seed\d"
FILTERS = (("open", "no filter"), ("20ppm", "20 ppm"), ("iso20ppm", "iso-20 ppm"))


def load_new(split="test") -> dict[str, dict]:
    out = {}
    for d in (f"cons-allck-{split}", f"cons-{split}"):
        for f in glob.glob(str(R / d / "*.json")):
            j = json.load(open(f))
            if re.fullmatch(NEW, j["name"]):
                out[j["name"]] = j
    return out


def cells(res, pattern, keys):
    out = {}
    for name, d in res.items():
        m = re.fullmatch(pattern, name)
        if m:
            for k in keys:
                if k in d["metrics"]:
                    out.setdefault(tuple(m.groupdict().values()), {}).setdefault(k, []).append(d["metrics"][k])
    return out


def curves(c, key):
    """{scale: [(steps, mean, sd), ...]} sorted by steps."""
    out = {}
    for (s, ck), v in c.items():
        if key in v:
            out.setdefault(s, []).append((int(ck) * 1000, *p.agg(v[key])))
    return {s: sorted(v) for s, v in out.items()}


def fig_scale(new, old):
    keys = [k for k, _ in p.METRICS]
    c = cells(new, NEW, keys)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, name) in zip(axes, p.METRICS):
        for ck, style in ((220, dict(ls="-", mfc=None)), (540, dict(ls=":", mfc="white"))):
            pts = [(s, p.agg(c[(s, f"{ck:03d}")][key])) for s in ORDER if (s, f"{ck:03d}") in c]
            x = [PARAMS[s] for s, _ in pts]; y = [m for _, (m, _) in pts]
            ax.errorbar(x, y, yerr=[sd for _, (_, sd) in pts], color=p.MUTED, lw=1.3, ls=style["ls"],
                        capsize=3, zorder=2, label=f"checkpoint {ck}k")
            for (s, _), xi, yi in zip(pts, x, y):
                ax.scatter([xi], [yi], s=55, zorder=3, edgecolor=COLOUR[s],
                           color=COLOUR[s] if style["mfc"] is None else "white", linewidth=1.5)
            p.label(ax, x, y, below=(ck == 540))
        zs = p.zeroshot_best(key)
        zx = [PARAMS[s] for s in ORDER if s in zs]; zy = [zs[s] for s in ORDER if s in zs]
        if zx:
            ax.plot(zx, zy, ls="--", lw=1.1, color=p.MUTED, marker="o", mfc="white", ms=6,
                    label="frozen, best layer (220k)")
        p.references(ax, old, key)
        ticks = [PARAMS[s] for s in ORDER]
        ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels(ORDER); ax.minorticks_off()
        p.finish(ax, "model size", name, f"{name}: consensus recipe, checkpoint 220k (solid) and 540k (open)")
    fig.tight_layout(); out = FIGS / "c100k_cons_scale.png"; fig.savefig(out); plt.close(fig)
    return out


def draw_curves(ax, cv, dashed=False, labels=True):
    for s in ORDER:
        if s not in cv:
            continue
        x = [t for t, _, _ in cv[s]]; y = [m for _, m, _ in cv[s]]
        ax.errorbar(x, y, yerr=[sd for _, _, sd in cv[s]], marker="o", ms=5 if not dashed else 4.5,
                    lw=1.6 if not dashed else 1.2, ls="--" if dashed else "-", capsize=3, color=COLOUR[s],
                    mfc="white" if dashed else COLOUR[s], label=(f"{s}" + (" (paper)" if dashed else "")))
        if labels:  # first and last point only: the intermediate checkpoints sit too close to label
            ends = [0, len(x) - 1] if len(x) > 1 else [0]
            p.label(ax, [x[i] for i in ends], [y[i] for i in ends], below=(s in ("50m", "25m")))


def end_labels(ax, cv):
    """Value labels beside the first and last point of each scale's curve (zoomed panels)."""
    for s in ORDER:
        if s not in cv:
            continue
        (x0, y0, _), (x1, y1, _) = cv[s][0], cv[s][-1]
        ax.annotate(f"{y1:.3f}", (x1, y1), textcoords="offset points", xytext=(8, -3), fontsize=7.2,
                    color=COLOUR[s])
        if len(cv[s]) > 1:
            ax.annotate(f"{y0:.3f}", (x0, y0), textcoords="offset points", xytext=(-8, -3), ha="right",
                        fontsize=7.2, color=COLOUR[s])


def fig_checkpoint(new, old):
    """Top row: the paper's framing (full axis with reference lines). Bottom row: the same curves zoomed."""
    c = cells(new, NEW, [k for k, _ in p.METRICS])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    for col, (key, name) in enumerate(p.METRICS):
        cv = curves(c, key)
        ax = axes[0][col]
        draw_curves(ax, cv, labels=False)
        p.references(ax, old, key)
        p.finish(ax, "pretraining grad steps", name, f"{name}: consensus recipe")
        ax = axes[1][col]
        draw_curves(ax, cv, labels=False)
        end_labels(ax, cv)
        p.finish(ax, "pretraining grad steps", name, f"{name}: zoomed")
        ax.margins(x=0.12)
    fig.tight_layout(); out = FIGS / "c100k_cons_checkpoint.png"; fig.savefig(out); plt.close(fig)
    return out


def fig_filters(new, old):
    rows = (("experimental/{f}/full/MAP@R", "experimental MAP@R"), ("library/{f}/full/Hit@1", "library search Hit@1"))
    keys = [r.format(f=f) for r, _ in rows for f, _ in FILTERS]
    c = cells(new, NEW, keys)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r, (row, rname) in enumerate(rows):
        for col, (f, fname) in enumerate(FILTERS):
            ax = axes[r][col]; key = row.format(f=f)
            draw_curves(ax, curves(c, key), labels=False)
            if f == "open" and r == 0:
                p.references(ax, old, "experimental/MAP@R")
            p.finish(ax, "pretraining grad steps", rname,
                     f"{rname}, {fname}" + ("" if (f == "open" and r == 0) else " (y axis zoomed)"))
    fig.suptitle("Consensus recipe, ms-contrastive-100k test: no query fails the 20 ppm filter here, so there is no "
                 "pass/fail split", x=0.01, ha="left", fontsize=9, color=p.MUTED)
    fig.tight_layout(); out = FIGS / "c100k_cons_filters.png"; fig.savefig(out); plt.close(fig)
    return out


def fig_old_vs_new(new, old):
    keys = [k for k, _ in p.METRICS]
    cn = cells(new, NEW, keys)
    co = cells(old, r"c[24]_s0*(?P<scale>\d+m)_ck(?P<ck>\d+)k_seed\d", keys)
    c7 = cells(old, r"c7b_cont0*(?P<scale>\d+m)_final_seed\d", keys)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (key, name) in zip(axes, p.METRICS):
        draw_curves(ax, curves(co, key), dashed=True, labels=False)
        draw_curves(ax, curves(cn, key), labels=False)
        for (s,), v in c7.items():
            if key in v:
                m, sd = p.agg(v[key])
                ax.errorbar([220_000], [m], yerr=[sd], marker="*", ms=13, color=COLOUR[s], mec=p.INK, mew=0.6,
                            ls="none", capsize=3, label=f"{s} C7 (paper's best: + 1 epoch ms-contrastive-100k)")
        p.references(ax, old, key)
        p.finish(ax, "pretraining grad steps", name, f"{name}: new recipe (solid) vs paper (dashed)")
        ax.legend(frameon=False, fontsize=6.8, ncol=2)
    fig.suptitle("Paper: replicate-corpus-only fine-tunes (C2/C4). New: ms-contrastive-100k + consensus, lr 4e-4, "
                 "P170xK2 (K163/K188). Same 25,137 test queries.", x=0.01, ha="left", fontsize=8.5, color=p.MUTED)
    fig.tight_layout(); out = FIGS / "c100k_old_vs_new.png"; fig.savefig(out); plt.close(fig)
    return out


def main() -> int:
    old = p.load()
    new = load_new()
    for f in (fig_scale, fig_checkpoint, fig_filters, fig_old_vs_new):
        print(f(new, old))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
