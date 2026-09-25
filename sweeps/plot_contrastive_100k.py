"""Contrastive figures on the ms-contrastive-100k test split (Stage 2 / 3 of PLAN.md).

    python sweeps/plot_contrastive_100k.py

Reads the per-model JSONs that msdelta.eval_grouped_retrieval wrote under
results/finetune/contrastive/grouped100k-test/ and draws, for experimental-spectrum
MAP@R (the headline) and Hit@1:

  c100k_scale.png        C2: model size at the frozen recipe (220k), replicate corpus
  c100k_checkpoint.png   C4 + C2: pretraining grad steps, one line per scale
  c100k_c7.png           C7: continuing on ms-contrastive-100k, x = steps into its epoch
  c100k_transfer.png     small replicate eval vs this test, every Stage-1 model

Every panel carries the untrained-encoder and binned-cosine references as horizontal
lines. Points are 3-seed means with seed sd as error bars, labelled with the mean.
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
RESULTS = REPO / "results" / "finetune" / "contrastive" / "grouped100k-test"
FIGS = REPO / "results" / "figures" / "C_contrastive"
RUNS = "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs"
INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"
SCALE_COLOUR = {"50m": "#93c5fd", "100m": "#60a5fa", "200m": "#2563eb", "400m": "#1e3a8a"}
ORDER = ["50m", "100m", "200m", "400m"]
PARAMS = {"50m": 50, "100m": 100, "200m": 200, "400m": 400}
METRICS = (("experimental/MAP@R", "MAP@R"), ("experimental/Hit@1", "Hit@1"))
BASELINE_STYLE = {"binned_w0.1": ("binned cosine 0.1 Da", "#b45309", "--"),
                  "binned_w1.0005": ("binned cosine 1 Da", "#d97706", ":")}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.7, "figure.facecolor": "white",
})


def load() -> dict[str, dict]:
    out = {}
    for f in glob.glob(str(RESULTS / "*.json")):
        d = json.load(open(f))
        out[d["name"]] = d
    return out


def agg(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def cells(res, pattern):
    """regex with named groups -> {groupdict tuple: {metric: [values]}}"""
    c = collections.defaultdict(lambda: collections.defaultdict(list))
    for name, d in res.items():
        m = re.fullmatch(pattern, name)
        if not m:
            continue
        for key, _ in METRICS:
            if key in d["metrics"]:
                c[tuple(m.groupdict().values())][key].append(d["metrics"][key])
    return c


def label(ax, xs, ys, below=False):
    """Mean above the point; below it for the lower of two near-overlapping lines."""
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, -13 if below else 7), ha="center", fontsize=7.2,
                    color=MUTED)


def references(ax, res, key):
    for name, (text, colour, style) in BASELINE_STYLE.items():
        if name in res:
            ax.axhline(res[name]["metrics"][key], color=colour, ls=style, lw=1.2,
                       label=text)
    base = [d["metrics"][key] for n, d in res.items()
            if n.startswith("base_") and key in d["metrics"]]
    if base:
        ax.axhspan(min(base), max(base), color="#9ca3af", alpha=0.18,
                   label="frozen, final layer")


def finish(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", pad=12)
    ax.grid(); ax.set_axisbelow(True); ax.margins(x=0.15, y=0.12)
    ax.legend(frameon=False, fontsize=7.8)


def fig_scale(res):
    c = cells(res, r"c2_s0*(?P<scale>\d+m)_ck220k_seed\d")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, name) in zip(axes, METRICS):
        pts = [(s, agg(c[(s,)][key])) for s in ORDER if (s,) in c]
        x = [PARAMS[s] for s, _ in pts]; y = [p[0] for _, p in pts]
        ax.errorbar(x, y, yerr=[p[1] for _, p in pts], color=MUTED, lw=1.4, capsize=3,
                    zorder=2)
        for (s, _), xi, yi in zip(pts, x, y):
            ax.scatter([xi], [yi], s=55, color=SCALE_COLOUR[s], zorder=3, label=s)
        label(ax, x, y)
        # "before": the frozen pretrained encoder's best block at 220k (zero-shot redo)
        zs = zeroshot_best(key)
        zx = [PARAMS[s] for s in ORDER if s in zs]; zy = [zs[s] for s in ORDER if s in zs]
        if zx:
            ax.plot(zx, zy, ls="--", lw=1.1, color=MUTED, marker="o", mfc="white",
                    ms=6, label="frozen, best layer")
            label(ax, zx, zy, below=True)
        references(ax, res, key)
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([s for s, _ in pts])
        finish(ax, "model size", name, f"{name}, checkpoint 220k")
    fig.tight_layout(); out = FIGS / "c100k_scale.png"; fig.savefig(out); plt.close(fig)
    return out


def zeroshot_best(key, ck="220k"):
    """Frozen encoder's best-block score per scale (eval_zeroshot_layers output)."""
    out = {}
    for f in glob.glob(str(REPO / "results" / "finetune" / "contrastive" / "zeroshot-layers"
                           / f"zs_*_ck{ck}.json")):
        d = json.load(open(f))
        m = re.match(r"zs_0*(\d+m)_ck", d["name"])
        vals = [v[key] for v in d["layers"].values() if key in v]
        if m and vals:
            out[m.group(1)] = max(vals)
    return out


def fig_checkpoint(res):
    c = cells(res, r"c[24]_s0*(?P<scale>\d+m)_ck(?P<ck>\d+)k_seed\d")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, name) in zip(axes, METRICS):
        for s in ORDER:
            pts = sorted((int(ck) * 1000, agg(v[key])) for (sc, ck), v in c.items()
                         if sc == s and key in v)
            if len(pts) < 2:
                continue
            x = [p[0] for p in pts]; y = [p[1][0] for p in pts]
            ax.errorbar(x, y, yerr=[p[1][1] for p in pts], marker="o", ms=5, lw=1.6,
                        capsize=3, color=SCALE_COLOUR[s], label=s)
            label(ax, x, y, below=(s == "50m"))
        references(ax, res, key)
        finish(ax, "pretraining grad steps", name, name)
    fig.tight_layout(); out = FIGS / "c100k_checkpoint.png"; fig.savefig(out); plt.close(fig)
    return out


def fig_c7(res):
    c = cells(res, r"c7b_cont0*(?P<scale>\d+m)_(?P<step>s\d+|final)_seed\d")
    start = {"50m": cells(res, r"s050m_t0002_pk256_ep24_seed\d"),
             "400m": cells(res, r"s400m_t0002_pk256_ep12_seed\d")}
    total = 1062
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, name) in zip(axes, METRICS):
        for s in ("50m", "400m"):
            pts = [(0, agg(v[key])) for v in start[s].values() if key in v]
            pts += sorted((total if st == "final" else int(st[1:]), agg(v[key]))
                          for (sc, st), v in c.items() if sc == s and key in v)
            if len(pts) < 2:
                continue
            x = [p[0] for p in pts]; y = [p[1][0] for p in pts]
            ax.errorbar(x, y, yerr=[p[1][1] for p in pts], marker="o", ms=5, lw=1.6,
                        capsize=3, color=SCALE_COLOUR[s], label=s)
            label(ax, x, y, below=(s == "50m"))
        references(ax, res, key)
        finish(ax, "steps on ms-contrastive-100k (1,062 = one epoch)", name, name)
    fig.tight_layout(); out = FIGS / "c100k_c7.png"; fig.savefig(out); plt.close(fig)
    return out


def fig_transfer(res):
    small, big, colour = [], [], []
    for name, d in res.items():
        path = d.get("path", "")
        if not path.endswith("/final") or "/runs/sweep-" not in path:
            continue
        p = path[: -len("/final")] + "/all_results.json"
        if not os.path.exists(p):
            continue
        a = json.load(open(p))
        if "retrieval/MAP@R" not in a or "c7b_" in name or "c9" in name:
            continue
        m = re.search(r"s0*(\d+m)", name)
        small.append(a["retrieval/MAP@R"]); big.append(d["metrics"]["experimental/MAP@R"])
        colour.append(SCALE_COLOUR.get(m.group(1) if m else "", MUTED))
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.scatter(small, big, s=22, c=colour, alpha=0.85)
    for s in ORDER:
        ax.scatter([], [], s=22, color=SCALE_COLOUR[s], label=s)
    if len(small) > 2:
        from scipy.stats import spearmanr
        rho = spearmanr(small, big)[0]
        ax.set_title(f"n={len(small)}, Spearman {rho:.2f}", loc="left", pad=12)
    finish(ax, "MAP@R, replicate-corpus eval (99 groups)",
           "exp MAP@R, ms-contrastive-100k test", ax.get_title(loc="left"))
    fig.tight_layout(); out = FIGS / "c100k_transfer.png"; fig.savefig(out); plt.close(fig)
    return out


def fig_zeroshot_layers(res=None):
    """Frozen pretrained encoders: exp MAP@R at every depth (eval_zeroshot_layers)."""
    import matplotlib.lines as mlines
    files = sorted(glob.glob(str(REPO / "results" / "finetune" / "contrastive"
                                 / "zeroshot-layers" / "zs_*.json")))
    if not files:
        return None
    key = "experimental/MAP@R"
    style = {"010k": ":", "220k": "-", "430k": "--", "540k": "--"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 4.4),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    best = collections.defaultdict(dict)
    for f in files:
        d = json.load(open(f))
        m = re.match(r"zs_0*(\d+m)_ck(\d+k)", d["name"])
        scale, ck = m.group(1), m.group(2)
        blocks = sorted(k for k in d["layers"] if k.startswith("block"))
        y = [d["layers"][b][key] for b in blocks]
        x = [i / (len(blocks) - 1) for i in range(len(blocks))]
        ax.plot(x, y, ls=style.get(ck, "-"), lw=1.6, color=SCALE_COLOUR[scale])
        i = int(np.argmax(y))
        ax.scatter([x[i]], [y[i]], s=22, color=SCALE_COLOUR[scale], zorder=3)
        best[ck][scale] = (y[i], d["layers"]["final"][key])
    handles = [mlines.Line2D([], [], color=SCALE_COLOUR[s], lw=2, label=s) for s in ORDER]
    handles += [mlines.Line2D([], [], color=MUTED, ls=style[c], label=f"ckpt {c}")
                for c in ("010k", "220k", "540k")]
    handles[-1].set_label("ckpt 540k (400m: 430k)")
    ax.legend(handles=handles, frameon=False, fontsize=7.8, ncol=2)
    ax.set_xlabel("relative depth (block / number of blocks)"); ax.set_ylabel("MAP@R")
    ax.set_title("frozen encoders, every block", loc="left", pad=12)
    ax.grid(); ax.set_axisbelow(True)
    for ck, marker in (("220k", "o"), ("latest", "s")):
        pts = best["220k"] if ck == "220k" else {**best.get("540k", {}), **best.get("430k", {})}
        xs = [PARAMS[s] for s in ORDER if s in pts]
        bx.plot(xs, [pts[s][0] for s in ORDER if s in pts], marker=marker, lw=1.5,
                color=INK if ck == "220k" else MUTED,
                label=f"best block, {'220k' if ck == '220k' else 'latest ckpt'}")
        bx.plot(xs, [pts[s][1] for s in ORDER if s in pts], marker=marker, lw=1, ls=":",
                color=INK if ck == "220k" else MUTED,
                label=f"final layer, {'220k' if ck == '220k' else 'latest ckpt'}")
        label(bx, xs, [pts[s][0] for s in ORDER if s in pts])
    bx.set_xscale("log"); bx.set_xticks([PARAMS[s] for s in ORDER])
    bx.set_xticklabels(ORDER)
    finish(bx, "model size", "MAP@R", "best block vs final layer")
    fig.tight_layout(); out = FIGS / "c100k_zeroshot_layers.png"
    fig.savefig(out); plt.close(fig)
    return out


def fig_zeroshot_abtt(res=None):
    """Frozen encoders with all-but-the-top (Mu & Viswanath 2018): mean + top-D principal
    directions (fitted on TRAIN spectra) removed before cosine. eval_zeroshot_layers --abtt.
    Also writes the numbers to zeroshot-layers-abtt/summary.csv."""
    import csv
    import matplotlib.lines as mlines
    base = REPO / "results" / "finetune" / "contrastive" / "zeroshot-layers-abtt"
    files = sorted(glob.glob(str(base / "zs_*.json")))
    if not files:
        return None
    key = "experimental/MAP@R"
    style = {"010k": ":", "220k": "-", "430k": "--", "540k": "--"}
    rows = []
    for f in files:
        d = json.load(open(f))
        m = re.match(r"zs_0*(\d+m)_ck(\d+k)", d["name"])
        scale, ck = m.group(1), m.group(2)
        raw = {k: v[key] for k, v in d["layers"].items()}
        blocks = sorted(k for k in raw if k.startswith("block"))
        best_block = max(blocks, key=raw.get)
        tr = d["abtt"]["train"]
        ds = sorted((k for k in tr if k != "center"), key=int)
        curve = [raw[best_block], tr["center"][best_block][key]] + [tr[D][best_block][key] for D in ds]
        ab_final = max(tr[D]["final"][key] for D in ds)
        ab_best, ab_d, ab_layer = max((tr[D][b][key], D, b) for D in ds for b in blocks + ["final"])
        rows.append(dict(scale=scale, ckpt=ck, best_block=best_block, raw_final=raw["final"],
                         raw_best=raw[best_block], abtt_final=ab_final, abtt_best=ab_best,
                         abtt_best_D=int(ab_d), abtt_best_layer=ab_layer, curve=curve, Ds=ds))
    with open(base / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scale", "ckpt", "raw_final", "raw_best_block", "best_block",
                    "abtt_final", "abtt_best", "abtt_best_D", "abtt_best_layer"])
        for r in rows:
            w.writerow([r["scale"], r["ckpt"], f"{r['raw_final']:.4f}", f"{r['raw_best']:.4f}",
                        r["best_block"], f"{r['abtt_final']:.4f}", f"{r['abtt_best']:.4f}",
                        r["abtt_best_D"], r["abtt_best_layer"]])
    rows.sort(key=lambda r: (ORDER.index(r["scale"]), r["ckpt"]))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(rows)); w = 0.2
    for i, (k, lab, alpha) in enumerate((("raw_final", "raw, final layer", 0.35),
                                         ("raw_best", "raw, best block", 0.6),
                                         ("abtt_final", "ABTT, final layer", 0.8),
                                         ("abtt_best", "ABTT, best block (best D)", 1.0))):
        ax.bar(x + (i - 1.5) * w, [r[k] for r in rows], w, alpha=alpha,
               color=[SCALE_COLOUR[r["scale"]] for r in rows],
               edgecolor=INK if k.startswith("abtt") else "none", linewidth=0.6, label=lab)
    for yref, lab in ((0.730, "binned cosine 0.730"), (0.868, "trained C7 400m 0.868")):
        ax.axhline(yref, color=MUTED, ls="--", lw=1)
        ax.text(len(rows) - 0.5, yref + 0.01, lab, ha="right", fontsize=7.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels([f"{r['scale']}\n@{r['ckpt']}" for r in rows], fontsize=8)
    ax.set_ylabel("MAP@R (experimental)"); ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left", bbox_to_anchor=(0, 0.66))
    ax.set_title("frozen encoders: raw vs all-but-the-top", loc="left", pad=10)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    for r in rows:
        xs = np.arange(len(r["curve"]))
        bx.plot(xs, r["curve"], marker="o", ms=3, lw=1.5, ls=style.get(r["ckpt"], "-"),
                color=SCALE_COLOUR[r["scale"]])
    labels = ["raw", "centre"] + [f"D={D}" for D in rows[0]["Ds"]]
    bx.set_xticks(range(len(labels))); bx.set_xticklabels(labels, fontsize=8)
    handles = [mlines.Line2D([], [], color=SCALE_COLOUR[s], lw=2, label=s) for s in ORDER
               if any(r["scale"] == s for r in rows)]
    handles += [mlines.Line2D([], [], color=MUTED, ls=style[c], label=f"ckpt {c}")
                for c in ("010k", "220k", "540k")]
    finish(bx, "top principal directions removed (fit on train)", "MAP@R, best block",
           "best block vs D")
    bx.legend(handles=handles, frameon=False, fontsize=7.5, ncol=2, loc="lower right")
    fig.tight_layout(); out = FIGS / "c100k_zeroshot_abtt.png"
    fig.savefig(out); plt.close(fig)
    return out


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    res = load()
    if not res:
        raise SystemExit(f"no results under {RESULTS}")
    print(f"  {len(res)} scored models")
    for f in (fig_scale, fig_checkpoint, fig_c7, fig_transfer, fig_zeroshot_layers,
              fig_zeroshot_abtt):
        out = f(res)
        if out is not None:
            print(f"  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
