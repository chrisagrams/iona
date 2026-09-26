"""Package the C (contrastive spectrum embedding) results: one long CSV + four figures.

    .venv/bin/python sweeps/package_contrastive.py

Everything is read from the per-run result JSONs (results/finetune/contrastive/<benchmark>/)
and GLEAMS's metrics (baselines/<benchmark>/gleams_metrics.json on Lustre); nothing is typed
in. Metric: experimental-spectrum MAP@R (and Hit@1). Plotting only (login node).

    results/contrastive_results.csv                      every run on every benchmark
    results/contrastive_zeroshot.csv                     frozen encoders, raw vs ABTT
    results/figures/SUMMARY/C_benchmarks.png             ours vs GLEAMS vs binned, 4 benchmarks
    results/figures/SUMMARY/C_pretraining_scaling.png    replicate-corpus recipe x checkpoint x size
    results/figures/SUMMARY/C_zeroshot.png               frozen encoders, raw vs ABTT
    results/figures/SUMMARY/C_transfer.png               C7 fine-tuning trajectory, in-dist vs unseen
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
RES = REPO / "results" / "finetune" / "contrastive"
BASE = Path("/lus/flare/projects/UIC-HPC/khuss/msdelta/baselines")
FIG = REPO / "results" / "figures" / "SUMMARY"
# benchmark key -> (result dir, GLEAMS dir, label)
BENCH = {
    "ms-contrastive-100k": ("grouped100k-test", "gleams", "ms-contrastive-100k test\n(in-distribution)"),
    "hek-lowres": ("c11_cap20", "c11_cap20", "HEK\n(unseen, low-res MS2)"),
    "yeast-full": ("nine_yeast", "nine_yeast", "nine-species yeast\n(unseen, high-res)"),
    "yeast-20k": ("nine20k", "nine_yeast20k", "yeast 20k subset\n(unseen, high-res)"),
    "oodval-8species": ("oodval20k", None, "8 other species\n(OOD validation)"),
}
INK, MUTED, GRIDC = "#1f2937", "#6b7280", "#e5e7eb"
OURS, OURS_DARK, OURS_LIGHT, GLEAMS, BINNED = "#2563eb", "#1e3a8a", "#93c5fd", "#f59e0b", "#9ca3af"
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9ca3af",
         "axes.linewidth": 0.8, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
         "savefig.bbox": "tight", "figure.dpi": 200}


BIN = {"1.0005": "1 Da", "0.1": "0.1 Da"}   # 1.0005 Da is the standard 1 Da bin width


def describe(name):
    """Run file stem -> (method, scale, pretrain ckpt, stage/step, seed)."""
    seed = re.search(r"_seed(\d+)$", name)
    seed = int(seed.group(1)) if seed else ""
    stem = re.sub(r"_seed\d+$", "", name)
    if m := re.match(r"c7b_cont(\d+m)_(s\d+|final)$", stem):
        return "C7 (replicate corpus -> ms-contrastive-100k)", m.group(1).lstrip("0"), "220k", m.group(2), seed
    if m := re.match(r"s(\d+m)_t\d+_pk\d+_ep(\d+)$", stem):
        return f"replicate corpus only ({int(m.group(2))} ep)", m.group(1).lstrip("0"), "220k", "final", seed
    if m := re.match(r"c[24]_s(\d+m)_ck(\d+k)$", stem):
        return "replicate corpus only (24 ep, C2/C4)", m.group(1).lstrip("0"), m.group(2).lstrip("0"), "final", seed
    if m := re.match(r"base_s(\d+m)_ck(\d+k)$", stem):
        return "frozen encoder (final layer)", m.group(1).lstrip("0"), m.group(2).lstrip("0"), "", seed
    if m := re.match(r"binned_w([\d.]+)$", stem):
        return f"binned cosine ({BIN[m.group(1)]} bins)", "", "", "", seed
    if m := re.match(r"pca_w([\d.]+)_d(\d+)$", stem):
        return f"binned + PCA ({BIN[m.group(1)]}, d={m.group(2)})", "", "", "", seed
    if m := re.match(r"c9q?_s(\d+m)(?:_ck(\d+k))?_head\d+(?:_ep(\d+))?$", stem):
        return f"replicate corpus + projection head (C9, {int(m.group(3) or 24)} ep)", m.group(1).lstrip("0"), "220k", "final", seed
    return stem, "", "", "", seed


def exp_metrics(d):
    m = d.get("metrics", d)
    return m.get("experimental/MAP@R"), m.get("experimental/Hit@1"), m.get("experimental/queries")


def load_rows():
    rows = []
    for key, (sub, gdir, _) in BENCH.items():
        for f in sorted((RES / sub).glob("*.json")):
            d = json.loads(f.read_text())
            mapr, hit1, q = exp_metrics(d)
            if mapr is None:
                continue
            method, scale, ck, stage, seed = describe(f.stem)
            rows.append(dict(benchmark=key, method=method, scale=scale, pretrain_ckpt=ck, stage=stage,
                             seed=seed, map_at_r=mapr, hit_at_1=hit1, queries=q,
                             source=str(f.relative_to(REPO))))
        if gdir and (BASE / gdir / "gleams_metrics.json").exists():
            g = json.loads((BASE / gdir / "gleams_metrics.json").read_text())["mp512/experimental"]
            rows.append(dict(benchmark=key, method="GLEAMS (pretrained)", scale="", pretrain_ckpt="", stage="",
                             seed="", map_at_r=g["cos_MAP@R"], hit_at_1=g["cos_Hit@1"], queries=g["queries"],
                             source=f"{BASE / gdir / 'gleams_metrics.json'}"))
    return rows


def pick(rows, bench, method, scale="", ck="", stage="", seed=None):
    return [r["map_at_r"] for r in rows if r["benchmark"] == bench and r["method"] == method
            and (not scale or r["scale"] == scale) and (not ck or r["pretrain_ckpt"] == ck)
            and (not stage or r["stage"] == stage) and (seed is None or r["seed"] == seed)]


# End-of-epoch seed chosen on the ms-contrastive-100k VALIDATION split (grouped100k-validation MAP@R:
# 400M seeds 0.8639 / 0.8629 / 0.8634 -> seed 0; 50M 0.8314 / 0.8357 / 0.8342 -> seed 1).
BEST_SEED = {"400m": 0, "50m": 1}
# "replicate corpus only" = the stage-1 model each fine-tuned model started from
# (configs/sweep-con100k-best: 400M from the 12-epoch run, 50M from the 24-epoch run).
STAGE1 = {"400m": "replicate corpus only (12 ep)", "50m": "replicate corpus only (24 ep)"}


def msd(v):
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def fig_benchmarks(rows):
    order = ["ms-contrastive-100k", "hek-lowres", "yeast-full", "yeast-20k"]
    series = [("ours: C7 400M", OURS, lambda b: pick(rows, b, "C7 (replicate corpus -> ms-contrastive-100k)", "400m", stage="final")),
              ("ours: replicate corpus only 400M", OURS_LIGHT, lambda b: pick(rows, b, "replicate corpus only (12 ep)", "400m")),
              ("GLEAMS", GLEAMS, lambda b: pick(rows, b, "GLEAMS (pretrained)")),
              ("binned cosine (best bin width)", BINNED, lambda b: [max(pick(rows, b, "binned cosine (1 Da bins)")
                                                                       + pick(rows, b, "binned cosine (0.1 Da bins)"))])]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4.4))
        ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
        w = 0.19
        for i, (lab, col, get) in enumerate(series):
            for j, b in enumerate(order):
                v = get(b)
                if not v:
                    continue
                m, s = msd(v); x = j + (i - 1.5) * w
                ax.bar(x, m, w * 0.92, color=col, yerr=s if len(v) > 1 else None, capsize=3,
                       error_kw=dict(elinewidth=1, capthick=1, ecolor=INK), zorder=2)
                ax.text(x, m + (s if len(v) > 1 else 0) + 0.012, f"{m:.2f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([BENCH[b][2] for b in order])
        ax.set_ylim(0, 1.0); ax.set_ylabel("MAP@R (experimental spectra)")
        ax.set_title("Spectrum retrieval: ours vs GLEAMS and binned cosine", loc="left", fontsize=13,
                     fontweight="bold", pad=12)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in series],
                  labels=[l for l, _, _ in series], frameon=False, fontsize=8.5, ncol=4,
                  loc="upper center", bbox_to_anchor=(0.5, -0.14))
        fig.text(0.01, -0.13, "Ours: mean ± sd over 3 seeds (replicate-only on yeast 20k: 1 seed). Binned cosine: better of 1 Da and 0.1 Da bins per benchmark.",
                 fontsize=8, color=MUTED)
        fig.savefig(FIG / "C_benchmarks.png"); plt.close(fig)


def fig_pretraining(rows):
    b, meth = "ms-contrastive-100k", "replicate corpus only (24 ep, C2/C4)"
    cks = [10, 120, 220, 330, 430, 540]
    cols = {"50m": "#93c5fd", "100m": "#3b82f6", "200m": "#1d4ed8", "400m": "#1e3a8a"}
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
        handles = []
        mk = {"50m": "o", "100m": "o", "200m": "D", "400m": "s"}
        dx = {"50m": 0, "100m": 0, "200m": -14, "400m": 14}
        for sc in ["400m", "200m", "100m", "50m"]:
            pts = [(c, pick(rows, b, meth, sc, f"{c}k")) for c in cks]
            pts = [(c, msd(v)) for c, v in pts if v]
            x = [p[0] + dx[sc] for p in pts]; m = [p[1][0] for p in pts]; s = [p[1][1] for p in pts]
            ax.errorbar(x, m, yerr=s, color=cols[sc], lw=2.0, marker=mk[sc], ms=7, mec="white", mew=1.2,
                        capsize=3.5, elinewidth=1.2, zorder=4 if len(pts) == 1 else 3)
            handles.append(Line2D([], [], color=cols[sc], lw=2 if len(pts) > 1 else 0, marker=mk[sc], ms=7, mec="white", mew=1.2,
                                  label=sc.upper() + ("" if len(pts) > 1 else "  (220k only)")))
        ax.set_xticks(cks); ax.set_xticklabels([f"{c}k" for c in cks])
        ax.set_xlabel("pretraining steps of the starting checkpoint", labelpad=6)
        ax.set_ylabel("MAP@R, ms-contrastive-100k test", labelpad=6)
        ax.set_title("Pretraining and scale improve spectrum retrieval", loc="left", fontsize=13,
                     fontweight="bold", pad=12)
        ax.legend(handles=handles, title="model size", frameon=False, fontsize=9, title_fontsize=9.5,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.text(0.01, -0.04, "Replicate-corpus recipe (24 epochs), mean ± sd over 3 seeds.",
                 fontsize=8, color=MUTED)
        fig.savefig(FIG / "C_pretraining_scaling.png"); plt.close(fig)


def zeroshot_rows():
    out = []
    with open(RES / "zeroshot-layers-abtt" / "summary.csv") as fh:
        for r in csv.DictReader(fh):
            out.append(dict(benchmark="ms-contrastive-100k", encoder=f"{r['scale']}@{r['ckpt'].lstrip('0')}",
                            raw_final=float(r["raw_final"]), raw_best_layer=float(r["raw_best_block"]),
                            abtt_best=float(r["abtt_best"]), abtt_D=r["abtt_best_D"], abtt_layer=r["abtt_best_layer"],
                            abtt_fit="ms-contrastive-100k train"))
    for f in sorted((RES / "nine20k_zeroshot").glob("zs_*.json")):
        d = json.loads(f.read_text())
        sc, ck = re.match(r"zs_0*(\d+m)_ck0*(\d+k)", f.stem).groups()
        tr = d["abtt"]["train"]
        raw = tr["center"] if "layers" not in d else d["layers"]
        raw_best = max(v["experimental/MAP@R"] for v in d.get("layers", tr["center"]).values())
        raw_final = d.get("layers", tr["center"]).get("final", {}).get("experimental/MAP@R", float("nan"))
        best = max(((D, lay, v["experimental/MAP@R"]) for D, L in tr.items() if D != "center"
                    for lay, v in L.items()), key=lambda t: t[2])
        out.append(dict(benchmark="yeast-20k", encoder=f"{sc}@{ck}", raw_final=raw_final, raw_best_layer=raw_best,
                        abtt_best=best[2], abtt_D=best[0], abtt_layer=best[1], abtt_fit="8 other species (train split)"))
    return out


def fig_zeroshot(zs):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for ax, bench, title in ((axes[0], "ms-contrastive-100k", "ms-contrastive-100k test (in-distribution)"),
                                 (axes[1], "yeast-20k", "yeast 20k subset (unseen)")):
            ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
            rs = {r["encoder"]: r for r in zs if r["benchmark"] == bench}
            enc = sorted(rs, key=lambda e: (int(e.split("m@")[0]), int(e.split("@")[1][:-1])))
            x = np.arange(len(enc))
            ax.bar(x - 0.2, [rs[e]["raw_best_layer"] for e in enc], 0.38, color=BINNED, label="frozen, best layer", zorder=2)
            ax.bar(x + 0.2, [rs[e]["abtt_best"] for e in enc], 0.38, color=OURS, label="frozen, best layer + ABTT", zorder=2)
            for i, e in enumerate(enc):
                ax.text(i + 0.2, rs[e]["abtt_best"] + 0.01, f"{rs[e]['abtt_best']:.2f}", ha="center", fontsize=7.5)
            ax.set_xticks(x); ax.set_xticklabels([e.upper().replace("K", "k") for e in enc], rotation=30, ha="right")
            ax.set_title(title, loc="left", fontsize=10.5)
        axes[0].set_ylabel("MAP@R (experimental spectra)")
        axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
        fig.suptitle("Zero-shot retrieval from the pretrained encoder (no contrastive training)", x=0.07, ha="left",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.savefig(FIG / "C_zeroshot.png"); plt.close(fig)


def fig_transfer(rows, zs, panels=None, baselines=True, out="C_transfer.png",
                 title="Our models vs baselines, in-distribution and on unseen data", labels=None):
    """Per benchmark: every model of ours scored there (optionally vs GLEAMS and binned cosine)."""
    c7 = "C7 (replicate corpus -> ms-contrastive-100k)"
    def rep(b, sc):
        return pick(rows, b, STAGE1[sc], sc)
    def frozen(b):
        r = [r for r in zs if r["benchmark"] == b]
        if not r:
            return [], ""
        best = max(r, key=lambda t: t["abtt_best"])
        return [best["abtt_best"]], best["encoder"].upper().replace("K", "k")
    series = [("fine-tuned 400M", "#1e3a8a", lambda b: (pick(rows, b, c7, "400m", stage="final", seed=BEST_SEED["400m"]), "")),
              ("fine-tuned 50M", "#93c5fd", lambda b: (pick(rows, b, c7, "50m", stage="final", seed=BEST_SEED["50m"]), "")),
              ("replicate corpus only 400M", "#0d9488", lambda b: (rep(b, "400m"), "")),
              ("replicate corpus only 50M", "#5eead4", lambda b: (rep(b, "50m"), "")),
              ("frozen + ABTT (best encoder)", "#7c3aed", frozen),
              ("GLEAMS", GLEAMS, lambda b: (pick(rows, b, "GLEAMS (pretrained)"), "")),
              ("binned cosine", BINNED, lambda b: ([max(pick(rows, b, "binned cosine (1 Da bins)")
                                                       + pick(rows, b, "binned cosine (0.1 Da bins)"))]
                                                  if pick(rows, b, "binned cosine (1 Da bins)") + pick(rows, b, "binned cosine (0.1 Da bins)") else [], ""))]
    panels = panels or [("ms-contrastive-100k", "ms-contrastive-100k test\n(in-distribution)", 8),
                        ("oodval-8species", "8 other species\n(unseen, validation)", 4),
                        ("yeast-20k", "yeast 20k subset\n(unseen, test)", 8)]
    if not baselines:
        series = [t for t in series if t[0] not in ("GLEAMS", "binned cosine")]
    if labels:
        series = [(labels.get(n, n), c, g) for n, c, g in series]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels) + (0 if baselines else -1), 4.8),
                                 sharey=True, gridspec_kw={"width_ratios": [w for *_, w in panels]})
        for ax, (b, ptitle, _) in zip(axes, panels):
            ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
            x = 0
            for name, col, get in series:
                vals, sub = get(b)
                if not vals:
                    continue
                if name == "GLEAMS" or (name == "binned cosine" and not pick(rows, b, "GLEAMS (pretrained)")):
                    x += 0.5                                   # gap between ours and the baselines
                m, s_ = msd(vals)
                ax.bar(x, m, 0.8, color=col, yerr=s_ if len(vals) > 1 else None, capsize=3,
                       error_kw=dict(elinewidth=1, capthick=1, ecolor=INK), zorder=2)
                ax.text(x, m + (s_ if len(vals) > 1 else 0) + 0.015, f"{m:.2f}", ha="center", fontsize=8)
                if sub:
                    ax.text(x, 0.02, sub, rotation=90, ha="center", va="bottom", fontsize=7.5, color="white")
                x += 1
            ax.set_xticks([]); ax.set_xlim(-0.7, x - 0.3); ax.set_title(ptitle, loc="left", fontsize=10.5)
        axes[0].set_ylim(0, 1.0); axes[0].set_ylabel("MAP@R (experimental spectra)")
        fig.suptitle(title, x=0.07, ha="left",
                     fontsize=13, fontweight="bold", y=1.03)
        fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in series],
                   labels=[n for n, _, _ in series], frameon=False, fontsize=8.5, ncol=4 if baselines else 3,
                   loc="upper center", bbox_to_anchor=(0.5, 0.03))
        fig.savefig(FIG / out); plt.close(fig)


REP_LABEL = {"400m": "fine-tuned 400M (replicate corpus only)", "50m": "fine-tuned 50M (replicate corpus only)"}


RUNS = Path("/lus/flare/projects/UIC-HPC/khuss/msdelta/runs")
# C3: identical contrastive training from the pretrained vs a RANDOM encoder (random_init, verified
# per run in its wandb config). Older recipe (t 0.07, lr 1e-4, KL 10) and the small replicate
# eval (MAP@100 / Hit@1), each scale at its original checkpoint; 6 seeds. Later jobs are retries.
ABL_JOBS = {"8853557": "pretrained", "8854412": "pretrained", "8853558": "random init", "8854760": "random init"}


def ablation_rows():
    cells = {}
    for job, kind in ABL_JOBS.items():
        for d in sorted(RUNS.glob(f"sweep-*-{job}")):
            f = d / "all_results.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            if "retrieval/MAP@100" not in r:
                continue
            arm = re.match(r"sweep-(.+)-\d+$", d.name).group(1)
            sc, seed = re.match(r"s0*(\d+m)_seed(\d+)$", arm).groups()
            cells[(kind, sc, int(seed))] = dict(init=kind, scale=sc, seed=int(seed), job=job,
                                                map_at_100=r["retrieval/MAP@100"], hit_at_1=r["retrieval/Hit@1"])
    return sorted(cells.values(), key=lambda r: (r["init"], int(r["scale"][:-1]), r["seed"]))


def fig_ablation(ab):
    scales = ["50m", "100m", "200m", "400m"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for ax, key, lab in ((axes[0], "map_at_100", "MAP@100"), (axes[1], "hit_at_1", "Hit@1")):
            ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
            x = np.arange(len(scales))
            for k, (kind, col) in enumerate((("random init", BINNED), ("pretrained", OURS))):
                vals = [[r[key] for r in ab if r["init"] == kind and r["scale"] == sc] for sc in scales]
                m = [np.mean(v) for v in vals]; e = [np.std(v, ddof=1) for v in vals]
                xx = x + (k - 0.5) * 0.38
                ax.bar(xx, m, 0.36, yerr=e, capsize=3, color=col, label=f"{kind} + contrastive training",
                       error_kw=dict(elinewidth=1, capthick=1, ecolor=INK), zorder=2)
                for xi, mi, ei in zip(xx, m, e):
                    ax.text(xi, mi + ei + 0.012, f"{mi:.2f}", ha="center", fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels([sc.upper() for sc in scales]); ax.set_ylabel(lab)
            ax.set_xlabel("model size")
        axes[0].set_ylim(0, 0.52); axes[1].set_ylim(0, 0.92)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Pretraining is what makes contrastive training work", x=0.07, ha="left", fontsize=13,
                     fontweight="bold", y=1.03)
        fig.savefig(FIG / "C_pretraining_ablation.png"); plt.close(fig)


C7 = "C7 (replicate corpus -> ms-contrastive-100k)"
PLOT_KEYS = ["benchmark", "model", "scale", "pretrain_ckpt", "finetune_stage", "seed", "map_at_r", "hit_at_1", "note"]


def sel(rows, bench, method, scale="", ck="", stage="", seed=None):
    return [r for r in rows if r["benchmark"] == bench and r["method"] == method
            and (not scale or r["scale"] == scale) and (not ck or r["pretrain_ckpt"] == ck)
            and (not stage or r["stage"] == stage) and (seed is None or r["seed"] == seed)]


def as_plot_row(r, model, note=""):
    return dict(benchmark=r["benchmark"], model=model, scale=r["scale"], pretrain_ckpt=r["pretrain_ckpt"],
                finetune_stage=r["stage"], seed=r["seed"], map_at_r=r["map_at_r"], hit_at_1=r["hit_at_1"], note=note)


def write_plot_csv(name, out_rows, keys=PLOT_KEYS):
    with open(FIG / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader()
        for r in out_rows:
            w.writerow({k: (f"{r[k]:.5f}" if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})


def export_plot_data(rows, zs, ab):
    """One CSV per figure holding exactly the per-seed rows that figure plots (same selection as the plots)."""
    # C_transfer_ours: our models on the in-distribution test and on yeast 20k
    out = []
    for b in ("ms-contrastive-100k", "yeast-20k"):
        for model, rs in (("fine-tuned 400M", sel(rows, b, C7, "400m", stage="final", seed=BEST_SEED["400m"])),
                          ("fine-tuned 50M", sel(rows, b, C7, "50m", stage="final", seed=BEST_SEED["50m"])),
                          ("replicate corpus only 400M", sel(rows, b, STAGE1["400m"], "400m")),
                          ("replicate corpus only 50M", sel(rows, b, STAGE1["50m"], "50m"))):
            out += [as_plot_row(r, model, r["method"]) for r in rs]
        z = max((r for r in zs if r["benchmark"] == b), key=lambda t: t["abtt_best"])
        sc, ck = z["encoder"].split("@")
        out.append(dict(benchmark=b, model="frozen + ABTT (best encoder)", scale=sc, pretrain_ckpt=ck, finetune_stage="",
                        seed="", map_at_r=z["abtt_best"], hit_at_1="",
                        note=f"frozen encoder, layer {z['abtt_layer']}, ABTT D={z['abtt_D']} (fit: {z['abtt_fit']}); "
                             f"encoder/layer/D chosen on this benchmark"))
    write_plot_csv("C_transfer_ours.csv", out)
    # C_transfer: the same rows (replicate-only models relabelled) + GLEAMS and binned cosine (both widths)
    rel = {"replicate corpus only 400M": REP_LABEL["400m"], "replicate corpus only 50M": REP_LABEL["50m"]}
    tr = [dict(r, model=rel.get(r["model"], r["model"])) for r in out]
    for b in ("ms-contrastive-100k", "yeast-20k"):
        tr += [as_plot_row(r, "GLEAMS") for r in sel(rows, b, "GLEAMS (pretrained)")]
        bins = sel(rows, b, "binned cosine (1 Da bins)") + sel(rows, b, "binned cosine (0.1 Da bins)")
        best = max(r["map_at_r"] for r in bins)
        tr += [as_plot_row(r, "binned cosine", r["method"] + (" [plotted: best width]" if r["map_at_r"] == best else ""))
               for r in bins]
    write_plot_csv("C_transfer.csv", tr)
    # C_pretraining_scaling: replicate-corpus recipe (24 ep) across sizes and pretraining checkpoints
    write_plot_csv("C_pretraining_scaling.csv",
                   [as_plot_row(r, f"replicate corpus only {r['scale'].upper()}", r["method"])
                    for r in sel(rows, "ms-contrastive-100k", "replicate corpus only (24 ep, C2/C4)")])
    # C_pretraining_ablation
    write_plot_csv("C_pretraining_ablation.csv", ab, list(ab[0]))
    # C_zeroshot
    write_plot_csv("C_zeroshot.csv", zs, list(zs[0]))
    # C_benchmarks: the four plotted series; binned cosine at both bin widths (the plot shows the better one)
    out = []
    for b in ("ms-contrastive-100k", "hek-lowres", "yeast-full", "yeast-20k"):
        out += [as_plot_row(r, "ours: C7 400M") for r in sel(rows, b, C7, "400m", stage="final")]
        out += [as_plot_row(r, "ours: replicate corpus only 400M") for r in sel(rows, b, "replicate corpus only (12 ep)", "400m")]
        out += [as_plot_row(r, "GLEAMS") for r in sel(rows, b, "GLEAMS (pretrained)")]
        bins = sel(rows, b, "binned cosine (1 Da bins)") + sel(rows, b, "binned cosine (0.1 Da bins)")
        best = max(r["map_at_r"] for r in bins)
        out += [as_plot_row(r, "binned cosine", r["method"] + (" [plotted: best width]" if r["map_at_r"] == best else ""))
                for r in bins]
    write_plot_csv("C_benchmarks.csv", out)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    keys = ["benchmark", "method", "scale", "pretrain_ckpt", "stage", "seed", "map_at_r", "hit_at_1", "queries", "source"]
    with open(REPO / "results" / "contrastive_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.5f}" if k in ("map_at_r", "hit_at_1") and r[k] is not None else r[k]) for k in keys})
    zs = zeroshot_rows()
    with open(REPO / "results" / "contrastive_zeroshot.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(zs[0])); w.writeheader()
        for r in zs:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    ab = ablation_rows()
    with open(REPO / "results" / "contrastive_pretraining_ablation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(ab[0])); w.writeheader()
        for r in ab:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("ablation rows", len(ab))
    fig_benchmarks(rows); fig_pretraining(rows); fig_zeroshot(zs); fig_ablation(ab)
    # C_transfer: only the benchmarks every method was scored on (the 8-species OOD validation had no GLEAMS/frozen)
    fig_transfer(rows, zs, panels=[("ms-contrastive-100k", "ms-contrastive-100k test\n(in-distribution)", 8),
                                   ("yeast-20k", "yeast 20k subset\n(unseen)", 8)],
                 labels={"replicate corpus only 400M": REP_LABEL["400m"], "replicate corpus only 50M": REP_LABEL["50m"]})
    fig_transfer(rows, zs, panels=[("ms-contrastive-100k", "ms-contrastive-100k test\n(in-distribution)", 1),
                                   ("yeast-20k", "yeast 20k subset\n(unseen)", 1)],
                 baselines=False, out="C_transfer_ours.png", title="Our models in-distribution vs unseen")
    export_plot_data(rows, zs, ab)
    print(f"rows {len(rows)}, zero-shot rows {len(zs)}")
    agg = defaultdict(list)
    for r in rows:
        agg[(r["benchmark"], r["method"], r["scale"], r["pretrain_ckpt"], r["stage"])].append(r["map_at_r"])
    for k in sorted(agg, key=str):
        m, s = msd(agg[k]); print(f"  {k[0]:20s} {k[1]:48s} {k[2]:5s} {k[3]:5s} {k[4]:6s} n={len(agg[k])} {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
