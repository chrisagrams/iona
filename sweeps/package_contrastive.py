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


def pick(rows, bench, method, scale="", ck="", stage=""):
    return [r["map_at_r"] for r in rows if r["benchmark"] == bench and r["method"] == method
            and (not scale or r["scale"] == scale) and (not ck or r["pretrain_ckpt"] == ck)
            and (not stage or r["stage"] == stage)]


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
        fig.text(0.01, -0.04, "Replicate-corpus recipe (24 epochs), mean ± sd over 3 seeds. A random-init encoder trained the same way stays at chance.",
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
        fig.text(0.07, -0.08, "ABTT = remove the mean and top principal components (fit on training spectra). "
                 "Layer and number of components are the best on each benchmark, so these are upper bounds.",
                 fontsize=8, color=MUTED)
        fig.savefig(FIG / "C_zeroshot.png"); plt.close(fig)


def fig_transfer(rows):
    steps = ["s300", "s600", "s900", "final"]
    meth = "C7 (replicate corpus -> ms-contrastive-100k)"
    lines = [("ms-contrastive-100k", "in-distribution test", OURS),
             ("oodval-8species", "unseen: 8 other species (validation)", "#7c3aed"),
             ("yeast-20k", "unseen: yeast 20k (test)", "#db2777")]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
        for b, lab, col in lines:
            pts = [(i, pick(rows, b, meth, "400m", stage=s)) for i, s in enumerate(steps)]
            pts = [(i, msd(v)) for i, v in pts if v]
            ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts], yerr=[p[1][1] for p in pts], color=col,
                        lw=2, marker="o", ms=6.5, mec="white", mew=1.2, capsize=3.5, elinewidth=1.2, label=lab, zorder=3)
        g = pick(rows, "yeast-20k", "GLEAMS (pretrained)")
        if g:
            ax.axhline(g[0], color=GLEAMS, lw=1.2, ls="--"); ax.text(3.05, g[0], "GLEAMS\n(yeast 20k)", va="center", fontsize=8, color=GLEAMS)
        ax.set_xticks(range(4)); ax.set_xticklabels(["300", "600", "900", "end (1 epoch)"])
        ax.set_xlabel("ms-contrastive-100k fine-tuning step", labelpad=6); ax.set_ylabel("MAP@R", labelpad=6)
        ax.set_title("Fine-tuning improves in-distribution, peaks early on unseen data", loc="left", fontsize=12,
                     fontweight="bold", pad=12)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        fig.text(0.01, -0.04, "C7 400M. Mean ± sd over 3 seeds where available (yeast 20k steps 300–900: seed 0 only).",
                 fontsize=8, color=MUTED)
        fig.savefig(FIG / "C_transfer.png"); plt.close(fig)


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
    fig_benchmarks(rows); fig_pretraining(rows); fig_zeroshot(zs); fig_transfer(rows)
    print(f"rows {len(rows)}, zero-shot rows {len(zs)}")
    agg = defaultdict(list)
    for r in rows:
        agg[(r["benchmark"], r["method"], r["scale"], r["pretrain_ckpt"], r["stage"])].append(r["map_at_r"])
    for k in sorted(agg, key=str):
        m, s = msd(agg[k]); print(f"  {k[0]:20s} {k[1]:48s} {k[2]:5s} {k[3]:5s} {k[4]:6s} n={len(agg[k])} {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
