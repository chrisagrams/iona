"""Summary figures + tables for D / C / A / R (2026-09-25). Numbers are the recorded results in
notes/OBSERVATIONS.md (job ids in the comments); run on the login node (plotting only).

    .venv/bin/python sweeps/plot_summary.py   # -> results/figures/SUMMARY/*.png, results/SUMMARY_TABLES.md
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "figures" / "SUMMARY"
TABLES = REPO / "results" / "SUMMARY_TABLES.md"
OURS, BASE, OTHER, NULL, LIGHT = "#2563eb", "#9ca3af", "#f59e0b", "#d1d5db", "#93c5fd"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})

# ---------------------------------------------------------------- D (denoise)
D_SCALE = {"50m": 0.9317, "100m": 0.9400, "200m": 0.9447, "400m": 0.9434}      # D1, 6 seeds (FT5, 8845262)
D_SCALE_SD = {"50m": 0.00055, "100m": 0.00029, "200m": 0.00025, "400m": 0.00045}
# from scratch at the same config at every scale (lr 2e-4, eff. batch 12, 4 epochs). 100m-400m: 3 seeds
# each (8847663 + resume 8856558); 50m: ONE run (arm lr2e4_ep4_b12 of grid 8841984) -- no error bar.
D_SCRATCH_SEEDS = {"50m": [0.8821], "100m": [0.89065, 0.88986, 0.88539],
                   "200m": [0.89673, 0.89610, 0.89593], "400m": [0.89564, 0.89939, 0.89927]}
D_SCRATCH = {k: float(np.mean(v)) for k, v in D_SCRATCH_SEEDS.items()}

# ---------------------------------------------------------------- C (spectrum embeddings)
# exp MAP@R. ms-contrastive-100k test (in-distribution for C7); unseen: HEK (C11, low-res MS2),
# nine-species yeast full (C13, 8866804) and 20k subset (diagnostic 8867207).
C_BENCH = {  # dataset: {method: value or (min, mean, max)}
    "ms-contrastive-100k\n(in-distribution)": {
        "ours C7 400m": (0.868, 0.868, 0.869), "ours replicate-only 400m": (0.711, 0.713, 0.714),
        "GLEAMS": 0.646, "binned cosine": 0.730},
    "HEK (unseen,\nlow-res MS2)": {
        "ours C7 400m": (0.167, 0.170, 0.173), "ours replicate-only 400m": (0.016, 0.017, 0.018),
        "GLEAMS": 0.530, "binned cosine": 0.554},
    "nine-species yeast\n(unseen, high-res)": {
        "ours C7 400m": (0.475, 0.518, 0.565), "ours replicate-only 400m": (0.444, 0.474, 0.503),
        "GLEAMS": 0.676, "binned cosine": 0.790},
    "yeast 20k subset\n(unseen, high-res)": {
        "ours C7 400m": (0.550, 0.600, 0.656), "ours replicate-only 400m": (0.521, 0.521, 0.521),
        "ours frozen 400m + ABTT": 0.709, "GLEAMS": 0.770, "binned cosine": 0.916},
}
C_SCALE_FT = {"50m": 0.657, "100m": 0.665, "200m": 0.658, "400m": 0.703}        # C2, replicate corpus
C_SCALE_C7 = {"50m": 0.839, "400m": 0.868}                                       # C7
C_ZS_ID = {"50m": (0.208, 0.401), "100m": (0.140, 0.328), "200m": (0.191, 0.432),
           "400m": (0.216, 0.414)}   # frozen best block raw -> ABTT, 100k test (400m @10k; 200m @540k)
C_ZS_OOD = {"50m": (0.385, 0.602), "100m": (0.178, 0.421), "200m": (0.339, 0.654),
            "400m": (0.510, 0.709)}  # yeast 20k, frozen best block raw -> ABTT (50m/100m/400m @220k, 200m @540k);
                                     # from results/finetune/contrastive/nine20k_zeroshot/
C_TRAJ_OOD = {0: [0.606, 0.655, 0.596, 0.596]}  # seed0 yeast20k at step 300/600/900/final

# ---------------------------------------------------------------- A (peptide embeddings)
A_TEST = [("A1 (teacher C7 50m)", 0.898), ("A4 (A1 + LiT/hard neg.)", 0.895),
          ("A-oodsel (teacher C7 400m s600, OOD-selected)", 0.918),
          ("A2 (teacher C7 400m final)", 0.923)]
A_YH = {  # dataset -> window -> {model: Hit@1}
    "in-distribution\n(ms-contrastive-100k)": {
        "open": {"yHydra": 0.196, "A1": 0.901, "A2": 0.925, "A-oodsel": 0.921},
        "±1.1 Da": {"yHydra": 0.754, "A1": 0.973, "A2": 0.979},
        "20 ppm": {"yHydra": 0.942, "A1": 0.993, "A2": 0.994, "A-oodsel": 0.994}},
    "HEK (unseen, low-res)": {
        "open": {"yHydra": 0.016, "A1": 0.061},
        "±1.1 Da": {"yHydra": 0.345, "A1": 0.601},
        "20 ppm": {"yHydra": 0.606, "A1": 0.679}},
    "nine-species yeast\n(unseen, high-res)": {
        "open": {"yHydra": 0.060, "A1": 0.253, "A2": 0.388, "A-oodsel": 0.410},
        "±1.1 Da": {"yHydra": 0.650, "A1": 0.537, "A2": 0.639, "A-oodsel": 0.659},
        "20 ppm": {"yHydra": 0.890, "A1": 0.691, "A2": 0.761, "A-oodsel": 0.765}},
}

# ---------------------------------------------------------------- R (reranking)
# PSMs at 1% FDR, 8 runs (6 HEK + 2 HCT116), unless noted.
R_BENCH = [("MSFragger (e-value)", 89693, BASE), ("ours global MLP, MSFragger feat.", 93206, LIGHT),
           ("ours global MLP, lab feat.", 104785, LIGHT), ("MS2Rescore search-only", 100974, OTHER),
           ("ours per-run, MSFragger feat.", 99154, LIGHT), ("MS2Rescore full (MS2PIP+DeepLC)", 128211, OTHER),
           ("MS2Rescore full + A2 emb.", 128909, OURS), ("ours per-run, lab feat.", 129041, LIGHT),
           ("ours per-run, lab + A2 emb.", 129618, OURS)]
# per-run linear, real - null (3 seeds) : base -> embedding -> list
R_ABL = {"lab (8 runs)": {"A1": [398, 738, 473], "A2": [564, 924, 544], "A-oodsel": [901, 912, 572]},
         "MSFragger (8 runs)": {"A1": [1495, 1215, 1186], "A2": [2705, 2373, 2386],
                                "A-oodsel": [2096, 1912, 1872]}}
R_BASE8 = {"lab (8 runs)": 128909, "MSFragger (8 runs)": 99247}   # mean base over seeds
# HCT116 x 18 (high-res, unseen), A2, per-run, seed 0 (8867598)
R_HCT = {"MSFragger (e-value)": 330139, "per-run, MSFragger feat.": 370982,
         "per-run, MSFragger + A2 emb.": 392350, "per-run, MSFragger + null": 371587,
         "per-run, lab feat.": 522539, "per-run, lab + A2 emb.": 530977, "per-run, lab + null": 522353}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT / name); plt.close(fig)
    print("wrote", (OUT / name).relative_to(REPO))


D_CKPTS = [10000, 120000, 220000, 330000, 430000, 540423]   # pretraining steps shared by all four scales
D_CSV = REPO / "results" / "denoise_scaling_pretraining.csv"


def denoise_cells(metric="test_auroc"):
    """(scale, pretraining steps; 0 = from scratch) -> per-seed test `metric`, read from the run dirs.
    Pretrained: plot_ladder.collect (fixed config lr 2e-4 / es 0.5 / 4 ep / head 512 / eff. batch 12).
    Scratch: random encoder at lr 2e-4 / eff. batch 12 / 4 ep -- 100m-400m 3 seeds (8847663),
    50m one run (arm lr2e4_ep4_b12 of 8841984)."""
    import sys; sys.path.insert(0, str(REPO / "sweeps"))
    import json, plot_ladder
    cov = plot_ladder.collect()
    cells = {(sc, c): list(cov[(sc, c)][metric]) for sc in D_SCALE for c in D_CKPTS if (sc, c) in cov}
    runs = Path(plot_ladder.RUNS)
    for sc in D_SCALE:
        dirs = ([runs / "sweep-lr2e4_ep4_b12-8841984"] if sc == "50m"
                else [runs / f"sweep-{sc}_ep04_seed{i}-8847663" for i in (1, 2, 3)])
        cells[(sc, 0)] = [json.loads((d / "all_results.json").read_text())[metric] for d in dirs]
    return cells


def fig_d():
    from matplotlib.lines import Line2D
    cells = denoise_cells()
    xs = list(D_SCALE); xi = np.arange(len(xs))
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    ink, muted = "#1f2937", "#6b7280"
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9ca3af",
             "axes.linewidth": 0.8, "axes.labelcolor": ink, "text.color": ink,
             "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
             "axes.spines.right": False, "savefig.bbox": "tight", "figure.dpi": 200}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.yaxis.grid(True, color="#e5e7eb", lw=0.8); ax.set_axisbelow(True)
        cmap = plt.get_cmap("viridis_r")
        cols = {c: cmap(0.25 + 0.7 * i / (len(D_CKPTS) - 1)) for i, c in enumerate(D_CKPTS)}
        series = [(0, "#9ca3af", "from scratch", "s", "--")] + \
                 [(c, cols[c], f"{c // 1000}k steps", "o", "-") for c in D_CKPTS]
        means = {c: np.array([np.mean(cells[(x, c)]) for x in xs]) for c, *_ in series}
        handles = []
        for c, col, lab, mk, ls in series:
            ax.errorbar(xi, means[c], yerr=[sd(cells[(x, c)]) for x in xs], color=col, lw=2.0, ls=ls,
                        marker=mk, ms=6.5, mfc=col, mec="white", mew=1.2, capsize=3.5, capthick=1.2,
                        elinewidth=1.2, zorder=3)
            handles.append(Line2D([], [], color=col, lw=2.0, ls=ls, marker=mk, ms=6.5, mec="white", mew=1.2, label=lab))
        ax.set_xticks(xi); ax.set_xticklabels([x.upper() for x in xs])
        ax.set_xlim(-0.3, len(xs) - 0.7)
        ax.set_xlabel("model size (parameters)", labelpad=6); ax.set_ylabel("test AUROC", labelpad=6)
        ax.set_title("Scaling improvements for denoising", loc="left", fontsize=13, fontweight="bold", pad=12)
        leg = ax.legend(handles=handles[::-1], title="pretraining", frameon=False, fontsize=9,
                        title_fontsize=9.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        handlelength=2.6, labelspacing=0.7)
        leg._legend_box.align = "left"
        fig.text(0.01, -0.02, "Mean ± sd over 3–5 seeds per point; 50M from scratch is a single run.",
                 fontsize=8, color=muted)
        save(fig, "D_denoise.png")
    # AUPRC: noise is the positive class, so chance = the test noise fraction (0.533), not 0.5.
    prc = denoise_cells("test_auprc")
    rows = ["scale,pretraining_steps,n_seeds,mean_auroc,sd_auroc,min_auroc,max_auroc,per_seed_auroc,"
            "mean_auprc,sd_auprc,per_seed_auprc"]
    for x in xs:
        for c in [0] + D_CKPTS:
            u, a = cells[(x, c)], prc[(x, c)]
            assert len(a) == len(u), (x, c)
            rows.append(f"{x},{c},{len(u)},{np.mean(u):.5f},{sd(u) if len(u) > 1 else float('nan'):.5f},"
                        f"{min(u):.5f},{max(u):.5f},{' '.join(f'{t:.5f}' for t in u)},"
                        f"{np.mean(a):.5f},{sd(a) if len(a) > 1 else float('nan'):.5f},{' '.join(f'{t:.5f}' for t in a)}")
    D_CSV.write_text("\n".join(rows).replace("nan", "") + "\n"); print("wrote", D_CSV.relative_to(REPO))


def fig_c_bench():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    methods = ["ours C7 400m", "ours replicate-only 400m", "ours frozen 400m + ABTT", "GLEAMS", "binned cosine"]
    colors = [OURS, LIGHT, "#1e3a8a", OTHER, BASE]
    w = 0.16
    for j, (ds, vals) in enumerate(C_BENCH.items()):
        for i, (m, c) in enumerate(zip(methods, colors)):
            if m not in vals:
                continue
            v = vals[m]; x = j + (i - 2) * w
            if isinstance(v, tuple):
                ax.bar(x, v[1], w, color=c, label=m if j == 0 or m == "ours frozen 400m + ABTT" and j == 3 else None,
                       yerr=[[v[1] - v[0]], [v[2] - v[1]]], capsize=2)
            else:
                ax.bar(x, v, w, color=c, label=m if (j == 0 or (m == "ours frozen 400m + ABTT" and j == 3)) else None)
    ax.set_xticks(range(len(C_BENCH))); ax.set_xticklabels(list(C_BENCH))
    ax.set_ylabel("MAP@R (experimental)"); ax.set_ylim(0, 1)
    ax.set_title("C: spectrum retrieval vs GLEAMS and binned cosine (error bars: 3 seeds)", loc="left")
    h, l = ax.get_legend_handles_labels(); ax.legend(h, l, frameon=False, ncol=5, fontsize=8, loc="upper right")
    save(fig, "C_benchmarks.png")


def fig_c_scaling():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.3))
    order = ["50m", "100m", "200m", "400m"]
    a.plot(order, [C_SCALE_FT[s] for s in order], "o-", color=LIGHT, label="fine-tuned, replicate corpus (C2)")
    a.plot(["50m", "400m"], [C_SCALE_C7[s] for s in ["50m", "400m"]], "o-", color=OURS, label="fine-tuned, + ms-contrastive-100k (C7)")
    a.set_title("C: scaling, in-distribution (100k test)", loc="left"); a.set_ylabel("MAP@R"); a.legend(frameon=False, fontsize=7)
    for ax, data, title in ((b, C_ZS_ID, "zero-shot (frozen), 100k test"), (c, C_ZS_OOD, "zero-shot (frozen), unseen yeast")):
        x = np.arange(4)
        ax.bar(x - 0.2, [data[s][0] or 0 for s in order], 0.4, color=BASE, label="raw, best block")
        ax.bar(x + 0.2, [data[s][1] for s in order], 0.4, color=OURS, label="+ ABTT")
        ax.set_xticks(x); ax.set_xticklabels(order); ax.set_title(f"C: {title}", loc="left"); ax.legend(frameon=False, fontsize=7)
    save(fig, "C_scaling_zeroshot.png")


def fig_c_transfer():
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot(["300", "600", "900", "final"], C_TRAJ_OOD[0], "o-", color=OURS, label="C7 400m seed 0 (fine-tuned)")
    for y, lab, col in ((0.709, "frozen 400m + ABTT", "#1e3a8a"), (0.770, "GLEAMS", OTHER), (0.916, "binned cosine", BASE)):
        ax.axhline(y, ls="--", color=col, lw=1); ax.text(3, y + 0.01, lab, ha="right", fontsize=7, color=col)
    ax.set_xlabel("ms-contrastive-100k training step"); ax.set_ylabel("MAP@R, unseen yeast 20k")
    ax.set_title("C: fine-tuning specialises (transfer peaks mid-epoch)", loc="left"); ax.set_ylim(0.5, 0.95)
    save(fig, "C_transfer.png")


def fig_a():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 2.2]})
    a.barh([n for n, _ in A_TEST], [v for _, v in A_TEST], color=[LIGHT, LIGHT, OURS, OURS])
    for i, (_, v) in enumerate(A_TEST):
        a.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    a.set_xlim(0.85, 0.94); a.set_title("A: test Hit@1 (100k test, 9,771 candidates)", loc="left")
    models = ["yHydra", "A1", "A2", "A-oodsel"]; cols = [OTHER, LIGHT, OURS, "#1e3a8a"]
    xt, xl = [], []; x0 = 0
    for ds, wins in A_YH.items():
        for k, (win, vals) in enumerate(wins.items()):
            for i, m in enumerate(models):
                if m in vals:
                    b.bar(x0 + i * 0.2, vals[m], 0.2, color=cols[i], label=m if (x0 == 0) else None)
            xt.append(x0 + 0.3); xl.append(win); x0 += 1
        x0 += 0.5
    b.set_xticks(xt); b.set_xticklabels(xl, fontsize=7)
    for j, ds in enumerate(A_YH):
        b.text(j * 3.5 + 1.3, 1.02, ds.replace("\n", " "), ha="center", fontsize=7)
    b.set_ylim(0, 1.1); b.set_ylabel("Hit@1 (spectrum → peptide)")
    b.set_title("A vs yHydra: open retrieval and precursor-mass windows", loc="left", pad=16)
    b.legend(frameon=False, ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, "A_peptide.png")


def fig_r():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    rb = sorted(R_BENCH, key=lambda r: r[1])
    names = [n for n, _, _ in rb]; vals = [v for _, v, _ in rb]
    a.barh(names, vals, color=[c for _, _, c in rb])
    for i, v in enumerate(vals):
        a.text(v + 500, i, f"{v:,}", va="center", fontsize=7)
    a.set_xlim(80000, 140000); a.set_xlabel("PSMs at 1% FDR (8 runs)")
    a.set_title("R: rescoring benchmarks", loc="left")
    x = np.arange(2); w = 0.25
    for i, (emb, col) in enumerate((("A1", LIGHT), ("A2", OURS), ("A-oodsel", "#1e3a8a"))):
        means = [np.mean(R_ABL[bk][emb]) / R_BASE8[bk] * 100 for bk in R_ABL]
        errs = [np.std(R_ABL[bk][emb]) / R_BASE8[bk] * 100 for bk in R_ABL]
        b.bar(x + (i - 1) * w, means, w, yerr=errs, capsize=2, color=col, label=emb)
    hct = [(530977 - 522353) / 522539 * 100, (392350 - 371587) / 370982 * 100]
    b.scatter(x + 0.42, hct, marker="D", color="#dc2626", zorder=3, label="A2, HCT116 ×18 (unseen, high-res)")
    b.set_xticks(x); b.set_xticklabels(["lab features\n(strong base ≈ MS2Rescore)", "MSFragger features"])
    b.set_ylabel("% PSMs added over null control"); b.legend(frameon=False, fontsize=7)
    b.set_title("R: what the embedding adds (per-run, 3 seeds)", loc="left")
    save(fig, "R_reranking.png")


def tables():
    L = ["# Summary tables (2026-09-25)", "", "Source: notes/OBSERVATIONS.md (job ids there). Figures: results/figures/SUMMARY/.", ""]
    L += ["## D: denoise", "", "| scale | pretrained + fine-tuned | from scratch | gain |", "|---|---|---|---|"]
    L += [f"| {s} | {D_SCALE[s]:.4f} | {D_SCRATCH[s]:.4f} | +{D_SCALE[s] - D_SCRATCH[s]:.3f} |" for s in D_SCALE]
    L += ["", "Pretrained: mean ± sd over 6 seeds. Scratch: 3 seeds at 100m–400m; 50m is a single run at the same config (lr 2e-4, eff. batch 12, 4 epochs).",
          "Caveat: the Hub cards show training-logged numbers; reloaded models re-evaluate ~0.014 AUROC lower (unresolved).", ""]
    L += ["## C: spectrum retrieval (experimental MAP@R)", "", "| dataset | ours C7 400m (3 seeds) | ours replicate-only | frozen + ABTT | GLEAMS | binned cosine |", "|---|---|---|---|---|---|"]
    for ds, v in C_BENCH.items():
        f = lambda k: ("—" if k not in v else (f"{v[k][1]:.3f} ({v[k][0]:.3f}–{v[k][2]:.3f})" if isinstance(v[k], tuple) else f"{v[k]:.3f}"))
        L.append(f"| {ds.replace(chr(10), ' ')} | {f('ours C7 400m')} | {f('ours replicate-only 400m')} | {f('ours frozen 400m + ABTT')} | {f('GLEAMS')} | {f('binned cosine')} |")
    L += ["", "| scale | C2 (replicate corpus) | C7 (+ms-contrastive-100k) | zero-shot raw → ABTT (100k test) | zero-shot raw → ABTT (unseen yeast) |", "|---|---|---|---|---|"]
    for s in ["50m", "100m", "200m", "400m"]:
        L.append(f"| {s} | {C_SCALE_FT[s]:.3f} | {C_SCALE_C7.get(s, float('nan')):.3f} | {C_ZS_ID[s][0]:.3f} → {C_ZS_ID[s][1]:.3f} | {"—" if C_ZS_OOD[s][0] is None else f"{C_ZS_OOD[s][0]:.3f}"} → {C_ZS_OOD[s][1]:.3f} |".replace("nan", "—"))
    L += ["", "Pretraining (C3): random init trained contrastively stays at chance at every scale.", ""]
    L += ["## A: peptide embeddings", "", "| model | test Hit@1 |", "|---|---|"] + [f"| {n} | {v:.3f} |" for n, v in A_TEST]
    L += ["", "| dataset | window | yHydra | A1 | A2 | A-oodsel |", "|---|---|---|---|---|---|"]
    for ds, wins in A_YH.items():
        for win, v in wins.items():
            g = lambda k: f"{v[k]:.3f}" if k in v else "—"
            L.append(f"| {ds.replace(chr(10), ' ')} | {win} | {g('yHydra')} | {g('A1')} | {g('A2')} | {g('A-oodsel')} |")
    L += ["", "## R: reranking (PSMs at 1% FDR)", "", "| method (8 runs) | PSMs |", "|---|---|"] + [f"| {n} | {v:,} |" for n, v, _ in R_BENCH]
    L += ["", "| base (per-run, 8 runs) | embedding | real − null per seed | mean |", "|---|---|---|---|"]
    for bk, d in R_ABL.items():
        for emb, xs in d.items():
            L.append(f"| {bk} | {emb} | {' / '.join(f'+{x:,}' for x in xs)} | +{np.mean(xs):,.0f} ({np.mean(xs) / R_BASE8[bk] * 100:.2f}%) |")
    L += ["", "| HCT116 ×18 (unseen, high-res), A2, per-run, seed 0 | PSMs |", "|---|---|"] + [f"| {k} | {v:,} |" for k, v in R_HCT.items()]
    L += ["", f"HCT116: lab + A2 real − null = +{530977 - 522353:,} (+{(530977 - 522353) / 522539 * 100:.2f}%); MSFragger features + A2 real − null = +{392350 - 371587:,} (+{(392350 - 371587) / 370982 * 100:.2f}%). Leakage AUROC 0.519 (cosine), 0.499 (null).",
          "Controls: shuffled labels → 0 PSMs; per-run fold seeds 128,991–129,041 (lab); MS2Rescore repeat 128,211 vs 128,209.", ""]
    TABLES.write_text("\n".join(L) + "\n"); print("wrote", TABLES.relative_to(REPO))


if __name__ == "__main__":
    fig_d(); fig_a(); fig_r(); tables()   # C figures: sweeps/package_contrastive.py
