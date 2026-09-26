"""Parallel-coordinates view of the 50m denoise hyperparameter grid (job 8840408, 216 arms).

    .venv/bin/python sweeps/plot_hp_parallel.py   # -> results/processed/figures/SUMMARY/D_hp_parallel.png + .csv

One vertical axis per hyperparameter, one line per arm, coloured by test AUROC; the winning
combination is drawn on top. Values come from the committed grid table
results/raw/finetune/denoise/grid_denoise_50m.txt (b = effective batch).
Plotting only (login node).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parent.parent
TABLE = REPO / "results" / "raw" / "finetune" / "denoise" / "grid_denoise_50m.txt"   # job 8840408, 216 arms
OUT = REPO / "results" / "processed" / "figures" / "SUMMARY"
AXES = [("learning_rate", "learning rate", lambda v: float(v)),
        ("encoder_lr_scale", "encoder LR scale", lambda v: float(v)),
        ("num_train_epochs", "epochs", lambda v: int(v)),
        ("head_hidden_size", "head width", lambda v: int(v)),
        ("eff_batch", "effective batch", lambda v: int(v))]
FMT = {"learning_rate": lambda v: f"{v:.0e}".replace("e-0", "e-"),
       "encoder_lr_scale": lambda v: "frozen" if v == 0 else f"{v:g}×"}


def load():
    """Rows of the committed grid table (the canonical record: 4 run dirs no longer hold
    all_results.json). Its b column is the effective batch; blank = 48."""
    rows = []
    for line in TABLE.read_text().splitlines():
        f = line.split()
        if len(f) == 8 and f[0].startswith("lr") and "_es" in f[0]:
            f = f[:5] + ["48"] + f[5:]        # blank b = the default, per-device 4 x 12 ranks
        if len(f) == 9 and f[0].startswith("lr") and "_es" in f[0]:
            rows.append(dict(arm=f[0], learning_rate=float(f[1]), encoder_lr_scale=float(f[2]),
                             num_train_epochs=int(f[3]), head_hidden_size=int(f[4]), eff_batch=int(f[5]),
                             auroc=float(f[6]), auprc=float(f[8]), spectra=8584))
    return rows


def main():
    rows = load()
    assert len(rows) == 216, len(rows)
    keys = [k for k, _, _ in AXES]
    levels = {k: sorted({r[k] for r in rows}) for k in keys}
    lo, hi = 0.90, max(r["auroc"] for r in rows)
    norm, cmap = Normalize(lo, hi, clip=True), plt.get_cmap("viridis")
    ink, muted = "#1f2937", "#6b7280"
    style = {"font.family": "DejaVu Sans", "font.size": 10, "text.color": ink, "figure.dpi": 200,
             "savefig.bbox": "tight"}
    rng = np.random.default_rng(0)
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10, 5.2))
        n = len(keys)
        xs = np.arange(n + 1)
        y_auroc = lambda v: (v - lo) / (hi - lo)
        def ypos(r):
            ys = [levels[k].index(r[k]) / (len(levels[k]) - 1) for k in keys]
            ys = [y + rng.uniform(-0.025, 0.025) for y in ys]           # small jitter to separate lines
            return ys + [min(max(y_auroc(r["auroc"]), -0.06), 1.0)]
        best = max(rows, key=lambda r: r["auroc"])
        for r in sorted(rows, key=lambda r: r["auroc"]):
            if r is best:
                continue
            good = r["auroc"] >= lo
            ax.plot(xs, ypos(r), color=cmap(norm(r["auroc"])) if good else "#d1d5db",
                    lw=1.1 if good else 0.6, alpha=0.75 if good else 0.35, zorder=2 if good else 1)
        yb = [levels[k].index(best[k]) / (len(levels[k]) - 1) for k in keys] + [1.0]
        ax.plot(xs, yb, color="white", lw=5.5, zorder=4, solid_capstyle="round")
        ax.plot(xs, yb, color="#dc2626", lw=2.6, zorder=5, solid_capstyle="round", label="best combination")
        for i, k in enumerate(keys):
            ax.axvline(i, color="#9ca3af", lw=0.9, zorder=3)
            for j, v in enumerate(levels[k]):
                y = j / (len(levels[k]) - 1)
                lab = FMT.get(k, str)(v)
                ax.text(i - 0.05, y, lab, ha="right", va="center", fontsize=8.5, color=ink,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85), zorder=6)
        ax.axvline(n, color="#9ca3af", lw=0.9, zorder=3)
        for v in np.arange(0.90, hi + 1e-9, 0.01):
            ax.text(n + 0.05, y_auroc(v), f"{v:.2f}", ha="left", va="center", fontsize=8.5, color=ink)
        ax.text(n + 0.05, -0.06, f"< {lo:.2f}", ha="left", va="center", fontsize=8, color=muted)
        ax.set_xticks(xs); ax.set_xticklabels([l for _, l, _ in AXES] + ["test AUROC"], fontsize=9.5)
        ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=8)
        ax.set_yticks([]); ax.set_ylim(-0.1, 1.06); ax.set_xlim(-0.55, n + 0.35)
        for s in ax.spines.values():
            s.set_visible(False)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.06); cb.set_label("test AUROC", fontsize=9)
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.5, colors="#6b7280", length=0)
        fig.suptitle("Denoising hyperparameter search (50M, 216 configurations)", x=0.125, ha="left",
                     fontsize=13, fontweight="bold", y=1.04)
        bl = (f"best: lr {FMT['learning_rate'](best['learning_rate'])}, encoder {best['encoder_lr_scale']:g}×, "
              f"{best['num_train_epochs']} epochs, head {best['head_hidden_size']}, batch {best['eff_batch']}"
              f" → AUROC {best['auroc']:.4f}.\nGrey: the {sum(r['auroc'] < lo for r in rows)} runs below {lo:.2f} "
              f"(every lr 1e-6 and frozen-encoder run, and all but 3 at lr 1e-5).")
        fig.text(0.125, -0.05, bl, fontsize=8.5, color=muted)
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "D_hp_parallel.png"); plt.close(fig)
    csv = OUT / "D_hp_parallel.csv"
    csv.write_text("arm," + ",".join(keys) + ",test_auroc,test_auprc,test_spectra_scored\n" + "".join(
        f"{r['arm']}," + ",".join(str(r[k]) for k in keys) + f",{r['auroc']:.5f},{r['auprc']:.4f},{r['spectra']}\n"
        for r in sorted(rows, key=lambda r: -r["auroc"])))
    print("wrote", OUT / "D_hp_parallel.png", csv)
    print("best", best["arm"], round(best["auroc"], 4), "| below", lo, sum(r["auroc"] < lo for r in rows))
    print("scored spectra:", sorted({r["spectra"] for r in rows}, key=str))


if __name__ == "__main__":
    main()
