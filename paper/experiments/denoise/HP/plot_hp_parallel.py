"""Regenerate D_hp_parallel.png from D_hp_parallel.csv (this folder).

    python plot_hp_parallel.py        # needs matplotlib + numpy

Parallel coordinates over the 216-configuration 50M denoise grid: one vertical axis per
hyperparameter, one line per configuration coloured by test AUROC, the best combination in red.
Runs below 0.90 are drawn grey. Lines get a small fixed-seed vertical jitter so overlapping
configurations stay distinguishable.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
AXES = [("learning_rate", "learning rate", float), ("encoder_lr_scale", "encoder LR scale", float),
        ("num_train_epochs", "epochs", int), ("head_hidden_size", "head width", int), ("eff_batch", "effective batch", int)]
FMT = {"learning_rate": lambda v: f"{v:.0e}".replace("e-0", "e-"),
       "encoder_lr_scale": lambda v: "frozen" if v == 0 else f"{v:g}×"}
INK, MUTED = "#1f2937", "#6b7280"
LO = 0.90


def main():
    rows = []
    for r in csv.DictReader(open(HERE / "D_hp_parallel.csv")):
        row = {k: f(r[k]) for k, _, f in AXES}
        row.update(arm=r["arm"], auroc=float(r["test_auroc"]))
        rows.append(row)
    keys = [k for k, _, _ in AXES]
    levels = {k: sorted({r[k] for r in rows}) for k in keys}
    hi = max(r["auroc"] for r in rows)
    norm, cmap = Normalize(LO, hi, clip=True), plt.get_cmap("viridis")
    rng = np.random.default_rng(0)
    n = len(keys); xs = np.arange(n + 1)
    y_auroc = lambda v: (v - LO) / (hi - LO)

    def ypos(r):
        ys = [levels[k].index(r[k]) / (len(levels[k]) - 1) + rng.uniform(-0.025, 0.025) for k in keys]
        return ys + [min(max(y_auroc(r["auroc"]), -0.06), 1.0)]

    best = max(rows, key=lambda r: r["auroc"])
    style = {"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "figure.dpi": 200, "savefig.bbox": "tight"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10, 5.2))
        for r in sorted(rows, key=lambda r: r["auroc"]):
            if r is best:
                continue
            good = r["auroc"] >= LO
            ax.plot(xs, ypos(r), color=cmap(norm(r["auroc"])) if good else "#d1d5db",
                    lw=1.1 if good else 0.6, alpha=0.75 if good else 0.35, zorder=2 if good else 1)
        yb = [levels[k].index(best[k]) / (len(levels[k]) - 1) for k in keys] + [1.0]
        ax.plot(xs, yb, color="white", lw=5.5, zorder=4, solid_capstyle="round")
        ax.plot(xs, yb, color="#dc2626", lw=2.6, zorder=5, solid_capstyle="round")
        for i, k in enumerate(keys):
            ax.axvline(i, color="#9ca3af", lw=0.9, zorder=3)
            for j, v in enumerate(levels[k]):
                ax.text(i - 0.05, j / (len(levels[k]) - 1), FMT.get(k, str)(v), ha="right", va="center", fontsize=8.5,
                        color=INK, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85), zorder=6)
        ax.axvline(n, color="#9ca3af", lw=0.9, zorder=3)
        for v in np.arange(LO, hi + 1e-9, 0.01):
            ax.text(n + 0.05, y_auroc(v), f"{v:.2f}", ha="left", va="center", fontsize=8.5, color=INK)
        ax.text(n + 0.05, -0.06, f"< {LO:.2f}", ha="left", va="center", fontsize=8, color=MUTED)
        ax.set_xticks(xs); ax.set_xticklabels([l for _, l, _ in AXES] + ["test AUROC"], fontsize=9.5)
        ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=8)
        ax.set_yticks([]); ax.set_ylim(-0.1, 1.06); ax.set_xlim(-0.55, n + 0.35)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.025, pad=0.06)
        cb.set_label("test AUROC", fontsize=9); cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8.5, colors=MUTED, length=0)
        fig.suptitle(f"Denoising hyperparameter search (50M, {len(rows)} configurations)", x=0.125, ha="left",
                     fontsize=13, fontweight="bold", y=1.04)
        n_low = sum(r["auroc"] < LO for r in rows)
        fig.text(0.125, -0.05,
                 f"best: lr {FMT['learning_rate'](best['learning_rate'])}, encoder {best['encoder_lr_scale']:g}×, "
                 f"{best['num_train_epochs']} epochs, head {best['head_hidden_size']}, batch {best['eff_batch']} "
                 f"→ AUROC {best['auroc']:.4f}.\nGrey: the {n_low} runs below {LO:.2f} "
                 f"(every lr 1e-6 and frozen-encoder run, and all but 3 at lr 1e-5).", fontsize=8.5, color=MUTED)
        fig.savefig(HERE / "D_hp_parallel.png"); plt.close(fig)


if __name__ == "__main__":
    main()
