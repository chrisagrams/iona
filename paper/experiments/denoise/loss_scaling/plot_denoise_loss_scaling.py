"""Regenerate D_denoise_loss_scaling.png from D_denoise_loss_scaling.csv (this folder).

    python plot_denoise_loss_scaling.py      # needs matplotlib, numpy, scipy

Fits L(N, S) = E_N + B * (S / 1e5)^-beta to the per-point mean denoise test loss (one floor E_N per
model size, one power law in pretraining steps S shared by all sizes; weighted by the standard error
over seeds, floored at 5e-4), and plots the data with the fitted curves (left) and the reducible
loss L - E_N on log-log axes (right). Also prints the fit and a held-out check (fit without the
final checkpoint, predict it).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
SCALES = ["50m", "100m", "200m", "400m"]
COLS = {"50m": "#93c5fd", "100m": "#3b82f6", "200m": "#1d4ed8", "400m": "#1e3a8a"}


def load():
    import csv
    rows = list(csv.DictReader(open(HERE / "D_denoise_loss_scaling.csv")))
    sc = np.array([r["scale"] for r in rows])
    s = np.array([float(r["pretraining_steps"]) for r in rows])
    L = np.array([float(r["mean_test_loss"]) for r in rows])
    se = np.array([float(r["se_test_loss"]) for r in rows])
    return sc, s, L, se


def model(X, E0, E1, E2, E3, B, b):
    s, si = X
    return np.array([E0, E1, E2, E3])[si.astype(int)] + B * (s / 1e5) ** (-b)


def main():
    sc, s, L, se = load()
    w = np.maximum(se, 5e-4)
    si = np.array([SCALES.index(x) for x in sc], float)
    p, cv = curve_fit(model, (s, si), L, p0=(0.3, 0.3, 0.3, 0.3, 0.02, 0.5), sigma=w, maxfev=100000)
    pe = np.sqrt(np.diag(cv))
    E = dict(zip(SCALES, p[:4])); B, b = p[4], p[5]
    r = L - model((s, si), *p); r2 = 1 - np.sum(r ** 2) / np.sum((L - L.mean()) ** 2)
    last = s.max(); m = s < last
    q, _ = curve_fit(model, (s[m], si[m]), L[m], p0=p, sigma=w[m], maxfev=100000)
    ho = np.abs(model((s[~m], si[~m]), *q) - L[~m]).max()
    print(f"L = E_N + {B:.4f} (S/1e5)^-{b:.3f} (beta +- {pe[5]:.3f}); floors:",
          {k: f"{E[k]:.4f}+-{e:.4f}" for k, e in zip(SCALES, pe[:4])},
          f"R2 {r2:.4f}; fit without the final checkpoint predicts it to within {ho:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ss = np.logspace(np.log10(8e3), np.log10(7e5), 200)
    for k in SCALES:
        i = sc == k; o = np.argsort(s[i])
        ax[0].errorbar(s[i][o], L[i][o], yerr=se[i][o], fmt="o", color=COLS[k], ms=5, capsize=2,
                       label=k.upper(), zorder=3)
        ax[0].plot(ss, E[k] + B * (ss / 1e5) ** (-b), color=COLS[k], lw=1.4, ls=":")
        ax[1].errorbar(s[i][o], L[i][o] - E[k], yerr=se[i][o], fmt="o", color=COLS[k], ms=5, capsize=2,
                       label=k.upper(), zorder=3)
    ax[0].set_xscale("log"); ax[0].set_xlabel("pretraining steps")
    ax[0].set_ylabel("denoise test loss (per-peak BCE)")
    ax[0].set_title("Test loss vs pretraining (mean ± SE over seeds; dotted: fit)", loc="left", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8.5)
    ax[1].plot(ss, B * (ss / 1e5) ** (-b), color="k", lw=1.4, ls=":", label=f"fit: {B:.3f}·(S/10⁵)^−{b:.2f}")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("pretraining steps")
    ax[1].set_ylabel("reducible loss  L − E_N")
    ax[1].set_title(f"Power law in pretraining steps (all sizes, β = {b:.2f} ± {pe[5]:.2f})", loc="left", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8.5)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(True, color="#e5e7eb", lw=0.6, which="both")
    fig.suptitle("Denoising loss scales with pretraining", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(HERE / "D_denoise_loss_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
