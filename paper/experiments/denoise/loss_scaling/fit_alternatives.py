"""Every functional form tried for the denoise loss scaling, fitted to the same data, with a figure each.

    python fit_alternatives.py      # needs matplotlib, numpy, scipy

Reads D_denoise_loss_scaling.csv (this folder; 29 points = 4 sizes x 7-8 pretraining checkpoints, mean
test loss over seeds). N = parameters, S = pretraining steps (proportional to pretraining data: every
size used the same global batch), compute C = N x S (proportional to FLOPs). Weighted least squares,
weights = standard error over seeds floored at 5e-4.

Forms:
  per_size_floor   L = E_N + B (S/1e5)^-b          one floor per size, shared step power law   <- used
  chinchilla       L = E + A (N/1e8)^-a + B (S/1e5)^-b
  compute          L = E + A (C/1e13)^-g           one curve in compute through every point
  pure_power       L = A (N/1e8)^-a (S/1e5)^-b     no floor

Writes fit_comparison.csv (parameters, fit quality, held-out errors) and one figure per alternative:
D_loss_fit_chinchilla.png, D_loss_fit_compute.png, D_loss_fit_pure_power.png. The per-size-floor fit's
figure is D_denoise_loss_scaling.png (plot_denoise_loss_scaling.py).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
SCALES = ["50m", "100m", "200m", "400m"]
COLS = {"50m": "#93c5fd", "100m": "#3b82f6", "200m": "#1d4ed8", "400m": "#1e3a8a"}

rows = list(csv.DictReader(open(HERE / "D_denoise_loss_scaling.csv")))
sc = np.array([r["scale"] for r in rows]); N = np.array([float(r["parameters"]) for r in rows])
S = np.array([float(r["pretraining_steps"]) for r in rows]); L = np.array([float(r["mean_test_loss"]) for r in rows])
SE = np.array([float(r["se_test_loss"]) for r in rows]); W = np.maximum(SE, 5e-4)
SI = np.array([SCALES.index(x) for x in sc], float)
X = (N, S, SI)

FORMS = {
    "per_size_floor": (lambda X, E0, E1, E2, E3, B, b: np.array([E0, E1, E2, E3])[X[2].astype(int)] + B * (X[1] / 1e5) ** -b,
                       (0.3, 0.3, 0.3, 0.3, 0.02, 0.5), ["E_50m", "E_100m", "E_200m", "E_400m", "B", "beta"]),
    "chinchilla": (lambda X, E, A, a, B, b: E + A * (X[0] / 1e8) ** -a + B * (X[1] / 1e5) ** -b,
                   (0.29, 0.01, 0.5, 0.02, 0.5), ["E", "A", "alpha", "B", "beta"]),
    "compute": (lambda X, E, A, g: E + A * (X[0] * X[1] / 1e13) ** -g, (0.29, 0.02, 0.4), ["E", "A", "gamma"]),
    "pure_power": (lambda X, A, a, b: A * (X[0] / 1e8) ** -a * (X[1] / 1e5) ** -b, (0.31, 0.02, 0.04), ["A", "alpha", "beta"]),
}


def fit(f, p0, mask):
    return curve_fit(f, tuple(x[mask] for x in X), L[mask], p0=p0, sigma=W[mask], maxfev=200000)


results = {}
with open(HERE / "fit_comparison.csv", "w") as fh:
    fh.write("form,parameters,values,std_errors,R2,RMSE,chi2_per_dof,max_err_540k_held_out,max_err_400m_held_out\n")
    for name, (f, p0, names) in FORMS.items():
        p, cv = fit(f, p0, np.ones(len(L), bool)); pe = np.sqrt(np.diag(cv))
        r = L - f(X, *p); r2 = 1 - np.sum(r ** 2) / np.sum((L - L.mean()) ** 2)
        chi = np.sum((r / W) ** 2) / (len(L) - len(p))
        held = []
        for label, mask in (("540k", S < S.max()), ("400m", sc != "400m")):
            if name == "per_size_floor" and label == "400m":
                held.append(float("nan")); continue       # a per-size floor cannot predict an unseen size
            q, _ = fit(f, p, mask)
            held.append(float(np.abs(f(tuple(x[~mask] for x in X), *q) - L[~mask]).max()))
        results[name] = (f, p)
        fh.write(f"{name},{' '.join(names)},{' '.join(f'{v:.5g}' for v in p)},{' '.join(f'{v:.3g}' for v in pe)},"
                 f"{r2:.4f},{np.sqrt(np.mean(r ** 2)):.5f},{chi:.2f},{held[0]:.4f},{held[1]:.4f}\n")
        print(f"{name:15s} R2 {r2:.4f} RMSE {np.sqrt(np.mean(r**2)):.5f} chi2/dof {chi:.2f} held-out 540k {held[0]:.4f} 400M {held[1]:.4f}")

ss = np.logspace(np.log10(8e3), np.log10(7e5), 200)
for name in ("chinchilla", "pure_power"):
    f, p = results[name]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for k in SCALES:
        m = sc == k; o = np.argsort(S[m]); n = N[m][0]
        ax.errorbar(S[m][o], L[m][o], yerr=SE[m][o], fmt="o", color=COLS[k], ms=5, capsize=2, label=k.upper(), zorder=3)
        ax.plot(ss, f((np.full_like(ss, n), ss, np.full_like(ss, SCALES.index(k))), *p), ":", color=COLS[k], lw=1.4)
    ax.set_xscale("log"); ax.set_xlabel("pretraining steps"); ax.set_ylabel("denoise test loss (per-peak BCE)")
    ax.set_title({"chinchilla": "Alternative: E + A·N^−α + B·S^−β (α ≈ 2, not used)",
                  "pure_power": "Alternative: A·N^−α·S^−β, no floor (not used)"}[name], loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e5e7eb", lw=0.6, which="both")
    fig.savefig(HERE / f"D_loss_fit_{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

f, p = results["compute"]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
C = N * S; cc = np.logspace(np.log10(C.min() * 0.7), np.log10(C.max() * 1.4), 200)
for k in SCALES:
    m = sc == k
    ax.errorbar(C[m], L[m], yerr=SE[m], fmt="o", color=COLS[k], ms=5, capsize=2, label=k.upper(), zorder=3)
ax.plot(cc, p[0] + p[1] * (cc / 1e13) ** -p[2], "k:", lw=1.5,
        label=f"fit: {p[0]:.3f} + {p[1]:.3f}·(C/10¹³)^−{p[2]:.2f}")
ax.set_xscale("log"); ax.set_xlabel("pretraining compute C = parameters × steps (∝ FLOPs)")
ax.set_ylabel("denoise test loss (per-peak BCE)")
ax.set_title("Alternative: one power law in compute (not used)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8.5); ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, color="#e5e7eb", lw=0.6, which="both")
fig.savefig(HERE / "D_loss_fit_compute.png", dpi=150, bbox_inches="tight"); plt.close(fig)
