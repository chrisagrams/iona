"""Pretraining scaling law with an irreducible-loss term (Chinchilla form), vs the form without it.

    python plot_modified_scaling_law.py      # needs matplotlib, numpy, scipy

Reads pretrain_eval_loss.csv (this folder) and fits, to the 40 post-warm-up checkpoints (step > 2,000)
of the five production runs, in log space,

    with E:     L(N, D) = E + A (N / 1e8)^-alpha + B (D / 1e10)^-beta
    without E:  L(N, D) =     A (N / 1e8)^-alpha + B (D / 1e10)^-beta

N = parameters, D = peaks seen (steps x 512 spectra x 512 peaks), compute C = 6 N D -- the conventions
of scripts/plot_pretrain_efficiency_powerlaw.py (repository root), whose no-E fit this reproduces.
Writes three figures and prints both fits (with AIC and held-out checks) and fit_parameters.csv.

Also (end of training only, 540,423 steps, one point per size): final_loss_vs_parameters.png and
final_loss_fit.csv, fitting L(N) = A (N/1e8)^-alpha with and without a floor E; and the crossover tables
equal_steps.csv (every size's loss at each logged step) and equal_compute.csv (every size's loss at
shared compute budgets, log-log interpolated between logged steps, and which size is best).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
MODELS = ["25M", "50M", "100M", "200M", "400M"]
COLS = dict(zip(MODELS, plt.get_cmap("viridis")(np.linspace(0.05, 0.85, 5))))
N_SCALE, D_SCALE = 1e8, 1e10
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
         "savefig.bbox": "tight", "figure.dpi": 200}


def load():
    rows = list(csv.DictReader(open(HERE / "pretrain_eval_loss.csv")))
    for r in rows:
        r["parameters"] = float(r["parameters"]); r["peaks_seen"] = float(r["peaks_seen"])
        r["flops"] = float(r["flops"]); r["eval_loss"] = float(r["eval_loss"]); r["step"] = int(r["step"])
    return rows


def law(n, d, q, with_e):
    """q holds logs (positivity): [log A, log B, log alpha, log beta, (log E)]."""
    e = np.exp(q[4]) if with_e else 0.0
    return e + np.exp(q[0]) * (n / N_SCALE) ** -np.exp(q[2]) + np.exp(q[1]) * (d / D_SCALE) ** -np.exp(q[3])


def fit(n, d, loss, with_e):
    best = None
    for a0 in (0.1, 0.3, 0.6):
        for b0 in (0.3, 0.6, 1.0):
            x0 = np.log([0.05, 0.03, a0, b0] + ([0.04] if with_e else []))
            r = least_squares(lambda q: np.log(law(n, d, q, with_e)) - np.log(loss), x0, max_nfev=20000)
            if best is None or r.cost < best.cost:
                best = r
    res = best.fun
    r2 = 1 - np.sum(res ** 2) / np.sum((np.log(loss) - np.log(loss).mean()) ** 2)
    aic = len(loss) * np.log(np.sum(res ** 2) / len(loss)) + 2 * len(best.x)
    return best.x, r2, aic


def frontier(q, with_e, compute):
    """Lowest loss the fitted law allows at each compute budget C = 6 N D (numerical minimum over N)."""
    nn = np.logspace(6.5, 11, 3000)
    return np.array([np.min(law(nn, c / (6 * nn), q, with_e)) for c in compute])


def params(q, with_e):
    out = {"A": np.exp(q[0]), "alpha": np.exp(q[2]), "B": np.exp(q[1]), "beta": np.exp(q[3])}
    if with_e:
        out = {"E": np.exp(q[4]), **out}
    return out


def main():
    rows = load()
    post = [r for r in rows if r["post_warmup"] == "1"]
    n = np.array([r["parameters"] for r in post]); d = np.array([r["peaks_seen"] for r in post])
    loss = np.array([r["eval_loss"] for r in post]); model = np.array([r["model"] for r in post])
    fits = {}
    for with_e in (False, True):
        q, r2, aic = fit(n, d, loss, with_e)
        held = {}
        for name, keep in (("400M held out", model != "400M"), ("final two checkpoints held out", d < 500000 * 512 * 512)):
            qh, _, _ = fit(n[keep], d[keep], loss[keep], with_e)
            held[name] = np.abs(law(n[~keep], d[~keep], qh, with_e) - loss[~keep]).max()
        fits[with_e] = (q, r2, aic, held)
        print(("with E   " if with_e else "without E"), {k: round(v, 4) for k, v in params(q, with_e).items()},
              f"log R2 {r2:.4f} AIC {aic:.1f}", {k: round(v, 4) for k, v in held.items()})
    with open(HERE / "fit_parameters.csv", "w") as fh:
        fh.write("form,E,A,alpha,B,beta,log_R2,AIC,max_err_400M_held_out,max_err_final_checkpoints_held_out\n")
        for with_e in (False, True):
            q, r2, aic, held = fits[with_e]; p = params(q, with_e)
            fh.write(f"{'with_E' if with_e else 'without_E'},{p.get('E', 0):.5f},{p['A']:.5f},{p['alpha']:.4f},"
                     f"{p['B']:.5f},{p['beta']:.4f},{r2:.4f},{aic:.2f},"
                     f"{held['400M held out']:.5f},{held['final two checkpoints held out']:.5f}\n")
    q0 = fits[False][0]; q1 = fits[True][0]; E = np.exp(q1[4])
    by = {m: sorted([r for r in rows if r["model"] == m], key=lambda r: r["step"]) for m in MODELS}

    with plt.rc_context(STYLE):
        # 1. compute-optimal frontier implied by each fit
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for m in MODELS:
            p = [r for r in by[m] if r["post_warmup"] == "1"]
            ax.plot([r["flops"] for r in p], [r["eval_loss"] for r in p], "o-", color=COLS[m], ms=4.5, lw=1.2,
                    label=m, zorder=3)
        cs = np.logspace(17.2, np.log10(4e20), 200)
        ax.plot(cs, frontier(q0, False, cs), ":", color="#d97706", lw=2,
                label=f"fit without E (α = {np.exp(q0[2]):.2f}, β = {np.exp(q0[3]):.2f})")
        ax.plot(cs, frontier(q1, True, cs), "--", color="k", lw=1.6,
                label=f"fit with E = {E:.3f} (α = {np.exp(q1[2]):.2f}, β = {np.exp(q1[3]):.2f})")
        ax.axhline(E, color="#6b7280", lw=0.8, ls=(0, (1, 3)))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2e17, 4e20); ax.set_ylim(0.035, 0.17)
        ax.set_xlabel("cumulative training compute (FLOPs)"); ax.set_ylabel("evaluation loss")
        ax.set_title("Compute-optimal frontier with and without an irreducible loss", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7.8); ax.grid(True, color="#e5e7eb", lw=0.6, which="both")
        fig.savefig(HERE / "scaling_law_frontier_E_vs_noE.png"); plt.close(fig)

        # 2. loss vs compute, each size with its fitted curve (with E) dashed
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ss = np.logspace(np.log10(4e3), np.log10(6.5e5), 200) * 512 * 512
        for m in MODELS:
            p = [r for r in by[m] if r["post_warmup"] == "1"]; nm = p[0]["parameters"]
            ax.plot([r["flops"] for r in p], [r["eval_loss"] for r in p], "o-", color=COLS[m], ms=4.5, lw=1.2,
                    label=m, zorder=3)
            ax.plot(6 * nm * ss, law(np.full_like(ss, nm), ss, q1, True), "--", color=COLS[m], lw=1.1)
        ax.plot(cs, frontier(q1, True, cs), "--", color="k", lw=1.6, label="fit: compute-optimal frontier")
        ax.axhline(E, color="#6b7280", lw=0.8, ls=(0, (1, 3)))
        ax.text(2.4e17, E * 1.025, f"E = {E:.3f}", color="#6b7280", fontsize=8.5)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2e17, 4e20); ax.set_ylim(0.035, 0.2)
        ax.set_xlabel("cumulative training compute (FLOPs)"); ax.set_ylabel("evaluation loss")
        ax.set_title("Evaluation loss vs compute (dashed: fitted law with E)", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8, title="model size", title_fontsize=8)
        ax.grid(True, color="#e5e7eb", lw=0.6, which="both")
        fig.savefig(HERE / "scaling_law_loss_vs_flops.png"); plt.close(fig)

        # 3. the fit with E along both axes: loss vs data per size, loss vs size per data budget
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
        for m in MODELS:
            p = [r for r in by[m] if r["post_warmup"] == "1"]; nm = p[0]["parameters"]
            ax[0].plot([r["peaks_seen"] for r in p], [r["eval_loss"] for r in p], "o", color=COLS[m], ms=5,
                       label=m, zorder=3)
            ax[0].plot(ss, law(np.full_like(ss, nm), ss, q1, True), "--", color=COLS[m], lw=1.3)
        nn = np.logspace(np.log10(1.5e7), 9, 200)
        for st, c in ((10000, "#d1d5db"), (50000, "#9ca3af"), (200000, "#6b7280"), (540423, "#111827")):
            pts = [(r["parameters"], r["eval_loss"]) for r in rows if r["step"] == st]
            ax[1].plot([x[0] for x in pts], [x[1] for x in pts], "o", color=c, ms=6, label=f"{st // 1000}k steps", zorder=3)
            ax[1].plot(nn, law(nn, np.full_like(nn, st * 512 * 512), q1, True), "--", color=c, lw=1.3)
        for a in ax:
            a.axhline(E, color="#6b7280", lw=0.8, ls=(0, (1, 3)))
            a.set_xscale("log"); a.set_yscale("log"); a.grid(True, color="#e5e7eb", lw=0.6, which="both")
            a.legend(frameon=False, fontsize=8.5)
        ax[0].set_xlabel("peaks seen D"); ax[0].set_ylabel("evaluation loss")
        ax[0].set_title("Loss vs data, per model size", loc="left", fontsize=10)
        ax[1].set_xlabel("parameters N"); ax[1].set_title("Loss vs model size, per data budget", loc="left", fontsize=10)
        fig.suptitle(f"L = {E:.3f} + {np.exp(q1[0]):.4f}·(N/10⁸)^−{np.exp(q1[2]):.2f} + "
                     f"{np.exp(q1[1]):.4f}·(D/10¹⁰)^−{np.exp(q1[3]):.2f}", x=0.01, ha="left", fontweight="bold", fontsize=11)
        fig.tight_layout(); fig.savefig(HERE / "scaling_law_fit_loglog.png"); plt.close(fig)


def extras():
    from scipy.optimize import curve_fit
    rows = load()
    by = {m: sorted([r for r in rows if r["model"] == m], key=lambda r: r["step"]) for m in MODELS}
    # final loss vs parameters
    fin = [by[m][-1] for m in MODELS]
    n = np.array([r["parameters"] for r in fin]); l = np.array([r["eval_loss"] for r in fin])
    pp, cp = curve_fit(lambda x, A, a: A * (x / N_SCALE) ** -a, n, l, p0=(0.06, 0.1))
    pf, cf = curve_fit(lambda x, E, A, a: E + A * (x / N_SCALE) ** -a, n, l, p0=(0.05, 0.005, 0.5), maxfev=100000)
    ep, ef = np.sqrt(np.diag(cp)), np.sqrt(np.diag(cf))
    rp = l - pp[0] * (n / N_SCALE) ** -pp[1]; rf = l - (pf[0] + pf[1] * (n / N_SCALE) ** -pf[2])
    with open(HERE / "final_loss_fit.csv", "w") as fh:
        fh.write("form,E,E_se,A,alpha,alpha_se,RMSE,residuals_25M_50M_100M_200M_400M\n")
        fh.write(f"power_law,,,{pp[0]:.5f},{pp[1]:.4f},{ep[1]:.4f},{np.sqrt(np.mean(rp**2)):.5f},{' '.join(f'{x:+.5f}' for x in rp)}\n")
        fh.write(f"power_law_plus_floor,{pf[0]:.5f},{ef[0]:.5f},{pf[1]:.5f},{pf[2]:.4f},{ef[2]:.4f},"
                 f"{np.sqrt(np.mean(rf**2)):.5f},{' '.join(f'{x:+.5f}' for x in rf)}\n")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 4.6)); nn = np.logspace(np.log10(1.8e7), np.log10(8e8), 200)
        ax.plot(nn, pp[0] * (nn / N_SCALE) ** -pp[1], "--", color="#9ca3af", lw=1.4,
                label=f"power law: {pp[0]:.4f}·(N/10⁸)^−{pp[1]:.3f}")
        ax.plot(nn, pf[0] + pf[1] * (nn / N_SCALE) ** -pf[2], ":", color="#1e3a8a", lw=1.6,
                label=f"power law + floor: {pf[0]:.4f} + {pf[1]:.4f}·(N/10⁸)^−{pf[2]:.2f}")
        ax.axhline(pf[0], color="#1e3a8a", lw=0.8, alpha=0.4)
        ax.plot(n, l, "o", color="#1e3a8a", ms=7, zorder=3)
        for x, y, m in zip(n, l, MODELS):
            ax.annotate(f"{m}\n{y:.4f}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("parameters N")
        ax.set_ylabel("evaluation loss, end of training")
        ax.set_title("End-of-training loss vs model size (same data, 3 epochs)", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8); ax.grid(True, color="#e5e7eb", lw=0.6, which="both")
        fig.savefig(HERE / "final_loss_vs_parameters.png"); plt.close(fig)
    # crossover tables
    steps = [r["step"] for r in by["50M"]]
    with open(HERE / "equal_steps.csv", "w") as fh:
        fh.write("step," + ",".join(MODELS) + ",best\n")
        for st in steps:
            v = {m: next(r["eval_loss"] for r in by[m] if r["step"] == st) for m in MODELS}
            fh.write(f"{st}," + ",".join(f"{v[m]:.5f}" for m in MODELS) + f",{min(v, key=v.get)}\n")
    with open(HERE / "equal_compute.csv", "w") as fh:
        fh.write("flops," + ",".join(MODELS) + ",best\n")
        for c in np.logspace(17, np.log10(by["400M"][-1]["flops"]), 19):
            v = {}
            for m in MODELS:
                f = np.array([r["flops"] for r in by[m]]); y = np.array([r["eval_loss"] for r in by[m]])
                if f[0] <= c <= f[-1] * 1.0001:
                    v[m] = float(np.exp(np.interp(np.log(c), np.log(f), np.log(y))))
            fh.write(f"{c:.3e}," + ",".join(f"{v[m]:.5f}" if m in v else "" for m in MODELS)
                     + f",{min(v, key=v.get) if v else ''}\n")
    print("final-loss fits:", np.round(pp, 4), np.round(pf, 4))


if __name__ == "__main__":
    main()
    extras()
