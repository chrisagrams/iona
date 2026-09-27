"""Denoise loss scaling (test loss vs pretraining steps, all sizes): export the data, then plot.

    .venv/bin/python sweeps/plot_denoise_loss_scaling.py

Writes paper/experiments/denoise/loss_scaling/D_denoise_loss_scaling.csv (one row per (size,
pretraining checkpoint): mean / SE / per-seed test loss of the ladder recipe, from plot_ladder.collect)
and runs the standalone plot script in that folder, which does the fit and the figure; the result is
copied to results/processed/figures/SUMMARY/. Plotting only (login node).
"""
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "experiments" / "denoise" / "loss_scaling"
N = {"50m": 49814027, "100m": 100785547, "200m": 202858769, "400m": 395525397}   # from the checkpoints


def main():
    sys.path.insert(0, str(REPO / "sweeps")); import plot_ladder
    cov = plot_ladder.collect()
    rows = ["scale,parameters,pretraining_steps,n_seeds,mean_test_loss,se_test_loss,per_seed_test_loss"]
    for (sc, st), m in sorted(cov.items(), key=lambda kv: (N.get(kv[0][0], 0), kv[0][1])):
        if sc not in N:
            continue          # a scale without its full ladder (25m: one point) is not fitted here
        loss = np.array(m["test_loss"])
        rows.append(f"{sc},{N[sc]},{st},{len(loss)},{loss.mean():.5f},{loss.std(ddof=1) / np.sqrt(len(loss)):.5f},"
                    + " ".join(f"{x:.5f}" for x in loss))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "D_denoise_loss_scaling.csv").write_text("\n".join(rows) + "\n")
    runpy.run_path(str(OUT / "plot_denoise_loss_scaling.py"), run_name="__main__")
    shutil.copy(OUT / "D_denoise_loss_scaling.png", REPO / "results" / "processed" / "figures" / "SUMMARY")
    print("wrote", OUT / "D_denoise_loss_scaling.csv", "and the figure")


if __name__ == "__main__":
    main()
