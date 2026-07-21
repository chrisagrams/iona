"""Standalone Fourier-encoding resolution probe — NO model, checkpoint, or data.

Rationalizes the otherwise-arbitrary Fourier ranges used by PeakEmbed (intensity)
and PrecursorEmbed (precursor m/z). Because ``FourierFeatures`` uses FIXED
frequencies, the encoding is a deterministic function of the input scalar, so we
can test a config in isolation: freeze the featurizer, fit a throwaway readout
MLP on a grid of scalars, and measure reconstruction MAE on scalars *between* the
grid points (interpolation). Two failure modes both surface this way:

  * f_min too low  -> features ~constant over the domain -> can't separate values
  * f_max too high -> oscillation aliases between grid points -> can't interpolate

Run:  python scripts/fourier_probe.py
Emits a printed sweep table and a grid figure (f_min x f_max MAE heatmap per
input, current config marked) to docs/figures/fourier_probe.png.

Run:  python scripts/fourier_probe.py --learnable
Instead makes the frequencies trainable (initialized at the current wide range)
and plots where they migrate under the reconstruction objective -> an
independent read on the data-preferred range. Writes docs/figures/
fourier_learnable.png. Caveat: this optimizes the standalone *reconstruction*
proxy, and frequency space is non-convex, so the histogram confirms the range
the heatmap sweep found; it does not replace a real-model training run.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from msdelta.fourier import FourierFeatures

torch.manual_seed(0)


# --- one reconstruction measurement ---------------------------------------

def interp_mae(n_freqs: int, f_min: float, f_max: float, lo: float, hi: float,
               n_grid: int, steps: int = 800, width: int = 96) -> float:
    """Held-out interpolation MAE (native units) for one Fourier config.

    Fit a small MLP to map FourierFeatures(x) -> x on an evenly spaced grid over
    [lo, hi], then evaluate on the unseen midpoints between grid points.
    """
    span = hi - lo
    x_tr = torch.linspace(lo, hi, n_grid)
    x_te = (x_tr[:-1] + x_tr[1:]) / 2
    y_tr = (x_tr - lo) / span            # target normalized to [0,1]
    y_te = (x_te - lo) / span

    ff = FourierFeatures(n_freqs, f_min, f_max)
    with torch.no_grad():
        ftr, fte = ff(x_tr), ff(x_te)

    net = nn.Sequential(
        nn.Linear(ftr.shape[1], width), nn.GELU(),
        nn.Linear(width, width), nn.GELU(),
        nn.Linear(width, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(net(ftr).squeeze(-1), y_tr).backward()
        opt.step()
    with torch.no_grad():
        return (net(fte).squeeze(-1) - y_te).abs().mean().item() * span


def dead_freqs(n_freqs: int, f_min: float, f_max: float, span: float) -> int:
    """How many freqs complete < 0.5 cycles over the domain (~constant, wasted)."""
    freqs = torch.logspace(math.log10(f_min), math.log10(f_max), n_freqs)
    return int(((freqs * span) < 0.5).sum())


# --- learnable frequencies (tier 2) ---------------------------------------

class LogLearnableFourier(nn.Module):
    """Fourier features whose frequencies are trainable in LOG space.

    We optimize log10(freq), not freq directly (as FourierFeatures(learnable=True)
    would): with Adam an additive step is the same size for every parameter, so a
    linear-space step that suits a ~1e3 frequency would fling a ~1e-4 frequency
    negative. In log space each step is multiplicative, which is the scale-
    invariant behavior needed to move frequencies spanning many decades at once.
    Init matches FourierFeatures exactly (log-spaced in [f_min, f_max]).
    """

    def __init__(self, n_freqs: int, f_min: float, f_max: float):
        super().__init__()
        self.log_freqs = nn.Parameter(
            torch.linspace(math.log10(f_min), math.log10(f_max), n_freqs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * x.unsqueeze(-1) * (10.0 ** self.log_freqs)
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


def learn_freqs(task: "Task", steps: int = 3000, freq_lr: float = 1e-2,
                readout_lr: float = 3e-3, width: int = 96):
    """Train frequencies + readout on the interpolation objective.

    Frequencies start at the current (deliberately wide) range and are free to
    migrate. Returns (init_freqs, final_freqs, held_out_mae).
    """
    lo, hi, span = task.lo, task.hi, task.span
    x_tr = torch.linspace(lo, hi, task.n_grid)
    x_te = (x_tr[:-1] + x_tr[1:]) / 2
    y_tr = (x_tr - lo) / span
    y_te = (x_te - lo) / span

    ff = LogLearnableFourier(task.n_freqs, *task.current)
    init_freqs = (10.0 ** ff.log_freqs.detach()).clone()
    readout = nn.Sequential(
        nn.Linear(2 * task.n_freqs, width), nn.GELU(),
        nn.Linear(width, width), nn.GELU(),
        nn.Linear(width, 1),
    )
    opt = torch.optim.Adam([
        {"params": ff.parameters(), "lr": freq_lr},
        {"params": readout.parameters(), "lr": readout_lr},
    ])
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(readout(ff(x_tr)).squeeze(-1), y_tr).backward()
        opt.step()
    with torch.no_grad():
        final_freqs = (10.0 ** ff.log_freqs).clone()
        mae = (readout(ff(x_te)).squeeze(-1) - y_te).abs().mean().item() * span
    return init_freqs.numpy(), final_freqs.numpy(), mae


# --- tasks -----------------------------------------------------------------

class Task:
    def __init__(self, name, lo, hi, n_grid, n_freqs, current,
                 fmin_decades, fmax_decades):
        self.name = name
        self.lo, self.hi = lo, hi
        self.span = hi - lo
        self.n_grid = n_grid
        self.n_freqs = n_freqs
        self.current = current            # (f_min, f_max)
        self.fmins = np.logspace(*fmin_decades[:2], fmin_decades[2])
        self.fmaxs = np.logspace(*fmax_decades[:2], fmax_decades[2])


TASKS = [
    # log_int = log1p(intensity)/max lives in [0, 1] (msdelta/data.py)
    Task("intensity (log_int in [0,1])", 0.0, 1.0, n_grid=200, n_freqs=16,
         current=(1e-2, 1e2), fmin_decades=(-2, 1, 7), fmax_decades=(0, 2, 7)),
    # precursor m/z: raw m/z ~ [100, 2000] Da
    Task("precursor m/z (Da)", 100.0, 2000.0, n_grid=400, n_freqs=64,
         current=(1e-2, 1e3), fmin_decades=(-4, -1, 7), fmax_decades=(-1, 3, 7)),
]


def sweep(task: Task) -> np.ndarray:
    """MAE over the f_min x f_max grid; cells with f_min >= f_max are NaN."""
    grid = np.full((len(task.fmins), len(task.fmaxs)), np.nan)
    for i, fmin in enumerate(task.fmins):
        for j, fmax in enumerate(task.fmaxs):
            if fmin >= fmax:
                continue
            grid[i, j] = interp_mae(task.n_freqs, float(fmin), float(fmax),
                                    task.lo, task.hi, task.n_grid)
    return grid


# --- reporting -------------------------------------------------------------

def print_table(task: Task, grid: np.ndarray) -> None:
    cur_mae = interp_mae(task.n_freqs, *task.current, task.lo, task.hi, task.n_grid)
    best = np.unravel_index(np.nanargmin(grid), grid.shape)
    print(f"\n=== {task.name}: domain [{task.lo}, {task.hi}], "
          f"n_freqs={task.n_freqs} ===")
    print(f"  current  f_min={task.current[0]:.0e} f_max={task.current[1]:.0e}"
          f"  MAE={cur_mae:.4g}  dead={dead_freqs(task.n_freqs, *task.current, task.span)}"
          f"/{task.n_freqs}")
    print(f"  best     f_min={task.fmins[best[0]]:.0e} f_max={task.fmaxs[best[1]]:.0e}"
          f"  MAE={grid[best]:.4g}  dead={dead_freqs(task.n_freqs, float(task.fmins[best[0]]), float(task.fmaxs[best[1]]), task.span)}"
          f"/{task.n_freqs}   ({cur_mae/grid[best]:.0f}x better)")


def make_figure(tasks, grids, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.4 * len(tasks), 5.2))
    if len(tasks) == 1:
        axes = [axes]
    for ax, task, grid in zip(axes, tasks, grids):
        img = np.log10(grid)                       # log MAE for legible dynamic range
        im = ax.imshow(img, origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(task.fmaxs)))
        ax.set_xticklabels([f"{v:.0e}" for v in task.fmaxs], rotation=45, ha="right")
        ax.set_yticks(range(len(task.fmins)))
        ax.set_yticklabels([f"{v:.0e}" for v in task.fmins])
        ax.set_xlabel("f_max")
        ax.set_ylabel("f_min")
        ax.set_title(task.name)

        # Mark the current config at its nearest grid cell.
        ci = int(np.argmin(np.abs(np.log10(task.fmins) - math.log10(task.current[0]))))
        cj = int(np.argmin(np.abs(np.log10(task.fmaxs) - math.log10(task.current[1]))))
        ax.scatter([cj], [ci], marker="*", s=320, c="red", edgecolors="white",
                   linewidths=1.2, label="current", zorder=5)
        # Mark the best cell.
        bi, bj = np.unravel_index(np.nanargmin(grid), grid.shape)
        ax.scatter([bj], [bi], marker="o", s=140, facecolors="none",
                   edgecolors="white", linewidths=2.0, label="best", zorder=5)
        ax.legend(loc="upper left", framealpha=0.85)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("log10  interpolation MAE (native units)")

    fig.suptitle("Fourier-encoding resolution: interpolation MAE over f_min x f_max\n"
                 "(lower = better; red star = current config, white circle = best)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nwrote {out_path}")


def make_learnable_figure(tasks, results, out_path: Path) -> None:
    """Init-vs-final frequency histograms (log x-axis), one panel per input."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.4 * len(tasks), 4.6))
    if len(tasks) == 1:
        axes = [axes]
    for ax, task, (init_f, final_f, mae) in zip(axes, tasks, results):
        both = np.concatenate([init_f, final_f])
        bins = np.logspace(np.log10(both.min()) - 0.1, np.log10(both.max()) + 0.1, 24)
        ax.hist(init_f, bins=bins, alpha=0.45, color="gray", label="init (current range)")
        ax.hist(final_f, bins=bins, alpha=0.7, color="tab:blue", label="learned")
        # Rug of learned positions so individual frequencies are visible.
        ax.plot(final_f, np.full_like(final_f, -0.5), "|", color="tab:blue", ms=10)
        ax.set_xscale("log")
        ax.set_xlabel("frequency")
        ax.set_ylabel("count")
        ax.set_title(f"{task.name}\nn_freqs={task.n_freqs}, held-out MAE={mae:.3g}")
        ax.legend(loc="upper right", framealpha=0.85)

    fig.suptitle("Learnable Fourier frequencies: where they migrate from the current range\n"
                 "(gray = initialization, blue = after training on reconstruction)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nwrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--learnable", action="store_true",
                    help="train the frequencies and plot init-vs-final histograms")
    ap.add_argument("--out", type=Path, default=None,
                    help="figure path (defaults per mode under docs/figures/)")
    args = ap.parse_args()

    figdir = Path(__file__).resolve().parents[1] / "docs" / "figures"

    if args.learnable:
        results = []
        for task in TASKS:
            init_f, final_f, mae = learn_freqs(task)
            print(f"\n=== {task.name}: learnable frequencies ===")
            print(f"  init  range [{init_f.min():.2e}, {init_f.max():.2e}]  (current)")
            print(f"  final range [{final_f.min():.2e}, {final_f.max():.2e}]  "
                  f"median={np.median(final_f):.2e}  held-out MAE={mae:.4g}")
            results.append((init_f, final_f, mae))
        make_learnable_figure(TASKS, results, args.out or figdir / "fourier_learnable.png")
        return

    grids = []
    for task in TASKS:
        grid = sweep(task)
        print_table(task, grid)
        grids.append(grid)
    make_figure(TASKS, grids, args.out or figdir / "fourier_probe.png")


if __name__ == "__main__":
    main()
