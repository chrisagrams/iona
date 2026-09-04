"""Render bias curves and calculate attention entropy."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from msdelta.data.chemistry import ISOTOPES, NEUTRAL_LOSSES, RESIDUES_AA20
from msdelta.model.modeling import DeltaMZBias


def _references_in_range(lo: float, hi: float) -> list[tuple[str, float, str]]:
    """Return references in the specified mass range."""
    refs: list[tuple[str, float, str]] = []
    for name, m in ISOTOPES.items():
        if lo <= m <= hi:
            refs.append((name, m, "tab:blue"))
    for name, m in NEUTRAL_LOSSES.items():
        if lo <= m <= hi:
            refs.append((name, m, "tab:orange"))
    for name, m in RESIDUES_AA20.items():
        if lo <= m <= hi:
            refs.append((name, m, "tab:green"))
    return refs


@torch.no_grad()
def plot_bias_curves(
    bias_module: DeltaMZBias,
    dm_lo: float,
    dm_hi: float,
    step: float,
    title: str,
) -> plt.Figure:
    """Render the bias curve for each attention head."""
    device = next(bias_module.parameters()).device
    grid = torch.arange(dm_lo, dm_hi + step / 2, step, dtype=torch.float32, device=device)
    curves = bias_module.evaluate(grid).cpu().numpy()
    grid_np = grid.cpu().numpy()
    n_heads = curves.shape[1]
    refs = _references_in_range(max(dm_lo, 0.0), dm_hi)

    ncols = 4
    nrows = (n_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.2 * nrows), squeeze=False)
    for h in range(n_heads):
        ax = axes[h // ncols][h % ncols]
        ax.plot(grid_np, curves[:, h], linewidth=0.8)
        for name, m, color in refs:
            ax.axvline(m, color=color, linewidth=0.4, alpha=0.6)
            if len(refs) <= 25:
                ax.text(
                    m,
                    ax.get_ylim()[1],
                    name,
                    fontsize=6,
                    rotation=90,
                    va="top",
                    ha="right",
                    color=color,
                    alpha=0.8,
                )
        ax.set_title(f"head {h}", fontsize=8)
        ax.tick_params(labelsize=6)
    for h in range(n_heads, nrows * ncols):
        axes[h // ncols][h % ncols].axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def render_bias_panels(bias_module: DeltaMZBias, step: int) -> dict[str, plt.Figure]:
    """Render fine and coarse bias panels."""
    fine = plot_bias_curves(
        bias_module,
        dm_lo=-5.0,
        dm_hi=5.0,
        step=0.001,
        title=f"Δm/z bias — fine [-5, 5] Da @ step {step} (Δm=0 not used)",
    )
    coarse = plot_bias_curves(
        bias_module,
        dm_lo=-200.0,
        dm_hi=200.0,
        step=0.01,
        title=f"Δm/z bias — coarse [-200, 200] Da @ step {step} (Δm=0 not used)",
    )
    return {"bias/fine": fine, "bias/coarse": coarse}
