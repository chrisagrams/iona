"""Bias-curve visualization for periodic logging."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import DeltaMZBias


# Chemically meaningful Δm reference values (Da). Sources: sec 6.2(a) of the
# high-level plan.
ISOTOPES: dict[str, float] = {
    "¹³C": 1.003,
    "2×¹³C": 2.005,
}

NEUTRAL_LOSSES: dict[str, float] = {
    "NH₃": 17.027,
    "H₂O": 18.011,
    "CO": 27.995,
    "CO₂": 43.990,
    "HPO₃": 79.966,
    "H₃PO₄": 97.977,
    "Hexose": 162.053,
}

RESIDUES_AA20: dict[str, float] = {
    "G": 57.021, "A": 71.037, "S": 87.032, "P": 97.053, "V": 99.068,
    "T": 101.048, "C": 103.009, "L/I": 113.084, "N": 114.043, "D": 115.027,
    "Q": 128.059, "K": 128.095, "E": 129.043, "M": 131.040, "H": 137.059,
    "F": 147.068, "R": 156.101, "Y": 163.063, "W": 186.079,
}


def _references_in_range(lo: float, hi: float) -> list[tuple[str, float, str]]:
    """Pick references that fall inside |Δm| ∈ [lo, hi]."""
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
    """Render per-head bias curves on a dense Δm grid (positive side shown).

    Plots ``bias_module`` evaluated on Δm ∈ [dm_lo, dm_hi]. Overlays vertical
    lines at chemically meaningful residue / isotope / loss positions.
    """
    device = next(bias_module.parameters()).device
    grid = torch.arange(dm_lo, dm_hi + step / 2, step, dtype=torch.float32, device=device)
    curves = bias_module.evaluate(grid).cpu().numpy()  # (N, H)
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
            # Label only for sparse residue/isotope plots to keep readable.
            if len(refs) <= 25:
                ax.text(m, ax.get_ylim()[1], name, fontsize=6, rotation=90,
                        va="top", ha="right", color=color, alpha=0.8)
        ax.set_title(f"head {h}", fontsize=8)
        ax.tick_params(labelsize=6)
    for h in range(n_heads, nrows * ncols):
        axes[h // ncols][h % ncols].axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def render_bias_panels(bias_module: DeltaMZBias, step: int) -> dict[str, plt.Figure]:
    """Render both the fine (isotope-scale) and coarse (residue-scale) panels.

    Note: the MLP value at Δm=0 is shown but NOT used in attention (the
    diagonal is masked out in MSEncoder.forward), so it gets no gradient
    and its plotted value is unconstrained — treat with suspicion.
    """
    fine = plot_bias_curves(bias_module, dm_lo=-5.0, dm_hi=5.0, step=0.001,
                            title=f"Δm/z bias — fine [-5, 5] Da @ step {step} (Δm=0 not used)")
    coarse = plot_bias_curves(bias_module, dm_lo=-200.0, dm_hi=200.0, step=0.01,
                              title=f"Δm/z bias — coarse [-200, 200] Da @ step {step} (Δm=0 not used)")
    return {"bias/fine": fine, "bias/coarse": coarse}


def attention_entropy_per_head(blocks) -> np.ndarray | None:
    """Average attention entropy per head from blocks that saved attn weights.

    Returns array of shape (n_layers, n_heads) or None if no attn saved.
    """
    rows = []
    for blk in blocks:
        attn = blk.attn.last_attn
        if attn is None:
            return None
        # attn: (B, H, K, K). Entropy along last dim, mean over (B, K).
        eps = 1e-12
        ent = -(attn * (attn + eps).log()).sum(dim=-1).mean(dim=(0, 2))
        rows.append(ent.float().cpu().numpy())
    return np.stack(rows, axis=0) if rows else None
