"""Bias-curve visualization for periodic logging."""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .model import DeltaMZBias


# Chemically meaningful Δm reference values (Da). Sources: sec 6.2(a) of the
# high-level plan.
# Charge-aware: the ¹³C M→M+1 spacing in *m/z* is 1.00335/z, so multiply-
# charged peaks (the bulk of the data, z=2/3) show isotope structure at
# 0.5 and 0.33 Da — not 1.003. Score all the distinct M+1/M+2 spacings for
# z=1,2,3 or we miss the dominant (multiply-charged) isotope signal.
_C13 = 1.0033548
ISOTOPES: dict[str, float] = {
    "¹³C z3": _C13 / 3,        # 0.334  (z=3, M+1)
    "¹³C z2": _C13 / 2,        # 0.502  (z=2, M+1)
    "2¹³C z3": 2 * _C13 / 3,   # 0.669  (z=3, M+2)
    "¹³C": _C13,               # 1.003  (z=1 M+1; also z=2 M+2)
    "2¹³C": 2 * _C13,          # 2.007  (z=1, M+2)
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

    Note: the bucket value at Δm=0 is shown but NOT used in attention (the
    diagonal is masked out in MSEncoder.forward), so it gets no gradient
    and its plotted value is unconstrained — treat with suspicion.
    """
    fine = plot_bias_curves(bias_module, dm_lo=-5.0, dm_hi=5.0, step=0.001,
                            title=f"Δm/z bias — fine [-5, 5] Da @ step {step} (Δm=0 not used)")
    coarse = plot_bias_curves(bias_module, dm_lo=-200.0, dm_hi=200.0, step=0.01,
                              title=f"Δm/z bias — coarse [-200, 200] Da @ step {step} (Δm=0 not used)")
    return {"bias/fine": fine, "bias/coarse": coarse}


class AttentionRecorder:
    """Record per-layer attention weights for forward passes run inside it.

    SDPA never materializes attention weights, so a forward pre-hook on each
    ``BiasedMHA`` captures the layer's inputs and recomputes
    ``softmax(q·kᵀ/√d + bias)`` under no_grad — the pre-dropout weights the
    layer itself uses. Costs one extra QK matmul + softmax per layer, only
    while recording.

        with AttentionRecorder(encoder.blocks) as rec:
            encoder(...)
        rec.attn  # list of (B, H, K, K), one per layer per forward
    """

    def __init__(self, blocks):
        self._modules = [blk.attn for blk in blocks]
        self._handles: list = []
        self.attn: list[Tensor] = []

    def __enter__(self) -> "AttentionRecorder":
        self.attn.clear()
        for m in self._modules:
            self._handles.append(m.register_forward_pre_hook(self._hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def _hook(self, module, args) -> None:
        x, bias, key_padding_mask = args
        B, K, _ = x.shape
        qkv = module.qkv(x).reshape(B, K, 3, module.n_heads, module.d_head)
        q, k, _ = qkv.unbind(dim=2)
        logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)
        logits = logits / math.sqrt(module.d_head)
        logits = logits + bias.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        self.attn.append(F.softmax(logits, dim=-1))


def attention_entropy_per_head(attn_layers: list[Tensor]) -> np.ndarray | None:
    """Average attention entropy per head from recorded attention weights.

    attn_layers: one (B, H, K, K) tensor per layer (AttentionRecorder.attn).
    Returns array of shape (n_layers, n_heads) or None if nothing recorded.
    """
    rows = []
    for attn in attn_layers:
        # attn: (B, H, K, K). Entropy along last dim, mean over (B, K).
        eps = 1e-12
        ent = -(attn * (attn + eps).log()).sum(dim=-1).mean(dim=(0, 2))
        rows.append(ent.float().cpu().numpy())
    return np.stack(rows, axis=0) if rows else None
