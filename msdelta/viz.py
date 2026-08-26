"""Render bias curves and calculate attention entropy."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from msdelta.chemistry import ISOTOPES, NEUTRAL_LOSSES, RESIDUES_AA20
from msdelta.model import DeltaMZBias


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


class AttentionRecorder:
    """Record attention weights during a forward pass."""

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
    """Calculate the mean attention entropy for each head."""
    rows = []
    for attn in attn_layers:
        eps = 1e-12
        ent = -(attn * (attn + eps).log()).sum(dim=-1).mean(dim=(0, 2))
        rows.append(ent.float().cpu().numpy())
    return np.stack(rows, axis=0) if rows else None
