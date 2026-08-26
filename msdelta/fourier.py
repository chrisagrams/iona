"""Provide Fourier features and frequency quality metrics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FourierFeatures(nn.Module):
    """Convert scalar values to sine and cosine features."""

    def __init__(
        self,
        n_freqs: int,
        f_min: float,
        f_max: float,
        log_spaced: bool = True,
        learnable: bool = False,
        clamp_abs: float = 2000.0,
    ):
        super().__init__()
        if log_spaced:
            freqs = torch.logspace(math.log10(f_min), math.log10(f_max), n_freqs)
        else:
            freqs = torch.linspace(f_min, f_max, n_freqs)

        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs, persistent=True)

        self.n_freqs = n_freqs
        self.out_dim = 2 * n_freqs
        self.clamp_abs = clamp_abs

    def forward(self, x: Tensor) -> Tensor:
        """Return float32 features with shape ``(..., 2 * n_freqs)``."""
        x = x.float().clamp(-self.clamp_abs, self.clamp_abs)
        phase = 2.0 * math.pi * x.unsqueeze(-1) * self.freqs.float().abs()
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


def interp_mae(
    freqs: Tensor, x: Tensor, *, steps: int = 800, width: int = 96, seed: int = 0
) -> float:
    """Measure the reconstruction error for a set of frequencies."""
    freqs = freqs.detach().float().cpu()
    x = x.detach().float().cpu().flatten()
    if x.numel() < 8:
        return float("nan")
    lo, hi = float(x.min()), float(x.max())
    span = hi - lo
    if span <= 0:
        return float("nan")

    ff = FourierFeatures(freqs.numel(), 1.0, 2.0)
    with torch.no_grad():
        ff.freqs.copy_(freqs)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.numel(), generator=g)
    n_te = max(1, x.numel() // 5)
    x_te, x_tr = x[perm[:n_te]], x[perm[n_te:]]
    y_tr = (x_tr - lo) / span
    y_te = (x_te - lo) / span

    with torch.no_grad():
        ftr, fte = ff(x_tr), ff(x_te)
    net = nn.Sequential(
        nn.Linear(ftr.shape[1], width),
        nn.GELU(),
        nn.Linear(width, width),
        nn.GELU(),
        nn.Linear(width, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(net(ftr).squeeze(-1), y_tr).backward()
        opt.step()
    with torch.no_grad():
        return (net(fte).squeeze(-1) - y_te).abs().mean().item() * span


def dead_freqs(freqs: Tensor, span: float) -> int:
    """Count frequencies that complete less than half a cycle."""
    return int(((freqs.detach().abs().cpu() * span) < 0.5).sum())


def freq_drift(freqs: Tensor, init_freqs: Tensor) -> float:
    """Calculate the mean frequency change on a base-10 log scale."""
    f = freqs.detach().abs().cpu().clamp_min(1e-12)
    f0 = init_freqs.detach().abs().cpu().clamp_min(1e-12)
    return (f.log10() - f0.log10()).abs().mean().item()
