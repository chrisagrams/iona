"""Shared Fourier-feature embedding used for m/z, log-intensity, and Δm/z."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FourierFeatures(nn.Module):
    """Log-spaced Fourier features: x → concat[sin(2π f x), cos(2π f x)].

    Output last dim is 2 * n_freqs. Frequencies are placed log-spaced in
    [f_min, f_max]. Inputs are clamped to ±clamp_abs before featurization to
    prevent NaNs from overflow in sin/cos.
    """

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
        """x: (...,) → (..., 2 * n_freqs)."""
        x = x.clamp(-self.clamp_abs, self.clamp_abs)
        # Broadcast: (...,) × (n_freqs,) → (..., n_freqs)
        phase = 2.0 * math.pi * x.unsqueeze(-1) * self.freqs
        return torch.cat([phase.sin(), phase.cos()], dim=-1)
