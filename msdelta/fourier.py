"""Provide Fourier features and frequency quality metrics."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FourierFeatures(nn.Module):
    """Convert scalar values to sine and cosine features."""

    freqs: Tensor

    def __init__(
        self,
        n_freqs: int,
        f_min: float,
        f_max: float,
        log_spaced: bool = True,
        clamp_abs: float = 2000.0,
    ):
        super().__init__()
        if log_spaced:
            freqs = torch.logspace(math.log10(f_min), math.log10(f_max), n_freqs)
        else:
            freqs = torch.linspace(f_min, f_max, n_freqs)

        self.register_buffer("freqs", freqs, persistent=True)

        self.n_freqs = n_freqs
        self.out_dim = 2 * n_freqs
        self.clamp_abs = clamp_abs

    def forward(self, x: Tensor) -> Tensor:
        """Return float32 features with shape ``(..., 2 * n_freqs)``."""
        x = x.float().clamp(-self.clamp_abs, self.clamp_abs)
        phase = 2.0 * math.pi * x.unsqueeze(-1) * self.freqs.float()
        return torch.cat([phase.sin(), phase.cos()], dim=-1)
