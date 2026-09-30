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
        clamp_abs: float = 2000.0,
    ):
        super().__init__()
        freqs = torch.logspace(math.log10(f_min), math.log10(f_max), n_freqs)

        self.register_buffer("freqs", freqs, persistent=True)

        self.n_freqs = n_freqs
        self.out_dim = 2 * n_freqs
        self.clamp_abs = clamp_abs

    def forward(
        self, x: Tensor, dtype: torch.dtype = torch.float32, chunk_size: int = 32
    ) -> Tensor:
        """Return features with shape ``(..., 2 * n_freqs)``; phases are always computed in float32."""
        x = x.float().clamp(-self.clamp_abs, self.clamp_abs).unsqueeze(-1)
        feats = x.new_empty(*x.shape[:-1], self.out_dim, dtype=dtype)
        # Fill a few frequencies at a time so full-size float32 intermediates never exist.
        for start in range(0, self.n_freqs, chunk_size):
            stop = min(start + chunk_size, self.n_freqs)
            phase = 2.0 * math.pi * x * self.freqs[start:stop].float()
            feats[..., start:stop] = phase.sin()
            feats[..., self.n_freqs + start : self.n_freqs + stop] = phase.cos()
        return feats
