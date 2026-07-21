"""Shared Fourier-feature embedding used for m/z, log-intensity, and Δm/z.

Also holds the metrics that judge whether a set of frequencies is *effective*
(used by the inline ``FourierProbeCallback`` and the offline
``scripts/fourier_probe.py``): a featurizer is only useful if a downstream
readout can recover the original scalar from its encoding, so we fit a small
throwaway MLP to map ``Fourier(x) -> x`` and measure held-out reconstruction
error. Two failure modes surface as high error — frequencies too low (features
~constant, values collapse) or too high (oscillation aliases between samples).
The metric helpers take an explicit ``freqs`` tensor and value tensor, so the
same code judges a fixed config on a synthetic grid and the model's live learned
frequencies on real sampled values.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FourierFeatures(nn.Module):
    """Log-spaced Fourier features: x → concat[sin(2π f x), cos(2π f x)].

    Output last dim is 2 * n_freqs. Frequencies are initialized log-spaced in
    [f_min, f_max]. Inputs are clamped to ±clamp_abs before featurization to
    prevent NaNs from overflow in sin/cos.

    With ``learnable=True`` the frequencies become a trainable ``nn.Parameter``
    (linear space) that the model optimizes end-to-end. ``forward`` reads them
    through ``abs()`` so an optimizer step that pushes a frequency past zero
    can't produce a negative frequency: sin/cos are symmetric in the sign of f
    (cos is even, sin's sign flips but pairs with cos), so |f| is behavior-
    preserving at init and keeps the featurizer well-defined throughout training.
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
        """x: (...,) → (..., 2 * n_freqs), always fp32.

        m/z (and Δm) precision is load-bearing: isotope spacing is ~1 Da on m/z
        up to ~2000, where bf16's ulp is ~8 Da, so featurizing in bf16 would
        erase the very structure the Δm bias reads. We therefore compute in
        fp32 regardless of the surrounding precision mode (bf16 autocast, or
        DeepSpeed casting params/buffers to bf16); callers cast the result to
        their own weight dtype before the first matmul.
        """
        x = x.float().clamp(-self.clamp_abs, self.clamp_abs)
        # Broadcast: (...,) × (n_freqs,) → (..., n_freqs). abs() keeps a
        # learned frequency well-defined if optimization pushes it below zero.
        phase = 2.0 * math.pi * x.unsqueeze(-1) * self.freqs.float().abs()
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


# ---------- effectiveness metrics ----------

def interp_mae(freqs: Tensor, x: Tensor, *, steps: int = 800, width: int = 96,
               seed: int = 0) -> float:
    """Held-out reconstruction MAE (native units) for one set of frequencies.

    Encode the values with these frequencies, fit a small MLP to map
    ``Fourier(x) -> x`` on a random train split, then measure absolute error on
    the held-out split. Lower is better: the encoding keeps enough information to
    tell the values apart. Returns ``nan`` if there are too few / degenerate
    values to split.
    """
    freqs = freqs.detach().float().cpu()
    x = x.detach().float().cpu().flatten()
    if x.numel() < 8:
        return float("nan")
    lo, hi = float(x.min()), float(x.max())
    span = hi - lo
    if span <= 0:
        return float("nan")

    # Encode through the module so the metric sees exactly what the featurizer
    # computes (fp32, clamp, abs()); frequencies are fixed for this measurement.
    ff = FourierFeatures(freqs.numel(), 1.0, 2.0)   # placeholder init, overwritten
    with torch.no_grad():
        ff.freqs.copy_(freqs)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.numel(), generator=g)
    n_te = max(1, x.numel() // 5)
    x_te, x_tr = x[perm[:n_te]], x[perm[n_te:]]
    y_tr = (x_tr - lo) / span                    # target normalized to [0,1]
    y_te = (x_te - lo) / span

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


def dead_freqs(freqs: Tensor, span: float) -> int:
    """How many freqs complete < 0.5 cycles over the value range (~constant)."""
    return int(((freqs.detach().abs().cpu() * span) < 0.5).sum())


def freq_drift(freqs: Tensor, init_freqs: Tensor) -> float:
    """Mean |Δlog10 f| between current and initial frequencies (movement size)."""
    f = freqs.detach().abs().cpu().clamp_min(1e-12)
    f0 = init_freqs.detach().abs().cpu().clamp_min(1e-12)
    return (f.log10() - f0.log10()).abs().mean().item()
