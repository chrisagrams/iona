"""Peak-token transformer with learned per-head Δm/z attention bias + MPM heads."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .fourier import FourierFeatures


@dataclass
class FourierConfig:
    n_freqs: int
    f_min: float
    f_max: float


@dataclass
class DeltaBiasConfig:
    n_freqs: int = 64
    # Per-head hidden width. Each head gets its OWN MLP (no parameter
    # sharing across heads), so total bias capacity is n_heads * per_head_hidden.
    # Smaller than the old shared `hidden` because activation memory is
    # (B, K, K, n_heads * per_head_hidden) — set this to keep that under
    # ~6 GB at bf16 (per_head_hidden=32 with B=256, K=150, H=8).
    per_head_hidden: int = 32
    f_min: float = 1e-2
    f_max: float = 1e3
    # Bound the per-head bias to ±scale logits via scale*tanh(raw/scale).
    # Keeps the bias comparable to the content term (q·k/√d ~ O(1-2)) so
    # neither can steamroll the other — prevents the "bias dominates
    # content" runaway (high-level plan §6.3).
    scale: float = 3.0


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ffn_mult: int = 4
    dropout: float = 0.1
    max_peaks: int = 150
    fourier_mz: FourierConfig = field(default_factory=lambda: FourierConfig(64, 1e-2, 1e3))
    fourier_int: FourierConfig = field(default_factory=lambda: FourierConfig(16, 1e-2, 1e2))
    delta_bias: DeltaBiasConfig = field(default_factory=DeltaBiasConfig)


class PeakEmbed(nn.Module):
    """(m/z, log_int) → d_model token. No masking; denoising autoencoder."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ff_mz = FourierFeatures(cfg.fourier_mz.n_freqs, cfg.fourier_mz.f_min, cfg.fourier_mz.f_max)
        self.ff_int = FourierFeatures(cfg.fourier_int.n_freqs, cfg.fourier_int.f_min, cfg.fourier_int.f_max)
        in_dim = self.ff_mz.out_dim + self.ff_int.out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, mz: Tensor, log_int: Tensor) -> Tensor:
        """mz, log_int: (B, K) → (B, K, d_model)."""
        feats = torch.cat([self.ff_mz(mz), self.ff_int(log_int)], dim=-1)
        return self.mlp(feats)


class DeltaMZBias(nn.Module):
    """Learned per-head additive attention bias from signed Δm/z.

    Each head has its OWN independent MLP (no parameter sharing across
    heads). Implemented as stacked per-head Parameter tensors with einsum
    for efficient batched compute. Decoupling the gradient pools lets
    heads specialize on different Δm regions instead of all collapsing
    to whichever single feature dominates the data (e.g., suppression of
    small-Δm near-duplicate peaks).
    """

    def __init__(self, n_heads: int, cfg: DeltaBiasConfig):
        super().__init__()
        self.ff = FourierFeatures(cfg.n_freqs, cfg.f_min, cfg.f_max, log_spaced=True)
        self.n_heads = n_heads
        self.per_head_hidden = cfg.per_head_hidden
        self.scale = cfg.scale

        in_dim = self.ff.out_dim
        H, D_h = n_heads, cfg.per_head_hidden

        # Per-head first layer: 8 separate Linear(in_dim, per_head_hidden).
        self.w1 = nn.Parameter(torch.empty(H, in_dim, D_h))
        self.b1 = nn.Parameter(torch.zeros(H, D_h))
        # Per-head output projection: 8 separate Linear(per_head_hidden, 1).
        self.w2 = nn.Parameter(torch.zeros(H, D_h))     # zero-init → bias starts at 0
        self.b2 = nn.Parameter(torch.zeros(H))

        # Init each head's first layer the same way nn.Linear does (Kaiming).
        for h in range(H):
            nn.init.kaiming_uniform_(self.w1[h], a=math.sqrt(5))
        bound = 1 / math.sqrt(in_dim)
        nn.init.uniform_(self.b1, -bound, bound)

    def _bound(self, out: Tensor) -> Tensor:
        # Bound to ±scale logits so the bias can't steamroll the content term.
        return self.scale * torch.tanh(out / self.scale)

    def forward(self, mz: Tensor) -> Tensor:
        """mz: (B, K) → bias: (B, n_heads, K, K)."""
        dm = mz.unsqueeze(-1) - mz.unsqueeze(-2)              # (B, K, K), signed
        feats = self.ff(dm)                                   # (B, K, K, in_dim)
        # Per-head first layer: (B, K, K, H, D_h)
        h = torch.einsum("bijd,hde->bijhe", feats, self.w1) + self.b1
        h = F.gelu(h)
        # Per-head output: (B, K, K, H)
        out = torch.einsum("bijhe,he->bijh", h, self.w2) + self.b2
        out = self._bound(out)
        return out.permute(0, 3, 1, 2).contiguous()           # (B, H, K, K)

    def evaluate(self, dm_grid: Tensor) -> Tensor:
        """For visualization: evaluate per-head bias on a 1-D Δm grid.

        dm_grid: (N,) → (N, n_heads). Returns the bounded bias (what the
        attention actually sees).
        """
        feats = self.ff(dm_grid)                              # (N, in_dim)
        h = torch.einsum("nd,hde->nhe", feats, self.w1) + self.b1
        h = F.gelu(h)
        out = torch.einsum("nhe,he->nh", h, self.w2) + self.b2
        return self._bound(out)                               # (N, H)

    def l1_penalty(self, lo: float = -200.0, hi: float = 200.0, n: int = 4001) -> Tensor:
        """Mean |bias| over a uniform Δm grid, for an L1 sparsity penalty.

        Data-independent — penalizes the *shape* of the learned bias curve
        uniformly over Δm, so the broad locality bump (wide → lots of area)
        and the noise floor (everywhere) are taxed while a narrow chemistry
        spike costs almost nothing. Added to the training loss as
        λ · l1_penalty() to push the bias toward a few sharp spikes.
        """
        grid = torch.linspace(lo, hi, n, device=self.w1.device, dtype=self.w1.dtype)
        return self.evaluate(grid).abs().mean()


class BiasedMHA(nn.Module):
    """Multi-head attention with an additive (B, H, K, K) per-head bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = dropout
        self.proj_dropout = nn.Dropout(dropout)
        self._save_attn = False
        self.last_attn: Tensor | None = None  # (B, H, K, K), no grad

    def set_save_attn(self, save: bool) -> None:
        self._save_attn = save
        if not save:
            self.last_attn = None

    def forward(self, x: Tensor, bias: Tensor, key_padding_mask: Tensor) -> Tensor:
        """x: (B, K, D); bias: (B, H, K, K); key_padding_mask: (B, K) True=pad."""
        B, K, _ = x.shape
        qkv = self.qkv(x).reshape(B, K, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, K, H, d_head)
        q = q.transpose(1, 2)  # (B, H, K, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, K, K)
        logits = logits + bias

        # Mask out keys that are padding. mask: (B, 1, 1, K)
        mask = key_padding_mask[:, None, None, :]
        logits = logits.masked_fill(mask, float("-inf"))

        attn = F.softmax(logits, dim=-1)
        if self._save_attn:
            self.last_attn = attn.detach()
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        out = torch.matmul(attn, v)  # (B, H, K, d_head)
        out = out.transpose(1, 2).reshape(B, K, -1)
        return self.proj_dropout(self.out(out))


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = BiasedMHA(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        ffn_dim = cfg.d_model * cfg.ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(ffn_dim, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor, bias: Tensor, key_padding_mask: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), bias, key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class MSEncoder(nn.Module):
    """Peak transformer with shared Δm/z bias across layers."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = PeakEmbed(cfg)
        self.bias_module = DeltaMZBias(cfg.n_heads, cfg.delta_bias)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        mz: Tensor,
        log_int: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        tokens = self.embed(mz, log_int)
        bias = self.bias_module(mz)  # (B, H, K, K)
        # Zero the Δm bias on any pair touching a padded position so the
        # zero-padding sentinel doesn't pollute the bias gradient.
        real = ~key_padding_mask                                       # (B, K)
        bias_valid = real[:, None, :, None] & real[:, None, None, :]   # (B, 1, K, K)
        bias = bias * bias_valid
        # Also zero the diagonal so self-attention is never modulated by
        # the bias — forces self-suppression onto the content (Q/K) path
        # and frees the bias module to specialize on chemistry.
        K = mz.size(1)
        diag = torch.eye(K, dtype=torch.bool, device=mz.device).view(1, 1, K, K)
        bias = bias.masked_fill(diag, 0.0)
        for blk in self.blocks:
            tokens = blk(tokens, bias, key_padding_mask)
        return self.norm(tokens)

    def set_save_attn(self, save: bool) -> None:
        for blk in self.blocks:
            blk.attn.set_save_attn(save)


class DenoiseHead(nn.Module):
    """m/z denoising head.

    Predicts a per-peak (residual, log_var). The cleaned m/z is then
    ``mz_noisy + residual``; loss is Gaussian NLL on the residual against
    the true delta (mz_clean - mz_noisy). Residual parameterization keeps
    the target small (~0.1 Da Gaussian noise, ±1.003 isotope shifts) and
    makes "do nothing" a meaningful identity init.
    """

    def __init__(self, d_model: int, init_log_var: float = -2.0):
        super().__init__()
        self.head = nn.Linear(d_model, 2)  # (residual, log_var)
        # Initialize the residual prediction near zero (default identity
        # behavior) and the log_var bias to ~-2 so initial variance is
        # exp(-2) ≈ 0.14 — between the Gaussian noise σ=0.1 and the
        # isotope shift 1.003. Reasonable starting uncertainty.
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.bias[1] = init_log_var

    def loss(
        self,
        tokens: Tensor,
        mz_noisy: Tensor,
        mz_clean: Tensor,
        key_padding_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Per-peak Gaussian NLL on the cleaning residual, masked to real peaks."""
        real = ~key_padding_mask                       # (B, K)
        if not real.any():
            zero = tokens.new_zeros(())
            return zero, {"nll_mz": zero, "rmse_mz": zero, "mean_log_var": zero}

        out = self.head(tokens.float())                # (B, K, 2)
        residual = out[..., 0]                         # predicted (clean - noisy)
        log_var = out[..., 1].clamp(-10.0, 10.0)
        true_delta = mz_clean - mz_noisy               # (B, K)

        sq_err = (true_delta - residual) ** 2          # (B, K)
        # Gaussian NLL (drop the constant 0.5*log(2π))
        nll = 0.5 * log_var + 0.5 * sq_err / log_var.exp()

        nll = nll[real].mean()
        rmse = sq_err[real].mean().sqrt()
        return nll, {
            "nll_mz": nll.detach(),
            "rmse_mz": rmse.detach(),
            "mean_log_var": log_var[real].detach().mean(),
        }

    @torch.no_grad()
    def predict(self, tokens: Tensor, mz_noisy: Tensor) -> Tensor:
        """Return predicted clean m/z = mz_noisy + residual."""
        out = self.head(tokens.float())
        return mz_noisy + out[..., 0]
