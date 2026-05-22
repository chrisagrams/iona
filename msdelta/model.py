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
    hidden: int = 128
    f_min: float = 1e-2
    f_max: float = 1e3


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
    """(m/z, log_int) → d_model token, with a learned [MASK] swap-in."""

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
        self.mask_token = nn.Parameter(torch.randn(cfg.d_model) * 0.02)

    def forward(self, mz: Tensor, log_int: Tensor, mask_positions: Tensor) -> Tensor:
        """mz, log_int: (B, K); mask_positions: (B, K) bool → (B, K, d_model)."""
        feats = torch.cat([self.ff_mz(mz), self.ff_int(log_int)], dim=-1)
        tokens = self.mlp(feats)
        # Swap in mask_token at masked positions (do this AFTER MLP so the
        # mask embedding is shared regardless of the zeroed inputs).
        tokens = torch.where(mask_positions.unsqueeze(-1), self.mask_token, tokens)
        return tokens


class DeltaMZBias(nn.Module):
    """Learned per-head additive attention bias from signed Δm/z."""

    def __init__(self, n_heads: int, cfg: DeltaBiasConfig):
        super().__init__()
        self.ff = FourierFeatures(cfg.n_freqs, cfg.f_min, cfg.f_max, log_spaced=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.ff.out_dim, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, n_heads),
        )
        self.n_heads = n_heads
        # Initialize the final layer near zero so the bias starts ~0
        # and content attention dominates at step 0.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, mz: Tensor) -> Tensor:
        """mz: (B, K) → bias: (B, n_heads, K, K)."""
        dm = mz.unsqueeze(-1) - mz.unsqueeze(-2)  # (B, K, K), signed
        feats = self.ff(dm)
        bias = self.mlp(feats)  # (B, K, K, H)
        return bias.permute(0, 3, 1, 2).contiguous()

    def evaluate(self, dm_grid: Tensor) -> Tensor:
        """For visualization: evaluate bias on a 1-D Δm grid (no batch dim).

        dm_grid: (N,) → (N, n_heads). Uses no_grad implicitly via caller.
        """
        feats = self.ff(dm_grid)
        return self.mlp(feats)


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
        mask_positions: Tensor,
    ) -> Tensor:
        tokens = self.embed(mz, log_int, mask_positions)
        bias = self.bias_module(mz)  # (B, H, K, K)
        # Zero the Δm bias on any pair touching a masked or padded position.
        # Masked m/z values are real (collate no longer zeroes them) — this
        # prevents (a) a Δm≈0 cluster from masked×masked pairs, and (b)
        # leakage of the masked m/z back through the bias-→logits path.
        real = ~(key_padding_mask | mask_positions)            # (B, K)
        bias_valid = real[:, None, :, None] & real[:, None, None, :]  # (B, 1, K, K)
        bias = bias * bias_valid
        # Also zero the diagonal so self-attention is never modulated by
        # the bias. Without this, bias_module(0) becomes a free shortcut
        # for global self-attention suppression and captures all the
        # gradient that should be specializing the heads on chemistry.
        K = mz.size(1)
        diag = torch.eye(K, dtype=torch.bool, device=mz.device).view(1, 1, K, K)
        bias = bias.masked_fill(diag, 0.0)
        for blk in self.blocks:
            tokens = blk(tokens, bias, key_padding_mask)
        return self.norm(tokens)

    def set_save_attn(self, save: bool) -> None:
        for blk in self.blocks:
            blk.attn.set_save_attn(save)


class MPMHeads(nn.Module):
    """Masked-peak heads: Gaussian NLL for m/z, MSE for log-intensity."""

    def __init__(self, d_model: int, init_log_var: float = 10.0):
        super().__init__()
        self.mz_head = nn.Linear(d_model, 2)  # (mean, log_var)
        self.int_head = nn.Linear(d_model, 1)
        # Initialize log_var bias near the clamp ceiling so the initial
        # Gaussian has a wide variance, absorbing the prior prediction
        # error on raw m/z (~600 Da) without producing huge first-step
        # gradients. Loss clamps log_var to [-10, 10].
        with torch.no_grad():
            self.mz_head.bias[1] = init_log_var

    def gather(self, tokens: Tensor, target_index: Tensor) -> Tensor:
        """tokens: (B, K, D); target_index: (M, 2) of (batch, pos) → (M, D)."""
        return tokens[target_index[:, 0], target_index[:, 1]]

    def loss(
        self,
        tokens: Tensor,
        target_index: Tensor,
        target_mz: Tensor,
        target_logint: Tensor,
        loss_int_weight: float = 1.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if target_index.numel() == 0:
            zero = tokens.new_zeros(())
            return zero, {"nll_mz": zero, "mse_int": zero, "mean_log_var": zero}

        h = self.gather(tokens, target_index).float()  # always do loss in fp32
        mz_out = self.mz_head(h)  # (M, 2)
        mean, log_var = mz_out[:, 0], mz_out[:, 1].clamp(-10.0, 10.0)
        # Gaussian NLL (drop constant 0.5*log(2π) for clarity)
        nll_mz = (0.5 * log_var + 0.5 * (target_mz - mean) ** 2 / log_var.exp()).mean()

        int_pred = self.int_head(h).squeeze(-1)
        mse_int = F.mse_loss(int_pred, target_logint)

        total = nll_mz + loss_int_weight * mse_int
        return total, {
            "nll_mz": nll_mz.detach(),
            "mse_int": mse_int.detach(),
            "mean_log_var": log_var.detach().mean(),
        }
