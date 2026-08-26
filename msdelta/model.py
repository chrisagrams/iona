"""Provide the transformer encoder and masked-intensity head."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from msdelta.fourier import FourierFeatures


@dataclass
class FourierConfig:
    n_freqs: int
    f_min: float
    f_max: float
    learnable: bool = True


@dataclass
class DeltaBiasConfig:
    n_freqs: int = 64
    per_head_hidden: int = 32
    f_min: float = 1e-2
    f_max: float = 1e3
    scale: float = 3.0
    learnable: bool = True


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ffn_mult: int = 4
    dropout: float = 0.1
    max_peaks: int = 150
    fourier_int: FourierConfig = field(default_factory=lambda: FourierConfig(16, 1e-2, 1e2))
    delta_bias: DeltaBiasConfig = field(default_factory=DeltaBiasConfig)
    zero_bias_diagonal: bool = True


class PeakEmbed(nn.Module):
    """Create m/z-free tokens from log intensity."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ff_int = FourierFeatures(
            cfg.fourier_int.n_freqs,
            cfg.fourier_int.f_min,
            cfg.fourier_int.f_max,
            learnable=cfg.fourier_int.learnable,
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.ff_int.out_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.mask_token = nn.Parameter(torch.randn(cfg.d_model) * 0.02)

    def forward(self, log_int: Tensor, mask_positions: Tensor | None = None) -> Tensor:
        """Create peak tokens and insert mask tokens."""
        feats = self.ff_int(log_int)
        tokens = self.mlp(feats.to(self.mlp[0].weight.dtype))
        if mask_positions is not None:
            tokens = torch.where(mask_positions.unsqueeze(-1), self.mask_token, tokens)
        return tokens


class DeltaMZBias(nn.Module):
    """Create a learned attention bias from signed delta m/z."""

    def __init__(self, n_heads: int, cfg: DeltaBiasConfig):
        super().__init__()
        self.ff = FourierFeatures(
            cfg.n_freqs, cfg.f_min, cfg.f_max, log_spaced=True, learnable=cfg.learnable
        )
        self.n_heads = n_heads
        self.per_head_hidden = cfg.per_head_hidden
        self.scale = cfg.scale

        H, D_h = n_heads, cfg.per_head_hidden

        self.fc1 = nn.Linear(self.ff.out_dim, H * D_h)
        self.w2 = nn.Parameter(torch.zeros(H, D_h))
        self.b2 = nn.Parameter(torch.zeros(H))

    def _bound(self, out: Tensor) -> Tensor:
        return self.scale * torch.tanh(out / self.scale)

    def _curve(self, feats: Tensor) -> Tensor:
        """Return a bounded bias for each head."""
        h = self.fc1(feats.to(self.fc1.weight.dtype)).unflatten(
            -1, (self.n_heads, self.per_head_hidden)
        )
        h = F.gelu(h)
        out = (h * self.w2).sum(-1) + self.b2
        return self._bound(out)

    def forward(self, mz: Tensor) -> Tensor:
        """Return the attention bias for an m/z batch."""
        dm = mz.unsqueeze(-1) - mz.unsqueeze(-2)
        curve = self._curve(self.ff(dm))
        return curve.permute(0, 3, 1, 2).contiguous()

    def evaluate(self, dm_grid: Tensor) -> Tensor:
        """Return the bias for each point in a delta m/z grid."""
        return self._curve(self.ff(dm_grid)).float()


class BiasedMHA(nn.Module):
    """Apply multi-head attention with a bias for each head."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = dropout
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, bias: Tensor, key_padding_mask: Tensor) -> Tensor:
        """Apply biased attention to one batch."""
        B, K, _ = x.shape
        qkv = self.qkv(x).reshape(B, K, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = bias.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
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
    """Encode peaks with a shared delta m/z bias."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = PeakEmbed(cfg)
        self.zero_bias_diagonal = cfg.zero_bias_diagonal
        self.bias_module = DeltaMZBias(cfg.n_heads, cfg.delta_bias)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        mz: Tensor,
        log_int: Tensor,
        key_padding_mask: Tensor,
        mask_positions: Tensor | None = None,
    ) -> Tensor:
        tokens = self.embed(log_int, mask_positions)

        bias = self.bias_module(mz)
        if self.zero_bias_diagonal:
            K = mz.size(1)
            diag = torch.eye(K, dtype=torch.bool, device=mz.device).view(1, 1, K, K)
            bias = bias.masked_fill(diag, 0.0)
        for blk in self.blocks:
            tokens = blk(tokens, bias, key_padding_mask)
        tokens = self.norm(tokens)
        return tokens


class IntensityHead(nn.Module):
    """Predict the intensity distribution for masked peaks."""

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 1)

    def loss(
        self,
        tokens: Tensor,
        intensity_prob_target: Tensor,
        mask_positions: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Calculate batch-mean KL loss on masked positions."""
        if not mask_positions.any():
            zero = tokens.new_zeros(())
            return zero, {"kl": zero}

        logits = self.head(tokens).squeeze(-1).float()
        m = mask_positions

        log_q = F.log_softmax(logits.masked_fill(~m, float("-inf")), dim=-1)
        log_q = log_q.masked_fill(~m, 0.0)

        p = intensity_prob_target.masked_fill(~m, 0.0)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        kl = F.kl_div(log_q, p, reduction="batchmean")

        return kl, {"kl": kl.detach()}
