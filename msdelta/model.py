"""Peak-token transformer with learned per-head Δm/z attention bias + MPM heads."""
from __future__ import annotations

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
    learnable: bool = True


@dataclass
class DeltaBiasConfig:
    n_freqs: int = 64
    # Per-head hidden width. Each head gets its OWN MLP (no parameter
    # sharing across heads), so total bias capacity is n_heads * per_head_hidden.
    # Smaller than the old shared `hidden` because activation memory is
    # (B, K, K, n_heads * per_head_hidden)
    per_head_hidden: int = 32
    f_min: float = 1e-2
    f_max: float = 1e3
    # Bound the per-head bias to ±scale logits via scale*tanh(raw/scale).
    # Keeps the bias comparable to the content term (q·k/√d ~ O(1-2)) so
    # neither can steamroll the other
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
    # Zero the (i, i) entry of the Δm bias so the bias module can't modulate
    # self-attention (Δm=0 → a per-head constant self-logit offset).
    zero_bias_diagonal: bool = True


class PeakEmbed(nn.Module):
    """m/z-FREE token embedding (v9): token = MLP(Fourier(log_int)).

    Tokens deliberately do NOT encode m/z — m/z flows only through the Δm
    bias, making the bias the sole carrier of m/z structure (ALiBi/T5-style
    relative-only position). A learned [MASK] vector replaces the token at
    masked positions for masked-intensity prediction.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ff_int = FourierFeatures(
            cfg.fourier_int.n_freqs, cfg.fourier_int.f_min, cfg.fourier_int.f_max,
            learnable=cfg.fourier_int.learnable)
        self.mlp = nn.Sequential(
            nn.Linear(self.ff_int.out_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.mask_token = nn.Parameter(torch.randn(cfg.d_model) * 0.02)

    def forward(self, log_int: Tensor, mask_positions: Tensor | None = None) -> Tensor:
        """log_int: (B, K); mask_positions: (B, K) bool or None → (B, K, d_model)."""
        feats = self.ff_int(log_int)  # fp32 Fourier features
        tokens = self.mlp(feats.to(self.mlp[0].weight.dtype))
        if mask_positions is not None:
            tokens = torch.where(mask_positions.unsqueeze(-1), self.mask_token, tokens)
        return tokens


class DeltaMZBias(nn.Module):
    """Learned per-head additive attention bias from signed Δm/z.

    The bias is H independent 2-layer MLPs of a single scalar (Δm/z, lifted
    through log-spaced Fourier features), one per attention head, bounded to
    ±scale by a tanh. Decoupling the heads lets each specialize on a
    different Δm region instead of all collapsing onto whichever feature
    dominates the data (e.g. suppression of small-Δm near-duplicate peaks).

    "H independent MLPs sharing one input" needs no special machinery:
      * Layer 1 (shared Fourier features → per-head hidden) is a plain
        ``Linear(in_dim, H*D_h)`` whose outputs we simply *view* as
        ``(H, D_h)`` — every Linear output unit is already an independent
        function of the whole input, which is exactly per-head independence.
      * Layer 2 (per-head hidden → per-head scalar) is a per-head weighted
        sum over the hidden dim: ``(h * w2).sum(-1) + b2``.
    ``forward`` (the Δm matrix) and ``evaluate`` (a 1-D grid) both route
    through :meth:`_curve`, so the whole architecture lives in one place.
    """

    def __init__(self, n_heads: int, cfg: DeltaBiasConfig):
        super().__init__()
        self.ff = FourierFeatures(cfg.n_freqs, cfg.f_min, cfg.f_max, log_spaced=True,
                                  learnable=cfg.learnable)
        self.n_heads = n_heads
        self.per_head_hidden = cfg.per_head_hidden
        self.scale = cfg.scale

        H, D_h = n_heads, cfg.per_head_hidden

        # Layer 1: one Linear whose H*D_h outputs we view as H per-head blocks.
        self.fc1 = nn.Linear(self.ff.out_dim, H * D_h)
        # Layer 2: per-head readout (D_h → scalar), applied as a weighted sum.
        # Zero-init so every head's bias starts at exactly 0.
        self.w2 = nn.Parameter(torch.zeros(H, D_h))
        self.b2 = nn.Parameter(torch.zeros(H))

    def _bound(self, out: Tensor) -> Tensor:
        # Bound to ±scale logits so the bias can't steamroll the content term.
        return self.scale * torch.tanh(out / self.scale)

    def _curve(self, feats: Tensor) -> Tensor:
        """feats: (..., ff_dim) → bounded per-head bias (..., H).

        The whole bias architecture. `feats` may carry any leading dims (the
        (B, K, K) Δm matrix in `forward`, a flat (N,) grid in `evaluate`).
        """
        h = self.fc1(feats.to(self.fc1.weight.dtype)).unflatten(-1, (self.n_heads, self.per_head_hidden))  # (..., H, D_h)
        h = F.gelu(h)
        out = (h * self.w2).sum(-1) + self.b2                                    # (..., H)
        return self._bound(out)

    def forward(self, mz: Tensor) -> Tensor:
        """mz: (B, K) → bias: (B, n_heads, K, K)."""
        dm = mz.unsqueeze(-1) - mz.unsqueeze(-2)             # (B, K, K), signed
        curve = self._curve(self.ff(dm))                     # (B, K, K, H)
        return curve.permute(0, 3, 1, 2).contiguous()        # (B, H, K, K)

    def evaluate(self, dm_grid: Tensor) -> Tensor:
        """Evaluate per-head bias on a 1-D Δm grid.

        dm_grid: (N,) → (N, n_heads). Bounded bias (what attention sees).
        """
        return self._curve(self.ff(dm_grid)).float()         # (N, H)


class BiasedMHA(nn.Module):
    """Multi-head attention with an additive (B, H, K, K) per-head bias."""

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
        """x: (B, K, D); bias: (B, H, K, K); key_padding_mask: (B, K) True=pad."""
        B, K, _ = x.shape
        qkv = self.qkv(x).reshape(B, K, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, K, H, d_head)
        q = q.transpose(1, 2)  # (B, H, K, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Fold key padding into the additive bias: one float mask for SDPA.
        attn_mask = bias.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
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
    """Peak transformer with shared Δm/z bias across layers."""

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
        tokens = self.embed(log_int, mask_positions)   # m/z-free tokens (B, K, D)

        bias = self.bias_module(mz)  # (B, H, K, K) — the only place m/z enters
        # Zero the diagonal so self-attention is never modulated by the
        # bias — forces self-suppression onto the content (Q/K) path and
        # frees the bias module to specialize on chemistry. Key-padding is
        # handled in BiasedMHA (folded into the SDPA attn_mask as -inf), so
        # the bias needs no separate padding mask here. Toggled off by the
        # diagonal-zero ablation (cfg.zero_bias_diagonal).
        if self.zero_bias_diagonal:
            K = mz.size(1)
            diag = torch.eye(K, dtype=torch.bool, device=mz.device).view(1, 1, K, K)
            bias = bias.masked_fill(diag, 0.0)
        for blk in self.blocks:
            tokens = blk(tokens, bias, key_padding_mask)
        tokens = self.norm(tokens)
        return tokens


class IntensityHead(nn.Module):
    """Masked-intensity prediction head (v13 — KL on the masked subset).

    Per-peak scalar logit; per-spectrum softmax across masked positions
    gives a predicted distribution over the masked subset, compared via
    KL against the true intensity distribution (raw intensity normalised
    to sum to 1 across the masked peaks). Tokens are m/z-free, so the only
    way the model can localise a masked peak is via the Δm bias — and
    framing the target as a distribution puts intensity *ratios* (M+0/M+1
    ≈ 5:1 for ¹³C, residue-ladder ratios, …) directly into the loss.

    Pre-v13 used MSE on the per-spectrum max-normalised log_int; that
    target was so concentrated (mean ≈ 0.76, var ≈ 0.007) that constant-
    predict was near-optimal and the bias chemistry never got real
    gradient pressure (see EXPERIMENTS.md §v13).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 1)

    def loss(
        self,
        tokens: Tensor,
        intensity_prob_target: Tensor,
        mask_positions: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """KL(p || q) on masked positions, batch-mean."""
        if not mask_positions.any():
            zero = tokens.new_zeros(())
            return zero, {"kl": zero}

        # Matmul in the token/weight dtype (bf16 under DeepSpeed), then upcast
        # the per-peak logits to fp32 for the numerically-sensitive KL below.
        logits = self.head(tokens).squeeze(-1).float()             # (B, K)
        m = mask_positions

        # Predicted distribution q over masked positions, per row. Setting
        # non-masked logits to -inf zeros them in the softmax denominator;
        # we then overwrite the -inf in log_q with 0 so the subsequent
        # multiply-by-zero target doesn't produce 0·(-inf) = NaN.
        log_q = F.log_softmax(logits.masked_fill(~m, float("-inf")), dim=-1)
        log_q = log_q.masked_fill(~m, 0.0)

        # Target distribution p over masked positions, per row. Re-normalise
        # the per-spectrum (sum-to-1) intensity probabilities to sum-to-1
        # over the masked subset only.
        p = intensity_prob_target.masked_fill(~m, 0.0)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # F.kl_div expects log-prob input + prob target; 'batchmean' = / B.
        # (The default 'mean' divides by B·K which is wrong for variable K.)
        kl = F.kl_div(log_q, p, reduction="batchmean")

        return kl, {"kl": kl.detach()}
