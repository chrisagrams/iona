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
    # v10: condition the bias on precursor charge. A learned charge embedding
    # adds a per-head term in the bias hidden layer, so each head can give a
    # charge-specific response (¹³C peak at 0.50 Da for z=2, 0.33 for z=3, …)
    # instead of smearing all spacings into one curve. Default 0 = OFF (= v9,
    # and keeps v9 checkpoints loadable); v10 configs set charge_dim: 16.
    charge_dim: int = 0
    n_charges: int = 8


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
    # v10: prepend a precursor anchor token carrying absolute m/z (+charge) —
    # the ONE token with m/z, resolving the absolute-frame ambiguity that
    # m/z-free tokens leave (fragment_mz_r2 0.89). Default False = v9 (and
    # keeps v9 checkpoints loadable); v10 configs set use_precursor: true.
    use_precursor: bool = False


class PeakEmbed(nn.Module):
    """m/z-FREE token embedding (v9): token = MLP(Fourier(log_int)).

    Tokens deliberately do NOT encode m/z — m/z flows only through the Δm
    bias, making the bias the sole carrier of m/z structure (ALiBi/T5-style
    relative-only position). A learned [MASK] vector replaces the token at
    masked positions for masked-intensity prediction.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ff_int = FourierFeatures(cfg.fourier_int.n_freqs, cfg.fourier_int.f_min, cfg.fourier_int.f_max)
        self.mlp = nn.Sequential(
            nn.Linear(self.ff_int.out_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.mask_token = nn.Parameter(torch.randn(cfg.d_model) * 0.02)

    def forward(self, log_int: Tensor, mask_positions: Tensor | None = None) -> Tensor:
        """log_int: (B, K); mask_positions: (B, K) bool or None → (B, K, d_model)."""
        tokens = self.mlp(self.ff_int(log_int))
        if mask_positions is not None:
            tokens = torch.where(mask_positions.unsqueeze(-1), self.mask_token, tokens)
        return tokens


class PrecursorEmbed(nn.Module):
    """Precursor anchor token (v10): MLP(Fourier(precursor_mz) ⊕ charge_emb).

    The ONE token carrying absolute m/z. Prepended to the m/z-free fragment
    tokens, it pins the absolute frame; the Δm bias then propagates absolute
    fragment m/z from it. Charge enters here too (the bias is charge-agnostic).
    """

    def __init__(self, cfg: ModelConfig, charge_dim: int = 16):
        super().__init__()
        self.ff_mz = FourierFeatures(cfg.fourier_mz.n_freqs, cfg.fourier_mz.f_min, cfg.fourier_mz.f_max)
        self.charge_emb = nn.Embedding(cfg.delta_bias.n_charges, charge_dim)
        nn.init.normal_(self.charge_emb.weight, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(self.ff_mz.out_dim + charge_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, precursor_mz: Tensor, charge: Tensor) -> Tensor:
        """precursor_mz: (B,), charge: (B,) long → (B, d_model)."""
        feats = torch.cat([self.ff_mz(precursor_mz), self.charge_emb(charge)], dim=-1)
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
        self.charge_dim = cfg.charge_dim

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

        # Charge conditioning: per-charge embedding → per-head additive term in
        # the hidden layer. w1_charge zero-init so the model starts at the v9
        # (charge-agnostic) bias and *learns* charge-dependence.
        if self.charge_dim > 0:
            self.charge_emb = nn.Embedding(cfg.n_charges, cfg.charge_dim)
            nn.init.normal_(self.charge_emb.weight, std=0.02)
            self.w1_charge = nn.Parameter(torch.zeros(H, cfg.charge_dim, D_h))

    def _bound(self, out: Tensor) -> Tensor:
        # Bound to ±scale logits so the bias can't steamroll the content term.
        return self.scale * torch.tanh(out / self.scale)

    def forward(self, mz: Tensor, charge: Tensor | None = None) -> Tensor:
        """mz: (B, K); charge: (B,) long → bias: (B, n_heads, K, K)."""
        dm = mz.unsqueeze(-1) - mz.unsqueeze(-2)              # (B, K, K), signed
        feats = self.ff(dm)                                   # (B, K, K, ff_dim)
        h = torch.einsum("bijd,hde->bijhe", feats, self.w1)   # (B, K, K, H, D_h)
        if self.charge_dim > 0 and charge is not None:
            ce = self.charge_emb(charge)                      # (B, charge_dim)
            h_ch = torch.einsum("bc,hce->bhe", ce, self.w1_charge)  # (B, H, D_h)
            h = h + h_ch[:, None, None]                       # broadcast over (i, j)
        h = F.gelu(h + self.b1)
        out = torch.einsum("bijhe,he->bijh", h, self.w2) + self.b2
        out = self._bound(out)
        return out.permute(0, 3, 1, 2).contiguous()           # (B, H, K, K)

    def evaluate(self, dm_grid: Tensor, charge: int = 0) -> Tensor:
        """Evaluate per-head bias on a 1-D Δm grid at a given precursor charge.

        dm_grid: (N,) → (N, n_heads). Bounded bias (what attention sees).
        """
        feats = self.ff(dm_grid)                              # (N, ff_dim)
        h = torch.einsum("nd,hde->nhe", feats, self.w1)       # (N, H, D_h)
        if self.charge_dim > 0:
            ce = self.charge_emb(torch.tensor(charge, device=self.w1.device))  # (charge_dim,)
            h = h + torch.einsum("c,hce->he", ce, self.w1_charge)              # (H, D_h)
        h = F.gelu(h + self.b1)
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
        if self.charge_dim > 0:
            # average over the common charges so all charge-conditioned curves are penalized
            return torch.stack([self.evaluate(grid, z).abs().mean() for z in (1, 2, 3)]).mean()
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
        self.use_precursor = cfg.use_precursor
        if self.use_precursor:
            self.precursor_embed = PrecursorEmbed(cfg)
        self.bias_module = DeltaMZBias(cfg.n_heads, cfg.delta_bias)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        mz: Tensor,
        log_int: Tensor,
        key_padding_mask: Tensor,
        mask_positions: Tensor | None = None,
        charge: Tensor | None = None,
        precursor_mz: Tensor | None = None,
    ) -> Tensor:
        tokens = self.embed(log_int, mask_positions)   # m/z-free tokens (B, K, D)

        # v10: prepend a precursor anchor token (the only token with absolute m/z).
        prepended = self.use_precursor and precursor_mz is not None
        if prepended:
            B = mz.size(0)
            z = charge if charge is not None else torch.zeros(B, dtype=torch.long, device=mz.device)
            prec_tok = self.precursor_embed(precursor_mz, z)              # (B, D)
            tokens = torch.cat([prec_tok.unsqueeze(1), tokens], dim=1)    # (B, K+1, D)
            mz = torch.cat([precursor_mz.unsqueeze(1), mz], dim=1)        # (B, K+1)
            key_padding_mask = torch.cat(
                [torch.zeros(B, 1, dtype=torch.bool, device=mz.device), key_padding_mask], dim=1)

        bias = self.bias_module(mz, charge)  # (B, H, K(+1), K(+1)) — the only place m/z enters
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
        tokens = self.norm(tokens)
        if prepended:
            tokens = tokens[:, 1:]   # drop the precursor; return fragment tokens (B, K, D)
        return tokens

    def set_save_attn(self, save: bool) -> None:
        for blk in self.blocks:
            blk.attn.set_save_attn(save)


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

        logits = self.head(tokens.float()).squeeze(-1)              # (B, K)
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
