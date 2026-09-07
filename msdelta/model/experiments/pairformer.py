"""AlphaFold3-style Pairformer encoder for tandem mass spectra.

Motivation and full design rationale live in ``docs/PAIRFORMER.md``. The short version:
the baseline MSDelta keeps a *single* representation that carries only intensity and injects
m/z solely through a static per-head Δm/z attention bias. This variant instead maintains two
hidden states, refined together at every layer (AF3 Pairformer, Abramson et al. 2024):

    s : (B, N, hidden_size)     one vector per peak         -- "single"
    z : (B, N, N, pair_channels) one vector per peak pair   -- "pair"

Both states carry BOTH m/z and intensity information, which is the change requested over the
existing ``pair_stream`` experiments (there the pair was a pure function of Δm/z and the single
carried only intensity):

* ``s_init`` = MLP(Fourier(m/z) ++ Fourier(intensity) ++ ...)      -- section 3 of the spec
* ``z_init[i,j]`` = W_a s_i + W_b s_j + W_c pair_feats[i,j]        -- section 4
  where ``pair_feats`` couples the signed mass difference Δ_ij = mz_i - mz_j (Fourier + mass
  defect + a neutral-loss / residue soft-match dictionary + complementarity + isotope spacing)
  with the intensity ratio log(I_i / I_j).

z is injected into the single-stream attention as a per-head additive logit bias (as in the
baseline), and is itself refined across depth by triangle multiplication (B3), optionally
triangle attention (B4) and a single->pair outer-product-mean write-back (B5). A global
conditioning vector g (precursor m/z + charge) modulates the single stream through adaptive
LayerNorm (spec section 5.4/5.5).

The AF3 build-phase ladder (spec section 10) is reachable from one module via config:

    pair_update = "static"      -> B1  z frozen after init, only the per-layer readout learns
    pair_update = "transition"  -> B2  z refined by a SwiGLU transition only
    pair_update = "triangle"    -> B3  + triangle multiplication (the hypothesis)
    use_triangle_attention=True -> B4  + triangle attention  (EXPENSIVE, see the config docstring)
    use_writeback=True          -> B5  + single->pair outer-product-mean

Diagnostics compatibility: the pair module is exposed as ``encoder.bias_module`` with the same
``.ff`` (signed-Δ Fourier bank) and ``.evaluate(grid) -> (grid, heads)`` surface the baseline
``DeltaMZBias`` offers, so ``eval/viz.py``, ``eval/alignment.py`` and ``FourierProbeCallback``
keep working. ``evaluate`` reports the *pure-Δ component* of the learned bias (intensity
coupling, triangle mixing and write-back all zeroed) -- see ``PairStack.evaluate_layers``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput

from msdelta.model.configuration import MSDeltaConfig
from msdelta.model.fourier import FourierFeatures
from msdelta.model.modeling import (
    IntensityHead,
    MSDeltaForPreTraining,
    MSDeltaForPreTrainingOutput,
    MSDeltaPreTrainedModel,
)

# Monoisotopic masses (Da) for the neutral-loss / residue soft-match dictionary (spec p3).
# Hardcoded rather than imported from ``msdelta.data.chemistry`` because ``msdelta.model`` must
# import nothing from ``data/`` -- that independence is what keeps a checkpoint self-contained.
_NEUTRAL_LOSS_BANK: tuple[float, ...] = (
    # Small neutral losses.
    1.007825, 2.015650, 15.010899, 15.994915, 16.018724, 17.002740, 17.026549, 18.010565,
    18.034374, 19.989830, 26.003074, 27.994915, 28.006148, 28.031300, 30.010565, 31.989829,
    32.026215, 33.987721, 34.968853, 42.010565, 43.005814, 43.989829, 44.997654, 45.992904,
    46.005479, 46.968261, 62.963701, 63.961901, 79.956815, 79.966331, 97.976896, 115.913700,
    # Amino-acid residue masses (L and I share a mass).
    57.021464, 71.037114, 87.032028, 97.052764, 99.068414, 101.047678, 103.009185, 113.084064,
    114.042927, 115.026943, 128.058578, 128.094963, 129.042593, 131.040485, 137.058912,
    147.068414, 156.101111, 163.063329, 186.079313,
    # Common sugar / modification residues.
    79.966331, 132.042259, 146.057909, 162.052824, 176.032088, 203.079373,
)

_C13_SPACING = 1.003355  # Δm between the ¹³C and ¹²C isotopologues (Da).


class MSDeltaPairformerConfig(MSDeltaConfig):
    """MSDelta configuration plus the Pairformer (single + pair) settings."""

    model_type = "msdelta-pairformer"

    def __init__(
        self,
        pair_channels: int = 64,
        pair_transition_expansion: int = 2,
        tri_channels: int = 64,
        tri_attn_heads: int = 4,
        tri_attn_dim: int = 16,
        tri_attn_chunk: int = 32,
        opm_channels: int = 16,
        global_cond_dim: int = 128,
        pair_update: str = "triangle",
        use_triangle_attention: bool = False,
        use_writeback: bool = True,
        single_use_mz: bool = True,
        pair_use_intensity: bool = True,
        use_global_cond: bool = True,
        pair_dropout: float = 0.0,
        mass_defect_n_freqs: int = 32,
        loss_bank_sigma_ppm: float = 20.0,
        n_charges: int = 8,
        pair_bias_scale: float | None = None,
        **kwargs,
    ):
        # Set before super().__init__, which calls _validate().
        self.pair_channels = pair_channels
        self.pair_transition_expansion = pair_transition_expansion
        self.tri_channels = tri_channels
        self.tri_attn_heads = tri_attn_heads
        self.tri_attn_dim = tri_attn_dim
        self.tri_attn_chunk = tri_attn_chunk
        self.opm_channels = opm_channels
        self.global_cond_dim = global_cond_dim
        self.pair_update = pair_update
        self.use_triangle_attention = use_triangle_attention
        self.use_writeback = use_writeback
        self.single_use_mz = single_use_mz
        self.pair_use_intensity = pair_use_intensity
        self.use_global_cond = use_global_cond
        self.pair_dropout = pair_dropout
        self.mass_defect_n_freqs = mass_defect_n_freqs
        self.loss_bank_sigma_ppm = loss_bank_sigma_ppm
        self.n_charges = n_charges
        self.pair_bias_scale = pair_bias_scale
        super().__init__(**kwargs)

    def _validate(self) -> None:
        super()._validate()
        if self.pair_channels <= 0 or self.tri_channels <= 0:
            raise ValueError("pair_channels and tri_channels must be positive")
        if self.pair_transition_expansion <= 0:
            raise ValueError("pair_transition_expansion must be positive")
        if self.pair_update not in {"static", "transition", "triangle"}:
            raise ValueError("pair_update must be 'static', 'transition' or 'triangle'")
        if self.use_triangle_attention:
            if self.tri_attn_heads <= 0 or self.tri_attn_dim <= 0 or self.tri_attn_chunk <= 0:
                raise ValueError("tri_attn_heads, tri_attn_dim and tri_attn_chunk must be positive")
        if self.use_writeback and self.opm_channels <= 0:
            raise ValueError("opm_channels must be positive when use_writeback is set")
        if self.use_global_cond and (self.global_cond_dim <= 0 or self.n_charges <= 0):
            raise ValueError("global_cond_dim and n_charges must be positive with use_global_cond")
        if self.mass_defect_n_freqs <= 0:
            raise ValueError("mass_defect_n_freqs must be positive")
        if self.loss_bank_sigma_ppm <= 0:
            raise ValueError("loss_bank_sigma_ppm must be positive")
        if self.pair_bias_scale is not None and self.pair_bias_scale <= 0:
            raise ValueError("pair_bias_scale must be positive when set")


class Transition(nn.Module):
    """SwiGLU feed-forward (AF3 Alg. 11): ``LinearNoBias(swish(a) * b)`` with pre-LayerNorm.

    Applied residually by the caller (``x <- x + Transition(x)``).
    """

    def __init__(self, channels: int, expansion: int, eps: float):
        super().__init__()
        inner = expansion * channels
        self.norm = nn.LayerNorm(channels, eps=eps)
        self.a = nn.Linear(channels, inner, bias=False)
        self.b = nn.Linear(channels, inner, bias=False)
        self.out = nn.Linear(inner, channels, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.out(F.silu(self.a(x)) * self.b(x))


class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale/shift are produced from a global conditioning vector g.

    ``to_scale_shift`` is zero-initialised so the module starts as a plain (affine-free)
    LayerNorm; conditioning can only add signal from there.
    """

    def __init__(self, channels: int, cond_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False, eps=eps)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * channels)

    def zero_init(self) -> None:
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: Tensor, g: Tensor | None) -> Tensor:
        x = self.norm(x)
        if g is None:
            return x
        scale, shift = self.to_scale_shift(F.silu(g)).chunk(2, dim=-1)
        # g is (B, cond_dim); broadcast over the peak axis.
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class GlobalCond(nn.Module):
    """Build a global conditioning vector g from precursor m/z and charge (spec section 5.5)."""

    def __init__(self, config: MSDeltaPairformerConfig):
        super().__init__()
        self.n_charges = config.n_charges
        self.ff_prec = FourierFeatures(
            config.delta_bias_n_freqs,
            config.delta_bias_f_min,
            config.delta_bias_f_max,
            log_spaced=True,
            learnable=config.delta_bias_learnable,
            log_parameterized=config.fourier_log_parameterized,
        )
        in_dim = self.ff_prec.out_dim + config.n_charges
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, config.global_cond_dim),
            nn.GELU(),
            nn.Linear(config.global_cond_dim, config.global_cond_dim),
        )

    def forward(self, precursor_mz: Tensor, charge: Tensor) -> Tensor:
        prec = self.ff_prec(precursor_mz)  # (B, 2F)
        charge_oh = F.one_hot(charge.clamp(0, self.n_charges - 1), self.n_charges).to(prec.dtype)
        feats = torch.cat([prec, charge_oh], dim=-1)
        return self.mlp(feats.to(self.mlp[0].weight.dtype))


class PeakEmbedMZ(nn.Module):
    """Single-representation init from BOTH m/z and intensity (spec section 3).

    Keeps ``ff_int`` where ``FourierProbeCallback`` looks for it, and adds an ``ff_mz`` bank so
    m/z now enters the token directly (the baseline ``PeakEmbed`` used intensity only).
    """

    def __init__(self, config: MSDeltaPairformerConfig):
        super().__init__()
        self.use_mz = config.single_use_mz
        self.ff_int = FourierFeatures(
            config.fourier_int_n_freqs,
            config.fourier_int_f_min,
            config.fourier_int_f_max,
            learnable=config.fourier_int_learnable,
            log_parameterized=config.fourier_log_parameterized,
        )
        in_dim = self.ff_int.out_dim
        if self.use_mz:
            self.ff_mz = FourierFeatures(
                config.delta_bias_n_freqs,
                config.delta_bias_f_min,
                config.delta_bias_f_max,
                log_spaced=True,
                learnable=config.delta_bias_learnable,
                log_parameterized=config.fourier_log_parameterized,
            )
            in_dim += self.ff_mz.out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.mask_token = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, mz: Tensor, log_intensity: Tensor, mask_positions: Tensor | None) -> Tensor:
        feats = self.ff_int(log_intensity)
        if self.use_mz:
            feats = torch.cat([feats, self.ff_mz(mz)], dim=-1)
        tokens = self.mlp(feats.to(self.mlp[0].weight.dtype))
        if mask_positions is not None:
            tokens = torch.where(mask_positions.unsqueeze(-1), self.mask_token, tokens)
        return tokens


class PairFeatures(nn.Module):
    """Compute the raw per-pair feature block ``pair_feats[i, j]`` (spec section 4)."""

    def __init__(self, config: MSDeltaPairformerConfig):
        super().__init__()
        self.use_intensity = config.pair_use_intensity
        self.sigma_ppm = config.loss_bank_sigma_ppm
        # Signed Δ bank -- exposed as ``.ff`` for the diagnostics, so keep it first.
        self.ff = FourierFeatures(
            config.delta_bias_n_freqs,
            config.delta_bias_f_min,
            config.delta_bias_f_max,
            log_spaced=True,
            learnable=config.delta_bias_learnable,
            log_parameterized=config.fourier_log_parameterized,
        )
        # Finer bank for the mass defect frac(|Δ|) in [0, 1).
        self.ff_defect = FourierFeatures(
            config.mass_defect_n_freqs,
            1.0,
            float(config.mass_defect_n_freqs),
            log_spaced=True,
            learnable=config.delta_bias_learnable,
            log_parameterized=config.fourier_log_parameterized,
        )
        self.register_buffer(
            "loss_bank", torch.tensor(_NEUTRAL_LOSS_BANK, dtype=torch.float32), persistent=True
        )
        # p1 signed Δ, p2 mass defect, p3 loss dictionary, p4 complementarity,
        # p5 isotope (k in {1, 2}), p6 Δ log-intensity.
        self.out_dim = (
            self.ff.out_dim
            + self.ff_defect.out_dim
            + self.loss_bank.numel()
            + 1
            + 2
            + (1 if self.use_intensity else 0)
        )

    def forward(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        precursor_mz: Tensor | None,
        charge: Tensor | None,
    ) -> Tensor:
        delta = mz.unsqueeze(-1) - mz.unsqueeze(-2)  # (B, N, N), signed
        abs_delta = delta.abs()
        feats = [self.ff(delta), self.ff_defect(abs_delta - abs_delta.floor())]

        # sigma in Da, m/z-dependent: the tolerance grows with the heavier peak's mass.
        heavier = torch.maximum(mz.unsqueeze(-1), mz.unsqueeze(-2)).clamp_min(1.0)
        sigma = (self.sigma_ppm * 1e-6) * heavier
        two_var = 2.0 * (sigma * sigma).clamp_min(1e-12)

        # p3 neutral-loss / residue dictionary soft-match.
        bank = self.loss_bank.to(delta.dtype)
        diff = abs_delta.unsqueeze(-1) - bank  # (B, N, N, F_loss)
        p3 = torch.exp(-(diff * diff) / two_var.unsqueeze(-1))
        feats.append(p3)

        # p4 complementarity: fragment pair sums to the neutral precursor mass + 2 protons.
        proton = 1.007276
        if precursor_mz is not None and charge is not None:
            z = charge.clamp_min(1).to(mz.dtype)
            neutral_mass = (precursor_mz * z - z * proton).clamp_min(0.0)  # (B,)
            target = (neutral_mass + 2 * proton)[:, None, None]
            comp_diff = mz.unsqueeze(-1) + mz.unsqueeze(-2) - target
            p4 = torch.exp(-(comp_diff * comp_diff) / two_var)
            # A precursor of 0 (unparsed) carries no complementarity signal.
            p4 = p4 * (precursor_mz > 0)[:, None, None].to(p4.dtype)
        else:
            p4 = torch.zeros_like(abs_delta)
        feats.append(p4.unsqueeze(-1))

        # p5 isotope spacing for k in {1, 2}.
        iso = []
        for k in (1, 2):
            d = abs_delta - _C13_SPACING * k
            iso.append(torch.exp(-(d * d) / two_var))
        feats.append(torch.stack(iso, dim=-1))

        # p6 relative intensity, log(I_i / I_j). log_intensity is already log-scale.
        if self.use_intensity:
            rel = log_intensity.unsqueeze(-1) - log_intensity.unsqueeze(-2)
            feats.append(rel.unsqueeze(-1))

        return torch.cat([f.to(feats[0].dtype) for f in feats], dim=-1)


class TriangleMultiplication(nn.Module):
    """Triangle multiplicative update (AF3 Alg. 12/13), outgoing or incoming."""

    def __init__(self, config: MSDeltaPairformerConfig, outgoing: bool):
        super().__init__()
        c_z, c = config.pair_channels, config.tri_channels
        self.outgoing = outgoing
        self.norm = nn.LayerNorm(c_z, eps=config.layer_norm_eps)
        self.a_proj = nn.Linear(c_z, c, bias=False)
        self.a_gate = nn.Linear(c_z, c, bias=False)
        self.b_proj = nn.Linear(c_z, c, bias=False)
        self.b_gate = nn.Linear(c_z, c, bias=False)
        self.out_norm = nn.LayerNorm(c, eps=config.layer_norm_eps)
        self.out_proj = nn.Linear(c, c_z, bias=False)
        self.out_gate = nn.Linear(c_z, c_z, bias=False)

    def forward(self, z: Tensor, pair_mask: Tensor) -> Tensor:
        z = self.norm(z)
        a = torch.sigmoid(self.a_gate(z)) * self.a_proj(z)
        b = torch.sigmoid(self.b_gate(z)) * self.b_proj(z)
        # Mask BOTH endpoints so a padded peak contributes zero to the sum over k (trap #5).
        # Cast the mask to the (possibly bf16) operand dtype so the pair branch stays low-precision.
        mask = pair_mask.to(a.dtype)
        a = a * mask
        b = b * mask
        if self.outgoing:
            out = torch.einsum("bikc,bjkc->bijc", a, b)
        else:
            out = torch.einsum("bkic,bkjc->bijc", a, b)
        g = torch.sigmoid(self.out_gate(z))
        return g * self.out_proj(self.out_norm(out))


class TriangleAttention(nn.Module):
    """Triangle self-attention around a starting or ending node (AF3 Alg. 14/15).

    Chunked over the query row to bound the (B, chunk, N, N, H) logit tensor -- triangle
    attention is the memory-dominant operation (spec section 12); ``tri_attn_chunk`` trades
    peak memory for a Python loop.
    """

    def __init__(self, config: MSDeltaPairformerConfig, starting: bool):
        super().__init__()
        c_z = config.pair_channels
        self.starting = starting
        self.h = config.tri_attn_heads
        self.d = config.tri_attn_dim
        self.chunk = config.tri_attn_chunk
        inner = self.h * self.d
        self.norm = nn.LayerNorm(c_z, eps=config.layer_norm_eps)
        self.q = nn.Linear(c_z, inner, bias=False)
        self.k = nn.Linear(c_z, inner, bias=False)
        self.v = nn.Linear(c_z, inner, bias=False)
        self.bias = nn.Linear(c_z, self.h, bias=False)
        self.gate = nn.Linear(c_z, inner)
        self.out = nn.Linear(inner, c_z)

    def forward(self, z: Tensor, mask: Tensor) -> Tensor:
        # Ending-node attention is starting-node attention on the transposed pair tensor.
        if not self.starting:
            z = z.transpose(1, 2)
        z = self.norm(z)
        b, n, _, _ = z.shape
        q = self.q(z).view(b, n, n, self.h, self.d)
        k = self.k(z).view(b, n, n, self.h, self.d)
        v = self.v(z).view(b, n, n, self.h, self.d)
        bias = self.bias(z)  # (B, N(j), N(k), H): pair bias b_{jk}, broadcast over query row i.
        key_mask = (~mask)[:, None, None, :, None] * torch.finfo(z.dtype).min  # mask node k
        scale = 1.0 / math.sqrt(self.d)

        out = torch.empty_like(q)
        for s in range(0, n, self.chunk):
            e = min(s + self.chunk, n)
            qc = q[:, s:e]  # (B, C, N(j), H, d)
            logits = torch.einsum("bcjhd,bckhd->bcjkh", qc, k[:, s:e]) * scale
            logits = logits + bias[:, None, :, :, :] + key_mask
            # softmax upcasts to fp32 under autocast; cast back so the einsum operands
            # share a dtype (einsum is not autocast-managed and rejects fp32 x bf16).
            attn = torch.softmax(logits.float(), dim=3).to(v.dtype)
            out[:, s:e] = torch.einsum("bcjkh,bckhd->bcjhd", attn, v[:, s:e])

        out = out.reshape(b, n, n, self.h * self.d)
        out = torch.sigmoid(self.gate(z)) * out
        out = self.out(out)
        return out if self.starting else out.transpose(1, 2)


class OuterProductMean(nn.Module):
    """Single -> pair write-back (AF3 Alg. 9, single-sequence): z_ij += Linear(s_i (x) s_j).

    Output projection is zero-initialised so B5 starts as a no-op relative to B3.
    """

    def __init__(self, config: MSDeltaPairformerConfig):
        super().__init__()
        c = config.opm_channels
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.left = nn.Linear(config.hidden_size, c, bias=False)
        self.right = nn.Linear(config.hidden_size, c, bias=False)
        self.out = nn.Linear(c * c, config.pair_channels)

    def zero_init(self) -> None:
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, s: Tensor, mask: Tensor) -> Tensor:
        s = self.norm(s)
        a = self.left(s) * mask.unsqueeze(-1)
        b = self.right(s) * mask.unsqueeze(-1)
        outer = torch.einsum("bic,bjd->bijcd", a, b)
        outer = outer.reshape(*outer.shape[:3], -1)
        return self.out(outer)


class AttentionPairBias(nn.Module):
    """Single-stream attention with a per-head bias from z (AF3 Alg. 24, no diffusion cond.)."""

    def __init__(self, config: MSDeltaPairformerConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.d_head = config.hidden_size // config.num_attention_heads
        self.pre_norm = AdaLayerNorm(
            config.hidden_size, config.global_cond_dim, config.layer_norm_eps
        )
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.gate = nn.Linear(config.hidden_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = config.attention_probs_dropout_prob
        self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, s: Tensor, bias: Tensor, padding_mask: Tensor, g: Tensor | None) -> Tensor:
        b, n, _ = s.shape
        s = self.pre_norm(s, g)
        qkv = self.qkv(s).reshape(b, n, 3, self.n_heads, self.d_head)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_bias = bias.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_bias,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(b, n, -1)
        context = torch.sigmoid(self.gate(s)) * context
        return self.proj_dropout(self.out(context))


class PairformerBlock(nn.Module):
    """One Pairformer block (AF3 Alg. 17): refine z, then update s from z (and g)."""

    def __init__(self, config: MSDeltaPairformerConfig):
        super().__init__()
        self.pair_update = config.pair_update
        self.use_writeback = config.use_writeback
        self.use_triangle_attention = config.use_triangle_attention
        self.n_heads = config.num_attention_heads
        self.scale = config.pair_bias_scale
        self.dropout = nn.Dropout(config.pair_dropout)

        if self.use_writeback:
            self.opm = OuterProductMean(config)
        if self.pair_update == "triangle":
            self.tri_out = TriangleMultiplication(config, outgoing=True)
            self.tri_in = TriangleMultiplication(config, outgoing=False)
            if self.use_triangle_attention:
                self.tri_attn_start = TriangleAttention(config, starting=True)
                self.tri_attn_end = TriangleAttention(config, starting=False)
        if self.pair_update in {"triangle", "transition"}:
            self.pair_transition = Transition(
                config.pair_channels, config.pair_transition_expansion, config.layer_norm_eps
            )

        self.bias_norm = nn.LayerNorm(config.pair_channels, eps=config.layer_norm_eps)
        self.to_bias = nn.Linear(config.pair_channels, config.num_attention_heads, bias=False)

        self.attn = AttentionPairBias(config)
        self.single_transition = _SingleTransition(config)

    def refine_pair(self, z: Tensor, s: Tensor, mask: Tensor, pair_mask: Tensor) -> Tensor:
        if self.use_writeback:
            z = z + self.dropout(self.opm(s, mask))
        if self.pair_update == "triangle":
            z = z + self.dropout(self.tri_out(z, pair_mask))
            z = z + self.dropout(self.tri_in(z, pair_mask))
            if self.use_triangle_attention:
                z = z + self.dropout(self.tri_attn_start(z, mask))
                z = z + self.dropout(self.tri_attn_end(z, mask))
        if self.pair_update in {"triangle", "transition"}:
            z = z + self.pair_transition(z)
        return z

    def read_bias(self, z: Tensor) -> Tensor:
        """Per-head bias curve (..., heads) from the current pair state."""
        head_bias = self.to_bias(self.bias_norm(z))
        if self.scale is not None:
            head_bias = self.scale * torch.tanh(head_bias / self.scale)
        return head_bias

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        padding_mask: Tensor,
        g: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        z = self.refine_pair(z, s, mask, pair_mask)
        bias = self.read_bias(z).permute(0, 3, 1, 2).contiguous()
        s = s + self.attn(s, bias, padding_mask, g)
        s = s + self.single_transition(s, g)
        return s, z


class _SingleTransition(nn.Module):
    """SwiGLU transition on the single stream with AdaLayerNorm conditioning on g."""

    def __init__(self, config: MSDeltaPairformerConfig):
        super().__init__()
        inner = config.intermediate_size
        self.norm = AdaLayerNorm(
            config.hidden_size, config.global_cond_dim, config.layer_norm_eps
        )
        self.a = nn.Linear(config.hidden_size, inner, bias=False)
        self.b = nn.Linear(config.hidden_size, inner, bias=False)
        self.out = nn.Linear(inner, config.hidden_size, bias=False)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, s: Tensor, g: Tensor | None) -> Tensor:
        s = self.norm(s, g)
        return self.drop(self.out(F.silu(self.a(s)) * self.b(s)))


class PairStack(nn.Module):
    """Own the pair featurization and per-layer pair refinement (exposed as ``bias_module``).

    Holds ``.ff`` (signed-Δ Fourier bank) and ``.evaluate`` so the baseline diagnostics resolve.
    The per-layer refinement lives in ``PairformerBlock`` instances stored here; the single
    stream reads each block's pair bias in ``MSDeltaPairformerModel.forward``.
    """

    def __init__(self, config: MSDeltaPairformerConfig):
        super().__init__()
        self.n_layers = config.num_hidden_layers
        self.pair_feats = PairFeatures(config)
        self.w_a = nn.Linear(config.hidden_size, config.pair_channels, bias=False)
        self.w_b = nn.Linear(config.hidden_size, config.pair_channels, bias=False)
        self.w_c = nn.Linear(self.pair_feats.out_dim, config.pair_channels, bias=False)
        self.blocks = nn.ModuleList([PairformerBlock(config) for _ in range(self.n_layers)])

    @property
    def ff(self) -> FourierFeatures:
        """The signed-Δ Fourier bank, where ``FourierProbeCallback`` looks for it."""
        return self.pair_feats.ff

    def init_state(
        self,
        s_init: Tensor,
        mz: Tensor,
        log_intensity: Tensor,
        precursor_mz: Tensor | None,
        charge: Tensor | None,
    ) -> Tensor:
        """z_init[i,j] = W_a s_i + W_b s_j + W_c pair_feats[i,j] (asymmetric: W_a != W_b)."""
        feats = self.pair_feats(mz, log_intensity, precursor_mz, charge)
        outer_sum = self.w_a(s_init).unsqueeze(-2) + self.w_b(s_init).unsqueeze(-3)
        return outer_sum + self.w_c(feats.to(self.w_c.weight.dtype))

    @torch.no_grad()
    def evaluate_layers(self, delta_mz_grid: Tensor) -> Tensor:
        """Per-layer pure-Δ bias curves -> (layers, grid, heads).

        This is a DIAGNOSTIC projection: it drives z with the signed-Δ Fourier features alone
        (intensity coupling, the loss dictionary, complementarity, isotope terms, the outer-sum
        from s, triangle mixing and write-back all zeroed), then applies each block's pointwise
        transition and bias readout. It therefore reports the m/z-difference component of the
        learned bias -- exactly what ``eval/alignment.py`` and ``eval/viz.py`` interpret -- and
        deliberately does NOT reflect the cross-peak (triangle) refinement that only exists on
        real N x N spectra.
        """
        dtype = self.w_c.weight.dtype
        feats = self.ff(delta_mz_grid).to(dtype)
        # W_c restricted to the signed-Δ block (the first ff.out_dim columns of pair_feats).
        z = F.linear(feats, self.w_c.weight[:, : feats.shape[-1]])
        curves = []
        for block in self.blocks:
            if block.pair_update in {"triangle", "transition"}:
                z = z + block.pair_transition(z)
            curves.append(block.read_bias(z))
        return torch.stack(curves).float()

    def evaluate(self, delta_mz_grid: Tensor) -> Tensor:
        """Final-layer pure-Δ curves -> (grid, heads); drop-in for ``DeltaMZBias.evaluate``."""
        return self.evaluate_layers(delta_mz_grid)[-1]


class MSDeltaPairformerModel(MSDeltaPreTrainedModel):
    """Encode peaks with joint single + pair representations refined at every layer."""

    config_class = MSDeltaPairformerConfig
    _no_split_modules = ["PairformerBlock"]

    def __init__(self, config: MSDeltaPairformerConfig):
        super().__init__(config)
        self.embed = PeakEmbedMZ(config)
        self.bias_module = PairStack(config)
        self.global_cond = GlobalCond(config) if config.use_global_cond else None
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()
        self._zero_init_residual_readouts()

    def _zero_init_residual_readouts(self) -> None:
        """Start conditioning and the pair write-back as no-ops so every block begins clean."""
        for block in self.bias_module.blocks:
            block.attn.pre_norm.zero_init()
            block.single_transition.norm.zero_init()
            if block.use_writeback:
                block.opm.zero_init()

    def _init_weights(self, module: nn.Module) -> None:
        # The base handles Linear / LayerNorm (incl. the affine-free norm in AdaLayerNorm);
        # PeakEmbedMZ's mask token needs the same treatment the baseline gives PeakEmbed.
        super()._init_weights(module)
        if isinstance(module, PeakEmbedMZ):
            module.mask_token.data.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        attention_mask: Tensor | None = None,
        mask_positions: Tensor | None = None,
        precursor_mz: Tensor | None = None,
        charge: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutput | tuple[Tensor, ...]:
        if return_dict is None:
            return_dict = self.config.return_dict
        if mz.ndim != 2 or log_intensity.shape != mz.shape:
            raise ValueError("mz and log_intensity must have the same two-dimensional shape")
        if attention_mask is None:
            attention_mask = torch.ones_like(mz, dtype=torch.bool)
        elif attention_mask.shape != mz.shape:
            raise ValueError("attention_mask must have the same shape as mz")
        if mask_positions is not None and mask_positions.shape != mz.shape:
            raise ValueError("mask_positions must have the same shape as mz")

        mask = attention_mask.bool()
        padding_mask = ~mask
        # (B, N, N, 1) bool: a pair is real only if both peaks are real. Cast to the operand
        # dtype at each use site so bf16 pair-branch math is not silently promoted to fp32.
        pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1)

        g = None
        if self.global_cond is not None and precursor_mz is not None and charge is not None:
            g = self.global_cond(precursor_mz, charge)

        s = self.embed(mz, log_intensity, mask_positions)
        z = self.bias_module.init_state(s, mz, log_intensity, precursor_mz, charge)

        checkpointing = self.gradient_checkpointing and self.training
        for block in self.bias_module.blocks:
            if checkpointing:
                s, z = self._gradient_checkpointing_func(
                    block.__call__, s, z, mask, pair_mask, padding_mask, g
                )
            else:
                s, z = block(s, z, mask, pair_mask, padding_mask, g)
        s = self.norm(s)

        if not return_dict:
            return (s,)
        return BaseModelOutput(last_hidden_state=s)


class MSDeltaPairformerForPreTraining(MSDeltaForPreTraining):
    """Masked-intensity pretraining on the Pairformer encoder.

    Reuses ``MSDeltaForPreTraining``'s loss but overrides ``forward`` to thread the optional
    global-conditioning inputs (precursor m/z, charge) into the encoder. The baseline forward
    already accepts and ignores those keys, so the same pretraining collator feeds both.
    """

    config_class = MSDeltaPairformerConfig

    def __init__(self, config: MSDeltaPairformerConfig):
        MSDeltaPreTrainedModel.__init__(self, config)
        self.msdelta = MSDeltaPairformerModel(config)
        self.intensity_head = IntensityHead(config.hidden_size)
        self.post_init()
        # post_init re-runs weight init over the whole tree; re-assert the residual no-ops so
        # AdaLayerNorm conditioning and the pair write-back start as identity (matches the
        # defensive pattern in pair_stream_additive).
        self.msdelta._zero_init_residual_readouts()

    def forward(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        attention_mask: Tensor | None = None,
        mask_positions: Tensor | None = None,
        labels: Tensor | None = None,
        precursor_mz: Tensor | None = None,
        charge: Tensor | None = None,
        return_dict: bool | None = None,
    ):
        if return_dict is None:
            return_dict = self.config.return_dict
        outputs = self.msdelta(
            mz=mz,
            log_intensity=log_intensity,
            attention_mask=attention_mask,
            mask_positions=mask_positions,
            precursor_mz=precursor_mz,
            charge=charge,
            return_dict=True,
        )
        logits = self.intensity_head(outputs.last_hidden_state)
        loss = self._masked_intensity_loss(logits, labels, mask_positions)
        if not return_dict:
            result = (logits,)
            return ((loss,) + result) if loss is not None else result
        return MSDeltaForPreTrainingOutput(loss=loss, logits=logits)
