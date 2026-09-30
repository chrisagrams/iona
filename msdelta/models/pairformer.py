"""AlphaFold-3-style Pairformer encoder for tandem mass spectra (``architecture="pairformer"``).

Where it comes from
-------------------
Ported from the ``sweep/pairformer-aurora`` branch (commit bc2037b, file
``msdelta/model/experiments/pairformer.py``; first added on ``exp_pairformer`` in a1275c0),
which adapts the Pairformer stack of AlphaFold 3 (Abramson et al., Nature 2024; Supplementary
Algorithms 11, 12/13, 14/15, 17, 24) -- itself built on the triangle updates of AlphaFold 2
(Jumper et al., Nature 2021) -- from residue pairs to peak pairs. ``notes/METHODS.md`` has the
credit and the list of what was borrowed versus adapted.

What it computes
----------------
Two hidden states refined together at every layer::

    s : (B, N, hidden_size)      one vector per peak       -- "single"
    z : (B, N, N, pair_channels) one vector per peak pair  -- "pair"

* ``s_init``: the transformer's intensity token (``PeakEmbed``, same mask token) plus, when
  ``pair_single_use_mz``, a projection of the Fourier features of the peak's own m/z.
* ``z_init[i, j] = W_a s_i + W_b s_j + W_c pair_feats[i, j]`` (W_a != W_b, so z is directional).
  ``pair_feats`` starts with exactly the features the transformer's ``DeltaMZBias`` uses --
  the ``FourierFeatures`` bank of the signed difference ``mz_i - mz_j`` built from the same
  ``delta_bias_*`` config fields -- followed by the source branch's hand-built chemistry
  priors, each switchable: Fourier of the mass defect frac(|Δ|), a Gaussian soft match of |Δ|
  against a neutral-loss / residue dictionary (σ in ppm of the heavier peak), 13C isotope
  spacing (k = 1, 2), and the relative intensity log I_i - log I_j (masked peaks contribute a
  learned stand-in, never their true intensity, which is the pretraining label).
* Each layer (AF3 Alg. 17): refine z (optional outer-product-mean write-back s -> z,
  triangle multiplication outgoing + incoming, optional triangle attention starting + ending
  node, SwiGLU transition), read a per-head additive attention bias from z, then update s with
  gated biased self-attention and a SwiGLU transition.
* ``pair_update_every`` / ``pair_bias_lag`` (K114-P) decouple the two streams: the pair update
  runs on fewer layers than the single update, optionally with a one-update-stale bias so the
  two could run concurrently. See "Decoupled streams" below; the defaults (1, 0) are the model
  described above, unchanged.
* ``pair_update`` reproduces the source's ablation ladder: ``"static"`` (z frozen after init,
  only the per-layer bias readout learns), ``"transition"`` (pointwise refinement only),
  ``"triangle"`` (+ triangle multiplication, the default).

Deliberate differences from the source
--------------------------------------
* No precursor inputs: the source's complementarity feature (p4) and its precursor/charge
  global conditioning (AdaLayerNorm) are dropped. The encoder keeps this repo's
  ``(mz, log_intensity, attention_mask, mask_positions)`` interface so the processor,
  collators and every head work unchanged; the source's own ``pairformer_intrinsic`` variant
  also dropped both because the precursor in that data was derived from the identification
  label. LayerNorms are therefore plain (affine) LayerNorms.
* The intensity token is the transformer's scalar ``PeakEmbed`` rather than a Fourier bank of
  intensity (this repo's config has no intensity-Fourier fields).
* ``FourierFeatures`` here has fixed (non-learnable) log-spaced frequencies, as elsewhere in
  this package.

Why a config field and not a separate model class
-------------------------------------------------
Selected by ``MSDeltaConfig.architecture``, not by a new ``model_type``. Every entry point
(``finetune_denoise``, ``finetune_contrastive``, ``finetune_align``, the eval scripts) loads
``MSDeltaForPreTraining.from_pretrained`` and reads ``.msdelta``; ``MSDeltaForDenoising`` /
``MSDeltaForRetrieval`` build ``MSDeltaModel(config.encoder)``. With a config field all of
those construct a Pairformer from a Pairformer checkpoint with no code change, and the
default (``"transformer"``) builds exactly the old module tree, so old checkpoints load
unchanged (same state-dict keys, same RNG consumption at init, same saved ``config.json``).

The module tree mirrors the transformer's so the code that reaches inside the encoder keeps
working: ``encoder.embed`` (hooked by layer-mix pooling; holds ``mask_token``),
``encoder.blocks`` (one module per layer whose output is the single state, so forward hooks
capture ``(B, N, hidden)`` exactly as for ``EncoderBlock``), ``encoder.bias_module`` (the pair
stack; ``.ff`` and ``.evaluate(grid) -> (grid, heads)`` as on ``DeltaMZBias``, so
``render_bias_panels`` and the alignment diagnostics run), and ``encoder.norm``.
``evaluate`` reports only the pure-Δm/z component of the learned bias (see
``PairStack.evaluate_layers``).

Decoupled streams (K114-P, ``pair_update_every = k``, ``pair_bias_lag``)
-----------------------------------------------------------------------
The layers are grouped into ROUNDS of ``k`` consecutive layers (the last round is shorter when
``k`` does not divide ``num_hidden_layers``). The pair update -- everything in ``PairLayer``
except the readout: write-back, triangle multiplications, triangle attention, pair transition
-- runs once per round, on the round's FIRST layer (``i % k == 0``), before that layer's single
block, exactly where it runs in the per-layer model. The other layers of the round run their
single block only; the write-back is part of the update, so skipped layers do NOT write back.
Every layer keeps its own bias readout (``bias_norm``, ``to_bias``; 2c_z + c_z*H parameters,
<1% of the step), so each single block still reads its own per-head bias from the current z.
Only the update layers build update modules; the ``layers.{i}`` indices are kept, so at k = 1
the state-dict keys are exactly the original ones.

Why the first layer of a round and not the last: at lag 0 every single block, including layer
0, reads a refined z (never the raw ``z_init``), k = 1 reduces to the original order, and the
"one pair update per k single blocks" rounds line up with the lag-1 concurrency below.

``pair_bias_lag = 1``: the single blocks of round m read z(m), the pair state from BEFORE the
round's update (round 0 reads ``z_init``), while the update produces z(m+1) for round m + 1.
Within a round the update and the single blocks then depend only on the round's inputs (z(m)
and the s entering the round -- the write-back reads the same s as at lag 0), so they could run
on two streams. The last round's update would never be read, so it is not built (DDP would
otherwise see unused parameters); lag 1 therefore has one pair update fewer than lag 0 at the
same k, and needs at least two rounds.

What true concurrency would still need (not implemented): per round, fork a side stream for
``PairLayer`` update m (event on the main stream after s and z(m) are ready, ``record_stream``
on the tensors it reads so the caching allocator does not recycle them early), run the k single
blocks on the main stream, join with an event before round m + 1. Autograd runs each backward
op on its forward op's stream, so backward overlaps too if the device backend honours that
(check on XPU). Gradient checkpointing recomputes a segment on the stream that runs its
backward, which serialises the recompute unless the checkpoint function sets the stream
itself. On one tile the overlap is limited because the triangle ops are memory-bound; across
two tiles it is model parallelism with an s and bias exchange per round.

Cost
----
Memory is the constraint. ``z`` is ``B * N^2 * pair_channels``; triangle multiplication holds
several ``B * N^2 * pair_tri_channels`` activations and costs O(B * N^3 * pair_tri_channels)
FLOPs per layer (the source measured it at ~87% of step time); the write-back materialises
``B * N^2 * pair_opm_channels^2``; triangle attention materialises
``B * chunk * N^2 * heads`` logits per chunk. The feature block before ``W_c`` is
``B * N^2 * (2 * delta_bias_n_freqs + ...)`` -- the same size as the transformer's bias
input. At N = 150 and B = 32 one fp32 ``(B, N, N, 64)`` tensor is ~0.18 GB, per layer and per
saved activation. The config defaults are test-sized; experiment sizes are chosen per run.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from .configuration_msdelta import MSDeltaConfig
from .fourier import FourierFeatures
from .modeling_msdelta import PeakEmbed, ZeroInitLinear

# Monoisotopic masses (Da) for the neutral-loss / residue soft-match dictionary, copied from
# the source branch. Hardcoded rather than imported from ``msdelta.data.chemistry`` so the
# model package imports nothing from ``data/``. The H entry sits 4.47 mDa from the 13C
# spacing, so the loss-bank and isotope columns correlate at high m/z (kept on purpose in
# the source: they are physically different events, separable at low mass).
_NEUTRAL_LOSS_BANK: tuple[float, ...] = (
    # Small neutral losses.
    1.007825, 2.015650, 15.010899, 15.994915, 16.018724, 17.002740, 17.026549, 18.010565,
    18.034374, 19.989830, 26.003074, 27.994915, 28.006148, 28.031300, 30.010565, 31.989829,
    32.026215, 33.987721, 34.968853, 42.010565, 43.005814, 43.989829, 44.997654, 45.992904,
    46.005479, 46.968261, 62.963701, 63.961901, 79.956815, 79.966331, 97.976896, 115.913700,
    # Amino-acid residue masses (L and I share a mass).
    57.021464, 71.037114, 87.032028, 97.052764, 99.068414, 101.047678, 103.009185,
    113.084064, 114.042927, 115.026943, 128.058578, 128.094963, 129.042593, 131.040485,
    137.058912, 147.068414, 156.101111, 163.063329, 186.079313,
    # Common sugar / modification residues (HPO3 is already in the first block).
    132.042259, 146.057909, 162.052824, 176.032088, 203.079373,
)

_C13_SPACING = 1.003355  # Δm between the 13C and 12C isotopologues (Da).


class PairformerPeakEmbed(PeakEmbed):
    """Single-representation init: the transformer's intensity token plus Fourier(m/z).

    Subclasses ``PeakEmbed`` so the mask token, its initialisation and the scalar-input
    linear layer are exactly the transformer's.
    """

    def __init__(self, config: MSDeltaConfig):
        super().__init__(config)
        self.use_mz = config.pair_single_use_mz
        if self.use_mz:
            self.ff_mz = FourierFeatures(
                config.delta_bias_n_freqs, config.delta_bias_f_min, config.delta_bias_f_max
            )
            self.mz_proj = nn.Linear(self.ff_mz.out_dim, config.hidden_size, bias=False)

    def forward(
        self, mz: Tensor, log_intensity: Tensor, mask_positions: Tensor | None = None
    ) -> Tensor:
        intensity = log_intensity.unsqueeze(-1).to(dtype=self.mask_token.dtype)
        tokens = self.mlp(intensity)
        if self.use_mz:
            tokens = tokens + self.mz_proj(self.ff_mz(mz).to(self.mz_proj.weight.dtype))
        if mask_positions is not None:
            # As in the source, a masked peak's whole token is the mask token; its m/z still
            # reaches the model through the pair features.
            tokens = torch.where(mask_positions.unsqueeze(-1), self.mask_token, tokens)
        return tokens


class PairFeatures(nn.Module):
    """Raw per-pair features ``pair_feats[i, j]``; the signed-Δ Fourier block comes first."""

    def __init__(self, config: MSDeltaConfig):
        super().__init__()
        self.use_intensity = config.pair_use_intensity
        self.use_mass_defect = config.pair_use_mass_defect
        self.use_loss_bank = config.pair_use_loss_bank
        self.use_isotope = config.pair_use_isotope
        self.sigma_ppm = config.pair_loss_bank_sigma_ppm
        # Same bank (same config fields) as DeltaMZBias.ff; exposed as ``.ff``.
        self.ff = FourierFeatures(
            config.delta_bias_n_freqs, config.delta_bias_f_min, config.delta_bias_f_max
        )
        if self.use_mass_defect:
            n = config.pair_mass_defect_n_freqs
            self.ff_defect = FourierFeatures(n, 1.0, float(max(n, 2)))
        bank = _NEUTRAL_LOSS_BANK if self.use_loss_bank else ()
        self.register_buffer("loss_bank", torch.tensor(bank, dtype=torch.float32), persistent=True)
        if self.use_intensity:
            # Stand-in for a masked peak's log intensity. Without it the relative-intensity
            # feature would hand the model the very value masked pretraining asks it to
            # predict (the source measured that leak at skill 0.985).
            self.mask_log_intensity = nn.Parameter(torch.zeros(()))
        self.out_dim = (
            self.ff.out_dim
            + (self.ff_defect.out_dim if self.use_mass_defect else 0)
            + self.loss_bank.numel()
            + (2 if self.use_isotope else 0)
            + (1 if self.use_intensity else 0)
        )

    def forward(
        self, mz: Tensor, log_intensity: Tensor, mask_positions: Tensor | None = None
    ) -> Tensor:
        mz = mz.float()
        delta = mz.unsqueeze(-1) - mz.unsqueeze(-2)  # (B, N, N), signed
        abs_delta = delta.abs()
        feats = [self.ff(delta)]
        if self.use_mass_defect:
            feats.append(self.ff_defect(abs_delta - abs_delta.floor()))

        # Tolerance in Da grows with the heavier peak's m/z.
        heavier = torch.maximum(mz.unsqueeze(-1), mz.unsqueeze(-2)).clamp_min(1.0)
        sigma = (self.sigma_ppm * 1e-6) * heavier
        two_var = 2.0 * (sigma * sigma).clamp_min(1e-12)

        if self.use_loss_bank:
            diff = abs_delta.unsqueeze(-1) - self.loss_bank.float()
            feats.append(torch.exp(-(diff * diff) / two_var.unsqueeze(-1)))
        if self.use_isotope:
            iso = [torch.exp(-((abs_delta - _C13_SPACING * k) ** 2) / two_var) for k in (1, 2)]
            feats.append(torch.stack(iso, dim=-1))
        if self.use_intensity:
            stand_in = self.mask_log_intensity.float()
            visible = log_intensity.float()
            if mask_positions is not None:
                visible = torch.where(mask_positions.bool(), stand_in, visible)
            else:
                # Keep the stand-in in the graph (with zero effect) when nothing is masked,
                # so fine-tuning under DDP does not see it as an unused parameter.
                visible = visible + 0.0 * stand_in
            feats.append((visible.unsqueeze(-1) - visible.unsqueeze(-2)).unsqueeze(-1))
        return torch.cat(feats, dim=-1)


class Transition(nn.Module):
    """SwiGLU transition with pre-LayerNorm (AF3 Alg. 11); applied residually by the caller."""

    def __init__(self, channels: int, inner: int, eps: float, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)
        self.a = nn.Linear(channels, inner, bias=False)
        self.b = nn.Linear(channels, inner, bias=False)
        self.out = nn.Linear(inner, channels, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.drop(self.out(F.silu(self.a(x)) * self.b(x)))


class TriangleMultiplication(nn.Module):
    """Triangle multiplicative update, outgoing or incoming edges (AF3 Alg. 12/13)."""

    def __init__(self, config: MSDeltaConfig, outgoing: bool):
        super().__init__()
        c_z, c = config.pair_channels, config.pair_tri_channels
        eps = config.layer_norm_eps
        self.outgoing = outgoing
        self.norm = nn.LayerNorm(c_z, eps=eps)
        self.a_proj = nn.Linear(c_z, c, bias=False)
        self.a_gate = nn.Linear(c_z, c, bias=False)
        self.b_proj = nn.Linear(c_z, c, bias=False)
        self.b_gate = nn.Linear(c_z, c, bias=False)
        self.out_norm = nn.LayerNorm(c, eps=eps)
        self.out_proj = nn.Linear(c, c_z, bias=False)
        self.out_gate = nn.Linear(c_z, c_z, bias=False)

    def forward(self, z: Tensor, pair_mask: Tensor) -> Tensor:
        z = self.norm(z)
        mask = pair_mask.to(z.dtype)
        # Mask both endpoints so a padded peak k contributes nothing to the sum over k.
        a = torch.sigmoid(self.a_gate(z)) * self.a_proj(z) * mask
        b = torch.sigmoid(self.b_gate(z)) * self.b_proj(z) * mask
        if self.outgoing:
            out = torch.einsum("bikc,bjkc->bijc", a, b)
        else:
            out = torch.einsum("bkic,bkjc->bijc", a, b)
        return torch.sigmoid(self.out_gate(z)) * self.out_proj(self.out_norm(out))


class TriangleAttention(nn.Module):
    """Triangle self-attention around the starting or ending node (AF3 Alg. 14/15).

    Chunked over the query row to bound the ``(B, chunk, N, N, heads)`` logits. Two
    training-memory options (K102), both numerically equivalent to the default:

    - ``pair_tri_attn_checkpoint_chunks``: each chunk runs under
      ``torch.utils.checkpoint`` (non-reentrant) while grad is enabled, so autograd keeps
      only the chunk's inputs (views of q/k/v plus the shared bias) instead of every
      chunk's ``(B, chunk, N, N, H)`` attention weights; backward recomputes one chunk at a
      time.
    - ``pair_tri_attn_impl="sdpa"``: each chunk is one
      ``F.scaled_dot_product_attention`` call with the triangle bias plus the padded-key
      mask as its float ``attn_mask`` (broadcast over the query row), so a fused /
      memory-efficient backend can avoid materialising the weights. The 4-D call folds
      ``(B, chunk)`` into the batch dim, so the ``(H, N, N)`` mask is copied once per row.
    - ``pair_tri_attn_impl="sdpa_view"`` (K117): same SDPA call, but the rows of a chunk go
      into SDPA's second ("head") dim, ``(B*H, chunk, N, d)``, and the mask is a stride-0
      ``expand`` view over the rows: no per-row mask copy. Numerically equal to "sdpa"
      (fp32); K117 bench (job 8879977): forward 1.0-1.27x, peak memory ~2.5x lower, training
      step 0.98-1.08x (0.98x at B=8, N=256), so it is selectable but not the default.
    """

    def __init__(self, config: MSDeltaConfig, starting: bool):
        super().__init__()
        c_z = config.pair_channels
        self.starting = starting
        self.h = config.pair_tri_attn_heads
        self.d = config.pair_tri_attn_dim
        self.chunk = config.pair_tri_attn_chunk
        self.impl = getattr(config, "pair_tri_attn_impl", "naive")
        self.checkpoint_chunks = getattr(config, "pair_tri_attn_checkpoint_chunks", False)
        self.sdpa_flatten = True  # see _chunk_sdpa; not a config field (benchmarking only)
        inner = self.h * self.d
        self.norm = nn.LayerNorm(c_z, eps=config.layer_norm_eps)
        self.q = nn.Linear(c_z, inner, bias=False)
        self.k = nn.Linear(c_z, inner, bias=False)
        self.v = nn.Linear(c_z, inner, bias=False)
        self.bias = nn.Linear(c_z, self.h, bias=False)
        self.gate = nn.Linear(c_z, inner)
        self.out = nn.Linear(inner, c_z)

    def _chunk_naive(self, q: Tensor, k: Tensor, v: Tensor, bias: Tensor,
                     key_mask: Tensor) -> Tensor:
        """q/k/v ``(B, c, N, H, d)``; bias ``(B, N, N, H)``; key_mask ``(B, 1, 1, N, 1)``."""
        scale = 1.0 / math.sqrt(self.d)
        logits = torch.einsum("bcjhd,bckhd->bcjkh", q, k) * scale
        logits = logits.float() + bias[:, None].float() + key_mask
        attn = torch.softmax(logits, dim=3).to(v.dtype)
        return torch.einsum("bcjkh,bckhd->bcjhd", attn, v)

    def _chunk_sdpa(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor) -> Tensor:
        """q/k/v ``(B, c, N, H, d)``; attn_mask ``(B, 1, H, N, N)`` (broadcast over c).

        ``sdpa_flatten`` (default) folds ``(B, c)`` into one batch dim so the call is the 4-D
        form fused backends expect; the mask is then expanded to ``(B * c, H, N, N)`` (one
        chunk's worth, in the compute dtype). With it off the 5-D tensors go in as is and the
        mask broadcasts without a copy (backends may then fall back to the math path).
        """
        b, c, n = q.shape[:3]
        q, k, v = (t.permute(0, 1, 3, 2, 4) for t in (q, k, v))  # (B, c, H, N, d)
        if self.sdpa_flatten:
            q, k, v = (t.reshape(b * c, self.h, n, self.d) for t in (q, k, v))
            attn_mask = attn_mask.expand(b, c, self.h, n, n).reshape(b * c, self.h, n, n)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return out.view(b, c, self.h, n, self.d).permute(0, 1, 3, 2, 4)  # (B, c, N, H, d)

    def _chunk_sdpa_view(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor) -> Tensor:
        """q/k/v ``(B, c, N, H, d)``; attn_mask ``(B, H, N, N)``, contiguous.

        Rows of the chunk take SDPA's head dim and ``(B, H)`` its batch dim, so the mask is
        passed as ``(B*H, c, N, N)`` with stride 0 over the rows -- a view, not a copy.
        """
        b, c, n = q.shape[:3]
        q, k, v = (t.permute(0, 3, 1, 2, 4).reshape(b * self.h, c, n, self.d) for t in (q, k, v))
        attn_mask = attn_mask[:, :, None].expand(b, self.h, c, n, n).reshape(b * self.h, c, n, n)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return out.view(b, self.h, c, n, self.d).permute(0, 2, 3, 1, 4)  # (B, c, N, H, d)

    def forward(self, z: Tensor, mask: Tensor) -> Tensor:
        # Ending-node attention is starting-node attention on the transposed pair tensor.
        if not self.starting:
            z = z.transpose(1, 2)
        z = self.norm(z)
        b, n = z.shape[:2]
        q = self.q(z).view(b, n, n, self.h, self.d)
        k = self.k(z).view(b, n, n, self.h, self.d)
        v = self.v(z).view(b, n, n, self.h, self.d)
        bias = self.bias(z)  # (B, N(j), N(k), H), broadcast over the query row i.
        if self.impl == "sdpa":
            # Float mask (B, 1, H, N(j), N(k)) in the compute dtype: the bias with padded keys
            # set to that dtype's most negative value (masked_fill rather than +, so the sum
            # cannot overflow to -inf and a fully padded row stays finite, as in the naive path).
            attn_mask = bias.to(q.dtype).permute(0, 3, 1, 2)[:, None]
            attn_mask = attn_mask.masked_fill(~mask[:, None, None, None, :],
                                              torch.finfo(q.dtype).min)
            fn, extra = self._chunk_sdpa, (attn_mask,)
        elif self.impl == "sdpa_view":
            # Same mask as "sdpa", as (B, H, N(j), N(k)); contiguous so (B, H) merge into one
            # batch dim without a copy in _chunk_sdpa_view.
            attn_mask = bias.to(q.dtype).permute(0, 3, 1, 2)
            attn_mask = attn_mask.masked_fill(~mask[:, None, None, :],
                                              torch.finfo(q.dtype).min).contiguous()
            fn, extra = self._chunk_sdpa_view, (attn_mask,)
        else:
            key_mask = torch.zeros(mask.shape, dtype=torch.float32, device=z.device)
            key_mask = key_mask.masked_fill(~mask, torch.finfo(torch.float32).min)
            key_mask = key_mask[:, None, None, :, None]
            fn, extra = self._chunk_naive, (bias, key_mask)
        use_ckpt = self.checkpoint_chunks and torch.is_grad_enabled()
        out = torch.empty_like(q)
        for s in range(0, n, self.chunk):
            e = min(s + self.chunk, n)
            args = (q[:, s:e], k[:, s:e], v[:, s:e], *extra)
            if use_ckpt:
                out[:, s:e] = torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
            else:
                out[:, s:e] = fn(*args)
        out = torch.sigmoid(self.gate(z)) * out.reshape(b, n, n, -1)
        out = self.out(out)
        return out if self.starting else out.transpose(1, 2)


class OuterProductMean(nn.Module):
    """Single -> pair write-back, ``z_ij += Linear(a_i (x) b_j)`` (AF3 Alg. 9, one sequence).

    The output projection is a ``ZeroInitLinear`` (zeroed by
    ``MSDeltaPreTrainedModel._init_weights``) so the write-back starts as a no-op.
    """

    def __init__(self, config: MSDeltaConfig):
        super().__init__()
        c = config.pair_opm_channels
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.left = nn.Linear(config.hidden_size, c, bias=False)
        self.right = nn.Linear(config.hidden_size, c, bias=False)
        self.out = ZeroInitLinear(c * c, config.pair_channels)

    def forward(self, s: Tensor, mask: Tensor) -> Tensor:
        s = self.norm(s)
        m = mask.unsqueeze(-1).to(s.dtype)
        a = self.left(s) * m
        b = self.right(s) * m
        outer = torch.einsum("bic,bjd->bijcd", a, b)
        return self.out(outer.reshape(*outer.shape[:3], -1))


def pair_update_schedule(config: MSDeltaConfig) -> list[tuple[bool, bool]]:
    """Per layer ``(round_start, updates)`` for ``pair_update_every`` / ``pair_bias_lag``.

    ``round_start``: the layer opens a round of ``k`` layers (``i % k == 0``). ``updates``: the
    layer holds (and runs) a pair update -- every round start, except at lag 1 the last round's,
    whose output nothing would read. Defaults: every layer is ``(True, True)``.
    """
    n = config.num_hidden_layers
    k = getattr(config, "pair_update_every", 1)
    lag = getattr(config, "pair_bias_lag", 0)
    return [(i % k == 0, i % k == 0 and (lag == 0 or i + k < n)) for i in range(n)]


class PairLayer(nn.Module):
    """One layer's pair refinement (when ``updates``) plus its per-head attention-bias readout.

    With ``updates=False`` (a layer skipped by ``pair_update_every``, K114-P) only the readout
    is built and ``forward`` passes z through unchanged.
    """

    def __init__(self, config: MSDeltaConfig, updates: bool = True, round_start: bool = True):
        super().__init__()
        self.updates = updates
        self.round_start = round_start
        self.pair_update = config.pair_update if updates else "static"
        self.use_writeback = config.pair_use_writeback and updates
        self.use_triangle_attention = config.pair_use_triangle_attention and updates
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
        if self.pair_update in ("triangle", "transition"):
            self.transition = Transition(
                config.pair_channels,
                config.pair_transition_expansion * config.pair_channels,
                config.layer_norm_eps,
            )
        self.bias_norm = nn.LayerNorm(config.pair_channels, eps=config.layer_norm_eps)
        self.to_bias = nn.Linear(config.pair_channels, config.num_attention_heads, bias=False)

    def read_bias(self, z: Tensor) -> Tensor:
        """Per-head bias ``(..., heads)`` from the pair state."""
        bias = self.to_bias(self.bias_norm(z))
        if self.scale is not None:
            bias = self.scale * torch.tanh(bias / self.scale)
        return bias

    def forward(
        self, z: Tensor, s: Tensor, mask: Tensor, pair_mask: Tensor,
        z_read: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the refined ``z`` and the ``(B, heads, N, N)`` attention bias.

        The bias is read from the refined ``z``, or from ``z_read`` when given (the lagged
        state, ``pair_bias_lag=1``; then the update and the readout are independent).
        """
        if self.use_writeback:
            z = z + self.dropout(self.opm(s, mask))
        if self.pair_update == "triangle":
            z = z + self.dropout(self.tri_out(z, pair_mask))
            z = z + self.dropout(self.tri_in(z, pair_mask))
            if self.use_triangle_attention:
                z = z + self.dropout(self.tri_attn_start(z, mask))
                z = z + self.dropout(self.tri_attn_end(z, mask))
        if self.pair_update in ("triangle", "transition"):
            z = z + self.transition(z)
        bias = self.read_bias(z if z_read is None else z_read).permute(0, 3, 1, 2).contiguous()
        return z, bias


class PairStack(nn.Module):
    """Pair featurisation, ``z`` initialisation and the per-layer pair refinement.

    Exposed as ``MSDeltaModel.bias_module`` with the ``DeltaMZBias`` diagnostic surface
    (``.ff``, ``.evaluate``).
    """

    def __init__(self, config: MSDeltaConfig):
        super().__init__()
        self.pair_feats = PairFeatures(config)
        self.w_a = nn.Linear(config.hidden_size, config.pair_channels, bias=False)
        self.w_b = nn.Linear(config.hidden_size, config.pair_channels, bias=False)
        self.w_c = nn.Linear(self.pair_feats.out_dim, config.pair_channels, bias=False)
        self.bias_lag = getattr(config, "pair_bias_lag", 0)
        self.layers = nn.ModuleList(
            [PairLayer(config, updates=u, round_start=r) for r, u in pair_update_schedule(config)]
        )

    @property
    def ff(self) -> FourierFeatures:
        """The signed-Δm/z Fourier bank (same role as ``DeltaMZBias.ff``)."""
        return self.pair_feats.ff

    def init_state(
        self, s: Tensor, mz: Tensor, log_intensity: Tensor, mask_positions: Tensor | None
    ) -> Tensor:
        """``z[i, j] = W_a s_i + W_b s_j + W_c pair_feats[i, j]``."""
        feats = self.pair_feats(mz, log_intensity, mask_positions)
        outer_sum = self.w_a(s).unsqueeze(-2) + self.w_b(s).unsqueeze(-3)
        return outer_sum + self.w_c(feats.to(self.w_c.weight.dtype))

    @torch.no_grad()
    def evaluate_layers(self, delta_mz_grid: Tensor) -> Tensor:
        """Per-layer pure-Δm/z bias curves, ``(layers, grid, heads)``.

        A DIAGNOSTIC projection: z is driven by the signed-Δ Fourier features alone (every
        other pair feature, the outer sum from s, triangle mixing and write-back zeroed),
        then each layer's pointwise transition and bias readout are applied. It shows the
        m/z-difference component of the learned bias, not the full bias on a real spectrum.
        """
        feats = self.ff(delta_mz_grid).to(self.w_c.weight.dtype)
        z = F.linear(feats, self.w_c.weight[:, : feats.shape[-1]])
        curves = []
        z_read = z
        for layer in self.layers:
            if self.bias_lag and layer.round_start:
                z_read = z
            if layer.pair_update in ("triangle", "transition"):
                z = z + layer.transition(z)
            curves.append(layer.read_bias(z_read if self.bias_lag else z))
        return torch.stack(curves).float()

    def evaluate(self, delta_mz_grid: Tensor) -> Tensor:
        """Final-layer pure-Δm/z curves, ``(grid, heads)``; drop-in for ``DeltaMZBias.evaluate``."""
        return self.evaluate_layers(delta_mz_grid)[-1]


class PairformerSingleBlock(nn.Module):
    """Single-stream update of one Pairformer layer (AF3 Alg. 17, lines 5-6).

    Gated self-attention with the pair bias (AF3 Alg. 24 without diffusion conditioning)
    then a SwiGLU transition. Same call signature and output as ``EncoderBlock``:
    ``forward(hidden_states, bias, padding_mask) -> hidden_states``.
    """

    def __init__(self, config: MSDeltaConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.d_head = config.hidden_size // config.num_attention_heads
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.gate = nn.Linear(config.hidden_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = config.attention_probs_dropout_prob
        self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.transition = Transition(
            config.hidden_size,
            config.intermediate_size,
            config.layer_norm_eps,
            dropout=config.hidden_dropout_prob,
        )

    def forward(self, hidden_states: Tensor, bias: Tensor, padding_mask: Tensor) -> Tensor:
        b, n, _ = hidden_states.shape
        x = self.norm(hidden_states)
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.d_head)
        query, key, value = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        attn_bias = bias.to(query.dtype).masked_fill(padding_mask[:, None, None, :], float("-inf"))
        context = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_bias,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(b, n, -1)
        context = torch.sigmoid(self.gate(x)) * context
        hidden_states = hidden_states + self.proj_dropout(self.out(context))
        return hidden_states + self.transition(hidden_states)


def build_pairformer(model: nn.Module, config: MSDeltaConfig) -> None:
    """Attach the Pairformer submodules to an ``MSDeltaModel`` (called from its ``__init__``)."""
    model.embed = PairformerPeakEmbed(config)
    model.bias_module = PairStack(config)
    model.blocks = nn.ModuleList(
        [PairformerSingleBlock(config) for _ in range(config.num_hidden_layers)]
    )


def encode_pairformer(
    model: nn.Module,
    mz: Tensor,
    log_intensity: Tensor,
    attention_mask: Tensor,
    mask_positions: Tensor | None,
) -> Tensor:
    """Run the Pairformer layers of ``model`` and return the (un-normalised) single state."""
    mask = attention_mask.bool()
    padding_mask = ~mask
    pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1)  # (B, N, N, 1)
    s = model.embed(mz, log_intensity, mask_positions)
    z = model.bias_module.init_state(s, mz, log_intensity, mask_positions)
    checkpointing = model.gradient_checkpointing and model.training
    lag = model.bias_module.bias_lag
    z_read = z
    for pair_layer, block in zip(model.bias_module.layers, model.blocks):
        # pair_bias_lag=1: the single blocks of a round read the z that entered the round.
        extra = ()
        if lag:
            if pair_layer.round_start:
                z_read = z
            extra = (z_read,)
        if checkpointing:
            z, bias = model._gradient_checkpointing_func(
                pair_layer.__call__, z, s, mask, pair_mask, *extra
            )
            s = model._gradient_checkpointing_func(block.__call__, s, bias, padding_mask)
        else:
            z, bias = pair_layer(z, s, mask, pair_mask, *extra)
            s = block(s, bias, padding_mask)
    return s
