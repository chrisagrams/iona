"""Peak-token transformer with learned per-head Δm/z attention bias + MPM heads."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import flex_attention
from sentence_transformers.sentence_transformer.modules import Pooling
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from .fourier import FourierFeatures


@dataclass
class FourierConfig:
    n_freqs: int
    f_min: float
    f_max: float
    learnable: bool = True


@dataclass
class DeltaBiasConfig:
    hidden: int = 128
    resolution: float = 0.01
    max_distance: float = 2000.0
    coordinate_scale: float = 1.0
    # Bound the per-head bias to ±scale logits via scale*tanh(raw/scale).
    # Keeps the bias comparable to the content term (q·k/√d ~ O(1-2)) so
    # neither can steamroll the other
    scale: float = 3.0


class MSDeltaConfig(PretrainedConfig):
    """Self-describing Hugging Face configuration for every MSDelta task head."""

    model_type = "msdelta"

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        max_peaks: int = 150,
        zero_bias_diagonal: bool = True,
        fourier_int_n_freqs: int = 16,
        fourier_int_f_min: float = 1e-2,
        fourier_int_f_max: float = 1e2,
        fourier_int_learnable: bool = True,
        delta_bias_hidden: int = 128,
        delta_bias_resolution: float = 0.01,
        delta_bias_max_distance: float = 2000.0,
        delta_bias_coordinate_scale: float = 1.0,
        delta_bias_scale: float = 3.0,
        pooling_modes: tuple[str, ...] | list[str] = ("mean", "max"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_mult = ffn_mult
        self.dropout = dropout
        self.max_peaks = max_peaks
        self.zero_bias_diagonal = zero_bias_diagonal
        self.fourier_int_n_freqs = fourier_int_n_freqs
        self.fourier_int_f_min = fourier_int_f_min
        self.fourier_int_f_max = fourier_int_f_max
        self.fourier_int_learnable = fourier_int_learnable
        self.delta_bias_hidden = delta_bias_hidden
        self.delta_bias_resolution = delta_bias_resolution
        self.delta_bias_max_distance = delta_bias_max_distance
        self.delta_bias_coordinate_scale = delta_bias_coordinate_scale
        self.delta_bias_scale = delta_bias_scale
        self.pooling_modes = list(pooling_modes)

    @property
    def fourier_int(self) -> FourierConfig:
        return FourierConfig(
            self.fourier_int_n_freqs,
            self.fourier_int_f_min,
            self.fourier_int_f_max,
            self.fourier_int_learnable,
        )

    @property
    def delta_bias(self) -> DeltaBiasConfig:
        return DeltaBiasConfig(
            hidden=self.delta_bias_hidden,
            resolution=self.delta_bias_resolution,
            max_distance=self.delta_bias_max_distance,
            coordinate_scale=self.delta_bias_coordinate_scale,
            scale=self.delta_bias_scale,
        )


ModelConfig = MSDeltaConfig


@dataclass
class MSDeltaPretrainingOutput(ModelOutput):
    loss: Tensor | None = None
    kl: Tensor | None = None


@dataclass
class MSDeltaEmbeddingOutput(ModelOutput):
    embeddings: Tensor | None = None
    token_embeddings: Tensor | None = None


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
    """Swin-V2-style continuous relative-position bias for signed Δm/z.

    A small MLP maps a normalized signed-log mass coordinate to one additive
    bias per attention head. To avoid v15's ``(B, K, K, hidden)`` activation,
    the MLP is evaluated only on a fixed uniform grid of Δm/z knots. Pairwise
    differences linearly interpolate neighboring knot values, so the runtime
    pair tensor carries only the head dimension.
    """

    def __init__(self, n_heads: int, cfg: DeltaBiasConfig):
        super().__init__()
        n_intervals = round(2 * cfg.max_distance / cfg.resolution)
        self.n_heads = n_heads
        self.resolution = cfg.resolution
        self.max_distance = cfg.max_distance
        self.coordinate_scale = cfg.coordinate_scale
        self.scale = cfg.scale

        self.mlp = nn.Sequential(
            nn.Linear(1, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, n_heads, bias=False),
        )
        # Start from ordinary content attention, matching v15's zero-bias init.
        nn.init.zeros_(self.mlp[-1].weight)

        knot_dm = torch.linspace(
            -cfg.max_distance,
            cfg.max_distance,
            n_intervals + 1,
            dtype=torch.float32,
        )
        self.register_buffer("knot_dm", knot_dm, persistent=False)

    def _coordinate(self, dm: Tensor) -> Tensor:
        """Map physical Δm/z to a normalized signed-log coordinate in [-1, 1]."""
        dm = dm.float().clamp(-self.max_distance, self.max_distance)
        x = torch.sign(dm) * torch.log1p(dm.abs() / self.coordinate_scale)
        normalizer = math.log1p(self.max_distance / self.coordinate_scale)
        return x / normalizer

    def _curve(self, dm: Tensor) -> Tensor:
        """Evaluate the bounded continuous MLP: (...,) → (..., n_heads)."""
        x = self._coordinate(dm).unsqueeze(-1)
        raw = self.mlp(x.to(self.mlp[0].weight.dtype))
        return self.scale * torch.tanh(raw / self.scale)

    def _make_bias_table(self) -> Tensor:
        """Generate the current bounded bias curve at all fixed Δm/z knots."""
        return self._curve(self.knot_dm)

    def interpolation_coefficients(self) -> tuple[Tensor, Tensor]:
        """Return per-interval intercept and slope tables for FlexAttention."""
        table = self._make_bias_table()
        # Capture the fixed-layout table itself in score_mod. Capturing
        # ``table[:-1]`` creates a SliceView that Inductor cannot dynamically
        # index while rendering the eval FlexAttention template.
        return table, table[1:] - table[:-1]

    def _interpolate(self, dm: Tensor, table: Tensor) -> Tensor:
        """Linearly interpolate ``table`` at arbitrary continuous Δm/z values."""
        dm = dm.float().clamp(-self.max_distance, self.max_distance)
        position = (dm + self.max_distance) / self.resolution
        left = position.floor().long().clamp(0, table.shape[0] - 2)
        fraction = (position - left.to(position.dtype)).clamp(0.0, 1.0)
        weight = fraction.unsqueeze(-1).to(table.dtype)
        return torch.lerp(table[left], table[left + 1], weight)

    def forward(self, mz: Tensor) -> Tensor:
        """mz: (B, K) → bias: (B, n_heads, K, K)."""
        dm = mz.unsqueeze(-1) - mz.unsqueeze(-2)             # (B, K, K), signed
        curve = self._interpolate(dm, self._make_bias_table()) # (B, K, K, H)
        return curve.permute(0, 3, 1, 2).contiguous()        # (B, H, K, K)

    def evaluate(self, dm_grid: Tensor) -> Tensor:
        """Evaluate per-head bias on a 1-D Δm grid.

        dm_grid: (N,) → (N, n_heads). Bounded bias (what attention sees).
        """
        return self._interpolate(dm_grid, self._make_bias_table()).float()


@torch.compiler.disable
def _eval_interpolation_coefficients(
    bias_module: DeltaMZBias,
) -> tuple[Tensor, Tensor]:
    """Materialize bias tables before entering compiled eval attention.

    Inductor's eval FlexAttention template cannot dynamically index a table
    whose producer is still part of the same compiled graph: its loader reaches
    back into an unresolved-layout MLP buffer. A narrow graph boundary makes
    both tables concrete inputs to the resumed compiled graph. Training keeps
    the end-to-end compiled path and its gradients unchanged.
    """
    return bias_module.interpolation_coefficients()


class BiasedMHA(nn.Module):
    """FlexAttention with on-kernel interpolated continuous Δm/z bias."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.bias_resolution = cfg.delta_bias.resolution
        self.bias_max_distance = cfg.delta_bias.max_distance
        self.zero_bias_diagonal = cfg.zero_bias_diagonal
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.proj_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        query_mz: Tensor,
        key_mz: Tensor,
        bias_intercept: Tensor,
        bias_slope: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        """Apply attention without materializing pairwise scores or biases."""
        B, K, _ = x.shape
        qkv = self.qkv(x).reshape(B, K, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, K, H, d_head)
        q = q.transpose(1, 2)  # (B, H, K, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        resolution = self.bias_resolution
        max_distance = self.bias_max_distance
        # bias_intercept is the full knot table (one entry longer); bias_slope
        # determines the valid left-endpoint interval indices.
        n_intervals = bias_slope.shape[0]
        zero_diagonal = self.zero_bias_diagonal

        def score_mod(score, batch, head, q_idx, kv_idx):
            dm = query_mz[batch, q_idx] - key_mz[batch, kv_idx]
            dm = dm.clamp(-max_distance, max_distance)
            position = (dm + max_distance) / resolution
            left = position.floor().to(torch.int64).clamp(0, n_intervals - 1)
            fraction = (position - left.to(position.dtype)).to(bias_intercept.dtype)
            bias = (
                bias_intercept[left, head]
                + fraction * bias_slope[left, head]
            )
            if zero_diagonal:
                bias = torch.where(q_idx == kv_idx, 0.0, bias)
            score = score + bias.to(score.dtype)
            return torch.where(
                key_padding_mask[batch, kv_idx],
                -float("inf"),
                score,
            )

        # FlexAttention currently has no attention-probability dropout. The
        # projection and FFN dropout configured on the model remain active.
        out = flex_attention(q, k, v, score_mod=score_mod)
        out = out.transpose(1, 2).reshape(B, K, -1)
        return self.proj_dropout(self.out(out))


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = BiasedMHA(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        ffn_dim = cfg.d_model * cfg.ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(ffn_dim, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: Tensor,
        query_mz: Tensor,
        key_mz: Tensor,
        bias_intercept: Tensor,
        bias_slope: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        x = x + self.attn(
            self.norm1(x), query_mz, key_mz,
            bias_intercept, bias_slope, key_padding_mask,
        )
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
        if self.training:
            bias_intercept, bias_slope = self.bias_module.interpolation_coefficients()
        else:
            bias_intercept, bias_slope = _eval_interpolation_coefficients(
                self.bias_module
            )
        # FlexAttention currently permits one read per captured tensor in a
        # score modifier. Separate query/key copies keep each m/z capture to a
        # single indexed read; the clone is only O(B*K).
        key_mz = mz.clone()
        for blk in self.blocks:
            tokens = blk(
                tokens, mz, key_mz,
                bias_intercept, bias_slope, key_padding_mask,
            )
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


class MSDeltaPreTrainedModel(PreTrainedModel):
    config_class = MSDeltaConfig
    base_model_prefix = "encoder"

    def _init_weights(self, module: nn.Module) -> None:
        # Submodules already use the project's established PyTorch
        # initializers, including the intentionally zeroed bias MLP output.
        return


class _EmbeddingMixin:
    encoder: MSEncoder
    pooler: Pooling

    def _embedding_forward(
        self,
        mz: Tensor,
        log_int: Tensor,
        key_padding_mask: Tensor,
        *,
        return_token_embeddings: bool = False,
    ) -> MSDeltaEmbeddingOutput:
        tokens = self.encoder(mz, log_int, key_padding_mask, None)
        pooled = self.pooler({
            "token_embeddings": tokens,
            # Sentence Transformers uses 1=real while the encoder uses
            # True=padding.
            "attention_mask": (~key_padding_mask).long(),
        })["sentence_embedding"]
        # Sentence Transformers' max pool emits -inf for a completely padded
        # row; empty spectra have a defined zero embedding in this project.
        pooled = torch.nan_to_num(pooled, neginf=0.0, posinf=0.0)
        return MSDeltaEmbeddingOutput(
            embeddings=pooled,
            token_embeddings=tokens if return_token_embeddings else None,
        )


class MSDeltaForEmbedding(_EmbeddingMixin, MSDeltaPreTrainedModel):
    """Encoder plus canonical mean/max spectrum pooling for downstream use."""

    _keys_to_ignore_on_load_unexpected = [r"heads\..*"]

    def __init__(self, config: MSDeltaConfig):
        super().__init__(config)
        self.encoder = MSEncoder(config)
        self.pooler = Pooling(
            embedding_dimension=config.d_model,
            pooling_mode=tuple(config.pooling_modes),
        )
        self.post_init()

    def forward(
        self,
        mz: Tensor,
        log_int: Tensor,
        key_padding_mask: Tensor,
        return_token_embeddings: bool = False,
    ) -> MSDeltaEmbeddingOutput:
        return self._embedding_forward(
            mz,
            log_int,
            key_padding_mask,
            return_token_embeddings=return_token_embeddings,
        )


class MSDeltaForPretraining(_EmbeddingMixin, MSDeltaPreTrainedModel):
    """Masked-intensity pretraining model with a distributed embedding route."""

    def __init__(self, config: MSDeltaConfig):
        super().__init__(config)
        self.encoder = MSEncoder(config)
        self.pooler = Pooling(
            embedding_dimension=config.d_model,
            pooling_mode=tuple(config.pooling_modes),
        )
        self.heads = IntensityHead(config.d_model)
        self.post_init()

    def forward(
        self,
        mz: Tensor,
        log_int: Tensor,
        key_padding_mask: Tensor,
        mask_positions: Tensor | None = None,
        intensity_prob: Tensor | None = None,
        eval_mode: str = "pretraining",
    ) -> MSDeltaPretrainingOutput | MSDeltaEmbeddingOutput:
        if eval_mode == "embedding":
            return self._embedding_forward(mz, log_int, key_padding_mask)
        if eval_mode == "representations":
            return self._embedding_forward(
                mz,
                log_int,
                key_padding_mask,
                return_token_embeddings=True,
            )
        if eval_mode != "pretraining":
            raise ValueError(f"unknown eval_mode: {eval_mode}")
        if mask_positions is None or intensity_prob is None:
            raise ValueError(
                "mask_positions and intensity_prob are required for pretraining"
            )
        tokens = self.encoder(mz, log_int, key_padding_mask, mask_positions)
        loss, parts = self.heads.loss(tokens, intensity_prob, mask_positions)
        return MSDeltaPretrainingOutput(loss=loss, kl=parts["kl"])
