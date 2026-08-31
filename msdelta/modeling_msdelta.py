"""Hugging Face-native MSDelta model implementations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils.generic import ModelOutput

from .configuration_msdelta import MSDeltaConfig
from .fourier import FourierFeatures

MODEL_INPUT_WIDTH = 150


@dataclass
class MSDeltaForPreTrainingOutput(ModelOutput):
    """Output of masked-intensity pretraining."""

    loss: Tensor | None = None
    logits: Tensor | None = None


@dataclass
class MSDeltaForDenoisingOutput(ModelOutput):
    """Peak-level Monte Carlo influence and signal scores."""

    mean_influence: Tensor | None = None
    signal_score: Tensor | None = None
    std_influence: Tensor | None = None
    num_contexts: Tensor | None = None
    sem: Tensor | None = None
    z_score: Tensor | None = None
    original_peak_index: Tensor | None = None
    mz: Tensor | None = None
    intensity: Tensor | None = None
    log_intensity: Tensor | None = None
    num_views: int | None = None
    coverage_complete: bool | None = None


@dataclass(frozen=True)
class UniformViewSampler:
    """Sample uniformly random peak subsets while preserving spectrum order."""

    max_peaks_per_view: int = MODEL_INPUT_WIDTH

    def sample(self, num_peaks: int, generator: torch.Generator) -> Tensor:
        """Return CPU indices for one independently sampled view."""
        if num_peaks <= self.max_peaks_per_view:
            return torch.arange(num_peaks)
        selected = torch.randperm(num_peaks, generator=generator)[: self.max_peaks_per_view]
        return selected.sort().values


def _one_dimensional(values, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(values, dtype=dtype).reshape(-1).detach().cpu().contiguous()


def _masked_intensity_kl_per_row(
    logits: Tensor,
    labels: Tensor,
    mask_positions: Tensor,
) -> Tensor:
    """Match the model's masked-intensity KL objective without batch reduction."""
    selected = mask_positions.bool()
    log_prob = F.log_softmax(logits.masked_fill(~selected, float("-inf")), dim=-1).masked_fill(
        ~selected, 0.0
    )
    target = labels.float().masked_fill(~selected, 0.0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return F.kl_div(log_prob, target, reduction="none").sum(dim=-1)


def _pad_view(values: Tensor, width: int) -> Tensor:
    result = torch.zeros(width, dtype=values.dtype, device=values.device)
    result[: values.numel()] = values
    return result


def monte_carlo_loo_denoise(
    *,
    model: "MSDeltaForDenoising",
    mz,
    log_intensity,
    labels,
    intensity=None,
    attention_mask=None,
    original_peak_index=None,
    min_contexts_per_peak: int = 10,
    mask_fraction: float = 0.50,
    max_peaks_per_view: int = MODEL_INPUT_WIDTH,
    loo_batch_size: int = 64,
    max_views: int = 1000,
    seed: int = 42,
    eps: float = 1e-8,
) -> MSDeltaForDenoisingOutput:
    """Score peaks by their influence on masked-intensity reconstruction.

    Positive influence is more noise-like. Negative influence is more signal-like;
    ``signal_score`` is the negated mean influence.
    """
    mz_cpu = _one_dimensional(mz, torch.float32)
    log_intensity_cpu = _one_dimensional(log_intensity, torch.float32)
    labels_cpu = _one_dimensional(labels, torch.float32)
    num_peaks = mz_cpu.numel()
    intensity_cpu = (
        labels_cpu.clone() if intensity is None else _one_dimensional(intensity, torch.float32)
    )
    original_index_cpu = (
        torch.arange(num_peaks, dtype=torch.long)
        if original_peak_index is None
        else _one_dimensional(original_peak_index, torch.long)
    )
    device = next(model.parameters()).device
    sampler = UniformViewSampler(max_peaks_per_view=max_peaks_per_view)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    counts_cpu = torch.zeros(num_peaks, dtype=torch.long)
    sums = torch.zeros(num_peaks, dtype=torch.float64, device=device)
    squared_sums = torch.zeros_like(sums)
    num_views = 0
    was_training = model.training

    model.eval()
    try:
        with torch.inference_mode():
            while num_views < max_views and bool((counts_cpu < min_contexts_per_peak).any()):
                view_indices = sampler.sample(num_peaks, generator)
                view_size = view_indices.numel()
                num_probes = min(
                    view_size - 1,
                    max(2, int(round(view_size * mask_fraction))),
                )
                probe_positions = torch.randperm(view_size, generator=generator)[:num_probes]
                probe_mask_cpu = torch.zeros(view_size, dtype=torch.bool)
                probe_mask_cpu[probe_positions] = True
                candidate_positions = torch.nonzero(~probe_mask_cpu, as_tuple=False).squeeze(-1)

                baseline_mz = _pad_view(
                    mz_cpu[view_indices].to(device), MODEL_INPUT_WIDTH
                ).unsqueeze(0)
                baseline_log_intensity = _pad_view(
                    log_intensity_cpu[view_indices].to(device), MODEL_INPUT_WIDTH
                ).unsqueeze(0)
                baseline_labels = _pad_view(
                    labels_cpu[view_indices].to(device), MODEL_INPUT_WIDTH
                ).unsqueeze(0)
                baseline_attention = torch.zeros(
                    1, MODEL_INPUT_WIDTH, dtype=torch.bool, device=device
                )
                baseline_attention[:, :view_size] = True
                baseline_probes = torch.zeros_like(baseline_attention)
                baseline_probes[0, probe_positions.to(device)] = True

                baseline_loss = model(
                    mz=baseline_mz,
                    log_intensity=baseline_log_intensity,
                    attention_mask=baseline_attention,
                    mask_positions=baseline_probes,
                    labels=baseline_labels,
                    return_dict=True,
                ).loss

                for start in range(0, candidate_positions.numel(), loo_batch_size):
                    positions_cpu = candidate_positions[start : start + loo_batch_size]
                    positions = positions_cpu.to(device)
                    batch_size = positions.numel()
                    rows = torch.arange(batch_size, device=device)
                    loo_mz = baseline_mz.expand(batch_size, -1).clone()
                    loo_log_intensity = baseline_log_intensity.expand(batch_size, -1).clone()
                    loo_labels = baseline_labels.expand(batch_size, -1).clone()
                    loo_attention = baseline_attention.expand(batch_size, -1).clone()
                    loo_probes = baseline_probes.expand(batch_size, -1).clone()
                    loo_mz[rows, positions] = 0.0
                    loo_log_intensity[rows, positions] = 0.0
                    loo_labels[rows, positions] = 0.0
                    loo_attention[rows, positions] = False
                    loo_probes[rows, positions] = False

                    loo_logits = model(
                        mz=loo_mz,
                        log_intensity=loo_log_intensity,
                        attention_mask=loo_attention,
                        mask_positions=loo_probes,
                        return_dict=True,
                    ).logits
                    loo_losses = _masked_intensity_kl_per_row(loo_logits, loo_labels, loo_probes)
                    influences = (baseline_loss - loo_losses).to(torch.float64)
                    original_positions_cpu = view_indices[positions_cpu]
                    original_positions = original_positions_cpu.to(device)
                    sums.index_add_(0, original_positions, influences)
                    squared_sums.index_add_(0, original_positions, influences.square())
                    counts_cpu.index_add_(
                        0,
                        original_positions_cpu,
                        torch.ones_like(original_positions_cpu),
                    )
                num_views += 1
    finally:
        model.train(was_training)

    counts = counts_cpu.to(device)
    covered = counts > 0
    mean = torch.full_like(sums, torch.nan)
    std = torch.full_like(sums, torch.nan)
    mean[covered] = sums[covered] / counts[covered]
    variance = torch.zeros_like(sums)
    variance[covered] = squared_sums[covered] / counts[covered] - mean[covered].square()
    std[covered] = variance[covered].clamp_min(0.0).sqrt()
    sem = torch.full_like(sums, torch.nan)
    sem[covered] = std[covered] / counts[covered].sqrt()
    z_score = torch.full_like(sums, torch.nan)
    z_score[covered] = mean[covered] / (sem[covered] + eps)

    return MSDeltaForDenoisingOutput(
        mean_influence=mean.float().cpu(),
        signal_score=(-mean).float().cpu(),
        std_influence=std.float().cpu(),
        num_contexts=counts_cpu,
        sem=sem.float().cpu(),
        z_score=z_score.float().cpu(),
        original_peak_index=original_index_cpu,
        mz=mz_cpu,
        intensity=intensity_cpu,
        log_intensity=log_intensity_cpu,
        num_views=num_views,
        coverage_complete=bool((counts_cpu >= min_contexts_per_peak).all()),
    )


class PeakEmbed(nn.Module):
    """Create m/z-free tokens from normalized log intensity."""

    def __init__(self, config: MSDeltaConfig):
        super().__init__()
        self.ff_int = FourierFeatures(
            config.fourier_int_n_freqs,
            config.fourier_int_f_min,
            config.fourier_int_f_max,
            learnable=config.fourier_int_learnable,
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.ff_int.out_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.mask_token = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, log_intensity: Tensor, mask_positions: Tensor | None = None) -> Tensor:
        feats = self.ff_int(log_intensity)
        tokens = self.mlp(feats.to(self.mlp[0].weight.dtype))
        if mask_positions is not None:
            tokens = torch.where(mask_positions.unsqueeze(-1), self.mask_token, tokens)
        return tokens


class DeltaMZBias(nn.Module):
    """Create a learned attention bias from signed delta m/z."""

    def __init__(self, config: MSDeltaConfig):
        super().__init__()
        self.ff = FourierFeatures(
            config.delta_bias_n_freqs,
            config.delta_bias_f_min,
            config.delta_bias_f_max,
            log_spaced=True,
            learnable=config.delta_bias_learnable,
        )
        self.n_heads = config.num_attention_heads
        self.scale = config.delta_bias_scale
        self.head_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.ff.out_dim, config.delta_bias_per_head_hidden),
                    nn.GELU(),
                    nn.Linear(config.delta_bias_per_head_hidden, 1),
                )
                for _ in range(self.n_heads)
            ]
        )

    def _curve(self, feats: Tensor) -> Tensor:
        feats = feats.to(next(self.head_mlps.parameters()).dtype)
        out = torch.cat([mlp(feats) for mlp in self.head_mlps], dim=-1)
        return self.scale * torch.tanh(out / self.scale)

    def forward(self, mz: Tensor) -> Tensor:
        delta_mz = mz.unsqueeze(-1) - mz.unsqueeze(-2)
        curve = self._curve(self.ff(delta_mz))
        return curve.permute(0, 3, 1, 2).contiguous()

    def evaluate(self, delta_mz_grid: Tensor) -> Tensor:
        """Evaluate every attention-head bias curve on a delta m/z grid."""
        return self._curve(self.ff(delta_mz_grid)).float()


class BiasedMHA(nn.Module):
    """Apply multi-head attention with a learned per-head bias."""

    def __init__(self, config: MSDeltaConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.d_head = config.hidden_size // config.num_attention_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.attn_dropout = config.attention_probs_dropout_prob
        self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: Tensor,
        bias: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        batch_size, n_peaks, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(batch_size, n_peaks, 3, self.n_heads, self.d_head)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attention_bias = bias.masked_fill(padding_mask[:, None, None, :], float("-inf"))

        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(batch_size, n_peaks, -1)
        return self.proj_dropout(self.out(context))


class EncoderBlock(nn.Module):
    def __init__(self, config: MSDeltaConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = BiasedMHA(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

    def forward(
        self,
        hidden_states: Tensor,
        bias: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        attention_output = self.attn(self.norm1(hidden_states), bias, padding_mask)
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states))
        return hidden_states


class MSDeltaPreTrainedModel(PreTrainedModel):
    """Shared Hugging Face behavior for MSDelta model classes."""

    config_class = MSDeltaConfig
    base_model_prefix = "msdelta"
    main_input_name = "mz"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EncoderBlock"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PeakEmbed):
            module.mask_token.data.normal_(mean=0.0, std=self.config.initializer_range)


class MSDeltaModel(MSDeltaPreTrainedModel):
    """Encode mass-spectrum peaks with continuous relative-mass attention."""

    def __init__(self, config: MSDeltaConfig):
        super().__init__(config)
        self.embed = PeakEmbed(config)
        self.bias_module = DeltaMZBias(config)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        attention_mask: Tensor | None = None,
        mask_positions: Tensor | None = None,
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
        padding_mask = ~attention_mask.bool()

        hidden_states = self.embed(log_intensity, mask_positions)
        bias = self.bias_module(mz)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    block.__call__, hidden_states, bias, padding_mask
                )
            else:
                hidden_states = block(hidden_states, bias, padding_mask)
        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return (hidden_states,)
        return BaseModelOutput(last_hidden_state=hidden_states)


class IntensityHead(nn.Module):
    """Predict one masked-intensity logit per peak."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.projection(hidden_states).squeeze(-1).float()


class MSDeltaForPreTraining(MSDeltaPreTrainedModel):
    """MSDelta with the masked-intensity pretraining objective."""

    def __init__(self, config: MSDeltaConfig):
        super().__init__(config)
        self.msdelta = MSDeltaModel(config)
        self.intensity_head = IntensityHead(config.hidden_size)
        self.post_init()

    def forward(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        attention_mask: Tensor | None = None,
        mask_positions: Tensor | None = None,
        labels: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> MSDeltaForPreTrainingOutput | tuple[Tensor, ...]:
        if return_dict is None:
            return_dict = self.config.return_dict
        outputs = self.msdelta(
            mz=mz,
            log_intensity=log_intensity,
            attention_mask=attention_mask,
            mask_positions=mask_positions,
            return_dict=True,
        )
        logits = self.intensity_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            if labels.shape != logits.shape:
                raise ValueError("labels must have the same shape as mz")
            if mask_positions is None:
                raise ValueError("mask_positions must be provided with labels")
            selected = mask_positions.bool()
            if selected.any():
                log_prob = F.log_softmax(
                    logits.masked_fill(~selected, float("-inf")), dim=-1
                ).masked_fill(~selected, 0.0)
                target = labels.float().masked_fill(~selected, 0.0)
                target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                loss = F.kl_div(log_prob, target, reduction="batchmean")
            else:
                loss = logits.new_zeros(())

        if not return_dict:
            result = (logits,)
            return ((loss,) + result) if loss is not None else result
        return MSDeltaForPreTrainingOutput(
            loss=loss,
            logits=logits,
        )


class MSDeltaForDenoising(MSDeltaForPreTraining):
    """Pretrained MSDelta model with inference-time peak signal scoring.

    This class adds no parameters to :class:`MSDeltaForPreTraining`. Its inherited
    ``forward`` retains the masked-intensity objective, while :meth:`denoise`
    performs Monte Carlo leave-one-out inference over arbitrary-length spectra.
    """

    def denoise(
        self,
        mz,
        log_intensity,
        labels,
        intensity=None,
        attention_mask=None,
        original_peak_index=None,
        *,
        min_contexts_per_peak: int = 10,
        mask_fraction: float = 0.50,
        max_peaks_per_view: int = 150,
        loo_batch_size: int = 64,
        max_views: int = 1000,
        seed: int = 42,
        eps: float = 1e-8,
    ) -> MSDeltaForDenoisingOutput:
        """Return signal scores; larger values indicate more signal-like peaks."""
        return monte_carlo_loo_denoise(
            model=self,
            mz=mz,
            log_intensity=log_intensity,
            labels=labels,
            intensity=intensity,
            attention_mask=attention_mask,
            original_peak_index=original_peak_index,
            min_contexts_per_peak=min_contexts_per_peak,
            mask_fraction=mask_fraction,
            max_peaks_per_view=max_peaks_per_view,
            loo_batch_size=loo_batch_size,
            max_views=max_views,
            seed=seed,
            eps=eps,
        )


MSDeltaModel.register_for_auto_class("AutoModel")
MSDeltaForPreTraining.register_for_auto_class("AutoModelForPreTraining")
