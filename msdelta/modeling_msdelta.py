"""Hugging Face-native MSDelta model implementations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import SupConLoss
from sentence_transformers.sentence_transformer.modules import Pooling
from torch import Tensor, nn
from torch.distributed.nn.functional import all_gather as distributed_all_gather
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils.generic import ModelOutput

from .configuration_msdelta import (
    MSDeltaConfig,
    MSDeltaDenoisingConfig,
    MSDeltaRetrievalConfig,
)
from .fourier import FourierFeatures


@dataclass
class MSDeltaForPreTrainingOutput(ModelOutput):
    """Output of masked-intensity pretraining."""

    loss: Tensor | None = None
    logits: Tensor | None = None


class ScalarInputLinear(nn.Linear):
    """Linear layer that retains PyTorch's fan-in-aware initialization."""


@dataclass
class MSDeltaForDenoisingOutput(ModelOutput):
    """Output of peak-level signal/noise classification."""

    loss: Tensor | None = None
    logits: Tensor | None = None


@dataclass
class MSDeltaForRetrievalOutput(ModelOutput):
    """Output of spectrum-level contrastive retrieval."""

    loss: Tensor | None = None
    embeddings: Tensor | None = None


class PeakEmbed(nn.Module):
    """Create m/z-free tokens from normalized log intensity."""

    def __init__(self, config: MSDeltaConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            ScalarInputLinear(1, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.mask_token = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, log_intensity: Tensor, mask_positions: Tensor | None = None) -> Tensor:
        intensity = log_intensity.unsqueeze(-1).to(dtype=self.mask_token.dtype)
        tokens = self.mlp(intensity)
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
        )
        self.n_heads = config.num_attention_heads
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
        return torch.cat([mlp(feats) for mlp in self.head_mlps], dim=-1)

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

    config_class: type[PretrainedConfig] | None = MSDeltaConfig
    base_model_prefix = "msdelta"
    main_input_name = "mz"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EncoderBlock"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, ScalarInputLinear):
            return
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


class PeakDenoisingHead(nn.Module):
    """Predict one noise logit per encoded peak."""

    def __init__(self, config: MSDeltaDenoisingConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.encoder.hidden_size, config.head_hidden_size),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden_size, 1),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.projection(hidden_states).squeeze(-1).float()


class SpectrumRetrievalHead(nn.Module):
    """Pool peak tokens and project them into a normalized retrieval space."""

    def __init__(self, config: MSDeltaRetrievalConfig):
        super().__init__()
        self.pooling = Pooling(config.encoder.hidden_size, pooling_mode=("mean", "max"))
        self.projection = nn.Sequential(
            nn.Linear(2 * config.encoder.hidden_size, config.projection_hidden_size),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.projection_hidden_size, config.embedding_size),
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        mask = attention_mask.bool()
        pooled = self.pooling(
            {
                "token_embeddings": hidden_states,
                "attention_mask": mask,
                # Keep empty-spectrum mean pooling finite, including in float16.
                "token_weights_sum": mask.sum(dim=1).clamp_min(1).to(hidden_states.dtype),
            }
        )["sentence_embedding"]
        # Max pooling yields -inf when a spectrum has no valid peaks.
        pooled = torch.nan_to_num(pooled, neginf=0.0)
        embeddings = self.projection(pooled)
        return F.normalize(embeddings.float(), dim=-1)


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


class MSDeltaForDenoising(MSDeltaPreTrainedModel):
    """MSDelta encoder with a peak-level noise classifier."""

    config_class: type[PretrainedConfig] | None = MSDeltaDenoisingConfig

    def __init__(
        self,
        config: MSDeltaDenoisingConfig,
        encoder: MSDeltaModel | None = None,
        freeze_encoder: bool = False,
    ):
        super().__init__(config)
        self.msdelta = encoder if encoder is not None else MSDeltaModel(config.encoder)
        self.denoising_head = PeakDenoisingHead(config)
        self._encoder_is_frozen = False
        if encoder is None:
            self.post_init()
        else:
            self.denoising_head.apply(self._init_weights)
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze the encoder and keep its stochastic layers disabled."""
        self._encoder_is_frozen = True
        self.msdelta.requires_grad_(False)
        self.msdelta.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_is_frozen:
            self.msdelta.eval()
        return self

    def forward(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> MSDeltaForDenoisingOutput | tuple[Tensor, ...]:
        if return_dict is None:
            return_dict = self.config.return_dict
        if self._encoder_is_frozen:
            with torch.no_grad():
                outputs = self.msdelta(
                    mz=mz,
                    log_intensity=log_intensity,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        else:
            outputs = self.msdelta(
                mz=mz,
                log_intensity=log_intensity,
                attention_mask=attention_mask,
                return_dict=True,
            )
        logits = self.denoising_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            if labels.shape != logits.shape:
                raise ValueError("labels must have the same shape as mz")
            valid = labels != -100
            if attention_mask is not None:
                valid = valid & attention_mask.bool()
            loss = (
                F.binary_cross_entropy_with_logits(logits[valid], labels[valid].float())
                if valid.any()
                else logits.sum() * 0.0
            )

        if not return_dict:
            result = (logits,)
            return ((loss,) + result) if loss is not None else result
        return MSDeltaForDenoisingOutput(loss=loss, logits=logits)


class MSDeltaForRetrieval(MSDeltaPreTrainedModel):
    """MSDelta encoder with a spectrum-level contrastive retrieval head."""

    config_class: type[PretrainedConfig] | None = MSDeltaRetrievalConfig

    def __init__(
        self,
        config: MSDeltaRetrievalConfig,
        encoder: MSDeltaModel | None = None,
        freeze_encoder: bool = False,
    ):
        super().__init__(config)
        self.msdelta = encoder if encoder is not None else MSDeltaModel(config.encoder)
        self.retrieval_head = SpectrumRetrievalHead(config)
        self.contrastive_loss = SupConLoss(temperature=config.temperature)
        self._encoder_is_frozen = False
        if encoder is None:
            self.post_init()
        else:
            self.retrieval_head.apply(self._init_weights)
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze the encoder and keep its stochastic layers disabled."""
        self._encoder_is_frozen = True
        self.msdelta.requires_grad_(False)
        self.msdelta.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_is_frozen:
            self.msdelta.eval()
        return self

    def forward(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        attention_mask: Tensor | None = None,
        group_ids: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> MSDeltaForRetrievalOutput | tuple[Tensor, ...]:
        if return_dict is None:
            return_dict = self.config.return_dict
        if attention_mask is None:
            attention_mask = torch.ones_like(mz, dtype=torch.long)
        if self._encoder_is_frozen:
            with torch.no_grad():
                outputs = self.msdelta(
                    mz=mz,
                    log_intensity=log_intensity,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        else:
            outputs = self.msdelta(
                mz=mz,
                log_intensity=log_intensity,
                attention_mask=attention_mask,
                return_dict=True,
            )
        embeddings = self.retrieval_head(outputs.last_hidden_state, attention_mask)
        loss = None
        if group_ids is not None:
            if group_ids.ndim != 1 or group_ids.shape[0] != embeddings.shape[0]:
                raise ValueError("group_ids must contain one value per spectrum")
            loss_embeddings = embeddings
            labels = group_ids.long()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                # Each rank's collator assigns its own batch-local group IDs.
                labels = labels * world_size + rank
                # Gather with autograd so remote candidates receive gradients.
                loss_embeddings = torch.cat(distributed_all_gather(embeddings), dim=0)
                gathered_labels = [torch.empty_like(labels) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_labels, labels)
                labels = torch.cat(gathered_labels)
            loss = self.contrastive_loss(loss_embeddings, labels)

        if not return_dict:
            result = (embeddings,)
            return ((loss,) + result) if loss is not None else result
        return MSDeltaForRetrievalOutput(loss=loss, embeddings=embeddings)


MSDeltaModel.register_for_auto_class("AutoModel")
MSDeltaForPreTraining.register_for_auto_class("AutoModelForPreTraining")
MSDeltaForDenoising.register_for_auto_class("AutoModelForTokenClassification")
MSDeltaForRetrieval.register_for_auto_class("AutoModel")
