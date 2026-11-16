"""Hugging Face-native Iona model implementations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import SupConLoss
from sentence_transformers.sentence_transformer.modules import Pooling
from torch import Tensor, nn
from torch.distributed.nn.functional import all_gather as distributed_all_gather
from transformers import PretrainedConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils.generic import ModelOutput

from .chemistry import ISOTOPES, PROTON_MASS, WATER_MASS
from .configuration_iona import (
    IonaConfig,
    IonaDenoisingConfig,
    IonaDenovoConfig,
    IonaRetrievalConfig,
)
from .fourier import FourierFeatures


@dataclass
class IonaForPreTrainingOutput(ModelOutput):
    """Output of masked-intensity pretraining."""

    loss: Tensor | None = None
    logits: Tensor | None = None


class ScalarInputLinear(nn.Linear):
    """Linear layer that retains PyTorch's fan-in-aware initialization."""


@dataclass
class IonaForDenoisingOutput(ModelOutput):
    """Output of peak-level signal/noise classification."""

    loss: Tensor | None = None
    logits: Tensor | None = None


@dataclass
class IonaForRetrievalOutput(ModelOutput):
    """Output of spectrum-level contrastive retrieval."""

    loss: Tensor | None = None
    embeddings: Tensor | None = None


class PeakEmbed(nn.Module):
    """Create m/z-free tokens from normalized log intensity."""

    def __init__(self, config: IonaConfig):
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

    def __init__(self, config: IonaConfig):
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

    def _curve(self, delta_mz: Tensor) -> Tensor:
        feats = self.ff(delta_mz).to(self.head_mlps[0][0].weight.dtype)
        # Equivalent to torch.cat([mlp(feats) for mlp in self.head_mlps], dim=-1),
        # but stacked into one GEMM.
        w_in = torch.cat([mlp[0].weight for mlp in self.head_mlps])
        b_in = torch.cat([mlp[0].bias for mlp in self.head_mlps])
        w_out = torch.cat([mlp[2].weight for mlp in self.head_mlps])
        b_out = torch.cat([mlp[2].bias for mlp in self.head_mlps])
        hidden = F.gelu(F.linear(feats, w_in, b_in)).unflatten(-1, (self.n_heads, -1))
        return (hidden * w_out).sum(-1) + b_out

    def forward(self, mz: Tensor) -> Tensor:
        delta_mz = mz.unsqueeze(-1) - mz.unsqueeze(-2)
        return self._curve(delta_mz).permute(0, 3, 1, 2).contiguous()

    def evaluate(self, delta_mz_grid: Tensor) -> Tensor:
        """Evaluate every attention-head bias curve on a delta m/z grid."""
        return self._curve(delta_mz_grid).float()


class BiasedMHA(nn.Module):
    """Apply multi-head attention with a learned per-head bias."""

    def __init__(self, config: IonaConfig):
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
        attention_bias: Tensor,
    ) -> Tensor:
        batch_size, n_peaks, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(batch_size, n_peaks, 3, self.n_heads, self.d_head)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

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
    def __init__(self, config: IonaConfig):
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
        attention_bias: Tensor,
    ) -> Tensor:
        attention_output = self.attn(self.norm1(hidden_states), attention_bias)
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states))
        return hidden_states


class IonaPreTrainedModel(PreTrainedModel):
    """Shared Hugging Face behavior for Iona model classes."""

    config_class: type[PretrainedConfig] | None = IonaConfig
    base_model_prefix = "iona"
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
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, PeakEmbed):
            module.mask_token.data.normal_(mean=0.0, std=self.config.initializer_range)


class IonaModel(IonaPreTrainedModel):
    """Encode mass-spectrum peaks with continuous relative-mass attention."""

    def __init__(self, config: IonaConfig):
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
        # Mask padded keys once so every layer shares one bias tensor instead of saving its own copy.
        attention_bias = self.bias_module(mz).masked_fill(
            padding_mask[:, None, None, :], float("-inf")
        )

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    block.__call__, hidden_states, attention_bias
                )
            else:
                hidden_states = block(hidden_states, attention_bias)
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

    def __init__(self, config: IonaDenoisingConfig):
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

    def __init__(self, config: IonaRetrievalConfig):
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


class IonaForPreTraining(IonaPreTrainedModel):
    """Iona with the masked-intensity pretraining objective."""

    def __init__(self, config: IonaConfig):
        super().__init__(config)
        self.iona = IonaModel(config)
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
    ) -> IonaForPreTrainingOutput | tuple[Tensor, ...]:
        if return_dict is None:
            return_dict = self.config.return_dict
        outputs = self.iona(
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
            log_prob = F.log_softmax(
                logits.masked_fill(~selected, float("-inf")), dim=-1
            ).masked_fill(~selected, 0.0)
            target = labels.float().masked_fill(~selected, 0.0)
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            loss = F.kl_div(log_prob, target, reduction="batchmean")

        if not return_dict:
            result = (logits,)
            return ((loss,) + result) if loss is not None else result
        return IonaForPreTrainingOutput(
            loss=loss,
            logits=logits,
        )


class IonaForDenoising(IonaPreTrainedModel):
    """Iona encoder with a peak-level noise classifier."""

    config_class: type[PretrainedConfig] | None = IonaDenoisingConfig

    def __init__(
        self,
        config: IonaDenoisingConfig,
        encoder: IonaModel | None = None,
        freeze_encoder: bool = False,
    ):
        super().__init__(config)
        self.iona = encoder if encoder is not None else IonaModel(config.encoder)
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
        self.iona.requires_grad_(False)
        self.iona.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_is_frozen:
            self.iona.eval()
        return self

    def forward(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> IonaForDenoisingOutput | tuple[Tensor, ...]:
        if return_dict is None:
            return_dict = self.config.return_dict
        if self._encoder_is_frozen:
            with torch.no_grad():
                outputs = self.iona(
                    mz=mz,
                    log_intensity=log_intensity,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        else:
            outputs = self.iona(
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
        return IonaForDenoisingOutput(loss=loss, logits=logits)


class IonaForRetrieval(IonaPreTrainedModel):
    """Iona encoder with a spectrum-level contrastive retrieval head."""

    config_class: type[PretrainedConfig] | None = IonaRetrievalConfig

    def __init__(
        self,
        config: IonaRetrievalConfig,
        encoder: IonaModel | None = None,
        freeze_encoder: bool = False,
    ):
        super().__init__(config)
        self.iona = encoder if encoder is not None else IonaModel(config.encoder)
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
        self.iona.requires_grad_(False)
        self.iona.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_is_frozen:
            self.iona.eval()
        return self

    def forward(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        attention_mask: Tensor | None = None,
        group_ids: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> IonaForRetrievalOutput | tuple[Tensor, ...]:
        if return_dict is None:
            return_dict = self.config.return_dict
        if attention_mask is None:
            attention_mask = torch.ones_like(mz, dtype=torch.long)
        if self._encoder_is_frozen:
            with torch.no_grad():
                outputs = self.iona(
                    mz=mz,
                    log_intensity=log_intensity,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        else:
            outputs = self.iona(
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
        return IonaForRetrievalOutput(loss=loss, embeddings=embeddings)


@dataclass
class IonaForDenovoOutput(ModelOutput):
    """Output of teacher-forced peptide decoding."""

    loss: Tensor | None = None
    logits: Tensor | None = None


@dataclass
class IonaDenovoGenerationOutput(ModelOutput):
    """Greedy-decoded peptide per spectrum.

    `sequences` holds token ids ending with EOS, padded; `token_scores` holds the probability
    of each chosen token, 0 at padding; `precursor_match` says whether the peptide mass fits
    the precursor.
    """

    sequences: Tensor | None = None
    token_scores: Tensor | None = None
    precursor_match: Tensor | None = None


class PrecursorEmbed(nn.Module):
    """Embed the precursor neutral mass and charge as the first decoder token."""

    def __init__(self, config: IonaDenovoConfig):
        super().__init__()
        hidden_size = config.decoder_hidden_size
        self.mass_features = FourierFeatures(
            config.precursor_n_freqs,
            config.precursor_f_min,
            config.precursor_f_max,
            clamp_abs=config.max_precursor_mass,
        )
        self.mass_mlp = nn.Sequential(
            nn.Linear(self.mass_features.out_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.charge = nn.Embedding(config.max_charge + 1, hidden_size)

    def forward(self, neutral_mass: Tensor, charge: Tensor) -> Tensor:
        # Fourier features are computed in float32; bf16 cannot resolve sub-dalton masses.
        features = self.mass_features(neutral_mass).to(self.mass_mlp[0].weight.dtype)
        return self.mass_mlp(features) + self.charge(charge)


class DecoderSelfAttention(nn.Module):
    """Causal multi-head self-attention over peptide tokens, with an optional KV cache."""

    def __init__(self, config: IonaDenovoConfig):
        super().__init__()
        hidden_size = config.decoder_hidden_size
        self.n_heads = config.decoder_num_heads
        self.d_head = hidden_size // config.decoder_num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = config.decoder_dropout
        self.proj_dropout = nn.Dropout(config.decoder_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        past_key_values: EncoderDecoderCache | None,
        layer_idx: int,
    ) -> Tensor:
        batch_size, n_tokens, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(batch_size, n_tokens, 3, self.n_heads, self.d_head)
        query, key, value = (x.transpose(1, 2) for x in qkv.unbind(dim=2))
        if past_key_values is not None:
            key, value = past_key_values.self_attention_cache.update(key, value, layer_idx)
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=n_tokens > 1,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(batch_size, n_tokens, -1)
        return self.proj_dropout(self.out(context))


class DecoderCrossAttention(nn.Module):
    """Multi-head attention from peptide tokens to encoded peaks."""

    def __init__(self, config: IonaDenovoConfig):
        super().__init__()
        hidden_size = config.decoder_hidden_size
        self.n_heads = config.decoder_num_heads
        self.d_head = hidden_size // config.decoder_num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key_value = nn.Linear(hidden_size, 2 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = config.decoder_dropout
        self.proj_dropout = nn.Dropout(config.decoder_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        past_key_values: EncoderDecoderCache | None,
        layer_idx: int,
    ) -> Tensor:
        """Attend from `hidden_states` to `memory`.

        With a cache, peak keys and values are projected on the first step and reused after.
        """
        batch_size, n_tokens, _ = hidden_states.shape
        if past_key_values is not None and past_key_values.is_updated.get(layer_idx):
            layer = past_key_values.cross_attention_cache.layers[layer_idx]
            key, value = layer.keys, layer.values
        else:
            key_value = self.key_value(memory).reshape(
                batch_size, memory.shape[1], 2, self.n_heads, self.d_head
            )
            key, value = (x.transpose(1, 2) for x in key_value.unbind(dim=2))
            if past_key_values is not None:
                past_key_values.cross_attention_cache.update(key, value, layer_idx)
                past_key_values.is_updated[layer_idx] = True
        query = self.query(hidden_states).reshape(batch_size, n_tokens, self.n_heads, self.d_head)
        context = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key,
            value,
            attn_mask=memory_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(batch_size, n_tokens, -1)
        return self.proj_dropout(self.out(context))


class DecoderBlock(nn.Module):
    def __init__(self, config: IonaDenovoConfig):
        super().__init__()
        hidden_size = config.decoder_hidden_size
        eps = config.encoder.layer_norm_eps
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.self_attn = DecoderSelfAttention(config)
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.cross_attn = DecoderCrossAttention(config)
        self.norm3 = nn.LayerNorm(hidden_size, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, config.decoder_intermediate_size),
            nn.GELU(),
            nn.Dropout(config.decoder_dropout),
            nn.Linear(config.decoder_intermediate_size, hidden_size),
            nn.Dropout(config.decoder_dropout),
        )

    def forward(
        self,
        hidden_states: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        past_key_values: EncoderDecoderCache | None,
        layer_idx: int,
    ) -> Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.norm1(hidden_states), past_key_values, layer_idx
        )
        hidden_states = hidden_states + self.cross_attn(
            self.norm2(hidden_states), memory, memory_mask, past_key_values, layer_idx
        )
        return hidden_states + self.ffn(self.norm3(hidden_states))


class PeptideDecoder(nn.Module):
    """Autoregressive transformer over peptide tokens, conditioned on the precursor and peaks.

    Position 0 is the precursor token; position t > 0 is the (t-1)th peptide token.
    """

    def __init__(self, config: IonaDenovoConfig):
        super().__init__()
        hidden_size = config.decoder_hidden_size
        eps = config.encoder.layer_norm_eps
        self.memory_projection = nn.Sequential(
            nn.Linear(config.encoder.hidden_size, hidden_size),
            nn.LayerNorm(hidden_size, eps=eps),
        )
        self.precursor = PrecursorEmbed(config)
        self.token_embedding = nn.Embedding(
            config.vocab_size, hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(config.max_peptide_length + 1, hidden_size)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.decoder_num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=eps)

    def embed_precursor(self, neutral_mass: Tensor, charge: Tensor) -> Tensor:
        position = self.position_embedding.weight[0]
        return (self.precursor(neutral_mass, charge) + position).unsqueeze(1)

    def embed_tokens(self, input_ids: Tensor, start_position: int) -> Tensor:
        positions = torch.arange(
            start_position, start_position + input_ids.shape[1], device=input_ids.device
        )
        return self.token_embedding(input_ids) + self.position_embedding(positions)

    def forward(
        self,
        hidden_states: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        past_key_values: EncoderDecoderCache | None = None,
    ) -> Tensor:
        """Decode `hidden_states` against projected peaks, updating `past_key_values` if given."""
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, memory, memory_mask, past_key_values, layer_idx)
        return self.norm(hidden_states)


class IonaForDenovo(IonaPreTrainedModel):
    """Iona encoder with an autoregressive peptide decoder for de novo sequencing."""

    config_class: type[PretrainedConfig] | None = IonaDenovoConfig

    def __init__(
        self,
        config: IonaDenovoConfig,
        encoder: IonaModel | None = None,
        freeze_encoder: bool = False,
    ):
        super().__init__(config)
        self.iona = encoder if encoder is not None else IonaModel(config.encoder)
        self.decoder = PeptideDecoder(config)
        self.lm_head = nn.Linear(config.decoder_hidden_size, config.vocab_size)
        self._encoder_is_frozen = False
        if encoder is None:
            self.post_init()
        else:
            self.decoder.apply(self._init_weights)
            self.lm_head.apply(self._init_weights)
        # The mask token is only used by masked-intensity pretraining.
        self.iona.embed.mask_token.requires_grad_(False)
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze the encoder and keep its stochastic layers disabled."""
        self._encoder_is_frozen = True
        self.iona.requires_grad_(False)
        self.iona.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_is_frozen:
            self.iona.eval()
        return self

    @staticmethod
    def neutral_mass(precursor_mz: Tensor, precursor_charge: Tensor) -> Tensor:
        return (precursor_mz - PROTON_MASS) * precursor_charge

    def _encode(
        self, mz: Tensor, log_intensity: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode peaks and project them to the decoder width, with their attention mask."""
        states = self.iona(
            mz=mz, log_intensity=log_intensity, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        memory_mask = attention_mask.bool()[:, None, None, :]
        return self.decoder.memory_projection(states), memory_mask

    def forward(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        precursor_mz: Tensor,
        precursor_charge: Tensor,
        labels: Tensor,
        attention_mask: Tensor,
    ) -> IonaForDenovoOutput:
        """Teacher-forced decoding: predict each label from the precursor and earlier labels."""
        shifted = labels[:, :-1]
        decoder_input_ids = shifted.masked_fill(shifted == -100, self.config.pad_token_id)

        memory, memory_mask = self._encode(mz, log_intensity, attention_mask)
        neutral_mass = self.neutral_mass(precursor_mz, precursor_charge)
        hidden_states = torch.cat(
            [
                self.decoder.embed_precursor(neutral_mass, precursor_charge),
                self.decoder.embed_tokens(decoder_input_ids, start_position=1),
            ],
            dim=1,
        )
        hidden_states = self.decoder(hidden_states, memory, memory_mask)
        logits = self.lm_head(hidden_states).float()
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        return IonaForDenovoOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        precursor_mz: Tensor,
        precursor_charge: Tensor,
        tokenizer,
        attention_mask: Tensor,
        precursor_tolerance_ppm: float = 50.0,
    ) -> IonaDenovoGenerationOutput:
        """Greedy decoding, as in Casanovo: take the most likely token until every peptide ends.

        Precursor mass does not constrain decoding; `precursor_match` only reports whether each
        peptide fits the precursor within tolerance, allowing a 0 or 1 isotope error.
        """
        config = self.config
        max_length = config.max_peptide_length
        device, eos, pad = mz.device, config.eos_token_id, config.pad_token_id
        # Special tokens never appear mid-peptide, and N-terminal mods only come first.
        blocked = torch.zeros(config.vocab_size, dtype=torch.bool, device=device)
        blocked[tokenizer.all_special_ids] = True
        n_terminal = torch.zeros_like(blocked)
        n_terminal[tokenizer.n_terminal_token_ids] = True
        later = blocked | n_terminal
        later[eos] = False

        memory, memory_mask = self._encode(mz, log_intensity, attention_mask)
        precursor_mass = self.neutral_mass(precursor_mz, precursor_charge)
        hidden_states = self.decoder.embed_precursor(precursor_mass, precursor_charge)
        cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        tokens, scores = [], []
        done = torch.zeros(mz.shape[0], dtype=torch.bool, device=device)
        for step in range(max_length + 1):
            hidden_states = self.decoder(hidden_states, memory, memory_mask, cache)
            logits = self.lm_head(hidden_states[:, -1])
            logits = logits.float().masked_fill(later if step else blocked, float("-inf"))
            token = logits.argmax(dim=-1)
            if step == max_length:
                token.fill_(eos)
            token = token.masked_fill(done, pad)
            score = logits.softmax(dim=-1).gather(1, token[:, None]).squeeze(1)
            tokens.append(token)
            scores.append(score.masked_fill(done, 0.0))
            done |= token == eos
            if done.all():
                break
            hidden_states = self.decoder.embed_tokens(token[:, None], start_position=step + 1)

        sequences = torch.stack(tokens, dim=1)
        token_masses = torch.tensor(tokenizer.token_masses, dtype=torch.float32, device=device)
        peptide_mass = token_masses[sequences].sum(dim=1) + WATER_MASS
        shifts = torch.tensor([0.0, ISOTOPES["¹³C"]], device=device)
        error = (peptide_mass[:, None] + shifts - precursor_mass[:, None]).abs()
        tolerance = precursor_mass * precursor_tolerance_ppm * 1e-6
        return IonaDenovoGenerationOutput(
            sequences=sequences,
            token_scores=torch.stack(scores, dim=1),
            precursor_match=(error <= tolerance[:, None]).any(dim=-1),
        )


IonaModel.register_for_auto_class("AutoModel")
IonaForPreTraining.register_for_auto_class("AutoModelForPreTraining")
IonaForDenoising.register_for_auto_class("AutoModelForTokenClassification")
IonaForRetrieval.register_for_auto_class("AutoModel")
IonaForDenovo.register_for_auto_class("AutoModel")
