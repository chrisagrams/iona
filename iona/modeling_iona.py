"""Hugging Face-native Iona model implementations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import SupConLoss
from torch import Tensor, nn
from torch.distributed.nn.functional import all_gather as distributed_all_gather
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils.generic import ModelOutput

from .configuration_iona import (
    POOLING_MODES,
    IonaConfig,
    IonaDenoisingConfig,
    IonaPeptideConfig,
    IonaRetrievalConfig,
)
from .fourier import FourierFeatures

# 20 standard residues plus `n`, which this corpus uses as an N-terminal marker.
RESIDUES = "ACDEFGHIKLMNPQRSTVWYn"
PAD, UNK = 0, 1
RESIDUE_TO_ID = {residue: index + 2 for index, residue in enumerate(RESIDUES)}
VOCAB_SIZE = len(RESIDUE_TO_ID) + 2


def pool_sequence(tokens: Tensor, mask: Tensor, mode: str = "mean+max") -> Tensor:
    """Reduce variable-length token embeddings to one vector. Both towers must use the same mode."""
    if mode not in POOLING_MODES:
        raise ValueError(f"pooling must be one of {POOLING_MODES}, got {mode!r}")
    mask = mask.bool().unsqueeze(-1)
    mean = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
    if mode == "mean":
        return mean
    maximum = torch.nan_to_num(tokens.masked_fill(~mask, float("-inf")).max(1).values,
                               neginf=0.0)
    return torch.cat([mean, maximum], dim=-1)


def pooled_width(hidden_size: int, mode: str) -> int:
    """Width `pool_sequence` produces, so the projection can be sized without a forward."""
    return 2 * hidden_size if mode == "mean+max" else hidden_size


def head_kl(logits: Tensor, reference_logits: Tensor, attention_mask: Tensor) -> Tensor:
    """KL(reference || current) over each spectrum's distribution across its real peaks."""
    valid = attention_mask.bool()
    current = logits.float().masked_fill(~valid, float("-inf")).log_softmax(dim=-1)
    reference = reference_logits.float().masked_fill(~valid, float("-inf")).log_softmax(dim=-1)
    # Termwise rather than F.kl_div, which gives NaN at the -inf padded positions.
    terms = reference.exp() * (reference - current)
    return torch.where(valid, terms, torch.zeros_like(terms)).sum(-1).mean()


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
    contrastive: Tensor | None = None
    kl: Tensor | None = None


@dataclass
class IonaPeptideEncoderOutput(ModelOutput):
    """Unit-length peptide embeddings."""

    embeddings: Tensor | None = None


@dataclass
class IonaPeptideForAlignmentOutput(ModelOutput):
    """Output of aligning peptide embeddings onto spectrum-encoder targets."""

    loss: Tensor | None = None
    embeddings: Tensor | None = None
    target: Tensor | None = None


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
        bias: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        attention_output = self.attn(self.norm1(hidden_states), bias, padding_mask)
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
    """Project pooled peak tokens into the retrieval space."""

    def __init__(self, config: IonaRetrievalConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(pooled_width(config.encoder.hidden_size, config.pooling),
                      config.projection_hidden_size),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.projection_hidden_size, config.embedding_size),
        )

    def forward(self, pooled: Tensor) -> Tensor:
        return self.projection(pooled)


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
    """Iona encoder trained with supervised contrastive loss for spectrum retrieval.

    Covers both the frozen-encoder probe (projection head, encoder frozen) and encoder
    fine-tuning (no head, embedding is the pooled encoder output). With `kl_weight > 0`,
    a KL term to a frozen reference model's intensity head keeps the encoder's peak
    predictions; the reference is never saved.
    """

    config_class: type[PretrainedConfig] | None = IonaRetrievalConfig

    def __init__(
        self,
        config: IonaRetrievalConfig,
        encoder: IonaModel | None = None,
        freeze_encoder: bool = False,
        intensity_head: IntensityHead | None = None,
        reference: IonaForPreTraining | None = None,
    ):
        super().__init__(config)
        if config.kl_weight <= 0 and (intensity_head is not None or reference is not None):
            raise ValueError("intensity_head and reference are only used when kl_weight > 0")
        self.iona = encoder if encoder is not None else IonaModel(config.encoder)
        self.retrieval_head = SpectrumRetrievalHead(config) if config.projection_head else None
        new_intensity_head = config.kl_weight > 0 and intensity_head is None
        if config.kl_weight > 0:
            self.intensity_head = (intensity_head if intensity_head is not None
                                   else IntensityHead(config.encoder.hidden_size))
        self.contrastive_loss = SupConLoss(temperature=config.temperature)
        self._encoder_is_frozen = False
        if encoder is None:
            self.post_init()
        else:
            if self.retrieval_head is not None:
                self.retrieval_head.apply(self._init_weights)
            if new_intensity_head:
                self.intensity_head.apply(self._init_weights)
        self.reference = reference
        if reference is not None:
            # Frozen and in eval mode so the KL target does not move.
            reference.requires_grad_(False)
            reference.eval()
            self._keys_to_ignore_on_save = {f"reference.{k}" for k in reference.state_dict()}
        if freeze_encoder:
            self.freeze_encoder()

    @classmethod
    def from_pretraining(
        cls,
        pretrained: IonaForPreTraining,
        config: IonaRetrievalConfig,
        reference: IonaForPreTraining | None = None,
        freeze_encoder: bool = False,
    ) -> IonaForRetrieval:
        """Wrap a pretrained model, sharing its encoder (and intensity head, for the KL term)."""
        return cls(
            config,
            encoder=pretrained.iona,
            freeze_encoder=freeze_encoder,
            intensity_head=pretrained.intensity_head if config.kl_weight > 0 else None,
            reference=reference,
        )

    def freeze_encoder(self) -> None:
        """Freeze the encoder and keep its stochastic layers disabled."""
        self._encoder_is_frozen = True
        self.iona.requires_grad_(False)
        self.iona.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_is_frozen:
            self.iona.eval()
        if self.reference is not None:
            self.reference.eval()
        return self

    def _encode(self, mz: Tensor, log_intensity: Tensor, attention_mask: Tensor) -> Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self._encoder_is_frozen):
            return self.iona(
                mz=mz, log_intensity=log_intensity, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state

    def _pool(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        embeddings = pool_sequence(hidden, attention_mask, self.config.pooling)
        if self.retrieval_head is not None:
            embeddings = self.retrieval_head(embeddings)
        return F.normalize(embeddings.float(), dim=-1)

    def embed(
        self, mz: Tensor, log_intensity: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Return unit-length spectrum embeddings."""
        if attention_mask is None:
            attention_mask = torch.ones_like(mz, dtype=torch.long)
        return self._pool(self._encode(mz, log_intensity, attention_mask), attention_mask)

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
        hidden = self._encode(mz, log_intensity, attention_mask)
        embeddings = self._pool(hidden, attention_mask)
        if group_ids is None:
            return IonaForRetrievalOutput(embeddings=embeddings) if return_dict else (embeddings,)

        if group_ids.ndim != 1 or group_ids.shape[0] != embeddings.shape[0]:
            raise ValueError("group_ids must contain one value per spectrum")
        loss_embeddings = embeddings
        labels = group_ids.long()
        if (self.config.gather_across_ranks and torch.distributed.is_available()
                and torch.distributed.is_initialized()):
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            # Each rank's collator assigns its own batch-local group IDs.
            labels = labels * world_size + rank
            # Gather with autograd so remote candidates receive gradients.
            loss_embeddings = torch.cat(distributed_all_gather(embeddings), dim=0)
            gathered_labels = [torch.empty_like(labels) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_labels, labels)
            labels = torch.cat(gathered_labels)
        contrastive = self.contrastive_loss(loss_embeddings, labels)
        kl = embeddings.new_zeros(())
        if self.config.kl_weight > 0:
            if self.reference is None:
                raise ValueError("kl_weight > 0 needs a reference model")
            with torch.no_grad():
                reference_hidden = self.reference.iona(
                    mz=mz, log_intensity=log_intensity, attention_mask=attention_mask
                ).last_hidden_state
                reference_logits = self.reference.intensity_head(reference_hidden)
            kl = head_kl(self.intensity_head(hidden), reference_logits, attention_mask)
        loss = contrastive + self.config.kl_weight * kl

        if not return_dict:
            return (loss, embeddings, contrastive.detach(), kl.detach())
        return IonaForRetrievalOutput(
            loss=loss, embeddings=embeddings, contrastive=contrastive.detach(), kl=kl.detach()
        )


class IonaPeptidePreTrainedModel(PreTrainedModel):
    """Shared Hugging Face behavior for the peptide encoder classes."""

    config_class: type[PretrainedConfig] | None = IonaPeptideConfig
    base_model_prefix = "sequence_encoder"
    main_input_name = "residues"

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        # PyTorch's default initialization, which the peptide encoder has always trained from.
        if isinstance(module, nn.MultiheadAttention):
            module._reset_parameters()
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()  # ty: ignore[call-non-callable]


class IonaPeptideEncoder(IonaPeptidePreTrainedModel):
    """Encode a modified peptide plus its charge into a fixed-size unit embedding."""

    def __init__(self, config: IonaPeptideConfig):
        super().__init__(config)
        hidden_size = config.hidden_size
        self.residue = nn.Embedding(VOCAB_SIZE, hidden_size, padding_idx=PAD)
        self.position = nn.Embedding(config.max_position_embeddings, hidden_size)
        self.charge = nn.Embedding(config.n_charges, hidden_size)
        # 1e-2..1e3 spans a whole modification down to fine isotopic structure.
        self.mod_features = FourierFeatures(config.mod_n_freqs, 1e-2, 1e3)
        self.mod_projection = nn.Linear(self.mod_features.out_dim, hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=config.num_attention_heads,
            dim_feedforward=4 * hidden_size, dropout=config.dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_hidden_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        width = pooled_width(hidden_size, config.pooling)
        self.projection = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(width, config.embedding_size),
        )
        self.post_init()

    def forward(
        self,
        residues: Tensor,
        modifications: Tensor,
        sequence_mask: Tensor,
        charge: Tensor,
        return_dict: bool | None = None,
    ) -> IonaPeptideEncoderOutput | tuple[Tensor, ...]:
        if return_dict is None:
            return_dict = self.config.return_dict
        length = residues.shape[1]
        position = torch.arange(length, device=residues.device).clamp_max(
            self.position.num_embeddings - 1
        )
        hidden = self.residue(residues) + self.position(position)[None]
        # Only modified residues get a mass contribution.
        modified = (modifications.abs() > 1e-6).unsqueeze(-1)
        mod = self.mod_projection(self.mod_features(modifications).to(hidden.dtype))
        hidden = hidden + mod * modified
        hidden = hidden + self.charge(charge.clamp(0, self.charge.num_embeddings - 1))[:, None]
        # Autocast off, activations cast to the weights' dtype: the fused eval fast path
        # ignores autocast on XPU and fails on a dtype mismatch.
        param_dtype = next(self.encoder.parameters()).dtype
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.to(param_dtype)
            hidden = self.norm(self.encoder(hidden, src_key_padding_mask=~sequence_mask.bool()))
        pooled = pool_sequence(hidden, sequence_mask, self.config.pooling)
        embeddings = F.normalize(self.projection(pooled).float(), dim=-1)

        if not return_dict:
            return (embeddings,)
        return IonaPeptideEncoderOutput(embeddings=embeddings)


class IonaPeptideForAlignment(IonaPeptidePreTrainedModel):
    """Peptide encoder trained onto precomputed teacher targets with L2 on unit vectors."""

    def __init__(self, config: IonaPeptideConfig):
        super().__init__(config)
        self.sequence_encoder = IonaPeptideEncoder(config)
        self.post_init()

    def forward(
        self,
        residues: Tensor,
        modifications: Tensor,
        sequence_mask: Tensor,
        charge: Tensor,
        target: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> IonaPeptideForAlignmentOutput | tuple[Tensor, ...]:
        if return_dict is None:
            return_dict = self.config.return_dict
        predicted = self.sequence_encoder(
            residues, modifications, sequence_mask, charge, return_dict=True
        ).embeddings
        loss = None
        if target is not None:
            target = F.normalize(target.float(), dim=-1)
            # Mean squared L2; equals 2 - 2cos on unit vectors.
            loss = ((predicted - target) ** 2).sum(dim=-1).mean()

        if not return_dict:
            result = (predicted,) if target is None else (predicted, target)
            return ((loss,) + result) if loss is not None else result
        return IonaPeptideForAlignmentOutput(loss=loss, embeddings=predicted, target=target)


IonaModel.register_for_auto_class("AutoModel")
IonaForPreTraining.register_for_auto_class("AutoModelForPreTraining")
IonaForDenoising.register_for_auto_class("AutoModelForTokenClassification")
IonaForRetrieval.register_for_auto_class("AutoModel")
IonaPeptideEncoder.register_for_auto_class("AutoModel")
