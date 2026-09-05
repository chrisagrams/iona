"""Pair stream layered ON TOP of the original DeltaMZBias, rather than replacing it.

`pair_stream.py` swaps `DeltaMZBias` out entirely: its A independent per-head MLPs
(each `2*n_freqs -> per_head_hidden -> 1`) are replaced by a single shared
`2*n_freqs -> pair_channels` projection that every head reads linearly. That is a real
capacity change as well as an architecture change, so a comparison against the baseline
confounds the two.

This variant keeps the original module untouched and adds a per-layer correction:

    base       = DeltaMZBias(delta)                     # (B, K, K, heads), layer-independent
    pair_0     = Linear(Fourier(delta))                 # (B, K, K, pair_channels)
    for layer l:
        pair   = pair + MLP_l(LayerNorm_l(pair))        # residual refinement
        bias_l = base + to_heads_l(pair)                # baseline PLUS a per-layer delta

`to_heads` is zero-initialised (`pair_zero_init`), so at step 0 the model is **numerically
identical to the baseline** and the stream can only add signal from there. That makes the
comparison a clean ablation of "does a per-layer correction help?" rather than a
comparison between two differently-shaped bias modules.

The Fourier featurizer is shared with the baseline module -- computed once, used for both
the base curve and the pair state -- so `encoder.bias_module.ff` still resolves for
`FourierProbeCallback`, and `base` is evaluated once rather than per layer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput

from msdelta.model.configuration import MSDeltaConfig
from msdelta.model.modeling import (
    DeltaMZBias,
    EncoderBlock,
    IntensityHead,
    MSDeltaForPreTraining,
    MSDeltaPreTrainedModel,
    PeakEmbed,
)


class MSDeltaPairStreamAdditiveConfig(MSDeltaConfig):
    """MSDelta configuration plus the additive pair-stream settings."""

    model_type = "msdelta-pair-stream-additive"

    def __init__(
        self,
        pair_channels: int = 16,
        pair_hidden: int = 64,
        pair_bias_scale: float | None = None,
        pair_zero_init: bool = True,
        **kwargs,
    ):
        # Set before super().__init__, which calls _validate().
        self.pair_channels = pair_channels
        self.pair_hidden = pair_hidden
        self.pair_bias_scale = pair_bias_scale
        self.pair_zero_init = pair_zero_init
        super().__init__(**kwargs)

    def _validate(self) -> None:
        super()._validate()
        if self.pair_channels <= 0 or self.pair_hidden <= 0:
            raise ValueError("pair_channels and pair_hidden must be positive")
        if self.pair_bias_scale is not None and self.pair_bias_scale <= 0:
            raise ValueError("pair_bias_scale must be positive when set")


class AdditivePairStream(nn.Module):
    """The baseline per-head bias, plus a residual pair state refined at every layer."""

    def __init__(self, config: MSDeltaPairStreamAdditiveConfig):
        super().__init__()
        channels = config.pair_channels
        hidden = config.pair_hidden
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        self.scale = config.pair_bias_scale

        # The original module, unchanged: A independent per-head MLPs over Fourier(delta).
        self.base = DeltaMZBias(config)

        self.proj_in = nn.Linear(self.base.ff.out_dim, channels)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(channels, eps=config.layer_norm_eps) for _ in range(self.n_layers)]
        )
        self.updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(channels, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, channels),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.to_heads = nn.ModuleList(
            [nn.Linear(channels, self.n_heads) for _ in range(self.n_layers)]
        )

    @property
    def ff(self):
        """Expose the shared Fourier featurizer where the diagnostics look for it."""
        return self.base.ff

    def zero_init_readout(self) -> None:
        """Make the stream contribute exactly zero, so the model starts as the baseline."""
        for head in self.to_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def init_state(self, delta_mz: Tensor) -> tuple[Tensor, Tensor]:
        """Return the initial pair state and the layer-independent baseline curve."""
        feats = self.base.ff(delta_mz)
        pair = self.proj_in(feats.to(self.proj_in.weight.dtype))
        base_curve = self.base._curve(feats)  # noqa: SLF001 - same package, computed once
        return pair, base_curve

    def step(self, pair: Tensor, layer_index: int, base_curve: Tensor) -> tuple[Tensor, Tensor]:
        """Refine the pair state and return ``base + per-layer correction``."""
        pair = pair + self.updates[layer_index](self.norms[layer_index](pair))
        head_bias = base_curve + self.to_heads[layer_index](pair)
        if self.scale is not None:
            head_bias = self.scale * torch.tanh(head_bias / self.scale)
        return pair, head_bias

    def evaluate_base(self, delta_mz_grid: Tensor) -> Tensor:
        """The baseline curves alone -> ``(grid, heads)``."""
        return self.base.evaluate(delta_mz_grid)

    def evaluate_layers(self, delta_mz_grid: Tensor) -> Tensor:
        """Total per-layer bias curves -> ``(layers, grid, heads)``."""
        pair, base_curve = self.init_state(delta_mz_grid)
        curves = []
        for layer_index in range(self.n_layers):
            pair, head_bias = self.step(pair, layer_index, base_curve)
            curves.append(head_bias)
        return torch.stack(curves).float()

    def evaluate(self, delta_mz_grid: Tensor) -> Tensor:
        """Final-layer total curves -> ``(grid, heads)``; drop-in for ``DeltaMZBias``."""
        return self.evaluate_layers(delta_mz_grid)[-1]


class MSDeltaPairStreamAdditiveModel(MSDeltaPreTrainedModel):
    """Encoder whose per-layer bias is the baseline curve plus a learned correction."""

    config_class = MSDeltaPairStreamAdditiveConfig

    def __init__(self, config: MSDeltaPairStreamAdditiveConfig):
        super().__init__(config)
        self.embed = PeakEmbed(config)
        # Named `bias_module` so the existing diagnostics keep finding it.
        self.bias_module = AdditivePairStream(config)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()
        if config.pair_zero_init:
            # After post_init, which would otherwise give the readout a normal init.
            self.bias_module.zero_init_readout()

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
        pair, base_curve = self.bias_module.init_state(mz.unsqueeze(-1) - mz.unsqueeze(-2))

        checkpointing = self.gradient_checkpointing and self.training
        for layer_index, block in enumerate(self.blocks):
            if checkpointing:
                pair, head_bias = self._gradient_checkpointing_func(
                    self.bias_module.step, pair, layer_index, base_curve
                )
            else:
                pair, head_bias = self.bias_module.step(pair, layer_index, base_curve)
            bias = head_bias.permute(0, 3, 1, 2).contiguous()
            if checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block.__call__, hidden_states, bias, padding_mask
                )
            else:
                hidden_states = block(hidden_states, bias, padding_mask)
        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return (hidden_states,)
        return BaseModelOutput(last_hidden_state=hidden_states)


class MSDeltaPairStreamAdditiveForPreTraining(MSDeltaForPreTraining):
    """Masked-intensity pretraining on the additive pair-stream encoder.

    Inherits ``MSDeltaForPreTraining.forward`` unchanged -- only the encoder differs -- so
    the loss stays defined in one place. ``MSDeltaPreTrainedModel.__init__`` is called
    directly to skip building the baseline encoder that would immediately be discarded.
    """

    config_class = MSDeltaPairStreamAdditiveConfig

    def __init__(self, config: MSDeltaPairStreamAdditiveConfig):
        MSDeltaPreTrainedModel.__init__(self, config)
        self.msdelta = MSDeltaPairStreamAdditiveModel(config)
        self.intensity_head = IntensityHead(config.hidden_size)
        self.post_init()
        if config.pair_zero_init:
            self.msdelta.bias_module.zero_init_readout()
