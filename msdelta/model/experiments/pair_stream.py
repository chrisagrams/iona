"""AlphaFold-style pair stream: a persistent per-pair state, updated at every layer.

The baseline `DeltaMZBias` computes one bias curve per head, once, and reuses that same
`(B, heads, K, K)` tensor in every encoder layer. Here the pair representation is instead
carried as state and refined layer by layer:

    delta = mz_i - mz_j              (B, K, K)
      -> FourierFeatures             (B, K, K, 2 * delta_bias_n_freqs)
      -> Linear                      (B, K, K, pair_channels)        <- the persistent state
    for each encoder layer l:
        pair  = pair + MLP_l(LayerNorm_l(pair))                      <- residual refinement
        bias  = to_heads_l(pair)     (B, K, K, heads) -> (B, heads, K, K)
        hidden_states = block_l(hidden_states, bias, ...)

Two properties worth keeping in mind:

* Every update is **pointwise** in the pair index -- `nn.Linear` only touches the last
  dimension -- so the bias remains a pure function of `delta`, just a *different* function
  per layer. `evaluate()` therefore still works, and `evaluate_layers()` exposes one curve
  set per layer. Interpretability (`eval/viz.py`, `eval/alignment.py`) is preserved.
* The Fourier featurization is kept. It costs `delta_bias_n_freqs` parameters and is what
  makes sharp structure at chemical mass offsets learnable at all; an MLP fed the raw
  scalar `delta` cannot represent it.

The pair state is the memory driver: `(B, K, K, pair_channels)` retained per layer, versus
a single `(B, heads, K, K)` tensor in the baseline. Keep `pair_channels` small and enable
`--gradient_checkpointing` for anything but a smoke test.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput

from msdelta.model.configuration import MSDeltaConfig
from msdelta.model.fourier import FourierFeatures
from msdelta.model.modeling import (
    EncoderBlock,
    IntensityHead,
    MSDeltaForPreTraining,
    MSDeltaPreTrainedModel,
    PeakEmbed,
)


class MSDeltaPairStreamConfig(MSDeltaConfig):
    """MSDelta configuration plus the pair-stream settings."""

    model_type = "msdelta-pair-stream"

    def __init__(
        self,
        pair_channels: int = 16,
        pair_hidden: int = 64,
        pair_bias_scale: float | None = None,
        **kwargs,
    ):
        # Set before super().__init__, which calls _validate().
        self.pair_channels = pair_channels
        self.pair_hidden = pair_hidden
        self.pair_bias_scale = pair_bias_scale
        super().__init__(**kwargs)

    def _validate(self) -> None:
        super()._validate()
        if self.pair_channels <= 0 or self.pair_hidden <= 0:
            raise ValueError("pair_channels and pair_hidden must be positive")
        if self.pair_bias_scale is not None and self.pair_bias_scale <= 0:
            raise ValueError("pair_bias_scale must be positive when set")


class PairStream(nn.Module):
    """Hold and refine a per-pair representation, emitting a per-head bias each layer."""

    def __init__(self, config: MSDeltaPairStreamConfig):
        super().__init__()
        channels = config.pair_channels
        hidden = config.pair_hidden
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        self.scale = config.pair_bias_scale

        self.ff = FourierFeatures(
            config.delta_bias_n_freqs,
            config.delta_bias_f_min,
            config.delta_bias_f_max,
            log_spaced=True,
            learnable=config.delta_bias_learnable,
        )
        self.proj_in = nn.Linear(self.ff.out_dim, channels)
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

    def init_state(self, delta_mz: Tensor) -> Tensor:
        """Featurize signed mass differences into the initial pair state ``(..., channels)``."""
        feats = self.ff(delta_mz)
        return self.proj_in(feats.to(self.proj_in.weight.dtype))

    def step(self, pair: Tensor, layer_index: int) -> tuple[Tensor, Tensor]:
        """Refine the pair state for one layer and read off its ``(..., heads)`` bias."""
        pair = pair + self.updates[layer_index](self.norms[layer_index](pair))
        head_bias = self.to_heads[layer_index](pair)
        if self.scale is not None:
            head_bias = self.scale * torch.tanh(head_bias / self.scale)
        return pair, head_bias

    def evaluate_layers(self, delta_mz_grid: Tensor) -> Tensor:
        """Evaluate every layer's per-head bias curve on a grid -> ``(layers, grid, heads)``."""
        pair = self.init_state(delta_mz_grid)
        curves = []
        for layer_index in range(self.n_layers):
            pair, head_bias = self.step(pair, layer_index)
            curves.append(head_bias)
        return torch.stack(curves).float()

    def evaluate(self, delta_mz_grid: Tensor) -> Tensor:
        """Evaluate the final layer's curves -> ``(grid, heads)``.

        Matches ``DeltaMZBias.evaluate`` so ``eval/viz.py`` and ``eval/alignment.py`` work
        unchanged. Use ``evaluate_layers`` to see the per-layer specialization.
        """
        return self.evaluate_layers(delta_mz_grid)[-1]


class MSDeltaPairStreamModel(MSDeltaPreTrainedModel):
    """Encode peaks with a pair representation refined at every layer."""

    config_class = MSDeltaPairStreamConfig

    def __init__(self, config: MSDeltaPairStreamConfig):
        super().__init__(config)
        self.embed = PeakEmbed(config)
        # Named `bias_module` so the existing diagnostics keep finding it.
        self.bias_module = PairStream(config)
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
        pair = self.bias_module.init_state(mz.unsqueeze(-1) - mz.unsqueeze(-2))

        checkpointing = self.gradient_checkpointing and self.training
        for layer_index, block in enumerate(self.blocks):
            if checkpointing:
                pair, head_bias = self._gradient_checkpointing_func(
                    self.bias_module.step, pair, layer_index
                )
            else:
                pair, head_bias = self.bias_module.step(pair, layer_index)
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


class MSDeltaPairStreamForPreTraining(MSDeltaForPreTraining):
    """Masked-intensity pretraining on the pair-stream encoder.

    Inherits ``MSDeltaForPreTraining.forward`` unchanged -- only the encoder differs -- so the
    loss stays defined in exactly one place. ``MSDeltaPreTrainedModel.__init__`` is called
    directly to skip building the baseline encoder that would immediately be discarded.
    """

    config_class = MSDeltaPairStreamConfig

    def __init__(self, config: MSDeltaPairStreamConfig):
        MSDeltaPreTrainedModel.__init__(self, config)
        self.msdelta = MSDeltaPairStreamModel(config)
        self.intensity_head = IntensityHead(config.hidden_size)
        self.post_init()
