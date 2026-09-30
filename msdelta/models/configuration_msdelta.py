"""Configuration for MSDelta models.

``MSDeltaConfig.architecture`` selects the spectrum encoder: ``"transformer"`` (the default,
and what every checkpoint written before the field existed loads as) or ``"pairformer"``
(the AlphaFold-3-style single + pair encoder in ``pairformer.py``). The ``pair_*`` fields are
read only by the Pairformer. A transformer config ignores them and leaves them (and
``architecture`` itself) out of ``to_dict()``, so its ``config.json``, its ``repr`` and every
run manifest built from ``to_dict()`` are byte-for-byte what they were before these fields
existed; a config file without them loads as the transformer.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class MSDeltaConfig(PretrainedConfig):
    """Store the architecture settings required to construct an MSDelta model."""

    model_type = "msdelta"

    def __init__(
        self,
        hidden_size: int = 256,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 6,
        intermediate_size: int = 1024,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        delta_bias_n_freqs: int = 256,
        delta_bias_per_head_hidden: int = 32,
        delta_bias_f_min: float = 1e-3,
        delta_bias_f_max: float = 190.0,
        architecture: str = "transformer",
        # Pairformer settings (ignored when architecture == "transformer"). The defaults are
        # deliberately small -- test-sized, not an experiment choice. Pair tensors are
        # O(B * peaks^2 * pair_channels); see the pairformer module docstring for the cost.
        pair_channels: int = 16,
        pair_transition_expansion: int = 2,
        pair_tri_channels: int = 16,
        pair_update: str = "triangle",
        pair_use_triangle_attention: bool = False,
        pair_tri_attn_heads: int = 2,
        pair_tri_attn_dim: int = 8,
        pair_tri_attn_chunk: int = 32,
        pair_tri_attn_impl: str = "sdpa_view",  # "naive" | "sdpa" (K102, K115) | "sdpa_view" (K117, copy-free mask; default since 2026-09-30: B=32 train step 1.01-1.06x, up to 1.55x less memory, jobs 8879977/8880031)
        pair_tri_attn_checkpoint_chunks: bool = False,  # K102
        pair_use_writeback: bool = True,
        pair_opm_channels: int = 8,
        # K152-P write-back form: "outer" = Linear(a_i (x) b_j) (the original), "pointwise" =
        # Linear(a_i * b_j) (width pair_opm_channels; cheaper, kept as a speed option).
        pair_writeback: str = "outer",
        # "factored" = the outer product computed without materialising the c_o^2 tensor (same
        # math and parameters, K152-P(a)); "materialize" = the original einsum (reference).
        pair_writeback_impl: str = "factored",
        # K151-P: which triangle multiplications run: "both" (original), "outgoing" or "incoming".
        pair_tri_mul: str = "both",
        pair_single_use_mz: bool = True,
        pair_use_intensity: bool = True,
        pair_use_mass_defect: bool = False,  # dropped by default (user, K91/K113, 2026-09-28)
        pair_mass_defect_n_freqs: int = 16,
        pair_use_loss_bank: bool = True,
        pair_loss_bank_sigma_ppm: float = 20.0,
        pair_use_isotope: bool = True,
        pair_dropout: float = 0.0,
        pair_bias_scale: float | None = None,
        # K114-P decoupled streams (defaults = the original model, one pair update per layer).
        # See the pairformer module docstring, "Decoupled streams".
        pair_update_every: int = 1,
        pair_bias_lag: int = 0,
        # K172-P: run pair update m on a side stream concurrently with round m's single blocks
        # (needs pair_bias_lag=1; same numerics as the sequential lag-1 loop).
        pair_concurrent: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.delta_bias_n_freqs = delta_bias_n_freqs
        self.delta_bias_per_head_hidden = delta_bias_per_head_hidden
        self.delta_bias_f_min = delta_bias_f_min
        self.delta_bias_f_max = delta_bias_f_max
        self.architecture = architecture
        self.pair_channels = pair_channels
        self.pair_transition_expansion = pair_transition_expansion
        self.pair_tri_channels = pair_tri_channels
        self.pair_update = pair_update
        self.pair_use_triangle_attention = pair_use_triangle_attention
        self.pair_tri_attn_heads = pair_tri_attn_heads
        self.pair_tri_attn_dim = pair_tri_attn_dim
        self.pair_tri_attn_chunk = pair_tri_attn_chunk
        self.pair_tri_attn_impl = pair_tri_attn_impl
        self.pair_tri_attn_checkpoint_chunks = pair_tri_attn_checkpoint_chunks
        self.pair_use_writeback = pair_use_writeback
        self.pair_opm_channels = pair_opm_channels
        self.pair_writeback = pair_writeback
        self.pair_writeback_impl = pair_writeback_impl
        self.pair_tri_mul = pair_tri_mul
        self.pair_single_use_mz = pair_single_use_mz
        self.pair_use_intensity = pair_use_intensity
        self.pair_use_mass_defect = pair_use_mass_defect
        self.pair_mass_defect_n_freqs = pair_mass_defect_n_freqs
        self.pair_use_loss_bank = pair_use_loss_bank
        self.pair_loss_bank_sigma_ppm = pair_loss_bank_sigma_ppm
        self.pair_use_isotope = pair_use_isotope
        self.pair_dropout = pair_dropout
        self.pair_bias_scale = pair_bias_scale
        self.pair_update_every = pair_update_every
        self.pair_bias_lag = pair_bias_lag
        self.pair_concurrent = pair_concurrent
        self._validate()

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        if output.get("architecture", "transformer") == "transformer":
            output.pop("architecture", None)
            for key in [k for k in output if k.startswith("pair_")]:
                del output[key]
        return output

    def _validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_hidden_layers <= 0 or self.intermediate_size <= 0:
            raise ValueError("num_hidden_layers and intermediate_size must be positive")
        if not 0.0 <= self.hidden_dropout_prob < 1.0:
            raise ValueError("hidden_dropout_prob must be in [0, 1)")
        if not 0.0 <= self.attention_probs_dropout_prob < 1.0:
            raise ValueError("attention_probs_dropout_prob must be in [0, 1)")
        for name in ("delta_bias_n_freqs", "delta_bias_per_head_hidden"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.delta_bias_f_min < self.delta_bias_f_max:
            raise ValueError("delta_bias_f_min and delta_bias_f_max must satisfy 0 < min < max")
        if self.architecture not in ("transformer", "pairformer"):
            raise ValueError("architecture must be 'transformer' or 'pairformer'")
        if self.architecture == "pairformer":
            self._validate_pairformer()

    def _validate_pairformer(self) -> None:
        for name in ("pair_channels", "pair_transition_expansion", "pair_tri_channels",
                     "pair_mass_defect_n_freqs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pair_update not in ("static", "transition", "triangle"):
            raise ValueError("pair_update must be 'static', 'transition' or 'triangle'")
        if self.pair_use_triangle_attention:
            if self.pair_update != "triangle":
                raise ValueError("pair_use_triangle_attention needs pair_update='triangle'")
            for name in ("pair_tri_attn_heads", "pair_tri_attn_dim", "pair_tri_attn_chunk"):
                if getattr(self, name) <= 0:
                    raise ValueError(f"{name} must be positive")
            if self.pair_tri_attn_impl not in ("naive", "sdpa", "sdpa_view"):
                raise ValueError("pair_tri_attn_impl must be 'naive', 'sdpa' or 'sdpa_view'")
        if self.pair_use_writeback and self.pair_opm_channels <= 0:
            raise ValueError("pair_opm_channels must be positive when pair_use_writeback is set")
        if self.pair_writeback not in ("outer", "pointwise"):
            raise ValueError("pair_writeback must be 'outer' or 'pointwise'")
        if self.pair_writeback_impl not in ("factored", "materialize"):
            raise ValueError("pair_writeback_impl must be 'factored' or 'materialize'")
        if self.pair_tri_mul not in ("both", "outgoing", "incoming"):
            raise ValueError("pair_tri_mul must be 'both', 'outgoing' or 'incoming'")
        if self.pair_loss_bank_sigma_ppm <= 0:
            raise ValueError("pair_loss_bank_sigma_ppm must be positive")
        if not 0.0 <= self.pair_dropout < 1.0:
            raise ValueError("pair_dropout must be in [0, 1)")
        if self.pair_bias_scale is not None and self.pair_bias_scale <= 0:
            raise ValueError("pair_bias_scale must be positive when set")
        k, lag = self.pair_update_every, self.pair_bias_lag
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= self.num_hidden_layers:
            raise ValueError("pair_update_every must be an int in [1, num_hidden_layers]")
        if isinstance(lag, bool) or lag not in (0, 1):
            raise ValueError("pair_bias_lag must be 0 or 1")
        if lag == 1 and k >= self.num_hidden_layers:
            # One round only: the single update would never be read, so nothing refines z.
            raise ValueError("pair_bias_lag=1 needs pair_update_every < num_hidden_layers "
                             "(at least two rounds), otherwise no pair update is ever read")
        if not isinstance(self.pair_concurrent, bool):
            raise ValueError("pair_concurrent must be a bool")
        if self.pair_concurrent and lag != 1:
            # At lag 0 the round's single blocks read the update's output: nothing to overlap.
            raise ValueError("pair_concurrent needs pair_bias_lag=1")


class MSDeltaDenoisingConfig(PretrainedConfig):
    """Compose an MSDelta encoder configuration with a denoising head."""

    model_type = "msdelta-denoising"
    sub_configs = {"encoder": MSDeltaConfig}

    def __init__(
        self,
        encoder: MSDeltaConfig | dict | None = None,
        head_hidden_size: int = 128,
        head_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = (
            encoder if isinstance(encoder, MSDeltaConfig) else MSDeltaConfig(**(encoder or {}))
        )
        self.head_hidden_size = head_hidden_size
        self.head_dropout = head_dropout

    @property
    def initializer_range(self) -> float:
        return self.encoder.initializer_range


class MSDeltaRetrievalConfig(PretrainedConfig):
    """Compose an MSDelta encoder configuration with a retrieval head."""

    model_type = "msdelta-retrieval"
    sub_configs = {"encoder": MSDeltaConfig}

    def __init__(
        self,
        encoder: MSDeltaConfig | dict | None = None,
        projection_hidden_size: int = 512,
        embedding_size: int = 256,
        head_dropout: float = 0.1,
        temperature: float = 0.07,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = (
            encoder if isinstance(encoder, MSDeltaConfig) else MSDeltaConfig(**(encoder or {}))
        )
        self.projection_hidden_size = projection_hidden_size
        self.embedding_size = embedding_size
        self.head_dropout = head_dropout
        self.temperature = temperature
        if projection_hidden_size <= 0 or embedding_size <= 0:
            raise ValueError("retrieval projection dimensions must be positive")
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1)")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

    @property
    def initializer_range(self) -> float:
        return self.encoder.initializer_range


MSDeltaConfig.register_for_auto_class()
MSDeltaDenoisingConfig.register_for_auto_class()
MSDeltaRetrievalConfig.register_for_auto_class()
