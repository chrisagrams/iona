"""Configuration for MSDelta models."""

from __future__ import annotations

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
        fourier_int_n_freqs: int = 16,
        fourier_int_f_min: float = 1e-2,
        fourier_int_f_max: float = 1e2,
        fourier_int_learnable: bool = True,
        delta_bias_n_freqs: int = 64,
        delta_bias_per_head_hidden: int = 32,
        delta_bias_f_min: float = 1e-2,
        delta_bias_f_max: float = 1e3,
        delta_bias_learnable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not getattr(self, "auto_map", None):
            self.auto_map = {
                "AutoConfig": "configuration_msdelta.MSDeltaConfig",
                "AutoModel": "modeling_msdelta.MSDeltaModel",
                "AutoModelForPreTraining": "modeling_msdelta.MSDeltaForPreTraining",
            }
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.fourier_int_n_freqs = fourier_int_n_freqs
        self.fourier_int_f_min = fourier_int_f_min
        self.fourier_int_f_max = fourier_int_f_max
        self.fourier_int_learnable = fourier_int_learnable
        self.delta_bias_n_freqs = delta_bias_n_freqs
        self.delta_bias_per_head_hidden = delta_bias_per_head_hidden
        self.delta_bias_f_min = delta_bias_f_min
        self.delta_bias_f_max = delta_bias_f_max
        self.delta_bias_learnable = delta_bias_learnable
        self._validate()

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
        for name in ("fourier_int_n_freqs", "delta_bias_n_freqs", "delta_bias_per_head_hidden"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for prefix in ("fourier_int", "delta_bias"):
            lo = getattr(self, f"{prefix}_f_min")
            hi = getattr(self, f"{prefix}_f_max")
            if not 0 < lo < hi:
                raise ValueError(f"{prefix}_f_min and {prefix}_f_max must satisfy 0 < min < max")


MSDeltaConfig.register_for_auto_class()
