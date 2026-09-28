"""Configuration for Iona models."""

from __future__ import annotations

from transformers import PretrainedConfig

POOLING_MODES = ("mean", "mean+max")


class IonaConfig(PretrainedConfig):
    """Store the architecture settings required to construct an Iona model."""

    model_type = "iona"

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
        for name in ("delta_bias_n_freqs", "delta_bias_per_head_hidden"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.delta_bias_f_min < self.delta_bias_f_max:
            raise ValueError("delta_bias_f_min and delta_bias_f_max must satisfy 0 < min < max")


class IonaDenoisingConfig(PretrainedConfig):
    """Compose an Iona encoder configuration with a denoising head."""

    model_type = "iona-denoising"
    sub_configs = {"encoder": IonaConfig}

    def __init__(
        self,
        encoder: IonaConfig | dict | None = None,
        head_hidden_size: int = 128,
        head_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder if isinstance(encoder, IonaConfig) else IonaConfig(**(encoder or {}))
        self.head_hidden_size = head_hidden_size
        self.head_dropout = head_dropout

    @property
    def initializer_range(self) -> float:
        return self.encoder.initializer_range


class IonaRetrievalConfig(PretrainedConfig):
    """Compose an Iona encoder configuration with a contrastive retrieval objective.

    Without `projection_head`, the embedding is the pooled encoder output and the
    projection settings are unused. `kl_weight > 0` adds a KL term to a frozen reference
    model's intensity head.
    """

    model_type = "iona-retrieval"
    sub_configs = {"encoder": IonaConfig}
    # Loss components, not predictions; keeps Trainer.predict returning only embeddings.
    keys_to_ignore_at_inference = ["contrastive", "kl"]

    def __init__(
        self,
        encoder: IonaConfig | dict | None = None,
        projection_head: bool = True,
        projection_hidden_size: int = 512,
        embedding_size: int = 256,
        head_dropout: float = 0.1,
        pooling: str = "mean+max",
        temperature: float = 0.07,
        kl_weight: float = 0.0,
        gather_across_ranks: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder if isinstance(encoder, IonaConfig) else IonaConfig(**(encoder or {}))
        self.projection_head = projection_head
        self.projection_hidden_size = projection_hidden_size
        self.embedding_size = embedding_size
        self.head_dropout = head_dropout
        self.pooling = pooling
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.gather_across_ranks = gather_across_ranks
        if projection_hidden_size <= 0 or embedding_size <= 0:
            raise ValueError("retrieval projection dimensions must be positive")
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1)")
        if pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, got {pooling!r}")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if kl_weight < 0.0:
            raise ValueError("kl_weight must be non-negative")

    @property
    def initializer_range(self) -> float:
        return self.encoder.initializer_range


class IonaPeptideConfig(PretrainedConfig):
    """Store the architecture settings of the peptide encoder."""

    model_type = "iona-peptide"

    def __init__(
        self,
        embedding_size: int = 512,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 64,
        n_charges: int = 8,
        mod_n_freqs: int = 16,
        dropout: float = 0.1,
        pooling: str = "mean+max",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.n_charges = n_charges
        self.mod_n_freqs = mod_n_freqs
        self.dropout = dropout
        self.pooling = pooling
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, got {pooling!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


IonaConfig.register_for_auto_class()
IonaDenoisingConfig.register_for_auto_class()
IonaRetrievalConfig.register_for_auto_class()
IonaPeptideConfig.register_for_auto_class()
