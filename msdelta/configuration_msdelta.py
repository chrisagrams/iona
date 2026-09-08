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
        delta_bias_n_freqs: int = 64,
        delta_bias_per_head_hidden: int = 32,
        delta_bias_f_min: float = 1e-2,
        delta_bias_f_max: float = 1e3,
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


class MSDeltaRerankingConfig(MSDeltaRetrievalConfig):
    """Compose a frozen spectrum encoder with a small peptide transformer."""

    model_type = "msdelta-reranking"

    def __init__(
        self,
        peptide_hidden_size: int = 256,
        peptide_num_hidden_layers: int = 3,
        peptide_num_attention_heads: int = 8,
        peptide_intermediate_size: int = 1024,
        peptide_max_length: int = 25,
        peptide_vocab: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.peptide_hidden_size = peptide_hidden_size
        self.peptide_num_hidden_layers = peptide_num_hidden_layers
        self.peptide_num_attention_heads = peptide_num_attention_heads
        self.peptide_intermediate_size = peptide_intermediate_size
        self.peptide_max_length = peptide_max_length
        self.peptide_vocab = peptide_vocab or [
            "[PAD]",
            *list("ACDEFGHIKLMNPQRSTVWY"),
            "C[57.0215]",
            "M[15.9949]",
        ]
        for value in (
            peptide_hidden_size,
            peptide_num_hidden_layers,
            peptide_num_attention_heads,
            peptide_intermediate_size,
            peptide_max_length,
        ):
            if value <= 0:
                raise ValueError("peptide dimensions must be positive")
        if peptide_hidden_size % peptide_num_attention_heads:
            raise ValueError("peptide_hidden_size must be divisible by peptide_num_attention_heads")
        if self.peptide_vocab[0] != "[PAD]" or len(set(self.peptide_vocab)) != len(
            self.peptide_vocab
        ):
            raise ValueError("peptide_vocab must be unique with [PAD] at index zero")


MSDeltaConfig.register_for_auto_class()
MSDeltaDenoisingConfig.register_for_auto_class()
MSDeltaRetrievalConfig.register_for_auto_class()
MSDeltaRerankingConfig.register_for_auto_class()
