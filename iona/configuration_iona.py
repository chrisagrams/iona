"""Configuration for Iona models."""

from __future__ import annotations

from transformers import PretrainedConfig


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
    """Compose an Iona encoder configuration with a retrieval head."""

    model_type = "iona-retrieval"
    sub_configs = {"encoder": IonaConfig}

    def __init__(
        self,
        encoder: IonaConfig | dict | None = None,
        projection_hidden_size: int = 512,
        embedding_size: int = 256,
        head_dropout: float = 0.1,
        temperature: float = 0.07,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder if isinstance(encoder, IonaConfig) else IonaConfig(**(encoder or {}))
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


class IonaDenovoConfig(PretrainedConfig):
    """Compose an Iona encoder configuration with an autoregressive peptide decoder."""

    model_type = "iona-denovo"
    sub_configs = {"encoder": IonaConfig}

    def __init__(
        self,
        encoder: IonaConfig | dict | None = None,
        vocab_size: int | None = None,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        decoder_hidden_size: int = 512,
        decoder_num_layers: int = 6,
        decoder_num_heads: int = 8,
        decoder_intermediate_size: int = 2048,
        decoder_dropout: float = 0.1,
        max_peptide_length: int = 64,
        max_charge: int = 10,
        precursor_n_freqs: int = 64,
        precursor_f_min: float = 1e-4,
        precursor_f_max: float = 10.0,
        max_precursor_mass: float = 10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder if isinstance(encoder, IonaConfig) else IonaConfig(**(encoder or {}))
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_layers = decoder_num_layers
        self.decoder_num_heads = decoder_num_heads
        self.decoder_intermediate_size = decoder_intermediate_size
        self.decoder_dropout = decoder_dropout
        self.max_peptide_length = max_peptide_length
        self.max_charge = max_charge
        self.precursor_n_freqs = precursor_n_freqs
        self.precursor_f_min = precursor_f_min
        self.precursor_f_max = precursor_f_max
        self.max_precursor_mass = max_precursor_mass
        if decoder_hidden_size <= 0 or decoder_num_layers <= 0 or decoder_intermediate_size <= 0:
            raise ValueError("decoder dimensions must be positive")
        if decoder_hidden_size % decoder_num_heads:
            raise ValueError("decoder_hidden_size must be divisible by decoder_num_heads")
        if not 0.0 <= decoder_dropout < 1.0:
            raise ValueError("decoder_dropout must be in [0, 1)")
        if max_peptide_length <= 0 or max_charge <= 0:
            raise ValueError("max_peptide_length and max_charge must be positive")
        if not 0 < precursor_f_min < precursor_f_max:
            raise ValueError("precursor_f_min and precursor_f_max must satisfy 0 < min < max")

    @property
    def initializer_range(self) -> float:
        return self.encoder.initializer_range


IonaConfig.register_for_auto_class()
IonaDenoisingConfig.register_for_auto_class()
IonaRetrievalConfig.register_for_auto_class()
IonaDenovoConfig.register_for_auto_class()
