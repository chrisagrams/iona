"""Model architecture: configuration, encoder, heads, and Fourier features.

`experimental` is deliberately NOT re-exported here — importing it re-registers the
Hugging Face auto-classes against different class objects. Import it explicitly if
you need it.
"""

from msdelta.model.configuration import MSDeltaConfig, MSDeltaDenoisingConfig
from msdelta.model.fourier import FourierFeatures, dead_freqs, freq_drift, interp_mae
from msdelta.model.modeling import (
    BiasedMHA,
    DeltaMZBias,
    EncoderBlock,
    IntensityHead,
    MSDeltaForDenoising,
    MSDeltaForDenoisingOutput,
    MSDeltaForPreTraining,
    MSDeltaForPreTrainingOutput,
    MSDeltaModel,
    MSDeltaPreTrainedModel,
    PeakDenoisingHead,
    PeakEmbed,
)

__all__ = [
    "BiasedMHA",
    "DeltaMZBias",
    "EncoderBlock",
    "FourierFeatures",
    "IntensityHead",
    "MSDeltaConfig",
    "MSDeltaDenoisingConfig",
    "MSDeltaForDenoising",
    "MSDeltaForDenoisingOutput",
    "MSDeltaForPreTraining",
    "MSDeltaForPreTrainingOutput",
    "MSDeltaModel",
    "MSDeltaPreTrainedModel",
    "PeakDenoisingHead",
    "PeakEmbed",
    "dead_freqs",
    "freq_drift",
    "interp_mae",
]
