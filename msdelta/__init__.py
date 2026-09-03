"""Provide a Hugging Face-native mass-spectrum transformer."""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from msdelta.configuration_msdelta import MSDeltaConfig
from msdelta.modeling_msdelta import (
    MSDeltaForDenoising,
    MSDeltaForDenoisingOutput,
    MSDeltaForPreTraining,
    MSDeltaForPreTrainingOutput,
    MSDeltaModel,
    MSDeltaPreTrainedModel,
)
from msdelta.processing_msdelta import (
    MSDeltaDataCollatorForDenoising,
    MSDeltaDataCollatorForPreTraining,
    MSDeltaProcessor,
)

__version__ = "0.1.0"

AutoConfig.register(MSDeltaConfig.model_type, MSDeltaConfig, exist_ok=True)
AutoModel.register(MSDeltaConfig, MSDeltaModel, exist_ok=True)
AutoModelForPreTraining.register(MSDeltaConfig, MSDeltaForPreTraining, exist_ok=True)
AutoModelForTokenClassification.register(MSDeltaConfig, MSDeltaForDenoising, exist_ok=True)
AutoProcessor.register(MSDeltaConfig, MSDeltaProcessor, exist_ok=True)

__all__ = [
    "MSDeltaConfig",
    "MSDeltaDataCollatorForDenoising",
    "MSDeltaDataCollatorForPreTraining",
    "MSDeltaForDenoising",
    "MSDeltaForDenoisingOutput",
    "MSDeltaForPreTraining",
    "MSDeltaForPreTrainingOutput",
    "MSDeltaModel",
    "MSDeltaPreTrainedModel",
    "MSDeltaProcessor",
]
