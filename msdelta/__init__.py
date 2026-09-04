"""Provide a Hugging Face-native mass-spectrum transformer."""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from msdelta.data.processing import (
    MSDeltaDataCollatorForPreTraining,
    MSDeltaProcessor,
)
from msdelta.model.configuration import MSDeltaConfig, MSDeltaDenoisingConfig
from msdelta.model.modeling import (
    MSDeltaForDenoising,
    MSDeltaForDenoisingOutput,
    MSDeltaForPreTraining,
    MSDeltaForPreTrainingOutput,
    MSDeltaModel,
    MSDeltaPreTrainedModel,
)

__version__ = "0.1.0"

AutoConfig.register(MSDeltaConfig.model_type, MSDeltaConfig, exist_ok=True)
AutoConfig.register(
    MSDeltaDenoisingConfig.model_type,
    MSDeltaDenoisingConfig,
    exist_ok=True,
)
AutoModel.register(MSDeltaConfig, MSDeltaModel, exist_ok=True)
AutoModelForPreTraining.register(MSDeltaConfig, MSDeltaForPreTraining, exist_ok=True)
AutoModelForTokenClassification.register(
    MSDeltaDenoisingConfig,
    MSDeltaForDenoising,
    exist_ok=True,
)
AutoProcessor.register(MSDeltaConfig, MSDeltaProcessor, exist_ok=True)
AutoProcessor.register(MSDeltaDenoisingConfig, MSDeltaProcessor, exist_ok=True)

__all__ = [
    "MSDeltaConfig",
    "MSDeltaDenoisingConfig",
    "MSDeltaDataCollatorForPreTraining",
    "MSDeltaForDenoising",
    "MSDeltaForDenoisingOutput",
    "MSDeltaForPreTraining",
    "MSDeltaForPreTrainingOutput",
    "MSDeltaModel",
    "MSDeltaPreTrainedModel",
    "MSDeltaProcessor",
]
