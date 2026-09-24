"""A foundation model for mass spectrometry."""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from iona.configuration_iona import (
    IonaConfig,
    IonaDenoisingConfig,
    IonaIntensityPredictionConfig,
    IonaRetrievalConfig,
)
from iona.modeling_iona import (
    IonaForDenoising,
    IonaForDenoisingOutput,
    IonaForIntensityPrediction,
    IonaForIntensityPredictionOutput,
    IonaForPreTraining,
    IonaForPreTrainingOutput,
    IonaForRetrieval,
    IonaForRetrievalOutput,
    IonaModel,
    IonaPreTrainedModel,
)
from iona.processing_iona import (
    IonaDataCollatorForPreTraining,
    IonaDataCollatorForRetrieval,
    IonaProcessor,
)

__version__ = "0.1.0"  # x-release-please-version

AutoConfig.register(IonaConfig.model_type, IonaConfig, exist_ok=True)
AutoConfig.register(
    IonaDenoisingConfig.model_type,
    IonaDenoisingConfig,
    exist_ok=True,
)
AutoConfig.register(
    IonaRetrievalConfig.model_type,
    IonaRetrievalConfig,
    exist_ok=True,
)
AutoConfig.register(
    IonaIntensityPredictionConfig.model_type,
    IonaIntensityPredictionConfig,
    exist_ok=True,
)
AutoModel.register(IonaConfig, IonaModel, exist_ok=True)
AutoModel.register(
    IonaIntensityPredictionConfig,
    IonaForIntensityPrediction,
    exist_ok=True,
)
AutoModel.register(IonaRetrievalConfig, IonaForRetrieval, exist_ok=True)
AutoModelForPreTraining.register(IonaConfig, IonaForPreTraining, exist_ok=True)
AutoModelForTokenClassification.register(
    IonaDenoisingConfig,
    IonaForDenoising,
    exist_ok=True,
)
AutoProcessor.register(IonaConfig, IonaProcessor, exist_ok=True)
AutoProcessor.register(IonaDenoisingConfig, IonaProcessor, exist_ok=True)
AutoProcessor.register(IonaRetrievalConfig, IonaProcessor, exist_ok=True)

__all__ = [
    "IonaConfig",
    "IonaDenoisingConfig",
    "IonaIntensityPredictionConfig",
    "IonaRetrievalConfig",
    "IonaDataCollatorForPreTraining",
    "IonaDataCollatorForRetrieval",
    "IonaForDenoising",
    "IonaForDenoisingOutput",
    "IonaForIntensityPrediction",
    "IonaForIntensityPredictionOutput",
    "IonaForPreTraining",
    "IonaForPreTrainingOutput",
    "IonaForRetrieval",
    "IonaForRetrievalOutput",
    "IonaModel",
    "IonaPreTrainedModel",
    "IonaProcessor",
]
