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
    IonaPeptideConfig,
    IonaRetrievalConfig,
)
from iona.modeling_iona import (
    IonaForDenoising,
    IonaForDenoisingOutput,
    IonaForPreTraining,
    IonaForPreTrainingOutput,
    IonaForRetrieval,
    IonaForRetrievalOutput,
    IonaModel,
    IonaPeptideEncoder,
    IonaPeptideEncoderOutput,
    IonaPeptideForAlignment,
    IonaPeptideForAlignmentOutput,
    IonaPeptidePreTrainedModel,
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
AutoConfig.register(IonaPeptideConfig.model_type, IonaPeptideConfig, exist_ok=True)
AutoModel.register(IonaConfig, IonaModel, exist_ok=True)
AutoModel.register(IonaRetrievalConfig, IonaForRetrieval, exist_ok=True)
AutoModel.register(IonaPeptideConfig, IonaPeptideEncoder, exist_ok=True)
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
    "IonaPeptideConfig",
    "IonaRetrievalConfig",
    "IonaDataCollatorForPreTraining",
    "IonaDataCollatorForRetrieval",
    "IonaForDenoising",
    "IonaForDenoisingOutput",
    "IonaForPreTraining",
    "IonaForPreTrainingOutput",
    "IonaForRetrieval",
    "IonaForRetrievalOutput",
    "IonaModel",
    "IonaPeptideEncoder",
    "IonaPeptideEncoderOutput",
    "IonaPeptideForAlignment",
    "IonaPeptideForAlignmentOutput",
    "IonaPeptidePreTrainedModel",
    "IonaPreTrainedModel",
    "IonaProcessor",
]
