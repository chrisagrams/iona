"""Provide a Hugging Face-native mass-spectrum transformer."""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from msdelta.configuration_msdelta import (
    MSDeltaConfig,
    MSDeltaDenoisingConfig,
    MSDeltaRerankingConfig,
    MSDeltaRetrievalConfig,
)
from msdelta.modeling_msdelta import (
    MSDeltaForDenoising,
    MSDeltaForDenoisingOutput,
    MSDeltaForPreTraining,
    MSDeltaForPreTrainingOutput,
    MSDeltaForReranking,
    MSDeltaForRerankingOutput,
    MSDeltaForRetrieval,
    MSDeltaForRetrievalOutput,
    MSDeltaModel,
    MSDeltaPreTrainedModel,
)
from msdelta.processing_msdelta import (
    MSDeltaDataCollatorForPreTraining,
    MSDeltaDataCollatorForReranking,
    MSDeltaDataCollatorForRetrieval,
    MSDeltaProcessor,
    MSDeltaRerankingProcessor,
)

__version__ = "0.1.0"

AutoConfig.register(MSDeltaConfig.model_type, MSDeltaConfig, exist_ok=True)
AutoConfig.register(
    MSDeltaDenoisingConfig.model_type,
    MSDeltaDenoisingConfig,
    exist_ok=True,
)
AutoConfig.register(
    MSDeltaRetrievalConfig.model_type,
    MSDeltaRetrievalConfig,
    exist_ok=True,
)
AutoModel.register(MSDeltaConfig, MSDeltaModel, exist_ok=True)
AutoModel.register(MSDeltaRetrievalConfig, MSDeltaForRetrieval, exist_ok=True)
AutoModelForPreTraining.register(MSDeltaConfig, MSDeltaForPreTraining, exist_ok=True)
AutoModelForTokenClassification.register(
    MSDeltaDenoisingConfig,
    MSDeltaForDenoising,
    exist_ok=True,
)
AutoProcessor.register(MSDeltaConfig, MSDeltaProcessor, exist_ok=True)
AutoProcessor.register(MSDeltaDenoisingConfig, MSDeltaProcessor, exist_ok=True)
AutoProcessor.register(MSDeltaRetrievalConfig, MSDeltaProcessor, exist_ok=True)

AutoConfig.register(MSDeltaRerankingConfig.model_type, MSDeltaRerankingConfig, exist_ok=True)
AutoModel.register(MSDeltaRerankingConfig, MSDeltaForReranking, exist_ok=True)
AutoProcessor.register(MSDeltaRerankingConfig, MSDeltaRerankingProcessor, exist_ok=True)

__all__ = [
    "MSDeltaRerankingConfig",
    "MSDeltaForReranking",
    "MSDeltaForRerankingOutput",
    "MSDeltaDataCollatorForReranking",
    "MSDeltaRerankingProcessor",
    "MSDeltaConfig",
    "MSDeltaDenoisingConfig",
    "MSDeltaRetrievalConfig",
    "MSDeltaDataCollatorForPreTraining",
    "MSDeltaDataCollatorForRetrieval",
    "MSDeltaForDenoising",
    "MSDeltaForDenoisingOutput",
    "MSDeltaForPreTraining",
    "MSDeltaForPreTrainingOutput",
    "MSDeltaForRetrieval",
    "MSDeltaForRetrievalOutput",
    "MSDeltaModel",
    "MSDeltaPreTrainedModel",
    "MSDeltaProcessor",
]
