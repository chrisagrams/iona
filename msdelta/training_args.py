"""Parse Hugging Face-style training arguments."""

from __future__ import annotations

from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """Arguments for loading and overriding an MSDelta model configuration."""

    config_name: str = field(
        metadata={"help": "Path to a pretrained MSDelta configuration directory."}
    )
    config_overrides: str | None = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated model configuration overrides, for example "
                "'hidden_size=768,num_hidden_layers=12'."
            )
        },
    )


@dataclass
class DataArguments:
    """Arguments for loading spectra and configuring preprocessing."""

    processor_name_or_path: str = field(
        metadata={"help": "Path to a pretrained MSDelta processor directory."}
    )
    dataset_root: str | None = None
    dataset_repo_id: str | None = None
    dataset_train_split: str = "train"
    dataset_validation_split: str = "val"
    num_validation_files: int = 2
    preprocessing_num_workers: int = 24
    intensity_threshold_frac: float | None = None
    max_peaks: int | None = None


@dataclass
class MSDeltaTrainingArguments(TrainingArguments):
    """TrainingArguments extended with MSDelta callback settings."""

    mask_ratio: float = 0.15
    validation_batches: int = 50
    bias_curve_steps: int = 5000
    probe_steps: int = 0
    probe_num_spectra: int = 3000
    replicate_retrieval_repo: str | None = None
    retrieval_steps: int = 0
    retrieval_dataset_repo: str = "chrisagrams/ms-contrastive-100k"
    retrieval_per_device_batch_size: int = 16
    retrieval_epochs: int = 1
    retrieval_learning_rate: float = 1e-3
    retrieval_weight_decay: float = 1e-2
    retrieval_projection_hidden_size: int = 512
    retrieval_embedding_size: int = 256
    retrieval_head_dropout: float = 0.1
    retrieval_temperature: float = 0.07
    retrieval_validation_analytes: int = 1000
    retrieval_num_workers: int = 4
    retrieval_seed: int = 0
    reranking_steps: int = 0
    reranking_dataset_repo: str = "chrisagrams/ms-contrastive-100k"
    reranking_dataset_revision: str | None = "613d9b1901debc66877264bedbaa6def46ac0861"
    reranking_per_device_batch_size: int = 16
    reranking_epochs: int = 1
    reranking_learning_rate: float = 1e-3
    reranking_weight_decay: float = 1e-2
    reranking_projection_hidden_size: int = 512
    reranking_embedding_size: int = 256
    reranking_head_dropout: float = 0.1
    reranking_temperature: float = 0.07
    reranking_peptide_hidden_size: int = 256
    reranking_peptide_num_hidden_layers: int = 3
    reranking_peptide_num_attention_heads: int = 8
    reranking_peptide_intermediate_size: int = 1024
    reranking_peptide_max_length: int = 25
    reranking_include_consensus: bool = False
    reranking_validation_analytes: int = 1000
    reranking_num_workers: int = 4
    reranking_seed: int = 0
    denoise_steps: int = 0
    denoise_dataset_repo: str = "chrisagrams/ms-denoise-100k"
    denoise_max_peaks: int = 1024
    denoise_peak_pair_budget: int = 4_194_304
    denoise_intensity_threshold_frac: float = 0.0
    denoise_head_hidden_size: int = 128
    denoise_head_dropout: float = 0.1
    denoise_epochs: int = 1
    denoise_learning_rate: float = 1e-3
    denoise_weight_decay: float = 1e-2
    denoise_num_workers: int = 4
    denoise_seed: int = 0
    wandb_project: str | None = None
