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
    wandb_project: str | None = None
