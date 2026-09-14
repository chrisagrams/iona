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
    dataset_repo_id: str = field(metadata={"help": "Hugging Face dataset repository ID."})
    dataset_train_split: str = "train"
    dataset_validation_split: str = "validation"
    dataset_cache_dir: str | None = field(
        default=None,
        metadata={"help": "Optional Hugging Face datasets cache directory."},
    )
    preprocessed_dataset_dir: str | None = field(
        default=None,
        metadata={"help": "Optional finalized dataset directory created by msdelta-preprocess."},
    )
    preprocessing_num_workers: int = 24
    max_peaks: int | None = None


@dataclass
class MSDeltaTrainingArguments(TrainingArguments):
    """TrainingArguments extended with MSDelta callback settings."""

    probe_execution: str = field(
        default="inline",
        metadata={
            "choices": ["inline", "sidecar", "off"],
            "help": "Execution of denoising/retrieval heads; diagnostics remain inline.",
        },
    )
    sidecar_launcher: str | None = field(
        default=None, metadata={"help": "Shell launcher for independent probe processes."}
    )
    sidecar_denoise_device: str | None = None
    sidecar_retrieval_device: str | None = None
    mask_ratio: float = 0.15
    logarithmic_eval_start_step: int | None = None
    bias_curve_steps: int = 5000
    probe_steps: int = 0
    probe_num_spectra: int = 3000
    probe_batch_size: int = 1
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
    denoise_steps: int = 0
    denoise_dataset_repo: str = "chrisagrams/ms-denoise-100k"
    denoise_max_peaks: int = 1024
    denoise_peak_pair_budget: int = 4_194_304
    denoise_head_hidden_size: int = 128
    denoise_head_dropout: float = 0.1
    denoise_epochs: int = 1
    denoise_learning_rate: float = 1e-3
    denoise_weight_decay: float = 1e-2
    denoise_num_workers: int = 4
    denoise_seed: int = 0
    wandb_project: str | None = None

    def __post_init__(self):
        if self.logarithmic_eval_start_step is not None and self.logarithmic_eval_start_step < 1:
            raise ValueError("logarithmic_eval_start_step must be positive")
        if self.probe_batch_size < 1:
            raise ValueError("probe_batch_size must be positive")
        if self.probe_execution not in {"inline", "sidecar", "off"}:
            raise ValueError("probe_execution must be inline, sidecar, or off")
        if self.probe_execution == "sidecar":
            devices = []
            probes = (
                ("denoise", self.denoise_steps, self.sidecar_denoise_device),
                ("retrieval", self.retrieval_steps, self.sidecar_retrieval_device),
            )
            for kind, every, device in probes:
                if not every:
                    continue
                parts = (device or "").split(":")
                if (
                    len(parts) != 2
                    or parts[0] not in {"xpu", "cuda"}
                    or not all(index.isascii() and index.isdigit() for index in parts[1].split(","))
                ):
                    raise ValueError(
                        f"sidecar_{kind}_device must be xpu:<tile>[,<tile>...] or cuda:<index>[,<index>...]"
                    )
                devices.extend((parts[0], int(index)) for index in parts[1].split(","))
            if len(devices) != len(set(devices)):
                raise ValueError("Sidecar probes must use distinct devices")
            intervals = [self.denoise_steps, self.retrieval_steps]
            if any(interval < 0 for interval in intervals):
                raise ValueError("posttraining intervals must be nonnegative")
            if any(intervals):
                if not self.sidecar_launcher:
                    raise ValueError("sidecar_launcher is required for sidecar probes")
                if self.save_strategy != "steps" or self.save_steps < 1:
                    raise ValueError("sidecars require checkpoint saves at integer step intervals")
                if any(interval and interval % self.save_steps for interval in intervals):
                    raise ValueError("posttraining intervals must be multiples of save_steps")
            if self.deepspeed or self.fsdp:
                raise ValueError("sidecars currently support unsharded DDP checkpoints")
        super().__post_init__()
