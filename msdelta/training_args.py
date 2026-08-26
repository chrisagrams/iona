"""Parse model, processor, and Trainer configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml
from transformers import HfArgumentParser, TrainingArguments

from msdelta.configuration_msdelta import MSDeltaConfig
from msdelta.processing_msdelta import MSDeltaProcessor


@dataclass
class MSDeltaTrainingArguments(TrainingArguments):
    """TrainingArguments extended with MSDelta data and callback settings."""

    dataset_root: str | None = None
    dataset_repo_id: str | None = None
    dataset_train_split: str = "train"
    dataset_validation_split: str = "val"
    num_validation_files: int = 2
    preprocessing_num_workers: int = 24
    mask_ratio: float = 0.15
    validation_batches: int = 50
    bias_curve_steps: int = 5000
    probe_steps: int = 0
    probe_num_spectra: int = 3000
    replicate_retrieval_repo: str | None = None
    wandb_project: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dataset_root is None and self.dataset_repo_id is None:
            raise ValueError("set either dataset_root or dataset_repo_id")
        if self.dataset_root is not None and self.dataset_repo_id is not None:
            raise ValueError("dataset_root and dataset_repo_id are mutually exclusive")
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1]")
        for name in (
            "num_validation_files",
            "preprocessing_num_workers",
            "validation_batches",
            "bias_curve_steps",
            "probe_steps",
            "probe_num_spectra",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")


def parse_config(
    argv: list[str] | None,
) -> tuple[argparse.Namespace, MSDeltaConfig, MSDeltaProcessor, MSDeltaTrainingArguments]:
    """Parse nested YAML and native TrainingArguments command-line overrides."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", required=True, type=Path)
    pre_parser.add_argument("--resume", type=Path, default=None)
    cli, overrides = pre_parser.parse_known_args(argv)

    with open(cli.config) as handle:
        raw = yaml.safe_load(handle) or {}
    allowed_sections = {"model", "processor", "training"}
    unknown_sections = set(raw) - allowed_sections
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(f"unknown top-level configuration sections: {names}")

    model_config = MSDeltaConfig(**(raw.get("model") or {}))
    processor = MSDeltaProcessor(**(raw.get("processor") or {}))
    parser = HfArgumentParser(MSDeltaTrainingArguments)
    parser.set_defaults(**(raw.get("training") or {}))
    (training_args,) = parser.parse_args_into_dataclasses(args=overrides)
    return cli, model_config, processor, training_args
