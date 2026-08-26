"""Parse training configuration and create Trainer arguments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from transformers import HfArgumentParser, TrainingArguments

from msdelta.data import PreprocessConfig
from msdelta.model import DeltaBiasConfig, FourierConfig, ModelConfig


@dataclass
class ModelArgs:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ffn_mult: int = 4
    dropout: float = 0.1
    max_peaks: int = 150
    zero_bias_diagonal: bool = True
    fourier_int_n_freqs: int = 16
    fourier_int_f_min: float = 1e-2
    fourier_int_f_max: float = 1e2
    fourier_int_learnable: bool = True
    delta_bias_n_freqs: int = 64
    delta_bias_per_head_hidden: int = 32
    delta_bias_f_min: float = 1e-2
    delta_bias_f_max: float = 1e3
    delta_bias_scale: float = 3.0
    delta_bias_learnable: bool = True

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ffn_mult=self.ffn_mult,
            dropout=self.dropout,
            max_peaks=self.max_peaks,
            fourier_int=FourierConfig(
                self.fourier_int_n_freqs,
                self.fourier_int_f_min,
                self.fourier_int_f_max,
                learnable=self.fourier_int_learnable,
            ),
            delta_bias=DeltaBiasConfig(
                n_freqs=self.delta_bias_n_freqs,
                per_head_hidden=self.delta_bias_per_head_hidden,
                f_min=self.delta_bias_f_min,
                f_max=self.delta_bias_f_max,
                scale=self.delta_bias_scale,
                learnable=self.delta_bias_learnable,
            ),
            zero_bias_diagonal=self.zero_bias_diagonal,
        )


@dataclass
class DataArgs:
    root: str | None = None
    hf_repo: str | None = None
    hf_train_split: str = "train"
    hf_val_split: str = "val"
    n_val_files: int = 2
    intensity_threshold_frac: float = 0.01
    top_n: int = 150
    mask_ratio: float = 0.15
    preprocess_num_workers: int = 24

    def to_source_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "hf_repo": self.hf_repo,
            "hf_train_split": self.hf_train_split,
            "hf_val_split": self.hf_val_split,
            "n_val_files": self.n_val_files,
        }

    def preprocess(self) -> PreprocessConfig:
        return PreprocessConfig(
            intensity_threshold_frac=self.intensity_threshold_frac, top_n=self.top_n
        )


@dataclass
class TrainArgs:
    batch_size: int = 256
    num_workers: int = 8
    lr: float = 1e-4
    warmup_steps: int = 2000
    total_steps: int = 50000
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    precision: str = "bf16"
    compile: bool = False
    val_batches: int = 50
    seed: int = 0
    save_total_limit: int = 3
    deepspeed: bool = False
    zero_stage: int = 2
    deepspeed_fp32_gradients: bool = False
    deepspeed_torch_autocast: bool = True
    device: str = "cuda"


@dataclass
class LogArgs:
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    log_every: int = 50
    val_every: int = 2000
    bias_curve_every: int = 5000
    ckpt_every: int = 10000
    probe_every: int = 0
    probe_n_spectra: int = 3000
    replicate_retrieval_repo: str | None = None
    out_dir: str = "./runs"


def parse_config(argv: list[str] | None):
    """Parse a flat YAML file and command-line overrides."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", required=True, type=Path)
    pre.add_argument(
        "--run-name",
        dest="run_name",
        default=None,
        help="override wandb_run_name and the output dir name",
    )
    pre.add_argument("--resume", type=Path, default=None)
    pre.add_argument("--local_rank", type=int, default=-1)
    cli, overrides = pre.parse_known_args(argv)

    with open(cli.config) as f:
        flat = yaml.safe_load(f) or {}
    if cli.run_name:
        flat["wandb_run_name"] = cli.run_name

    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs, LogArgs))
    parser.set_defaults(**flat)
    margs, dargs, targs, largs = parser.parse_args_into_dataclasses(args=overrides)
    return cli, margs, dargs, targs, largs


def build_deepspeed_config(
    enabled: bool,
    zero_stage: int,
    fp32_gradients: bool = False,
    torch_autocast: bool = True,
) -> dict | None:
    """Create a DeepSpeed configuration or return ``None``."""
    if not enabled:
        return None
    config = {
        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": "auto",
        "gradient_accumulation_steps": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
    }
    if torch_autocast:
        config["torch_autocast"] = {
            "enabled": True,
            "dtype": "bfloat16",
        }
    else:
        config["bf16"] = {"enabled": "auto"}
        config["fp16"] = {"enabled": "auto"}
    if fp32_gradients:
        config["data_types"] = {"grad_accum_dtype": "fp32"}
        config["communication_data_type"] = "fp32"
    return config


def build_training_arguments(
    targs: TrainArgs,
    largs: LogArgs,
    out_dir: Path,
    run_name: str,
    report_to: list[str],
) -> TrainingArguments:
    """Convert project settings to Trainer arguments."""
    return TrainingArguments(
        output_dir=str(out_dir),
        run_name=run_name,
        max_steps=targs.total_steps,
        per_device_train_batch_size=targs.batch_size,
        per_device_eval_batch_size=targs.batch_size,
        gradient_accumulation_steps=targs.grad_accum_steps,
        learning_rate=targs.lr,
        weight_decay=targs.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=targs.grad_clip,
        warmup_steps=targs.warmup_steps,
        lr_scheduler_type=targs.lr_scheduler_type,
        bf16=(targs.precision == "bf16"),
        fp16=(targs.precision == "fp16"),
        torch_compile=targs.compile,
        logging_steps=largs.log_every,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=largs.val_every,
        save_strategy="steps",
        save_steps=largs.ckpt_every,
        save_total_limit=targs.save_total_limit,
        dataloader_num_workers=targs.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=targs.seed,
        report_to=report_to,
        deepspeed=build_deepspeed_config(
            targs.deepspeed,
            targs.zero_stage,
            targs.deepspeed_fp32_gradients,
            targs.deepspeed_torch_autocast,
        ),
    )
