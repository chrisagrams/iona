"""Typed training config: schema, parsing, and translation to HF arguments.

Config is four flat dataclasses — `ModelArgs`, `DataArgs`, `TrainArgs`,
`LogArgs`. A flat YAML file supplies the base values (`--config`), and
`HfArgumentParser` turns every field into a `--flag`, so any value is
overridable on the command line. That is exactly how a `wandb agent` injects a
sweep: it appends `--lr=… --mask_ratio=…` via the sweep's `${args}`, with no
glue code. `parse_config` merges the two (YAML → defaults, CLI flags win) and
`build_training_arguments` is the single place our vocabulary maps onto HF's.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from transformers import HfArgumentParser, TrainingArguments

from .data import PreprocessConfig
from .model import DeltaBiasConfig, FourierConfig, ModelConfig


# ---------- schema ----------
#
# Flat (no nested dataclasses) so HfArgumentParser turns every field into a
# `--flag`, which is what makes a value overridable on the command line and
# therefore sweepable by a wandb agent. The nested model sub-blocks (fourier_int
# / delta_bias) are flattened with a prefix — `fourier_int_n_freqs`,
# `delta_bias_hidden`, … — for the same reason; `to_model_config()` re-assembles
# the nested `ModelConfig`.

@dataclass
class ModelArgs:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ffn_mult: int = 4
    dropout: float = 0.1
    max_peaks: int = 150
    zero_bias_diagonal: bool = True
    score_mod_debug_stage: int = 7
    fourier_int_n_freqs: int = 16
    fourier_int_f_min: float = 1e-2
    fourier_int_f_max: float = 1e2
    fourier_int_learnable: bool = True
    delta_bias_hidden: int = 128
    delta_bias_resolution: float = 0.01
    delta_bias_max_distance: float = 2000.0
    delta_bias_coordinate_scale: float = 1.0
    delta_bias_scale: float = 3.0

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ffn_mult=self.ffn_mult,
            dropout=self.dropout,
            max_peaks=self.max_peaks,
            score_mod_debug_stage=self.score_mod_debug_stage,
            fourier_int=FourierConfig(
                self.fourier_int_n_freqs, self.fourier_int_f_min, self.fourier_int_f_max,
                learnable=self.fourier_int_learnable),
            delta_bias=DeltaBiasConfig(
                hidden=self.delta_bias_hidden,
                resolution=self.delta_bias_resolution,
                max_distance=self.delta_bias_max_distance,
                coordinate_scale=self.delta_bias_coordinate_scale,
                scale=self.delta_bias_scale,
            ),
            zero_bias_diagonal=self.zero_bias_diagonal,
        )


@dataclass
class DataArgs:
    root: str | None = None            # local parquet root (used when hf_repo unset)
    hf_repo: str | None = None         # HF dataset id (takes precedence over root)
    hf_train_split: str = "train"
    hf_val_split: str = "val"
    n_val_files: int = 2
    intensity_threshold_frac: float = 0.01
    top_n: int = 150
    mask_ratio: float = 0.15
    # CPU processes used once by rank 0 for Dataset.map/filter. This is
    # intentionally separate from per-rank DataLoader workers.
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
            intensity_threshold_frac=self.intensity_threshold_frac, top_n=self.top_n)


@dataclass
class TrainArgs:
    batch_size: int = 256              # per-device (per-GPU) micro-batch
    num_workers: int = 8
    lr: float = 1e-4
    warmup_steps: int = 2000
    total_steps: int = 50000
    lr_scheduler_type: str = "cosine"  # cosine | constant_with_warmup | linear | …
    # An LR-range-test screen wants a flat post-warmup LR so configs are ranked
    # on training dynamics, not on where the cosine tail happens to land; set
    # `constant_with_warmup` for that (see pbs/lr_sweep.pbs).
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    precision: str = "bf16"            # bf16 | fp16 | fp32
    compile: bool = False              # torch.compile the training forward
    val_batches: int = 50
    seed: int = 0
    save_total_limit: int = 3
    deepspeed: bool = False            # opt-in; needs a torchrun launch (see build_deepspeed_config)
    zero_stage: int = 2               # keep ≤2: inline probes call the eager encoder directly
    deepspeed_fp32_gradients: bool = False  # accumulate + reduce gradients in fp32 under DeepSpeed
    # Default to PyTorch autocast under DeepSpeed. Native DeepSpeed bf16 casts
    # the live model to bf16 and produced a reproducibly degraded trajectory for
    # this model under both ZeRO-0 and ZeRO-2.
    deepspeed_torch_autocast: bool = True
    device: str = "cuda"              # accepted for back-compat; Trainer manages placement


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


# ---------- parsing ----------

def parse_config(argv: list[str] | None):
    """Return (cli, ModelArgs, DataArgs, TrainArgs, LogArgs).

    The YAML is a flat key/value map (one key per dataclass field — no sections,
    since `HfArgumentParser` populates dataclasses from top-level keys only).
    `--config` names that base file; every other flag overrides the value it
    loaded (this is the wandb-sweep path — the agent appends `--lr=…` etc). YAML
    values become argparse defaults, so a flag left off the command line keeps
    the file's value and an unset field falls back to the dataclass default.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", required=True, type=Path)
    pre.add_argument("--run-name", dest="run_name", default=None,
                     help="override wandb_run_name and the output dir name")
    pre.add_argument("--resume", type=Path, default=None)
    # deepspeed/torch launchers may inject --local_rank; consume it here.
    pre.add_argument("--local_rank", type=int, default=-1)
    cli, overrides = pre.parse_known_args(argv)

    with open(cli.config) as f:
        flat = yaml.safe_load(f) or {}
    if cli.run_name:
        flat["wandb_run_name"] = cli.run_name

    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs, LogArgs))
    parser.set_defaults(**flat)                 # YAML → defaults; CLI flags win
    margs, dargs, targs, largs = parser.parse_args_into_dataclasses(args=overrides)
    return cli, margs, dargs, targs, largs


# ---------- translation to HF arguments ----------

def build_deepspeed_config(
    enabled: bool, zero_stage: int, fp32_gradients: bool = False,
    torch_autocast: bool = True,
) -> dict | None:
    """A sensible ZeRO config for `TrainingArguments(deepspeed=...)`, or None
    when disabled. Optimizer and scheduler are left as ``auto`` so HF builds
    them from `TrainingArguments` (AdamW + cosine-with-warmup).

    Off by default: DeepSpeed needs a torchrun/deepspeed launcher to stand up
    the process group, so it would break the single-GPU ``msdelta-train``
    launches. Opt in with ``deepspeed: true`` on the multi-GPU configs that
    launch under torchrun (see configs/massivekb_xl.yaml).

    PyTorch bf16 autocast is the default because it keeps live parameters in
    fp32 and matches the healthy non-DeepSpeed training trajectory. Set
    ``deepspeed_torch_autocast: false`` only to reproduce the legacy native-bf16
    behavior."""
    if not enabled:
        return None
    config = {
        "zero_optimization": {
            # ZeRO-2 partitions optimizer state + grads but keeps a full copy
            # of every parameter on each rank — required for the inline probes,
            # which call the eager encoder directly (see ScienceCallback).
            "stage": zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": "auto",
        "gradient_accumulation_steps": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        # The synchronized throughput timer calls torch.cuda.synchronize(None)
        # from a model forward hook. TorchDynamo 2.12/2.13 cannot trace that
        # valid CUDA call, and the timer is not needed for training metrics.
        "timers": {"throughput": {"enabled": False}},
    }
    if torch_autocast:
        # DeepSpeed's native bf16 mode casts the live model to bf16. PyTorch
        # autocast instead keeps fp32 parameters and chooses the compute dtype
        # per operation, matching the healthy non-DeepSpeed Trainer path while
        # retaining ZeRO sharding. Native bf16/fp16 keys must not coexist with
        # DeepSpeed's torch_autocast mode.
        config["torch_autocast"] = {
            "enabled": True,
            "dtype": "bfloat16",
        }
    else:
        config["bf16"] = {"enabled": "auto"}
        config["fp16"] = {"enabled": "auto"}
    if fp32_gradients:
        # DeepSpeed-native bf16 otherwise accumulates gradients in the model
        # dtype and may use a reduced-precision communication dtype. Pin both
        # lossy summations to fp32 so this path matches PyTorch bf16 AMP more
        # closely; parameters and optimizer state remain managed by ZeRO.
        config["data_types"] = {"grad_accum_dtype": "fp32"}
        config["communication_data_type"] = "fp32"
    return config


def build_training_arguments(
    targs: TrainArgs, largs: LogArgs, out_dir: Path, run_name: str,
    report_to: list[str],
) -> TrainingArguments:
    """The single place our config vocabulary is translated into HF's."""
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
