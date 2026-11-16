"""Train a Prosit-style fragment-intensity head on a pretrained Iona encoder.

Data are the Prosit 2020 HCD splits (figshare article 12937092), converted once by
``scripts/convert_prosit_hdf5.py`` into memory-mapped ``.npy`` arrays under ``--data_dir``.
Training batches are built by ion budget (``--ion_pair_budget``/``--ion_budget``), so
``--per_device_train_batch_size`` is ignored. Watch ``eval_spectral_angle_median``; the
holdout is scored once, after the last step, into ``holdout_metrics.json``.

    python -m iona.finetune_intensity \
        --args_file configs/finetune/intensity-prosit-200m/training.args

The encoder is frozen by default; ``--freeze_encoder false`` fine-tunes it at its own
``--encoder_learning_rate``. Two controls make the headline number interpretable:
``--random_init true`` uses an untrained encoder of the same shape, and
``--use_encoder_states false`` drops the encoder states so the head sees only Prosit's
metadata.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from transformers import HfArgumentParser, TrainingArguments, set_seed

from iona.callbacks import WalltimeCheckpointCallback
from iona.configuration_iona import IonaConfig, IonaIntensityPredictionConfig
from iona.intensity import (
    SPLITS,
    IntensityTrainer,
    IonaIntensityProcessor,
    PrositIntensityDataset,
    evaluate_split,
    fit_intensity_prior,
    gather_arrays,
    intensity_metrics,
    summarize_spectral_angles,
)
from iona.modeling_iona import IonaForIntensityPrediction, IonaForPreTraining, IonaModel
from iona.wandb_distributed import init_wandb_run


@dataclass
class IntensityModelArguments:
    pretrained_path: str
    random_init: bool = False
    freeze_encoder: bool = True
    # False is the encoder-free control: the head sees only Prosit's metadata.
    use_encoder_states: bool = True
    head_hidden_size: int = 256
    head_dropout: float = 0.1


@dataclass
class IntensityDataArguments:
    data_dir: str
    max_train_samples: int | None = None
    prior_samples: int = 2_000_000
    eval_samples: int = 50_000
    max_holdout_samples: int | None = None
    # Max padded ions^2 summed over a training batch; bounds the attention-bias memory.
    ion_pair_budget: int = 2**21
    # Also cap padded ions (spectra x longest) per batch; needed when fine-tuning the encoder.
    ion_budget: int | None = None


@dataclass
class IntensityTrainingArguments(TrainingArguments):
    wandb_project: str | None = None
    wandb_entity: str | None = None
    # A fresh head wants ~1e-3 (--learning_rate); pretrained encoder weights far less.
    encoder_learning_rate: float = 1e-5


def load_encoder(model_args: IntensityModelArguments) -> IonaModel:
    """Lift the encoder out of a pretraining checkpoint, or build a same-shaped random one."""
    if model_args.random_init:
        return IonaModel(IonaConfig.from_pretrained(model_args.pretrained_path))
    # MSDelta-era checkpoints store msdelta.* keys; a no-op for Iona checkpoints.
    return IonaForPreTraining.from_pretrained(
        model_args.pretrained_path, key_mapping={r"^msdelta\.": "iona."}
    ).iona


def encoder_init(model_args: IntensityModelArguments) -> str:
    """The control label recorded with the holdout metrics."""
    if not model_args.use_encoder_states:
        return "none"
    return "random" if model_args.random_init else "pretrained"


def main(argv: list[str] | None = None) -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.xpu.is_available():
        torch.xpu.set_device(local_rank)

    parser = HfArgumentParser(
        (IntensityModelArguments, IntensityDataArguments, IntensityTrainingArguments)  # pyright: ignore[reportArgumentType]
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=argv, args_file_flag="--args_file"
    )
    if not model_args.freeze_encoder and not model_args.use_encoder_states:
        parser.error("--freeze_encoder false needs an encoder; drop --use_encoder_states false")
    out_dir = Path(training_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if training_args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", training_args.wandb_project)
        os.environ.setdefault("WANDB_DIR", str(out_dir))
    if training_args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", training_args.wandb_entity)
    training_args.report_to = ["wandb"] if training_args.wandb_project else []
    # The processor builds the model inputs; the loss needs only `labels`.
    training_args.remove_unused_columns = False
    training_args.label_names = ["labels"]
    # Checkpointing recomputes encoder activations, which only a trained encoder has.
    training_args.gradient_checkpointing = (
        training_args.gradient_checkpointing and not model_args.freeze_encoder
    )
    set_seed(training_args.seed)

    wandb_run = None
    if training_args.wandb_project:
        wandb_run = init_wandb_run(
            project=training_args.wandb_project,
            run_name=training_args.run_name or out_dir.name,
            entity=training_args.wandb_entity,
            config={},
        )
    # init_wandb_run returns None on every rank but each node's local rank 0.
    if wandb_run is not None:
        # A resumed job reattaches to the same run; its stored config comes back
        # JSON-round-tripped (e.g. int keys as strings), so allow value changes.
        wandb_run.config.update(
            {
                "model_args": asdict(model_args),
                "data": asdict(data_args),
                "training": training_args.to_dict(),
            },
            allow_val_change=True,
        )
    try:
        return train_and_evaluate(model_args, data_args, training_args, wandb_run)
    except BaseException:
        # Mark the W&B run as failed rather than finished.
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
            wandb_run = None
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def train_and_evaluate(
    model_args: IntensityModelArguments,
    data_args: IntensityDataArguments,
    training_args: IntensityTrainingArguments,
    wandb_run,
) -> int:
    seed = training_args.seed
    data_dir = Path(data_args.data_dir)
    out_dir = Path(training_args.output_dir)
    data = {split: PrositIntensityDataset(data_dir / split) for split in SPLITS}
    train = data["train"].subset(data_args.max_train_samples, seed)
    validation = data["val"].subset(data_args.eval_samples, seed)
    holdout = data["holdout"].subset(data_args.max_holdout_samples, seed)

    encoder = load_encoder(model_args)
    config = IonaIntensityPredictionConfig(
        encoder=cast(IonaConfig, encoder.config),
        head_hidden_size=model_args.head_hidden_size,
        head_dropout=model_args.head_dropout,
        use_encoder_states=model_args.use_encoder_states,
    )
    model = IonaForIntensityPrediction(
        config, encoder=encoder, freeze_encoder=model_args.freeze_encoder
    )
    processor = IonaIntensityProcessor(
        fit_intensity_prior(train.subset(data_args.prior_samples, seed + 1), config),
        max_fragment_charge=config.max_fragment_charge,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if training_args.process_index == 0:
        origin = "RANDOM INIT (control)" if model_args.random_init else model_args.pretrained_path
        print(
            f"[intensity] {trainable:,} trainable parameters; train spectra: {len(train):,}; "
            f"encoder from {origin}; freeze_encoder={model_args.freeze_encoder}; "
            f"use_encoder_states={model_args.use_encoder_states}",
            flush=True,
        )
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "model": config.to_dict(),
                "trainable_parameters": trainable,
                "splits": {"train": len(train), "val": len(validation), "holdout": len(holdout)},
            },
            allow_val_change=True,
        )

    trainer = IntensityTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=processor,
        processing_class=processor,
        compute_metrics=intensity_metrics,
        ion_pair_budget=data_args.ion_pair_budget,
        ion_budget=data_args.ion_budget,
        encoder_learning_rate=(
            None if model_args.freeze_encoder else training_args.encoder_learning_rate
        ),
    )
    if deadline := os.environ.get("IONA_JOB_DEADLINE_EPOCH"):
        margin = float(os.environ.get("IONA_CHECKPOINT_MARGIN_SECONDS", "900"))
        trainer.add_callback(WalltimeCheckpointCallback(float(deadline), margin))
    # Trainer.train() does not read args.resume_from_checkpoint on its own.
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if trainer.state.global_step < trainer.state.max_steps:
        # Stopped at the walltime margin after saving a checkpoint; the holdout
        # pass would not fit, so resume and evaluate in the next job.
        print(f"stopped early at step {trainer.state.global_step}; resume to finish")
        return 0
    trainer.save_model()

    # Each process scores its own slice of the holdout; rank 0 reports.
    accelerator = trainer.accelerator
    angles = gather_arrays(
        evaluate_split(
            model,
            holdout.shard(accelerator.process_index, accelerator.num_processes),
            processor,
            ion_pair_budget=data_args.ion_pair_budget,
            num_workers=training_args.dataloader_num_workers,
            autocast_dtype=torch.bfloat16 if training_args.bf16 else None,
        )
    )
    if not trainer.is_world_process_zero():
        return 0
    metrics = {
        **summarize_spectral_angles(angles["model"], angles["charge"], "holdout/"),
        **summarize_spectral_angles(angles["prior"], angles["charge"], "holdout/prior_"),
    }
    prosit = np.load(data_dir / "holdout" / "prosit_spectral_angle.npy", mmap_mode="r")
    metrics["holdout/prosit_rnn_spectral_angle_median"] = float(np.median(prosit[holdout.indices]))
    metrics["encoder_init"] = encoder_init(model_args)
    metrics["checkpoint"] = model_args.pretrained_path
    metrics["world_size"] = accelerator.num_processes
    print(json.dumps(metrics, indent=2))
    (out_dir / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2))
    if wandb_run is not None:
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        wandb_run.log({**numeric, "train/global_step": trainer.state.global_step})
        wandb_run.summary.update(numeric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
