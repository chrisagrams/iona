"""Fine-tune a pretrained encoder for per-peak noise classification.

Each peak gets one logit, trained with binary cross-entropy against the corpus's per-peak
boolean. **Noise is the positive class (label 1)** -- the convention `process_denoising_example`
and `denoising_metrics` already use, so precision/recall here read as "of the peaks called
noise, how many were".

This is a standalone fine-tune, distinct from the probe `posttraining.py` runs during
pretraining. That one is a sidecar: it freezes the encoder, trains only the head at batch
size 1, logs into the parent run and saves nothing. Here the encoder trains too, which is
the point -- zero-shot probing established that the frozen representation does not carry a
usable noise signal (AUROC 0.727 from predicted intensity against 0.755 from raw intensity
alone), so a frozen probe is measuring something we already know fails.

Three things about the setup that are decisions rather than defaults:

**The head is randomly initialised and the encoder is not.** Early gradients from an
untrained head are large, and letting them straight into pretrained weights is how
fine-tuning erases what it was meant to build on. `freeze_encoder_steps` holds the encoder
still until the head is sane, and `encoder_lr_scale` keeps it moving slower afterwards.

**Batches are bounded by padded attention area, not by a count.** Peak counts in this
corpus span 20 to 2611 and the pair branch is quadratic, so a fixed batch size either
wastes memory on short spectra or runs out on long ones.

**Oversized spectra are dropped, not truncated.** `build_denoising_datasets` filters
anything above `max_peaks`; at 1024 that is 1.62% of train. Worth remembering that the
dropped spectra are the largest, which are also the noisiest, so the retained corpus is
slightly easier than the real one.
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from transformers import DataCollatorWithPadding, HfArgumentParser, TrainingArguments, set_seed

from msdelta.configuration_msdelta import MSDeltaConfig, MSDeltaDenoisingConfig
from msdelta.data import build_denoising_datasets
from msdelta.denoising import DenoisingTrainer
from msdelta.modeling_msdelta import MSDeltaForDenoising, MSDeltaForPreTraining
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.wandb_distributed import init_wandb_run


@dataclass
class DenoiseModelArguments:
    """Where the pretrained encoder comes from and how gently to move it."""

    pretrained_path: str = field(
        metadata={
            "help": (
                "Directory holding a pretrained MSDelta checkpoint (a `final/` or "
                "`checkpoint-N/`). Its encoder is lifted out and given a fresh peak "
                "classifier; the pretraining intensity head is discarded."
            )
        }
    )
    head_hidden_size: int = 128
    head_dropout: float = 0.1
    freeze_encoder_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Train only the head for this many steps before unfreezing the encoder. "
                "A randomly initialised head produces large early gradients and letting "
                "them reach pretrained weights is how fine-tuning erases them. 0 trains "
                "everything from step one."
            )
        },
    )
    encoder_lr_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "Multiply the learning rate by this for encoder parameters only. Below 1 "
                "(0.1 is common) lets the head move quickly while the encoder is nudged."
            )
        },
    )


@dataclass
class DenoiseDataArguments:
    """Corpus and peak handling."""

    processor_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Processor config directory. Defaults to `pretrained_path`."},
    )
    dataset_repo: str = "chrisagrams/ms-denoise-100k"
    preprocessing_num_workers: int = 24
    max_peaks: int = field(
        default=1024,
        metadata={
            "help": (
                "Peak cap. Pretraining used 512, but the corpus's own denoise probe "
                "setting is 1024 and the task needs the weak peaks a tighter cap would "
                "discard. Spectra above the cap are DROPPED, not truncated: 1.62% of "
                "train at 1024 against 12.85% at 512."
            )
        },
    )


@dataclass
class DenoiseFinetuneArguments(TrainingArguments):
    """TrainingArguments plus the batching and reporting this task needs."""

    peak_pair_budget: int = field(
        default=4_194_304,
        metadata={
            "help": (
                "Maximum padded peaks^2 per batch. Batches are built to this budget "
                "rather than a fixed count because peak counts span two orders of "
                "magnitude and the pair branch is quadratic in them."
            )
        },
    )
    wandb_project: str | None = None
    wandb_entity: str | None = None
    eval_test_split: bool = field(
        default=True,
        metadata={"help": "Score the held-out test split once training finishes."},
    )


def denoise_metrics(prediction) -> dict[str, float]:
    """Peak-level metrics with noise as the positive class, hardened against the gather.

    Upstream's `denoising_metrics` filters only `labels != -100` and hands the rest
    straight to sklearn. Job 8839579 died at the first in-training evaluation with
    "multiclass format is not supported", meaning the gathered array held a third value.
    A single-process eval and a two-rank gloo eval both produce a clean {-100, 0, 1}
    here, so whatever introduces it only appears at 12 ranks with bf16 and a full split --
    which is exactly the configuration that is expensive to reproduce.

    Rather than keep guessing at it from the outside, this keeps only the rows that are
    genuinely 0 or 1 and REPORTS what it dropped as `label_dropped` and `label_extra`.
    A metric has no business terminating a four-hour fine-tune, and the next run tells us
    the answer instead of costing another slot to ask the question again.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, auc, balanced_accuracy_score, f1_score,
        precision_recall_curve, precision_score, recall_score, roc_auc_score,
    )

    logits = np.asarray(prediction.predictions, dtype=np.float64).reshape(-1)
    labels = np.asarray(prediction.label_ids, dtype=np.float64).reshape(-1)
    binary = (labels == 0.0) | (labels == 1.0)
    unexpected = ~binary & (labels != -100.0)

    metrics = {
        "label_dropped": float(unexpected.sum()),
        # The distinct offending values, so one run identifies the cause.
        "label_extra": float(len(np.unique(labels[unexpected]))) if unexpected.any() else 0.0,
    }
    if unexpected.any():
        print(f"[denoise] dropped {int(unexpected.sum())} labels outside {{0,1,-100}}: "
              f"{np.unique(labels[unexpected])[:8]}", flush=True)

    logits, labels = logits[binary], labels[binary].astype(np.int64)
    if labels.size == 0 or labels.min() == labels.max():
        # A slice with one class is not scorable; returning zeros keeps the run alive and
        # makes the degenerate eval obvious in the W&B curve.
        metrics.update({"accuracy": 0.0, "balanced_accuracy": 0.0, "precision": 0.0,
                        "recall": 0.0, "f1": 0.0, "auroc": 0.5, "auprc": 0.0,
                        "n_peaks": float(labels.size)})
        return metrics

    predicted = logits >= 0
    pr_precision, pr_recall, _ = precision_recall_curve(labels, logits)
    metrics.update({
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(labels, logits)),
        "auprc": float(auc(pr_recall, pr_precision)),
        "n_peaks": float(labels.size),
        "noise_fraction": float(labels.mean()),
    })
    return metrics


def build_denoising_model(
    pretrained_path: str, model_args: DenoiseModelArguments
) -> tuple[MSDeltaForDenoising, MSDeltaForPreTraining]:
    """Lift the encoder out of a pretraining checkpoint and attach a fresh classifier."""
    pretrained = MSDeltaForPreTraining.from_pretrained(pretrained_path)
    config = MSDeltaDenoisingConfig(
        encoder=copy.deepcopy(pretrained.config if isinstance(pretrained.config, MSDeltaConfig)
                              else MSDeltaConfig(**pretrained.config.to_dict())),
        head_hidden_size=model_args.head_hidden_size,
        head_dropout=model_args.head_dropout,
    )
    # freeze_encoder=False: this is a full fine-tune. The freezing that matters here is
    # the temporary kind, handled by the trainer's step schedule.
    model = MSDeltaForDenoising(config, encoder=pretrained.msdelta, freeze_encoder=False)
    return model, pretrained


class DenoiseFinetuneTrainer(DenoisingTrainer):
    """Budget batching, plus split learning rates and a delayed encoder unfreeze."""

    def __init__(self, *args, encoder_lr_scale: float = 1.0, freeze_encoder_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_lr_scale = encoder_lr_scale
        self.freeze_encoder_steps = freeze_encoder_steps
        self._unfrozen = freeze_encoder_steps <= 0
        if not self._unfrozen:
            self.model.msdelta.requires_grad_(False)

    def create_optimizer(self):
        """Two parameter groups so the encoder can be nudged while the head moves."""
        if self.optimizer is not None:
            return self.optimizer
        optimizer_class, kwargs = type(self).get_optimizer_cls_and_kwargs(self.args, self.model)
        kwargs.pop("lr", None)
        encoder, head = [], []
        for name, parameter in self.model.named_parameters():
            (encoder if name.startswith("msdelta.") else head).append(parameter)
        self.optimizer = optimizer_class(
            [
                {"params": encoder, "lr": self.args.learning_rate * self.encoder_lr_scale},
                {"params": head, "lr": self.args.learning_rate},
            ],
            **kwargs,
        )
        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self._unfrozen and self.state.global_step >= self.freeze_encoder_steps:
            # requires_grad_ only; the encoder was never put in eval mode, so dropout and
            # the rest resume exactly as a full fine-tune expects.
            unwrapped = getattr(model, "module", model)
            unwrapped.msdelta.requires_grad_(True)
            self._unfrozen = True
        return super().training_step(model, inputs, num_items_in_batch)


def main(argv: list[str] | None = None) -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.xpu.is_available():
        torch.xpu.set_device(local_rank)

    parser = HfArgumentParser(
        (DenoiseModelArguments, DenoiseDataArguments, DenoiseFinetuneArguments)  # pyright: ignore[reportArgumentType]
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=argv, args_file_flag="--args_file"
    )

    out_dir = Path(training_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if training_args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", training_args.wandb_project)
        os.environ.setdefault("WANDB_DIR", str(out_dir))
    if training_args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", training_args.wandb_entity)

    set_seed(training_args.seed)

    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks,
    )
    model, pretrained = build_denoising_model(model_args.pretrained_path, model_args)

    wandb_run = None
    if training_args.wandb_project:
        wandb_run = init_wandb_run(
            project=training_args.wandb_project,
            run_name=training_args.run_name,
            entity=training_args.wandb_entity,
            config={
                "model": model_args.pretrained_path,
                "encoder": pretrained.config.to_dict(),
                "data": asdict(data_args),
                "training": training_args.to_dict(),
            },
        )

    try:
        if training_args.process_index == 0:
            total = sum(p.numel() for p in model.parameters())
            head = sum(p.numel() for p in model.denoising_head.parameters())
            print(f"[denoise] {total / 1e6:.2f}M params ({head / 1e6:.3f}M in the head)", flush=True)
            print(
                f"[denoise] max_peaks={processor.max_peaks} "
                f"freeze_encoder_steps={model_args.freeze_encoder_steps} "
                f"encoder_lr_scale={model_args.encoder_lr_scale}",
                flush=True,
            )

        with training_args.main_process_first(local=False, desc="denoising data"):
            datasets = build_denoising_datasets(
                data_args.dataset_repo,
                processor,
                num_proc=data_args.preprocessing_num_workers or None,
            )
        if training_args.process_index == 0:
            print(
                "[denoise] "
                + " ".join(f"{split}={len(ds):,}" for split, ds in datasets.items()),
                flush=True,
            )

        trainer = DenoiseFinetuneTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"),
            data_collator=DataCollatorWithPadding(
                tokenizer=processor, padding=True, return_tensors="pt"
            ),
            processing_class=processor,
            compute_metrics=denoise_metrics,
            peak_pair_budget=training_args.peak_pair_budget,
            encoder_lr_scale=model_args.encoder_lr_scale,
            freeze_encoder_steps=model_args.freeze_encoder_steps,
        )
        trainer.train()

        if training_args.eval_test_split and datasets.get("test") is not None:
            metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")
            if trainer.is_world_process_zero():
                print(f"[denoise] test: {metrics}", flush=True)
                trainer.log(metrics)
                trainer.save_metrics("test", metrics)

        if trainer.is_world_process_zero():
            trainer.save_model(str(out_dir / "final"))
            processor.save_pretrained(str(out_dir / "final"))
            print(f"[denoise] saved to {out_dir / 'final'}", flush=True)
        return 0
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    sys.exit(main())
