"""Fine-tune a pretrained encoder for per-peak noise classification.

Each peak gets one logit trained with BCE. Noise is the positive class (label 1).
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from transformers import (DataCollatorWithPadding, HfArgumentParser, Trainer,
                          TrainingArguments, set_seed)

from iona.configuration_iona import IonaConfig, IonaDenoisingConfig
from iona.data import build_denoising_datasets
from iona.denoising import DenoisingTrainer
from iona.modeling_iona import IonaForDenoising, IonaForPreTraining
from iona.processing_iona import IonaProcessor
from iona.wandb_distributed import init_wandb_run


@dataclass
class DenoiseModelArguments:
    pretrained_path: str
    head_hidden_size: int = 128
    head_dropout: float = 0.1
    random_init: bool = False
    freeze_encoder_steps: int = 0
    encoder_lr_scale: float = 1.0


@dataclass
class DenoiseDataArguments:
    processor_name_or_path: str | None = None
    dataset_repo: str = "chrisagrams/ms-denoise-100k"
    preprocessing_num_workers: int = 24
    max_samples: int = 0
    max_peaks: int = 1024


@dataclass
class DenoiseFinetuneArguments(TrainingArguments):
    use_peak_budget_batching: bool = False
    peak_pair_budget: int = 4_194_304
    wandb_project: str | None = None
    wandb_entity: str | None = None
    eval_test_split: bool = True


def denoise_metrics(prediction) -> dict[str, float]:
    """Peak-level metrics with noise as the positive class.

    Labels outside {0, 1, -100} are dropped and counted rather than raising.
    """
    # Keep the 2-D form for per_spectrum_auroc.
    logits_2d = np.asarray(prediction.predictions, dtype=np.float64)
    labels_2d = np.asarray(prediction.label_ids, dtype=np.float64)

    logits = logits_2d.reshape(-1)
    labels = labels_2d.reshape(-1)
    binary = (labels == 0.0) | (labels == 1.0)
    unexpected = ~binary & (labels != -100.0)

    metrics = {
        "label_dropped": float(unexpected.sum()),
        "label_extra": float(len(np.unique(labels[unexpected]))) if unexpected.any() else 0.0,
    }
    if unexpected.any():
        print(f"[denoise] dropped {int(unexpected.sum())} labels outside {{0,1,-100}}: "
              f"{np.unique(labels[unexpected])[:8]}", flush=True)

    logits, labels = logits[binary], labels[binary].astype(np.int64)
    if labels.size == 0 or labels.min() == labels.max():
        # Not scorable with a single class.
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
    metrics.update(per_spectrum_auroc(logits_2d, labels_2d))
    return metrics


def per_spectrum_auroc(logits_2d, labels_2d) -> dict[str, float]:
    """AUROC computed within each spectrum, then averaged. Single-class spectra are counted as unscorable."""
    if logits_2d.ndim != 2:
        return {"auroc_per_spectrum": float("nan"), "spectra_scored": 0.0,
                "spectra_unscorable": 0.0}
    scores, unscorable = [], 0
    for row_logits, row_labels in zip(logits_2d, labels_2d):
        keep = (row_labels == 0.0) | (row_labels == 1.0)
        y, x = row_labels[keep], row_logits[keep]
        if y.size < 2 or y.min() == y.max():
            unscorable += 1
            continue
        scores.append(roc_auc_score(y.astype(np.int64), x))
    if not scores:
        return {"auroc_per_spectrum": float("nan"), "spectra_scored": 0.0,
                "spectra_unscorable": float(unscorable)}
    return {
        "auroc_per_spectrum": float(np.mean(scores)),
        "auroc_per_spectrum_sd": float(np.std(scores)),
        "auroc_per_spectrum_p10": float(np.percentile(scores, 10)),
        "spectra_scored": float(len(scores)),
        "spectra_unscorable": float(unscorable),
    }


def build_denoising_model(
    pretrained_path: str, model_args: DenoiseModelArguments
) -> tuple[IonaForDenoising, IonaForPreTraining]:
    """Lift the encoder out of a pretraining checkpoint and attach a fresh classifier."""
    if model_args.random_init:
        # Same architecture, no pretrained weights.
        encoder_config = IonaConfig.from_pretrained(pretrained_path)
        pretrained = IonaForPreTraining(encoder_config)
    else:
        pretrained = IonaForPreTraining.from_pretrained(pretrained_path)
    config = IonaDenoisingConfig(
        encoder=copy.deepcopy(pretrained.config if isinstance(pretrained.config, IonaConfig)
                              else IonaConfig(**pretrained.config.to_dict())),
        head_hidden_size=model_args.head_hidden_size,
        head_dropout=model_args.head_dropout,
    )
    # Full fine-tune; temporary freezing is handled by the trainer schedule.
    model = IonaForDenoising(config, encoder=pretrained.iona, freeze_encoder=False)
    # The mask token never receives a gradient here; freeze it so DDP does not complain
    # about unused parameters.
    model.iona.embed.mask_token.requires_grad_(False)
    return model, pretrained


class DenoiseFinetuneTrainer(DenoisingTrainer):
    """Separate encoder/head learning rates and an optional encoder freeze."""

    def __init__(self, *args, encoder_lr_scale: float = 1.0, freeze_encoder_steps: int = 0,
                 use_peak_budget_batching: bool = False, **kwargs):
        self.use_peak_budget_batching = use_peak_budget_batching
        super().__init__(*args, **kwargs)
        self.encoder_lr_scale = encoder_lr_scale
        self.freeze_encoder_steps = freeze_encoder_steps
    def create_optimizer(self):
        """Encoder and head parameter groups, with the encoder LR scaled."""
        if self.optimizer is not None:
            return self.optimizer
        optimizer_class, kwargs = type(self).get_optimizer_cls_and_kwargs(self.args, self.model)
        kwargs.pop("lr", None)
        encoder, head = [], []
        for name, parameter in self.model.named_parameters():
            (encoder if name.startswith("iona.") else head).append(parameter)
        self.optimizer = optimizer_class(
            [
                {"params": encoder, "lr": self.args.learning_rate * self.encoder_lr_scale},
                {"params": head, "lr": self.args.learning_rate},
            ],
            **kwargs,
        )
        return self.optimizer

    def get_train_dataloader(self):
        """Fixed-size batches unless use_peak_budget_batching is set."""
        if self.use_peak_budget_batching:
            return super().get_train_dataloader()
        return Trainer.get_train_dataloader(self)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Gate the encoder LR to zero for the first freeze_encoder_steps."""
        scheduler = super().create_scheduler(num_training_steps, optimizer)
        if self.freeze_encoder_steps <= 0:
            return scheduler
        if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            raise TypeError(
                f"encoder freezing needs a LambdaLR to gate per group, got {type(scheduler).__name__}"
            )
        groups = (optimizer or self.optimizer).param_groups
        if len(groups) != 2:
            raise ValueError(f"expected encoder and head groups, got {len(groups)}")

        base = scheduler.lr_lambdas[1]
        freeze = self.freeze_encoder_steps
        # Group 0 is the encoder.
        scheduler.lr_lambdas = [
            lambda step, shape=base: 0.0 if step < freeze else shape(step),
            base,
        ]
        return scheduler


def subset_splits(datasets: dict, max_samples: int, process_index: int = 0) -> dict:
    """Cap every split at max_samples rows (0 = no cap)."""
    if max_samples <= 0:
        return datasets
    capped = {name: split.select(range(min(max_samples, len(split))))
              for name, split in datasets.items() if split is not None}
    if process_index == 0:
        print("[subset] " + " ".join(f"{k}={len(v):,}" for k, v in capped.items()),
              flush=True)
    return capped


def main(argv: list[str] | None = None) -> int:
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

    processor = IonaProcessor.from_pretrained(
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
                "random_init": model_args.random_init,
                "encoder": pretrained.config.to_dict(),
                "data": asdict(data_args),
                "training": training_args.to_dict(),
            },
        )

    try:
        if training_args.process_index == 0:
            total = sum(p.numel() for p in model.parameters())
            head = sum(p.numel() for p in model.denoising_head.parameters())
            origin = "RANDOM INIT (control)" if model_args.random_init else model_args.pretrained_path
            print(f"[denoise] {total / 1e6:.2f}M params ({head / 1e6:.3f}M in the head)", flush=True)
            print(f"[denoise] encoder from: {origin}", flush=True)
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
        datasets = subset_splits(datasets, data_args.max_samples,
                                 training_args.process_index)
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
            use_peak_budget_batching=training_args.use_peak_budget_batching,
        )
        # Trainer.train() does not read args.resume_from_checkpoint on its own.
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

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
    except BaseException:
        # Mark the W&B run as failed rather than finished.
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
            wandb_run = None
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    sys.exit(main())
