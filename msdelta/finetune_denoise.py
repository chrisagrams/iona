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
from transformers import (DataCollatorWithPadding, HfArgumentParser, TrainerCallback,
                          TrainingArguments, set_seed)

from msdelta.configuration_msdelta import MSDeltaConfig, MSDeltaDenoisingConfig
from msdelta.data import build_denoising_datasets
from transformers import Trainer
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
    random_init: bool = field(
        default=False,
        metadata={
            "help": (
                "Take the ARCHITECTURE from `pretrained_path` but discard its weights and "
                "start from a fresh initialisation. The control the fine-tuned numbers "
                "need: if a randomly initialised encoder reaches the same score, "
                "pretraining contributed nothing to this task and the head is simply "
                "learning it from the labels."
            )
        },
    )
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
    max_samples: int = field(
        default=0,
        metadata={"help": "Keep at most this many rows per split (0 = all). For smoke "
                          "tests: a small dataset lets a run finish REAL epochs quickly, "
                          "so saving, save_total_limit rotation, load_best_model_at_end, "
                          "the test split and the final save all actually execute. "
                          "--max_steps skips every one of those."},
    )
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

    use_peak_budget_batching: bool = field(
        default=False,
        metadata={
            "help": (
                "Build batches to a padded-attention budget instead of a fixed count. "
                "Off by default, matching pretraining, which uses a plain fixed batch "
                "and is the only configuration on this codebase proven to all-reduce a "
                "full encoder under DDP. Budget batching gives every rank a different "
                "sequence length, so per-rank memory spikes differ -- untested territory "
                "for the reducer."
            )
        },
    )
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
    run_description: str | None = field(
        default=None,
        metadata={
            "help": (
                "One line saying what this run is FOR. Written to the W&B notes and to "
                "RUN.md beside the checkpoints. A run name encodes settings but not "
                "intent, and six months from now 'lr2e4_es10_ep4_h512' will not say "
                "whether it was a grid point, a control, or a debugging attempt."
            )
        },
    )
    eval_test_split: bool = field(
        default=True,
        metadata={"help": "Score the held-out test split once training finishes."},
    )


def select_device() -> None:
    """Bind this process to its tile under either ZE_AFFINITY_MASK convention.

    ZE_AFFINITY_MASK filters which tiles a process can see AND renumbers the survivors
    from zero, so the correct device index depends on how the launcher set it:

      job-wide mask   every rank sees all N tiles -> set_device(LOCAL_RANK)
      per-rank mask   each rank sees exactly one  -> set_device(0)

    Both are legitimate; the second is what a one-tile-per-arm sweep needs. Mixing them
    is what killed job 8839150 ("device index out of range... got 7" against a one-device
    world). Rather than encode a convention here, infer it -- and refuse anything that is
    neither, because a partial mask would otherwise silently collide two ranks onto one
    tile and corrupt both.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0 or not torch.xpu.is_available():
        return
    visible = torch.xpu.device_count()
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if visible == 1:
        torch.xpu.set_device(0)
    elif visible >= local_world:
        torch.xpu.set_device(local_rank)
    else:
        raise RuntimeError(
            f"{visible} visible tiles for {local_world} local ranks: neither a per-rank "
            "mask (1 tile) nor a job-wide mask (>= one tile per rank). Ranks would share "
            "a tile."
        )


def describe_run(model_args, data_args, training_args) -> tuple[str, list[str]]:
    """A human sentence and machine tags for one run, derived from its own settings.

    Generated rather than hand-written so that nothing can be left undescribed: a 72-arm
    sweep will not get 72 hand-written notes, and the arms that go undescribed are
    exactly the ones nobody remembers later. An explicit --run_description is appended
    when given, since intent is the one thing the settings cannot supply.
    """
    if model_args.random_init:
        origin = "randomly initialised encoder (CONTROL: no pretraining)"
    else:
        origin = f"encoder from {Path(model_args.pretrained_path).parent.name}"

    if model_args.encoder_lr_scale == 0:
        encoder = "encoder frozen throughout"
    else:
        encoder = (f"encoder at {model_args.encoder_lr_scale:g}x the head's rate"
                   f"{f', frozen for {model_args.freeze_encoder_steps} steps' if model_args.freeze_encoder_steps else ''}")

    sentence = (
        f"Per-peak noise classification (noise = positive class) on "
        f"{data_args.dataset_repo}, max_peaks={data_args.max_peaks}. {origin}. "
        f"lr={training_args.learning_rate:g}, {encoder}, "
        f"{training_args.num_train_epochs:g} epochs, head width "
        f"{model_args.head_hidden_size}, seed {training_args.seed}."
    )
    if training_args.run_description:
        sentence = f"{training_args.run_description} -- {sentence}"

    tags = [
        "denoise",
        "scratch" if model_args.random_init else "pretrained",
        f"lr{training_args.learning_rate:g}",
        f"els{model_args.encoder_lr_scale:g}",
        f"ep{training_args.num_train_epochs:g}",
        f"head{model_args.head_hidden_size}",
        f"seed{training_args.seed}",
        f"peaks{data_args.max_peaks}",
    ]
    if model_args.encoder_lr_scale == 0 and not model_args.random_init:
        tags.append("frozen-encoder")
    return sentence, tags


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

    # Keep the 2-D form before flattening: the head squeezes to (spectra, peaks), so
    # spectrum boundaries are present here and reshape(-1) is the only thing that
    # destroys them. per_spectrum_auroc needs them.
    logits_2d = np.asarray(prediction.predictions, dtype=np.float64)
    labels_2d = np.asarray(prediction.label_ids, dtype=np.float64)

    logits = logits_2d.reshape(-1)
    labels = labels_2d.reshape(-1)
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
    metrics.update(per_spectrum_auroc(logits_2d, labels_2d))
    return metrics


def per_spectrum_auroc(logits_2d, labels_2d) -> dict[str, float]:
    """AUROC computed WITHIN each spectrum, then averaged.

    The pooled `auroc` asks: take a random noise peak and a random signal peak from
    anywhere in the test set -- is the noise one ranked higher? Those two peaks usually
    come from DIFFERENT spectra. The task asks something narrower: given one spectrum,
    which of ITS peaks are noise. Every comparison that matters is within a spectrum.

    Those come apart whenever the model carries a per-spectrum offset. If it can tell
    that a spectrum is noisy overall -- plausible, the encoder sees the whole spectrum --
    it can shift all of that spectrum's logits up. Noisy spectra contribute more noise
    peaks, so the shift lands the right way round in the pooled ranking and INFLATES
    pooled AUROC while doing nothing for within-spectrum discrimination.

    This is not hypothetical in this project. The reranking embedding scored AUROC 0.846
    pooled over all (spectrum, candidate) pairs and COST 0.109 hit@1, because hit@1
    ranks within a spectrum and the errors were correlated inside each one. Same gap
    between a pooled metric and a within-group one, and trusting the pooled side is what
    went wrong.

    Spectra that are entirely one class cannot be scored and are counted, not silently
    dropped: if most spectra are unscorable the average is about a biased minority.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    if logits_2d.ndim != 2:
        # A single-spectrum eval, or an upstream change to the output shape. Say so
        # rather than reporting a number computed over the wrong axis.
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
        # The spread says whether the mean describes the population or hides a split
        # between spectra the model handles and spectra it does not.
        "auroc_per_spectrum_sd": float(np.std(scores)),
        "auroc_per_spectrum_p10": float(np.percentile(scores, 10)),
        "spectra_scored": float(len(scores)),
        "spectra_unscorable": float(unscorable),
    }


def build_denoising_model(
    pretrained_path: str, model_args: DenoiseModelArguments
) -> tuple[MSDeltaForDenoising, MSDeltaForPreTraining]:
    """Lift the encoder out of a pretraining checkpoint and attach a fresh classifier."""
    if model_args.random_init:
        # Same architecture, no pretrained weights. from_config rather than
        # from_pretrained so the checkpoint's tensors are never read at all -- loading
        # then re-initialising would leave any buffer the init does not touch carrying
        # pretrained values, which is a subtler thing to be wrong about than it looks.
        encoder_config = MSDeltaConfig.from_pretrained(pretrained_path)
        pretrained = MSDeltaForPreTraining(encoder_config)
    else:
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
    # The mask token is only read when mask_positions is supplied, and denoising never
    # supplies it, so it can never receive a gradient. Left trainable it makes DDP abort
    # with "parameters that were not used in producing loss" (job 8839946, param index 0).
    # Freezing it before the DDP wrapper is built keeps it out of the reducer entirely,
    # which is cheaper and more honest than find_unused_parameters=True.
    model.msdelta.embed.mask_token.requires_grad_(False)
    return model, pretrained


class MemoryProbe(TrainerCallback):
    """Log peak device memory every N steps.

    Added because three successive diagnoses of a GPU page fault were reasoned from
    first principles and all three were wrong. A page fault is an illegal access, not a
    clean allocation failure, so it does not say whether memory was the cause -- but
    peak-vs-capacity does, and it costs one number per interval to know.
    """

    def __init__(self, every: int = 10):
        self.every = every
        self.peak = 0.0

    def on_step_end(self, args, state, control, **kwargs):
        if not torch.xpu.is_available():
            return
        peak = torch.xpu.max_memory_allocated() / 1e9
        self.peak = max(self.peak, peak)
        if state.global_step % self.every == 0 and args.process_index == 0:
            total = torch.xpu.get_device_properties(torch.xpu.current_device()).total_memory / 1e9
            print(f"[mem] step {state.global_step}: peak {peak:.2f} GB "
                  f"reserved {torch.xpu.memory_reserved()/1e9:.2f} GB "
                  f"of {total:.1f} GB", flush=True)


class DenoiseFinetuneTrainer(DenoisingTrainer):
    """Split learning rates and a gated encoder; batching follows pretraining by default."""

    def __init__(self, *args, encoder_lr_scale: float = 1.0, freeze_encoder_steps: int = 0,
                 use_peak_budget_batching: bool = False, **kwargs):
        self.use_peak_budget_batching = use_peak_budget_batching
        super().__init__(*args, **kwargs)
        self.encoder_lr_scale = encoder_lr_scale
        self.freeze_encoder_steps = freeze_encoder_steps
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

    def get_train_dataloader(self):
        """Plain fixed-size batches unless budget batching is asked for explicitly.

        DenoisingTrainer's override builds variable-sized batches and sets
        accelerator.even_batches=False. That is right for a frozen-encoder probe, where
        no activations are retained and ranks cannot drift. With a trainable encoder it
        gives each rank a different sequence length and therefore a different memory
        profile, which is not how any working DDP run on this codebase is configured.
        """
        if self.use_peak_budget_batching:
            return super().get_train_dataloader()
        return Trainer.get_train_dataloader(self)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Freeze the encoder with a per-group LR multiplier, not by touching gradients.

        `LambdaLR` accepts one lambda per parameter group and sets each group's rate to
        `base_lr * lambda(step)`. It writes only `param_group["lr"]`, so unlike the two
        mechanisms this replaces it cannot interfere with DDP: toggling `requires_grad`
        breaks the reducer, which is built once at wrap time, and zeroing `.grad` after
        backward writes into the flat all-reduce bucket DDP owns -- that took GPU page
        faults on a write in jobs 8840007/8/10/11.

        The head keeps the schedule HF built. The encoder gets the same schedule gated to
        zero for the first `freeze_encoder_steps`, so its rate rejoins the normal curve
        the moment the gate opens rather than restarting a warmup of its own.
        """
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

        base = scheduler.lr_lambdas[1]          # the shape HF built, unmodified
        freeze = self.freeze_encoder_steps
        # Group 0 is the encoder; create_optimizer builds it first.
        scheduler.lr_lambdas = [
            lambda step, shape=base: 0.0 if step < freeze else shape(step),
            base,
        ]
        return scheduler



def load_description(argv: list[str] | None = None) -> str | None:
    """Human intent for a run, read from DESCRIPTION.md beside its args file.

    Not a --run_description in the args file itself: HfArgumentParser reads those with
    read_text().split(), so any value containing a space becomes several stray positional
    arguments. A sibling file has no such limit, sits with the settings it describes, and
    is visible in a diff when someone changes what an experiment is for.

    describe_run() already derives a sentence from the settings, which is what guarantees
    no run is undescribed. This supplies the one thing settings cannot: WHY the run exists.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    for flag, value in zip(argv, argv[1:]):
        if flag == "--args_file":
            path = Path(value).parent / "DESCRIPTION.md"
            if path.exists():
                return " ".join(path.read_text().split())
    return None


def subset_splits(datasets: dict, max_samples: int, process_index: int = 0) -> dict:
    """Cap every split, for smoke tests that must still run the whole pipeline.

    Capping ROWS rather than steps is the point. `--max_steps` stops training early, so
    the end-of-training machinery -- checkpoint writes, save_total_limit rotation,
    load_best_model_at_end, the test pass, the final save -- never runs, and a debug job
    reports success having exercised none of it. A small dataset instead lets the run
    finish real epochs in seconds and touch every one of those paths.
    """
    if max_samples <= 0:
        return datasets
    capped = {name: split.select(range(min(max_samples, len(split))))
              for name, split in datasets.items() if split is not None}
    if process_index == 0:
        print("[subset] " + " ".join(f"{k}={len(v):,}" for k, v in capped.items()),
              flush=True)
    return capped


def main(argv: list[str] | None = None) -> int:
    select_device()

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

    training_args.run_description = training_args.run_description or load_description()
    description, tags = describe_run(model_args, data_args, training_args)
    if training_args.process_index == 0:
        # Also on disk: a checkpoint directory found later should explain itself without
        # needing W&B access or the job log.
        (out_dir / "RUN.md").write_text(
            f"# {training_args.run_name}\n\n{description}\n\n"
            f"tags: {', '.join(tags)}\n"
        )
        print(f"[denoise] {description}", flush=True)

    wandb_run = None
    if training_args.wandb_project:
        wandb_run = init_wandb_run(
            project=training_args.wandb_project,
            run_name=training_args.run_name,
            entity=training_args.wandb_entity,
            notes=description,
            tags=tags,
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
        trainer.add_callback(MemoryProbe(every=10))
        # Pass it explicitly. Trainer.train() defaults resume_from_checkpoint to None
        # and never falls back to args.resume_from_checkpoint, so the CLI flag parses
        # cleanly and is then IGNORED -- the run restarts from scratch while looking as
        # though it resumed. See pbs/aurora-finetune-sweep.pbs RESUME_JOB.
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # FT31: a run whose sampler yields no batch "finishes" at step 0 and then writes
        # final/ and metrics exactly like a real one; the C2/C4 smoke 8859890 reported
        # 14/14 ok that way. Fail loudly instead.
        if trainer.state.global_step == 0:
            raise SystemExit("trained 0 optimizer steps -- too few groups/rows for one batch?")

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
        # W&B marks a run "crashed" by missing heartbeat, so a process that dies outright
        # is labelled correctly -- but an exception caught here would reach the bare
        # finish() below and stamp the run "finished". At sweep scale the run list is the
        # index, and an arm that died at step 500 must not sit beside a completed one
        # carrying plausible partial metrics.
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
            wandb_run = None
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    sys.exit(main())
