"""Score a saved denoise checkpoint on the held-out test split.

    python -m msdelta.eval_checkpoint \
        --args_file configs/sweep-denoise/lr2e4_es05_ep4_h512_b12/training.args \
        --checkpoint /lus/.../sweep-lr2e4_es05_ep4_h512_b12-8840408/checkpoint-29072

Written for two jobs that both need "evaluate weights that already exist":

Recovery. `load_best_model_at_end` fails on arms with many checkpoints -- the ranks
disagree about which one is best, and a rank asking for a checkpoint that
save_total_limit already rotated away raises "Can't find a valid checkpoint". That
happens AFTER training and after every evaluation is logged, so the only casualty is the
test number. Four arms of job 8840408 died exactly there. Re-running them would cost ~79
node-hours; re-evaluating them costs minutes.

Reporting. Once a winner is picked, its test number wants recomputing from the saved
weights without touching the training path.

WHICH WEIGHTS. This scores whatever checkpoint it is pointed at. For a recovered arm
that is the FINAL checkpoint, where a successful arm reports its BEST-VALIDATION one.
The difference is small but it is not zero, so the output says so on every line and
`test_results.json` records `weights_selected_by` -- a recovered number must never sit
in a table beside a best-validation number without the distinction being visible.

W&B is off. These runs are not training runs and must not appear in the sweep project as
if they were; the numbers go to stdout and to disk.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import DataCollatorWithPadding, HfArgumentParser

from msdelta.data import build_denoising_datasets
from msdelta.finetune_denoise import (DenoiseDataArguments, DenoiseFinetuneArguments,
                                      DenoiseFinetuneTrainer, DenoiseModelArguments,
                                      build_denoising_model, denoise_metrics,
                                      select_device, subset_splits)
from msdelta.processing_msdelta import MSDeltaProcessor


@dataclass
class EvalArguments:
    """Where the weights are, and where the verdict goes."""

    checkpoint: str = field(
        metadata={"help": "Directory holding model.safetensors, e.g. a checkpoint-N or "
                          "a saved final/ directory."},
    )
    results_into: str | None = field(
        default=None,
        metadata={"help": "Directory to write test_results.json into. Defaults to the "
                          "checkpoint's parent, which is the arm directory."},
    )
    weights_selected_by: str = field(
        default="final",
        metadata={"help": "Recorded in the output so a recovered number is never "
                          "mistaken for a best-validation one."},
    )


# An arm's training.args is a TRAINING config for twelve tiles. These flags either launch
# machinery this script has no launcher for, or would publish an evaluation as if it were
# a training run. Stripped from the token stream BEFORE parsing rather than unset after:
# TrainingArguments.__post_init__ builds the DeepSpeed plugin and the accelerator state as
# a side effect of parsing, and clearing the attribute afterwards leaves those behind.
STRIP = (
    "--deepspeed",              # single tile, no distributed launcher
    "--report_to",              # an eval is not a run
    "--wandb_project",
    "--wandb_entity",
    "--load_best_model_at_end",  # the failure being recovered from; we name the weights
    "--metric_for_best_model",
    "--greater_is_better",
    "--resume_from_checkpoint",
    "--output_dir",             # supplied by the caller, never the original run's
)


def expand_args_file(argv: list[str]) -> list[str]:
    """Splice --args_file in by hand, dropping the flags an eval must not inherit.

    The files are written as strict --flag value pairs (HfArgumentParser reads them with
    read_text().split(), so a value containing a space would break them anyway), which is
    what makes pairwise stripping safe. Verified against the generated grids.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--args_file" and i + 1 < len(argv):
            tokens = Path(argv[i + 1]).read_text().split()
            if len(tokens) % 2:
                raise SystemExit(f"{argv[i + 1]} is not flag/value pairs")
            for flag, value in zip(tokens[::2], tokens[1::2]):
                if flag not in STRIP:
                    out += [flag, value]
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def load_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    """Load a checkpoint's tensors, refusing anything partial.

    strict=True on purpose. A rotated-away or half-written checkpoint is exactly the
    failure this script exists to clean up after, and silently evaluating a model with
    randomly initialised head weights would produce a plausible-looking number that is
    worse than no number at all.
    """
    safetensors = checkpoint / "model.safetensors"
    binary = checkpoint / "pytorch_model.bin"
    if safetensors.exists():
        from safetensors.torch import load_file
        state = load_file(str(safetensors))
    elif binary.exists():
        state = torch.load(str(binary), map_location="cpu", weights_only=True)
    else:
        raise SystemExit(f"no model weights under {checkpoint}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # mask_token is frozen out of the graph during denoise training and some checkpoints
    # therefore omit it; it is never read on this path. Anything else missing is real.
    missing = [k for k in missing if not k.endswith("embed.mask_token")]
    if missing or unexpected:
        raise SystemExit(f"checkpoint does not match the model\n"
                         f"  missing:    {missing[:8]}\n"
                         f"  unexpected: {unexpected[:8]}")


def main(argv: list[str] | None = None) -> int:
    select_device()

    parser = HfArgumentParser(
        (DenoiseModelArguments, DenoiseDataArguments, DenoiseFinetuneArguments,
         EvalArguments)  # pyright: ignore[reportArgumentType]
    )
    argv = expand_args_file(list(sys.argv[1:] if argv is None else argv))
    model_args, data_args, training_args, eval_args = parser.parse_args_into_dataclasses(
        args=argv
    )
    # Stripped from the file, so the dataclass default ("all") would otherwise
    # reach for W&B. Safe to set here: integrations are read when the Trainer is
    # built, not during __post_init__.
    training_args.report_to = []
    training_args.do_train = False

    checkpoint = Path(eval_args.checkpoint)
    into = Path(eval_args.results_into or checkpoint.parent)
    into.mkdir(parents=True, exist_ok=True)

    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks,
    )
    # random_init would rebuild the encoder from scratch; the checkpoint overwrites every
    # tensor anyway, but reading the pretrained file is wasted work and a scratch arm has
    # no business touching it.
    model, _ = build_denoising_model(model_args.pretrained_path, model_args)
    load_weights(model, checkpoint)

    datasets = build_denoising_datasets(
        data_args.dataset_repo, processor,
        num_proc=data_args.preprocessing_num_workers or None,
    )
    datasets = subset_splits(datasets, data_args.max_samples, training_args.process_index)
    test = datasets.get("test")
    if test is None:
        raise SystemExit(f"{data_args.dataset_repo} has no test split")

    trainer = DenoiseFinetuneTrainer(
        model=model,
        args=training_args,
        eval_dataset=test,
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
    metrics = trainer.evaluate(test, metric_key_prefix="test")
    if not trainer.is_world_process_zero():
        return 0

    metrics["weights_selected_by"] = eval_args.weights_selected_by
    metrics["checkpoint"] = str(checkpoint)
    print(f"[eval] {checkpoint.parent.name} "
          f"({eval_args.weights_selected_by} weights, {checkpoint.name}): "
          f"test_auroc={metrics.get('test_auroc'):.4f} "
          f"test_auprc={metrics.get('test_auprc'):.4f}", flush=True)
    (into / "test_results.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[eval] wrote {into / 'test_results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
