"""Pretraining entrypoint: m/z denoising autoencoder with Δm/z-biased transformer.

Training runs on the Hugging Face `Trainer` with optional DeepSpeed. The Trainer
owns the things that used to be hand-rolled here — distributed launch, AdamW, the
cosine schedule, bf16 autocast, gradient clipping, checkpoint save/resume, and
wandb logging. Data is a Hugging Face `datasets.Dataset` with preprocessing
precomputed by `.map` (see `build_pretraining_datasets`), so the Trainer's native
sampling/sharding/eval apply and no per-spectrum transform runs in the loop.
Representation probes and retrieval use the same prepared, compiled model on
every rank. Rank-zero callbacks only inspect parameters and render plots.

The config schema and parsing live in `config.py` (`ModelArgs`/`DataArgs`/
`TrainArgs`/`LogArgs`, parsed by `HfArgumentParser`): a YAML file supplies the
base values and any remaining command-line flags override them, which is exactly
how a `wandb agent` injects a sweep (it appends `--lr=... --mask_ratio=...` via
the sweep's `${args}`).

Single GPU / dev (HF Trainer, no launcher needed):
    msdelta-train --config configs/v14_cap_S.yaml
    msdelta-train --config configs/v14_cap_S.yaml --lr 2e-4          # CLI override
Multi-GPU with DeepSpeed (set `deepspeed: true` in the config; torchrun stands
up the process group — the `deepspeed` launcher can't run a package's `-m`
entrypoint given the relative imports):
    torchrun --standalone --nproc_per_node=4 -m msdelta.train --config configs/massivekb_xl.yaml
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import wandb

from transformers import Trainer

from .callbacks import build_callbacks
from .config import build_training_arguments, parse_config
from .data import (
    MaskIntensityCollator,
    build_pretraining_datasets,
    resolve_dataset_paths,
)
from .evaluation import DiagnosticCorpora, DistributedDiagnosticRunner
from .model import MSDeltaForPretraining
from .retrieval import (
    build_external_retrieval_dataset,
    build_internal_retrieval_dataset,
)
from .viz import render_bias_panels


class MSDeltaTrainer(Trainer):
    """Trainer that exempts the learnable Fourier frequencies from weight decay.

    Stock HF only exempts LayerNorm/bias params from AdamW's decay. The Fourier
    `freqs` are a frequency *scale*, not a weight — decaying them shrinks every
    frequency toward 0 (flattening the encoding), so we drop any `.freqs`
    parameter from the decay group and it lands in the weight_decay=0.0 group.
    """

    def get_decay_parameter_names(self, model):
        return [n for n in super().get_decay_parameter_names(model)
                if not n.endswith(".freqs")]

    diagnostic_runner: DistributedDiagnosticRunner | None = None

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        if self.diagnostic_runner is None:
            return metrics
        diagnostic_metrics = self.diagnostic_runner.run()
        if self.is_world_process_zero() and diagnostic_metrics:
            self.log(diagnostic_metrics)
            metrics.update(diagnostic_metrics)
        return metrics


# ---------- training ----------

def main(argv: list[str] | None = None) -> int:
    cli, margs, dargs, targs, largs = parse_config(argv)

    run_name = largs.wandb_run_name or cli.config.stem
    out_dir = Path(largs.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(exist_ok=True)

    # wandb — the HF integration reads WANDB_PROJECT and TrainingArguments.run_name.
    report_to: list[str] = []
    if largs.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", largs.wandb_project)
        os.environ.setdefault("WANDB_DIR", str(out_dir))
        report_to = ["wandb"]

    training_args = build_training_arguments(targs, largs, out_dir, run_name, report_to)

    # Model — placement + precision are the Trainer/DeepSpeed engine's job.
    model = MSDeltaForPretraining(margs.to_model_config())
    if training_args.local_process_index == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {n_params/1e6:.2f}M params", flush=True)

    # Only global rank 0 performs the expensive map/filter. Other ranks wait,
    # then execute the same calls and immediately load rank 0's completed Arrow
    # cache. This requires the ranks to share the HF/datasets cache, as they do
    # on a single multi-GPU node (and on multi-node jobs with shared storage).
    with training_args.main_process_first(local=False, desc="dataset preprocessing"):
        if training_args.process_index == 0:
            print(
                f"[data] preprocessing with {dargs.preprocess_num_workers} CPU workers",
                flush=True,
            )
        train_paths, val_paths = resolve_dataset_paths(dargs.to_source_dict())
        train_ds, val_ds = build_pretraining_datasets(
            train_paths, val_paths, dargs.preprocess(),
            num_proc=dargs.preprocess_num_workers or None,
        )
    eval_size = targs.val_batches * targs.batch_size
    eval_ds = val_ds.select(range(min(len(val_ds), eval_size)))

    trainer = MSDeltaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=MaskIntensityCollator(mask_ratio=dargs.mask_ratio),
    )
    # The model always returns a loss but exposes no label columns for HF to
    # detect, so the eval loop would otherwise skip loss and report no eval_loss
    # (our val KL). Force it on.
    trainer.can_return_loss = True
    calibration_size = min(len(train_ds), targs.val_batches * targs.batch_size)
    probe_size = min(len(val_ds), largs.probe_n_spectra)
    calibration_ds = train_ds.select(range(calibration_size)).add_column(
        "_eval_id", list(range(calibration_size))
    )
    probe_ds = val_ds.select(range(probe_size)).add_column(
        "_eval_id", list(range(probe_size))
    )
    internal_retrieval_ds, internal_binned = build_internal_retrieval_dataset(val_ds)
    external_retrieval_ds = None
    if largs.replicate_retrieval_repo:
        with training_args.main_process_first(
            local=False, desc="retrieval benchmark preprocessing"
        ):
            external_retrieval_ds = build_external_retrieval_dataset(
                largs.replicate_retrieval_repo, dargs.preprocess()
            )
    trainer.diagnostic_runner = DistributedDiagnosticRunner(
        trainer,
        DiagnosticCorpora(
            calibration=calibration_ds,
            probes=probe_ds,
            internal_retrieval=internal_retrieval_ds,
            internal_binned=internal_binned,
            external_retrieval=external_retrieval_ds,
        ),
        max_peaks=margs.max_peaks,
        batch_size=targs.batch_size,
        num_workers=targs.num_workers,
    )
    resolved = {**asdict(margs), **asdict(dargs), **asdict(targs), **asdict(largs)}
    for cb in build_callbacks(model, val_ds, largs, resolved, out_dir):
        trainer.add_callback(cb)

    trainer.train(resume_from_checkpoint=str(cli.resume) if cli.resume else None)

    # Refit the deployment transform against the final encoder weights even if
    # the last training step was not an evaluation boundary.
    final_diagnostics = trainer.diagnostic_runner.run()
    if trainer.is_world_process_zero() and final_diagnostics:
        trainer.log(final_diagnostics)

    # Final artifacts — main process writes the model + a last bias-curve render.
    if trainer.is_world_process_zero():
        trainer.save_model(str(out_dir / "final"))
        transform_path = out_dir / "final" / "retrieval_transform.joblib"
        trainer.diagnostic_runner.save_transform(transform_path)
        if wandb.run is not None:
            artifact = wandb.Artifact(
                f"{run_name}-retrieval-transform", type="retrieval-transform"
            )
            artifact.add_file(str(transform_path))
            wandb.log_artifact(artifact)
        panels = render_bias_panels(model.encoder.bias_module, trainer.state.global_step)
        for name, fig in panels.items():
            fig.savefig(out_dir / "figs" / f"{name.replace('/', '_')}_final.png", dpi=110)
            plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
