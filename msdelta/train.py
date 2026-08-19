"""Pretraining entrypoint: m/z denoising autoencoder with Δm/z-biased transformer.

Training runs on the Hugging Face `Trainer` with optional DeepSpeed. The Trainer
owns the things that used to be hand-rolled here — distributed launch, AdamW, the
cosine schedule, bf16 autocast, gradient clipping, checkpoint save/resume, and
wandb logging. Data is a Hugging Face `datasets.Dataset` with preprocessing
precomputed by `.map` (see `build_pretraining_datasets`), so the Trainer's native
sampling/sharding/eval apply and no per-spectrum transform runs in the loop. What
stays project-specific is the inline science — frozen-encoder probes, Δm
bias-curve alignment, retrieval, and the bias-curve/attention-entropy panels,
each its own callback in `callbacks.py`.

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
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from transformers import Trainer

from .callbacks import build_callbacks
from .config import build_training_arguments, parse_config
from .data import (
    MaskIntensityCollator,
    build_pretraining_datasets,
    resolve_dataset_paths,
)
from .model import IntensityHead, MSEncoder, ModelConfig
from .viz import render_bias_panels


# ---------- model ----------

class MSDeltaForPretraining(nn.Module):
    """Encoder + intensity head as one `Trainer`-compatible module.

    `forward` consumes the collated batch keys as keyword args and returns
    ``{"loss", "kl"}`` so `Trainer.compute_loss` reads ``outputs["loss"]``
    directly (loss == the masked-intensity KL). The submodules stay plain
    `MSEncoder`/`IntensityHead`, so the probe/panel code reaches ``.encoder``
    and runs its own forwards unchanged.
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.encoder = MSEncoder(model_cfg)
        self.heads = IntensityHead(model_cfg.d_model)

    def forward(
        self,
        mz: torch.Tensor,
        log_int: torch.Tensor,
        key_padding_mask: torch.Tensor,
        mask_positions: torch.Tensor,
        intensity_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = self.encoder(mz, log_int, key_padding_mask, mask_positions)
        loss, parts = self.heads.loss(tokens, intensity_prob, mask_positions)
        return {"loss": loss, "kl": parts["kl"]}


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_counts = {"spectra": 0, "peaks": 0, "masked": 0}
        self._eval_counts = {"spectra": 0, "masked": 0}
        self._metrics_started = time.monotonic()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        counts = self._train_counts if model.training else self._eval_counts
        counts["spectra"] += int(inputs["mz"].shape[0])
        counts["masked"] += int(inputs["mask_positions"].sum().item())
        if model.training:
            counts["peaks"] += int((~inputs["key_padding_mask"]).sum().item())
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def log(self, logs, start_time=None):
        elapsed = max(time.monotonic() - self._metrics_started, 1e-9)
        local = torch.tensor(
            [self._train_counts["spectra"], self._train_counts["peaks"],
             self._train_counts["masked"]],
            dtype=torch.long, device=self.args.device,
        )
        spectra, peaks, masked = self.accelerator.reduce(local, reduction="sum").tolist()
        logs = {
            **logs,
            "training_spectra_seen": spectra,
            "training_peak_tokens_seen": peaks,
            "masked_peaks_seen": masked,
            "cumulative_estimated_flops": self.state.total_flos,
            "wall_clock_time": elapsed,
            "spectra_per_second": spectra / elapsed,
            "peak_tokens_per_second": peaks / elapsed,
        }
        if "loss" in logs:
            logs["training_loss"] = logs["loss"]
        if "eval_loss" in logs:
            logs["validation_loss"] = logs["eval_loss"]
        if hasattr(self, "parameter_count"):
            logs["parameter_count"] = self.parameter_count
        return super().log(logs, start_time=start_time)

    def evaluate(self, *args, **kwargs):
        self._eval_counts = {"spectra": 0, "masked": 0}
        metrics = super().evaluate(*args, **kwargs)
        prefix = kwargs.get("metric_key_prefix", "eval")
        loss_key = f"{prefix}_loss"
        local = torch.tensor(
            [self._eval_counts["spectra"], self._eval_counts["masked"]],
            dtype=torch.long, device=self.args.device,
        )
        eval_spectra, eval_masked = self.accelerator.reduce(local, reduction="sum").tolist()
        if loss_key in metrics and eval_masked:
            # Existing KL is batch-mean (per spectrum). Convert its accumulated
            # numerator to the required held-out per-masked-peak comparison.
            per_masked = (
                metrics[loss_key] * eval_spectra / eval_masked
            )
            metrics[f"{prefix}_loss_per_masked_peak"] = per_masked
            self.log({
                f"{prefix}_loss_per_masked_peak": per_masked,
                "validation_loss_per_masked_peak": per_masked,
            })
        return metrics


def parameter_counts(model: MSDeltaForPretraining, architecture_id: str) -> dict[str, int | str]:
    """Exact trainable-parameter accounting for an ablation run."""
    count = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
    embed = model.encoder.embed
    return {
        "architecture": architecture_id,
        "total_parameters": count(model),
        "encoder_parameters": count(model.encoder),
        "absolute_mz_parameters": (
            count(embed.ff_mz) + count(embed.mz_mlp) if embed.use_absolute_mz else 0
        ),
        "delta_bias_parameters": (
            count(model.encoder.bias_module) if model.encoder.bias_module is not None else 0
        ),
    }


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
    counts = parameter_counts(model, margs.architecture_id)
    if training_args.local_process_index == 0:
        print(f"[model] {counts}", flush=True)
        with open(out_dir / "parameter_counts.json", "w") as f:
            json.dump(counts, f, indent=2)

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
    trainer.parameter_count = counts["total_parameters"]
    # The model always returns a loss but exposes no label columns for HF to
    # detect, so the eval loop would otherwise skip loss and report no eval_loss
    # (our val KL). Force it on.
    trainer.can_return_loss = True
    resolved = {
        **asdict(margs), **asdict(dargs), **asdict(targs), **asdict(largs),
        **counts, "parameter_count": counts["total_parameters"],
    }
    for cb in build_callbacks(model, val_ds, dargs.preprocess(), largs, resolved, out_dir):
        trainer.add_callback(cb)

    trainer.train(resume_from_checkpoint=str(cli.resume) if cli.resume else None)

    # Final artifacts — main process writes the model + a last bias-curve render.
    if trainer.is_world_process_zero():
        trainer.save_model(str(out_dir / "final"))
        if model.encoder.bias_module is not None:
            panels = render_bias_panels(model.encoder.bias_module, trainer.state.global_step)
            for name, fig in panels.items():
                fig.savefig(out_dir / "figs" / f"{name.replace('/', '_')}_final.png", dpi=110)
                plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
