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

    # Datasets: resolve shard paths once (local parquet root or HF dataset), then
    # build the map-style HF datasets with preprocessing precomputed/cached. The
    # full preprocessed val dataset feeds both eval (a capped slice) and the
    # inline probes (which read the same preprocessed rows).
    train_paths, val_paths = resolve_dataset_paths(dargs.to_source_dict())
    train_ds, val_ds = build_pretraining_datasets(
        train_paths, val_paths, dargs.preprocess(),
        num_proc=targs.num_workers or None,
    )
    eval_size = targs.val_batches * targs.batch_size
    eval_ds = val_ds.select(range(min(len(val_ds), eval_size)))

    trainer = Trainer(
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
    resolved = {**asdict(margs), **asdict(dargs), **asdict(targs), **asdict(largs)}
    for cb in build_callbacks(model, val_ds, dargs.preprocess(), largs, resolved, out_dir):
        trainer.add_callback(cb)

    trainer.train(resume_from_checkpoint=str(cli.resume) if cli.resume else None)

    # Final artifacts — main process writes the model + a last bias-curve render.
    if trainer.is_world_process_zero():
        trainer.save_model(str(out_dir / "final"))
        panels = render_bias_panels(model.encoder.bias_module, trainer.state.global_step)
        for name, fig in panels.items():
            fig.savefig(out_dir / "figs" / f"{name.replace('/', '_')}_final.png", dpi=110)
            plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
