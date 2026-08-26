"""Train the masked-intensity model."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from transformers import Trainer

from msdelta.callbacks import build_callbacks
from msdelta.config import build_training_arguments, parse_config
from msdelta.data import (
    MaskIntensityCollator,
    build_pretraining_datasets,
    resolve_dataset_paths,
)
from msdelta.model import IntensityHead, ModelConfig, MSEncoder
from msdelta.viz import render_bias_panels


class MSDeltaForPretraining(nn.Module):
    """Combine the encoder and intensity head for Trainer."""

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
    """Exclude Fourier frequencies from weight decay."""

    def get_decay_parameter_names(self, model):
        return [n for n in super().get_decay_parameter_names(model) if not n.endswith(".freqs")]


def main(argv: list[str] | None = None) -> int:
    cli, margs, dargs, targs, largs = parse_config(argv)

    run_name = largs.wandb_run_name or cli.config.stem
    out_dir = Path(largs.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(exist_ok=True)

    report_to: list[str] = []
    if largs.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", largs.wandb_project)
        os.environ.setdefault("WANDB_DIR", str(out_dir))
        report_to = ["wandb"]

    training_args = build_training_arguments(targs, largs, out_dir, run_name, report_to)

    model = MSDeltaForPretraining(margs.to_model_config())
    if training_args.local_process_index == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {n_params / 1e6:.2f}M params", flush=True)

    # Create the shared dataset cache on rank 0.
    with training_args.main_process_first(local=False, desc="dataset preprocessing"):
        if training_args.process_index == 0:
            print(
                f"[data] preprocessing with {dargs.preprocess_num_workers} CPU workers",
                flush=True,
            )
        train_paths, val_paths = resolve_dataset_paths(dargs.to_source_dict())
        train_ds, val_ds = build_pretraining_datasets(
            train_paths,
            val_paths,
            dargs.preprocess(),
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
    # Force Trainer to report the validation loss.
    trainer.can_return_loss = True
    resolved = {**asdict(margs), **asdict(dargs), **asdict(targs), **asdict(largs)}
    for cb in build_callbacks(model, val_ds, dargs.preprocess(), largs, resolved, out_dir):
        trainer.add_callback(cb)

    trainer.train(resume_from_checkpoint=str(cli.resume) if cli.resume else None)

    if trainer.is_world_process_zero():
        trainer.save_model(str(out_dir / "final"))
        panels = render_bias_panels(model.encoder.bias_module, trainer.state.global_step)
        for name, fig in panels.items():
            fig.savefig(out_dir / "figs" / f"{name.replace('/', '_')}_final.png", dpi=110)
            plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
