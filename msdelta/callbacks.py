"""Provide Trainer callbacks for model diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import wandb
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedType
from torch import nn
from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainingArguments

from msdelta.alignment import alignment_metrics
from msdelta.denoising import run_denoising_probe
from msdelta.probe import run_all_probes
from msdelta.retrieval import run_retrieval_probe
from msdelta.viz import render_bias_panels


class _InlineCallback(TrainerCallback):
    """Run a diagnostic at a specified interval on the main process."""

    empty_cache_before: bool = False

    def __init__(self, module: nn.Module, every: int, *, dataset=None, out_dir: Path | None = None):
        self.module = module
        self.every = every
        self.dataset = dataset
        self.out_dir = out_dir

    @property
    def encoder(self):
        return self.module.msdelta

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters()).device

    def _wlog(self, payload: dict, step: int) -> None:
        if payload and wandb.run is not None:
            wandb.log({**payload, "train/global_step": step})

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or not self.every:
            return
        step = state.global_step
        if step <= 0 or step % self.every != 0:
            return
        if self.empty_cache_before and self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.run(step)

    def run(self, step: int) -> None:
        raise NotImplementedError


class LinearProbeCallback(_InlineCallback):
    """Run linear probes on the frozen encoder."""

    empty_cache_before = True

    def __init__(self, module, every, dataset, n_spectra):
        super().__init__(module, every, dataset=dataset)
        self.n_spectra = n_spectra

    def run(self, step):
        m = run_all_probes(self.encoder, self.dataset, self.device, n_spectra=self.n_spectra)
        self._wlog(m, step)

        def key(k):
            return m.get(k, float("nan"))

        print(
            f"  probe: precursor_r2={key('probe/precursor_mz_r2'):.3f} "
            f"fragment_mz_r2={key('probe/fragment_mz_r2'):.3f} "
            f"charge_acc={key('probe/charge_acc'):.3f} "
            f"nloss_auc={key('probe/neutral_loss_auc'):.3f} "
            f"iso_f1={key('probe/isotope_f1'):.3f}",
            flush=True,
        )


class AlignmentCallback(_InlineCallback):
    """Measure bias alignment with chemical mass differences."""

    def run(self, step):
        a = alignment_metrics(self.encoder)
        self._wlog(a, step)
        print(
            f"  align: n_sig05={a.get('align/n_sig05', 0):.0f} "
            f"n_sig01_bonf={a.get('align/n_sig01_bonf', 0):.0f} "
            f"best_p={a.get('align/best_p', 1):.1e}",
            flush=True,
        )


class BiasPanelCallback(_InlineCallback):
    """Render and log bias curves."""

    def run(self, step):
        panels = render_bias_panels(self.encoder.bias_module, step)
        payload: dict[str, Any] = {}
        for name, fig in panels.items():
            fig_path = self.out_dir / "figs" / f"{name.replace('/', '_')}_step{step:06d}.png"
            fig.savefig(fig_path, dpi=110)
            plt.close(fig)
            if wandb.run is not None:
                payload[name] = wandb.Image(str(fig_path))
        self._wlog(payload, step)


class DenoisingProbeCallback(TrainerCallback):
    """Post-train a fresh distributed denoising head at fixed intervals."""

    def __init__(self, module, every, datasets, pp, training_args, out_dir):
        self.module = module
        self.every = every
        self.datasets = datasets
        self.pp = pp
        self.training_args = training_args
        self.out_dir = out_dir
        self.last_step = -1
        self.probe_training_args = TrainingArguments(
            output_dir=str(out_dir / "denoise-probes"),
            num_train_epochs=training_args.denoise_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=training_args.denoise_learning_rate,
            weight_decay=training_args.denoise_weight_decay,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_num_workers=training_args.denoise_num_workers,
            bf16=training_args.bf16,
            fp16=training_args.fp16,
            seed=training_args.denoise_seed,
            data_seed=training_args.denoise_seed,
            report_to=[],
            ddp_find_unused_parameters=False,
        )

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters()).device

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if not self.every or step <= 0 or step % self.every or step == self.last_step:
            return
        self.last_step = step
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        destination = self.out_dir / "denoise-probes" / f"step-{step}"
        accelerator_state = AcceleratorState()
        use_named_plugins = (
            accelerator_state.distributed_type == DistributedType.DEEPSPEED
            and isinstance(accelerator_state.deepspeed_plugins, dict)
            and "denoise" in accelerator_state.deepspeed_plugins
        )
        if use_named_plugins:
            accelerator_state.select_deepspeed_plugin("denoise")
        try:
            metrics = run_denoising_probe(
                self.module,
                self.datasets["train"],
                self.datasets["validation"],
                output_dir=destination,
                processor=self.pp,
                peak_pair_budget=self.training_args.denoise_peak_pair_budget,
                hidden_size=self.training_args.denoise_head_hidden_size,
                dropout=self.training_args.denoise_head_dropout,
                training_args=self.probe_training_args,
            )
        finally:
            if use_named_plugins:
                accelerator_state.select_deepspeed_plugin("pretrain")
        if state.is_world_process_zero:
            if wandb.run is not None:
                wandb.log({**metrics, "train/global_step": step})
            tqdm.write(
                f"denoise: AUROC={metrics['denoise/auroc']:.3f} "
                f"AUPRC={metrics['denoise/auprc']:.3f} "
                f"F1={metrics['denoise/f1']:.3f} model={destination}"
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()


class RetrievalProbeCallback(TrainerCallback):
    """Post-train a fresh distributed retrieval head at fixed intervals."""

    def __init__(
        self,
        module,
        every,
        datasets,
        pp,
        training_args,
        out_dir,
        *,
        evaluation_datasets,
    ):
        self.module = module
        self.every = every
        self.datasets = datasets
        self.evaluation_datasets = evaluation_datasets
        self.pp = pp
        self.training_args = training_args
        self.out_dir = out_dir
        self.last_step = -1
        self.probe_training_args = TrainingArguments(
            output_dir=str(out_dir / "retrieval-probes"),
            num_train_epochs=training_args.retrieval_epochs,
            per_device_train_batch_size=training_args.retrieval_per_device_batch_size,
            per_device_eval_batch_size=training_args.retrieval_per_device_batch_size,
            learning_rate=training_args.retrieval_learning_rate,
            weight_decay=training_args.retrieval_weight_decay,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
            label_names=["group_ids"],
            dataloader_num_workers=training_args.retrieval_num_workers,
            dataloader_drop_last=True,
            prediction_loss_only=True,
            bf16=training_args.bf16,
            fp16=training_args.fp16,
            seed=training_args.retrieval_seed,
            data_seed=training_args.retrieval_seed,
            report_to=[],
            ddp_find_unused_parameters=False,
        )

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters()).device

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if not self.every or step <= 0 or step % self.every or step == self.last_step:
            return
        self.last_step = step
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        destination = self.out_dir / "retrieval-probes" / f"step-{step}"
        accelerator_state = AcceleratorState()
        use_named_plugins = (
            accelerator_state.distributed_type == DistributedType.DEEPSPEED
            and isinstance(accelerator_state.deepspeed_plugins, dict)
            and "retrieval" in accelerator_state.deepspeed_plugins
        )
        if use_named_plugins:
            accelerator_state.select_deepspeed_plugin("retrieval")
        try:
            metrics = run_retrieval_probe(
                self.module,
                self.datasets["train"],
                self.datasets["validation"],
                output_dir=destination,
                processor=self.pp,
                evaluation_datasets=self.evaluation_datasets,
                projection_hidden_size=self.training_args.retrieval_projection_hidden_size,
                embedding_size=self.training_args.retrieval_embedding_size,
                dropout=self.training_args.retrieval_head_dropout,
                temperature=self.training_args.retrieval_temperature,
                training_args=self.probe_training_args,
            )
        finally:
            if use_named_plugins:
                accelerator_state.select_deepspeed_plugin("pretrain")
        if state.is_world_process_zero:
            if wandb.run is not None:
                wandb.log({**metrics, "train/global_step": step})
            tqdm.write(
                f"retrieval: loss={metrics['retrieval/loss']:.3f} "
                f"Hit@1={metrics['retrieval/Hit@1']:.3f} "
                f"MAP@100={metrics['retrieval/MAP@100']:.3f} "
                f"R@5={metrics['retrieval/R@5']:.3f} model={destination}"
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()


def build_callbacks(
    module,
    val_dataset,
    pp,
    training_args,
    out_dir,
    denoising_datasets=None,
    denoising_processor=None,
    retrieval_datasets=None,
    retrieval_evaluation_datasets=None,
):
    """Create the callbacks enabled in the configuration."""
    cbs: list[TrainerCallback] = []
    if training_args.bias_curve_steps:
        cbs.append(BiasPanelCallback(module, training_args.bias_curve_steps, out_dir=out_dir))
    if training_args.probe_steps:
        cbs.append(
            LinearProbeCallback(
                module,
                training_args.probe_steps,
                val_dataset,
                training_args.probe_num_spectra,
            )
        )
        cbs.append(AlignmentCallback(module, training_args.probe_steps))
    if training_args.retrieval_steps:
        if retrieval_datasets is None:
            raise ValueError("retrieval datasets are required when retrieval_steps is enabled")
        cbs.append(
            RetrievalProbeCallback(
                module,
                training_args.retrieval_steps,
                retrieval_datasets,
                pp,
                training_args,
                out_dir,
                evaluation_datasets=retrieval_evaluation_datasets,
            )
        )
    if training_args.denoise_steps:
        if denoising_datasets is None:
            raise ValueError("denoising datasets are required when denoise_steps is enabled")
        if denoising_processor is None:
            raise ValueError("denoising processor is required when denoise_steps is enabled")
        cbs.append(
            DenoisingProbeCallback(
                module,
                training_args.denoise_steps,
                denoising_datasets,
                denoising_processor,
                training_args,
                out_dir,
            )
        )
    return cbs
