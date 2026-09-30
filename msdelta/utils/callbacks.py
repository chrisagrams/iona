"""Provide Trainer callbacks for model diagnostics."""

from __future__ import annotations

import os

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedType
from torch import nn
from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

import wandb
from msdelta.finetuning.alignment.alignment import alignment_metrics
from msdelta.finetuning.denoise.denoising import run_denoising_probe
from msdelta.pretraining.posttraining import denoise_training_args, retrieval_training_args
from msdelta.pretraining.probe import run_all_probes
from msdelta.eval.retrieval import run_retrieval_probe
from msdelta.pretraining.training_args import MSDeltaTrainingArguments
from msdelta.utils.viz import render_bias_panels

logger = logging.getLogger(__name__)


def is_logarithmic_eval_step(step: int, start_step: int) -> bool:
    """Return whether ``step`` lies on a 1-2-5 logarithmic schedule."""
    if step < start_step:
        return False
    scale = 10 ** (len(str(step)) - 1)
    return step in (scale, 2 * scale, 5 * scale)


class LogarithmicEvalCallback(TrainerCallback):
    """Request evaluation at 1-2-5 steps per decade and at the final step."""

    def __init__(self, start_step: int):
        self.start_step = start_step

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if is_logarithmic_eval_step(step, self.start_step) or step == state.max_steps:
            control.should_evaluate = True
        return control


class EvaluationCacheCallback(TrainerCallback):
    """Release accelerator cache immediately before and after evaluation."""

    def __init__(self, module: nn.Module):
        self.module = module

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters()).device

    def _clear_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        elif self.device.type == "xpu":
            torch.xpu.synchronize(self.device)
            torch.xpu.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            self._clear_cache()
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        self._clear_cache()
        return control


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
        if self.empty_cache_before:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "xpu":
                torch.xpu.empty_cache()
        self.run(step)

    def run(self, step: int) -> None:
        raise NotImplementedError


class LinearProbeCallback(_InlineCallback):
    """Run linear probes on the frozen encoder."""

    empty_cache_before = True

    def __init__(self, module, every, dataset, n_spectra, batch_size):
        super().__init__(module, every, dataset=dataset)
        self.n_spectra = n_spectra
        self.batch_size = batch_size

    def run(self, step):
        m = run_all_probes(
            self.encoder,
            self.dataset,
            self.device,
            n_spectra=self.n_spectra,
            batch_size=self.batch_size,
        )
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
        self.probe_training_args = denoise_training_args(training_args, out_dir)

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
        elif self.device.type == "xpu":
            torch.xpu.empty_cache()
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
        self.probe_training_args = retrieval_training_args(training_args, out_dir)

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
        elif self.device.type == "xpu":
            torch.xpu.empty_cache()
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


class SidecarCallback(TrainerCallback):
    """Launch independent probes at each configured checkpoint interval."""

    def __init__(self, out_dir: Path, resolved: dict):
        self.out_dir = out_dir
        self.resolved = resolved
        self.processes: list[tuple[str, subprocess.Popen]] = []

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not state.is_world_process_zero:
            return
        if not isinstance(args, MSDeltaTrainingArguments):
            raise TypeError("SidecarCallback requires MSDeltaTrainingArguments")
        step = state.global_step
        probes = (
            ("denoise", args.denoise_steps, args.sidecar_denoise_device),
            ("retrieval", args.retrieval_steps, args.sidecar_retrieval_device),
        )
        for kind, every, device in probes:
            if not every or step <= 0 or step % every:
                continue
            try:
                checkpoint = self.out_dir / f"checkpoint-{step}"
                log_dir = self.out_dir / f"{kind}-probes"
                log_dir.mkdir(parents=True, exist_ok=True)
                if device is None or args.sidecar_launcher is None:
                    raise ValueError("Sidecar probes require a device and launcher")
                command = [
                    "bash",
                    args.sidecar_launcher,
                    device,
                    sys.executable,
                    "--checkpoint",
                    str(checkpoint),
                    "--probe",
                    kind,
                    "--step",
                    str(step),
                    "--settings-json",
                    json.dumps(self.resolved),
                    "--device",
                    device.split(":")[0],
                ]
                if wandb.run is not None:
                    command.extend(
                        [
                            "--wandb",
                            "--wandb-run-id",
                            wandb.run.id,
                            "--wandb-project",
                            wandb.run.project,
                            "--wandb-entity",
                            wandb.run.entity,
                        ]
                    )
                with (log_dir / f"step-{step}.log").open("a") as log:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                self.processes.append((kind, process))
                logger.info("Started %s probe for checkpoint %s", kind, step)
            except Exception:
                logger.exception(
                    "Could not launch %s probe at step %s; pretraining continues", kind, step
                )

    def close(self) -> None:
        """Wait for active probes to finish before closing the primary W&B run."""
        for kind, process in self.processes:
            status = process.wait()
            if status:
                logger.warning("%s probe exited with status %s", kind, status)
        self.processes.clear()



class StopAtStepCallback(TrainerCallback):
    """K150-P: stop training at ``stop_step`` while the LR schedule stays defined over ``max_steps``.

    Lets a short run follow a long run's schedule exactly (e.g. the transformer's cosine over 540,423
    steps, stopped at 0.5 epoch) instead of compressing the schedule into the short run. Saves at the stop.
    Enabled only through the environment variable MSDELTA_STOP_AT_STEP (off by default).
    """

    def __init__(self, stop_step: int):
        if stop_step < 1:
            raise ValueError("MSDELTA_STOP_AT_STEP must be a positive integer")
        self.stop_step = stop_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            control.should_save = True
            control.should_training_stop = True
        return control


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
    *,
    include_probes: bool = True,
):
    """Create the callbacks enabled in the configuration."""
    cbs: list[TrainerCallback] = []
    stop_at = os.environ.get("MSDELTA_STOP_AT_STEP")
    if stop_at:
        cbs.append(StopAtStepCallback(int(stop_at)))
    if training_args.logarithmic_eval_start_step is not None:
        cbs.append(LogarithmicEvalCallback(training_args.logarithmic_eval_start_step))
    cbs.append(EvaluationCacheCallback(module))
    if training_args.bias_curve_steps:
        cbs.append(BiasPanelCallback(module, training_args.bias_curve_steps, out_dir=out_dir))
    if training_args.probe_steps:
        cbs.append(
            LinearProbeCallback(
                module,
                training_args.probe_steps,
                val_dataset,
                training_args.probe_num_spectra,
                training_args.probe_batch_size,
            )
        )
        cbs.append(AlignmentCallback(module, training_args.probe_steps))
    if include_probes and training_args.retrieval_steps:
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
    if include_probes and training_args.denoise_steps:
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
