"""Provide Trainer callbacks for model diagnostics."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn
from tqdm.auto import tqdm
from transformers import TrainerCallback

from msdelta.alignment import alignment_metrics
from msdelta.denoising import run_denoising_probe
from msdelta.fourier import dead_freqs, freq_drift, interp_mae
from msdelta.probe import run_all_probes
from msdelta.retrieval import replicate_retrieval_inline_metrics, retrieval_inline_metrics
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


class FourierProbeCallback(_InlineCallback):
    """Measure learned Fourier frequencies on validation data."""

    empty_cache_before = True

    def __init__(self, module, every, dataset, n_spectra):
        super().__init__(module, every, dataset=dataset)
        self.n_spectra = n_spectra
        self._vals: dict[str, torch.Tensor] | None = None
        self._init_freqs: dict[str, torch.Tensor] = {}

    def _sample_values(
        self, budget: int = 8192, max_spectra: int = 1000, pairs_per_spectrum: int = 64
    ) -> dict[str, torch.Tensor]:
        """Sample intensity and delta m/z values from validation data."""
        g = torch.Generator().manual_seed(0)
        li_pool, dm_pool = [], []
        n = min(self.n_spectra, max_spectra)
        for row in itertools.islice(self.dataset, n):
            mz = torch.as_tensor(row["mz"], dtype=torch.float32)
            li = torch.as_tensor(row["log_intensity"], dtype=torch.float32)
            if li.numel():
                li_pool.append(li)
            if mz.numel() >= 2:
                k = mz.numel()
                idx = torch.randint(0, k, (pairs_per_spectrum, 2), generator=g)
                idx = idx[idx[:, 0] != idx[:, 1]]
                dm_pool.append(mz[idx[:, 0]] - mz[idx[:, 1]])

        def _cat(pool):
            if not pool:
                return torch.empty(0)
            v = torch.cat(pool)
            if v.numel() > budget:
                sel = torch.randperm(v.numel(), generator=g)[:budget]
                v = v[sel]
            return v

        return {"int": _cat(li_pool), "dm": _cat(dm_pool)}

    def _featurizer_metrics(self, name: str, ff, vals: torch.Tensor) -> dict:
        freqs = ff.freqs
        if not isinstance(freqs, nn.Parameter):
            return {}
        if name not in self._init_freqs:
            self._init_freqs[name] = freqs.detach().abs().cpu().clone()
        if vals.numel() < 8:
            return {}
        span = float(vals.max() - vals.min())
        f = freqs.detach().abs().float().cpu()
        m = {
            f"fourier/{name}_mae": interp_mae(freqs, vals),
            f"fourier/{name}_dead": dead_freqs(freqs, span),
            f"fourier/{name}_drift_log10": freq_drift(freqs, self._init_freqs[name]),
            f"fourier/{name}_f_min": float(f.min()),
            f"fourier/{name}_f_max": float(f.max()),
        }
        if wandb.run is not None:
            m[f"fourier/{name}_log10_freqs"] = wandb.Histogram(f.clamp_min(1e-12).log10().numpy())
        return m

    def run(self, step):
        if self._vals is None:
            self._vals = self._sample_values()
        enc = self.encoder
        payload: dict[str, Any] = {}
        payload.update(self._featurizer_metrics("int", enc.embed.ff_int, self._vals["int"]))
        payload.update(self._featurizer_metrics("dm", enc.bias_module.ff, self._vals["dm"]))
        if not payload:
            return
        self._wlog(payload, step)

        def g(k):
            return payload.get(k, float("nan"))

        print(
            f"  fourier: int_mae={g('fourier/int_mae'):.4g} "
            f"int_dead={g('fourier/int_dead'):.0f} "
            f"dm_mae={g('fourier/dm_mae'):.4g} "
            f"dm_dead={g('fourier/dm_dead'):.0f}",
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


class RetrievalCallback(_InlineCallback):
    """Measure spectrum retrieval against a binned baseline."""

    empty_cache_before = True

    def run(self, step):
        r = retrieval_inline_metrics(self.encoder, self.dataset, self.device)
        self._wlog(r, step)
        print(
            f"  retrieval: mAP={r.get('retrieval/mAP', float('nan')):.3f} "
            f"binned={r.get('retrieval/binned_mAP', float('nan')):.3f} "
            f"gap={r.get('retrieval/gap_vs_binned', float('nan')):+.3f}",
            flush=True,
        )


class ReplicateRetrievalCallback(_InlineCallback):
    """Measure retrieval on the external replicate dataset."""

    empty_cache_before = True

    def __init__(self, module, every, pp, repo_id):
        super().__init__(module, every)
        self.pp = pp
        self.repo_id = repo_id

    def run(self, step):
        rr = replicate_retrieval_inline_metrics(self.encoder, self.repo_id, self.device, self.pp)
        if not rr:
            return
        self._wlog(rr, step)
        print(
            f"  replicate-retrieval: "
            f"Hit@1={rr.get('replicate_retrieval/Hit@1', float('nan')):.3f} "
            f"MAP={rr.get('replicate_retrieval/MAP', float('nan')):.3f} "
            f"R@5={rr.get('replicate_retrieval/R@5', float('nan')):.3f}",
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
        metrics = run_denoising_probe(
            self.module,
            self.datasets["train"],
            self.datasets["validation"],
            output_dir=destination,
            processor=self.pp,
            peak_pair_budget=self.training_args.denoise_peak_pair_budget,
            epochs=self.training_args.denoise_epochs,
            learning_rate=self.training_args.denoise_learning_rate,
            weight_decay=self.training_args.denoise_weight_decay,
            hidden_size=self.training_args.denoise_head_hidden_size,
            dropout=self.training_args.denoise_head_dropout,
            num_workers=self.training_args.denoise_num_workers,
            seed=self.training_args.denoise_seed,
            bf16=self.training_args.bf16,
            fp16=self.training_args.fp16,
        )
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


def build_callbacks(
    module,
    val_dataset,
    pp,
    training_args,
    out_dir,
    denoising_datasets=None,
    denoising_processor=None,
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
        cbs.append(
            FourierProbeCallback(
                module,
                training_args.probe_steps,
                val_dataset,
                training_args.probe_num_spectra,
            )
        )
        cbs.append(AlignmentCallback(module, training_args.probe_steps))
        cbs.append(RetrievalCallback(module, training_args.probe_steps, dataset=val_dataset))
        if training_args.replicate_retrieval_repo:
            cbs.append(
                ReplicateRetrievalCallback(
                    module,
                    training_args.probe_steps,
                    pp,
                    training_args.replicate_retrieval_repo,
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
