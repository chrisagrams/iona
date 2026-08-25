"""Trainer callbacks for the inline diagnostics — one callback per probe.

These callbacks now cover only rank-zero parameter/plot diagnostics. Forward-
heavy representation probes are owned by ``MSDeltaTrainer.evaluate`` so they
use the prepared compiled model and every data-parallel rank.

`build_callbacks` assembles the right set from the `LogArgs` cadences (a probe
is only registered when its cadence is set), so nothing runs as a no-op.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn
from transformers import TrainerCallback

from .fourier import dead_freqs, freq_drift, interp_mae
from .alignment import alignment_metrics
from .viz import render_bias_panels


# ---------- shared plumbing ----------

class _InlineCallback(TrainerCallback):
    """Base for a step-cadenced, main-process-only diagnostic.

    Subclasses set `empty_cache_before` (whether to free the allocator before a
    forward-heavy run) and implement `run(step)`. Not registered directly.
    """

    empty_cache_before: bool = False

    def __init__(self, module: nn.Module, every: int, *,
                 dataset=None, out_dir: Path | None = None):
        self.module = module
        self.every = every
        self.dataset = dataset      # preprocessed HF val dataset (probes/retrieval)
        self.out_dir = out_dir

    @property
    def encoder(self):
        return self.module.encoder

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters()).device

    def _wlog(self, payload: dict, step: int) -> None:
        # Log on the same x-axis (train/global_step) the HF WandbCallback uses,
        # without passing an explicit wandb step (avoids step-ordering clashes).
        if payload and wandb.run is not None:
            wandb.log({**payload, "train/global_step": step})

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or not self.every:
            return
        step = state.global_step
        if step <= 0 or step % self.every != 0:
            return
        if self.empty_cache_before and self.device.type == "cuda":
            # Return reserved-but-unallocated blocks so the forward gets
            # contiguous room (big tiers OOM'd here on a fragmented 40GB A100).
            torch.cuda.empty_cache()
        self.run(step)

    def run(self, step: int) -> None:
        raise NotImplementedError


# ---------- one callback per probe ----------

class FourierProbeCallback(_InlineCallback):
    """Are the learned Fourier frequencies effective?

    Evaluates each learnable featurizer on REAL values drawn from the val set —
    real log-intensities for `PeakEmbed.ff_int`, real intra-spectrum Δm/z (what
    `DeltaMZBias.ff` actually sees) for the bias — rather than a synthetic grid.
    Per featurizer we log reconstruction MAE (can a small MLP recover the scalar
    from its Fourier code — the effectiveness number), the dead-frequency count,
    how far the frequencies have drifted from init, their current range, and a
    histogram of where they sit. A frozen (non-learnable) featurizer is skipped.
    """

    empty_cache_before = True

    def __init__(self, module, every, dataset, n_spectra):
        super().__init__(module, every, dataset=dataset)
        self.n_spectra = n_spectra
        self._vals: dict[str, torch.Tensor] | None = None   # cached real samples
        self._init_freqs: dict[str, torch.Tensor] = {}      # snapshot at first run

    # ---- one-time real-value sampling ----

    def _sample_values(self, budget: int = 8192,
                       max_spectra: int = 1000) -> dict[str, torch.Tensor]:
        """Pool real log-intensity values from the validation set."""
        g = torch.Generator().manual_seed(0)
        li_pool = []
        n = min(self.n_spectra, max_spectra)
        for row in itertools.islice(self.dataset, n):
            li = torch.as_tensor(row["log_int"], dtype=torch.float32)
            if li.numel():
                li_pool.append(li)

        def _cat(pool):
            if not pool:
                return torch.empty(0)
            v = torch.cat(pool)
            if v.numel() > budget:
                sel = torch.randperm(v.numel(), generator=g)[:budget]
                v = v[sel]
            return v

        return {"int": _cat(li_pool)}

    # ---- per-featurizer metrics ----

    def _featurizer_metrics(self, name: str, ff, vals: torch.Tensor) -> dict:
        freqs = ff.freqs
        if not isinstance(freqs, nn.Parameter):
            return {}                       # frozen featurizer — nothing to watch
        if name not in self._init_freqs:
            self._init_freqs[name] = freqs.detach().abs().cpu().clone()
        if vals.numel() < 8:
            return {}
        span = float(vals.max() - vals.min())
        # .float(): freqs is a learnable nn.Parameter, so under bf16 training it
        # comes back as bfloat16 — which Tensor.numpy() (the histogram below)
        # rejects. Cast to fp32 at the source, matching interp_mae's idiom.
        f = freqs.detach().abs().float().cpu()
        m = {
            f"fourier/{name}_mae": interp_mae(freqs, vals),
            f"fourier/{name}_dead": dead_freqs(freqs, span),
            f"fourier/{name}_drift_log10": freq_drift(freqs, self._init_freqs[name]),
            f"fourier/{name}_f_min": float(f.min()),
            f"fourier/{name}_f_max": float(f.max()),
        }
        if wandb.run is not None:
            m[f"fourier/{name}_log10_freqs"] = wandb.Histogram(
                f.clamp_min(1e-12).log10().numpy())
        return m

    def run(self, step):
        if self._vals is None:
            self._vals = self._sample_values()
        enc = self.encoder
        payload: dict[str, Any] = {}
        payload.update(self._featurizer_metrics("int", enc.embed.ff_int, self._vals["int"]))
        if not payload:
            return
        self._wlog(payload, step)
        g = lambda k: payload.get(k, float("nan"))
        print(f"  fourier: int_mae={g('fourier/int_mae'):.4g} "
              f"int_dead={g('fourier/int_dead'):.0f}", flush=True)


class AlignmentCallback(_InlineCallback):
    """Δm bias-curve chemistry alignment — pure bias-curve analysis, no data."""

    def run(self, step):
        a = alignment_metrics(self.encoder)
        self._wlog(a, step)
        print(f"  align: n_sig05={a.get('align/n_sig05', 0):.0f} "
              f"n_sig01_bonf={a.get('align/n_sig01_bonf', 0):.0f} "
              f"best_p={a.get('align/best_p', 1):.1e}", flush=True)


class BiasPanelCallback(_InlineCallback):
    """Δm bias-curve panels (fine + coarse) rendered to figs/ and logged as images."""

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


class WandbConfigCallback(TrainerCallback):
    """Log the resolved config to the wandb run once training begins (the run is
    created by HF's WandbCallback, which fires before this)."""

    def __init__(self, resolved_config: dict[str, Any]):
        self.resolved_config = resolved_config

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and wandb.run is not None:
            wandb.config.update(self.resolved_config, allow_val_change=True)


# ---------- assembly ----------

def build_callbacks(module, val_dataset, log_args, resolved_config, out_dir):
    """Assemble the callbacks the config asks for. A probe is registered only
    when its cadence is set, so no callback ever fires as a no-op. `val_dataset`
    is the preprocessed HF validation dataset used for Fourier diagnostics."""
    cbs: list[TrainerCallback] = [WandbConfigCallback(resolved_config)]
    if log_args.bias_curve_every:
        cbs.append(BiasPanelCallback(module, log_args.bias_curve_every, out_dir=out_dir))
    if log_args.val_every:
        cbs.append(FourierProbeCallback(
            module, log_args.val_every, val_dataset, log_args.probe_n_spectra))
        cbs.append(AlignmentCallback(module, log_args.val_every))
    return cbs
