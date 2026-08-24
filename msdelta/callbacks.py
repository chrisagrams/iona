"""Trainer callbacks for the inline diagnostics — one callback per probe.

Each concrete callback runs a single diagnostic on the main process at its
configured cadence, reaching the eager `.encoder` shared with the training
engine for its own forwards. That sharing assumes DeepSpeed ZeRO stage ≤ 2:
under ZeRO-3 the params live in the engine and a bare `encoder(...)` would see
empty shells.

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

from .data import collate_preprocessed
from .fourier import dead_freqs, freq_drift, interp_mae
from .probe import run_all_probes
from .alignment import alignment_metrics
from .retrieval import retrieval_inline_metrics, replicate_retrieval_inline_metrics
from .viz import AttentionRecorder, attention_entropy_per_head, render_bias_panels


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

class LinearProbeCallback(_InlineCallback):
    """Frozen-encoder linear probes (precursor m/z, charge, neutral loss, …)."""

    empty_cache_before = True

    def __init__(self, module, every, dataset, n_spectra):
        super().__init__(module, every, dataset=dataset)
        self.n_spectra = n_spectra

    def run(self, step):
        m = run_all_probes(self.encoder, self.dataset, self.device,
                           n_spectra=self.n_spectra)
        self._wlog(m, step)
        key = lambda k: m.get(k, float("nan"))
        print(f"  probe: precursor_r2={key('probe/precursor_mz_r2'):.3f} "
              f"fragment_mz_r2={key('probe/fragment_mz_r2'):.3f} "
              f"charge_acc={key('probe/charge_acc'):.3f} "
              f"nloss_auc={key('probe/neutral_loss_auc'):.3f} "
              f"iso_f1={key('probe/isotope_f1'):.3f}", flush=True)


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


class RetrievalCallback(_InlineCallback):
    """Spectrum retrieval — the model as an embedding model, vs a binned baseline."""

    empty_cache_before = True

    def run(self, step):
        r = retrieval_inline_metrics(self.encoder, self.dataset, self.device)
        self._wlog(r, step)
        print(f"  retrieval: mAP={r.get('retrieval/mAP', float('nan')):.3f} "
              f"binned={r.get('retrieval/binned_mAP', float('nan')):.3f} "
              f"gap={r.get('retrieval/gap_vs_binned', float('nan')):+.3f}", flush=True)


class ReplicateRetrievalCallback(_InlineCallback):
    """External MS2 peptide-replicate-retrieval benchmark (Hit@1 / MAP / R@5)."""

    empty_cache_before = True

    def __init__(self, module, every, pp, repo_id):
        super().__init__(module, every)
        self.pp = pp                # external benchmark preprocessed on the fly
        self.repo_id = repo_id

    def run(self, step):
        rr = replicate_retrieval_inline_metrics(
            self.encoder, self.repo_id, self.device, self.pp)
        if not rr:
            return
        self._wlog(rr, step)
        print(f"  replicate-retrieval: "
              f"Hit@1={rr.get('replicate_retrieval/Hit@1', float('nan')):.3f} "
              f"MAP={rr.get('replicate_retrieval/MAP', float('nan')):.3f} "
              f"R@5={rr.get('replicate_retrieval/R@5', float('nan')):.3f}", flush=True)


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


class AttentionEntropyCallback(_InlineCallback):
    """Per-head attention entropy on one val batch (SDPA doesn't expose the
    weights, so `AttentionRecorder` recomputes them under no_grad)."""

    empty_cache_before = True
    batch_size = 64

    def run(self, step):
        ent = self._entropy()
        if ent is None:
            return
        payload = {}
        for layer_i, row in enumerate(ent):
            for h_i, e in enumerate(row):
                payload[f"attn_entropy/L{layer_i}_H{h_i}"] = float(e)
        self._wlog(payload, step)

    def _entropy(self):
        enc = self.encoder
        rows = list(itertools.islice(self.dataset, self.batch_size))
        if not rows:
            return None
        batch = collate_preprocessed(rows)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        was_training = enc.training
        enc.eval()
        with torch.no_grad(), AttentionRecorder(enc.blocks) as rec:
            enc(batch["mz"], batch["log_int"], batch["key_padding_mask"], None)
        if was_training:
            enc.train()
        return attention_entropy_per_head(rec.attn)


class WandbConfigCallback(TrainerCallback):
    """Log the resolved config to the wandb run once training begins (the run is
    created by HF's WandbCallback, which fires before this)."""

    def __init__(self, resolved_config: dict[str, Any]):
        self.resolved_config = resolved_config

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and wandb.run is not None:
            wandb.config.update(self.resolved_config, allow_val_change=True)


# ---------- assembly ----------

def build_callbacks(module, val_dataset, pp, log_args, resolved_config, out_dir):
    """Assemble the callbacks the config asks for. A probe is registered only
    when its cadence is set, so no callback ever fires as a no-op. `val_dataset`
    is the preprocessed HF val dataset the probes/retrieval read; `pp` is only
    for the external replicate-retrieval benchmark (its own dataset)."""
    cbs: list[TrainerCallback] = [WandbConfigCallback(resolved_config)]
    if log_args.bias_curve_every:
        cbs.append(BiasPanelCallback(module, log_args.bias_curve_every, out_dir=out_dir))
        cbs.append(AttentionEntropyCallback(
            module, log_args.bias_curve_every, dataset=val_dataset))
    if log_args.probe_every:
        cbs.append(LinearProbeCallback(
            module, log_args.probe_every, val_dataset, log_args.probe_n_spectra))
        cbs.append(FourierProbeCallback(
            module, log_args.probe_every, val_dataset, log_args.probe_n_spectra))
        cbs.append(AlignmentCallback(module, log_args.probe_every))
        cbs.append(RetrievalCallback(
            module, log_args.probe_every, dataset=val_dataset))
        if log_args.replicate_retrieval_repo:
            cbs.append(ReplicateRetrievalCallback(
                module, log_args.probe_every, pp, log_args.replicate_retrieval_repo))
    return cbs
