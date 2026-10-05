"""K114: per-block forward and backward time and peak memory, Pairformer vs transformer.

    python pbs/diag/pairformer_profile.py --out $MSDELTA_DIAG/pairformer_profile/<jobid>.json

Purpose (user, K114): calibrate how many single blocks take as long as one pair update, at
each N. Whole pretraining steps (MSDeltaForPreTraining, masked-intensity loss, bf16 autocast
over fp32 weights as with ``--bf16 true``, train mode so dropout is on) are timed; no
optimizer step.

Models (--models):
  pairformer            configs/diag/k114-pairformer-stage0/config.json, a verbatim copy of
                        ``git show stage0-prep:configs/stage0/pairformer/config.json`` (branch
                        stage0-prep, commit cde0905 "P1 Stage 0 prep"): 512 x 10, 8 heads,
                        c_z 64, tri-mult 64, write-back 16, triangle attention OFF.
  pairformer_triattn    the same with pair_use_triangle_attention=true, pair_tri_attn_impl="sdpa"
                        (4 heads x 16, chunk 32, as in that config).
  transformer           configs/msdelta-base-50m/config.json (640 x 10, 10 heads).
  --pair-config-overrides '{"pair_update_every": 3, "pair_bias_lag": 1}' applies the JSON on top
  of both Pairformer configs (K114-P decoupled streams; any MSDeltaConfig field works). With
  k > 1 the calibration also reports the pair update per UPDATE (a-f only, divided by the
  number of update layers).
Sizes: --batches 8,32 x --peaks 100,150,256,512. Each batch is a set of random spectra --
m/z uniform in [100, 2000] sorted, lengths uniform in [0.3 N, N] with one spectrum at exactly
N, log1p-normalised intensities -- padded and masked by the pretraining collator itself
(MSDeltaDataCollatorForPreTraining, mask_ratio 0.5 as in the Stage 0 args).

Per (model, B, N), each mode caught separately (OOM recorded, not fatal):
  plain_gc_off   median step time (fwd+bwd, no hooks) and peak memory, checkpointing OFF.
  plain_gc_on    the same with gradient checkpointing ON (model.gradient_checkpointing_enable(),
                 the Trainer's default kwargs), which Stage 0 trains with.
  profiled       checkpointing OFF, every block wrapped (BlockProfiler below), device sync
                 around each block: per-block forward and backward time (median over --reps
                 steps after --warmup) and the forward's allocated-memory growth per block.
If both plain modes OOM at some N, larger N are skipped for that (model, B).

How the per-block times are taken (no edits to model code or train.py): the profiler
replaces the ``forward`` (or a method such as ``init_state`` / ``read_bias``) of chosen module
INSTANCES with a wrapper. Forward: sync, time the call, sync. Backward: the wrapper passes
each grad-requiring input and output through an identity autograd Function whose backward
syncs and logs a timestamp; the backward span between consecutive log entries is charged to
the block whose output marker opened it (or, after a block's input marker, to its parent
block, else "other"). The engine runs backward in reverse creation order, so these spans tile
the whole backward (the per-block backward times sum to the measured total by construction).
Nested blocks report EXCLUSIVE time: "h_single_attention" is the single block minus its
transition, "pair_glue" is the pair layer minus its sub-blocks (residual adds, dropout,
bias permute), "z_init" is init_state minus pair features, "block_glue" is the transformer
block minus attention and FFN (the two pre-LNs and residuals). The wrappers are identities on
values: tests/test_pairformer_profile.py checks outputs and gradients are unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import statistics
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch

from msdelta.models.configuration_msdelta import MSDeltaConfig
from msdelta.models.modeling_msdelta import MSDeltaForPreTraining
from msdelta.models.processing_msdelta import MSDeltaDataCollatorForPreTraining

PAIRFORMER_CONFIG = "configs/diag/k114-pairformer-stage0/config.json"
TRANSFORMER_CONFIG = "configs/msdelta-base-50m/config.json"

PAIR_BLOCKS = ("a_writeback", "b_trimul_out", "c_trimul_in", "d_triattn_start",
               "e_triattn_end", "f_pair_transition", "g_bias_readout", "pair_glue")
SINGLE_BLOCKS = ("h_single_attention", "i_single_transition")
TRANSFORMER_BLOCKS = ("attention", "ffn", "block_glue")


def sync(dev: torch.device) -> None:
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def mem_allocated(dev: torch.device) -> int:
    if dev.type == "xpu":
        return torch.xpu.memory_allocated()
    if dev.type == "cuda":
        return torch.cuda.memory_allocated()
    return 0


class _Mark(torch.autograd.Function):
    """Identity whose backward logs (event, time) on the profiler."""

    @staticmethod
    def forward(ctx, x, prof, event):
        ctx.prof = prof
        ctx.event = event
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        ctx.prof._log_backward(ctx.event)
        return grad, None, None


class BlockProfiler:
    """Per-block forward/backward timing through instance-level forward wrappers."""

    def __init__(self, device: torch.device):
        self.dev = device
        self.active = False
        self.parent: dict[str, str | None] = {}
        self._wrapped: list[tuple[object, str]] = []
        self.reset()

    # -- wrapping ------------------------------------------------------------------------
    def wrap(self, owner, attr: str, label: str, parent: str | None = None) -> None:
        fn = getattr(owner, attr)
        self.parent[label] = parent
        prof = self

        def wrapped(*args, **kwargs):
            if not prof.active:
                return fn(*args, **kwargs)
            args = tuple(prof._mark(a, (label, "in")) for a in args)
            sync(prof.dev)
            m0 = mem_allocated(prof.dev)
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            sync(prof.dev)
            prof.fwd[label] += time.perf_counter() - t0
            prof.fwd_mem[label] += mem_allocated(prof.dev) - m0
            prof.calls[label] += 1
            if isinstance(out, tuple):
                return tuple(prof._mark(o, (label, "out")) for o in out)
            return prof._mark(out, (label, "out"))

        setattr(owner, attr, wrapped)
        self._wrapped.append((owner, attr))

    def remove(self) -> None:
        for owner, attr in reversed(self._wrapped):
            delattr(owner, attr)
        self._wrapped.clear()

    def _mark(self, x, event):
        if (isinstance(x, torch.Tensor) and x.requires_grad and x.is_floating_point()
                and torch.is_grad_enabled()):
            return _Mark.apply(x, self, event)
        return x

    def _log_backward(self, event) -> None:
        sync(self.dev)
        self.bwd_events.append((event, time.perf_counter()))

    # -- one step ------------------------------------------------------------------------
    def reset(self) -> None:
        self.fwd = defaultdict(float)
        self.fwd_mem = defaultdict(int)
        self.calls = defaultdict(int)
        self.bwd_events: list[tuple[tuple[str, str], float]] = []

    def exclusive_forward(self, total: float) -> dict[str, float]:
        excl = dict(self.fwd)
        for label, parent in self.parent.items():
            if parent is not None and label in self.fwd:
                excl[parent] = excl.get(parent, 0.0) - self.fwd[label]
        top = sum(v for k, v in self.fwd.items() if self.parent.get(k) is None)
        excl["other"] = total - top
        return excl

    def backward_attribution(self, t_start: float, t_end: float) -> dict[str, float]:
        res = defaultdict(float)
        events = self.bwd_events
        first = events[0][1] if events else t_end
        res["loss"] += first - t_start
        for i, ((label, kind), t) in enumerate(events):
            t_next = events[i + 1][1] if i + 1 < len(events) else t_end
            owner = label if kind == "out" else (self.parent.get(label) or "other")
            res[owner] += t_next - t
        return dict(res)


def instrument(model: MSDeltaForPreTraining, prof: BlockProfiler) -> list[str]:
    """Wrap the blocks of a Pairformer or transformer MSDeltaForPreTraining."""
    enc = model.msdelta
    arch = getattr(model.config, "architecture", "transformer")
    prof.wrap(enc.embed, "forward", "token_embed")
    if arch == "pairformer":
        stack = enc.bias_module
        prof.wrap(stack, "init_state", "z_init")
        prof.wrap(stack.pair_feats, "forward", "pair_features", parent="z_init")
        for layer in stack.layers:
            prof.wrap(layer, "forward", "pair_glue")
            for attr, label in (("opm", "a_writeback"), ("tri_out", "b_trimul_out"),
                                ("tri_in", "c_trimul_in"), ("tri_attn_start", "d_triattn_start"),
                                ("tri_attn_end", "e_triattn_end"),
                                ("transition", "f_pair_transition")):
                if hasattr(layer, attr):
                    prof.wrap(getattr(layer, attr), "forward", label, parent="pair_glue")
            prof.wrap(layer, "read_bias", "g_bias_readout", parent="pair_glue")
        for block in enc.blocks:
            prof.wrap(block, "forward", "h_single_attention")
            prof.wrap(block.transition, "forward", "i_single_transition",
                      parent="h_single_attention")
    else:
        prof.wrap(enc.bias_module, "forward", "dmz_bias")
        for block in enc.blocks:
            prof.wrap(block, "forward", "block_glue")
            prof.wrap(block.attn, "forward", "attention", parent="block_glue")
            prof.wrap(block.ffn, "forward", "ffn", parent="block_glue")
    prof.wrap(enc.norm, "forward", "final_ln")
    prof.wrap(model.intensity_head, "forward", "head")
    return list(prof.parent)


# ----------------------------------------------------------------------------------------
# Inputs and models.
# ----------------------------------------------------------------------------------------

def make_batch(b: int, n: int, seed: int = 0, mask_ratio: float = 0.5) -> dict:
    g = torch.Generator().manual_seed(seed * 100_003 + b * 1009 + n)
    lo = max(2, int(0.3 * n))
    feats = []
    for row in range(b):
        length = n if row == 0 else int(torch.randint(lo, n + 1, (1,), generator=g))
        mz = torch.sort(100.0 + 1900.0 * torch.rand(length, generator=g)).values
        # Exponential(1) intensities (heavy tail, a few dominant peaks).
        intensity = -torch.log(torch.rand(length, generator=g).clamp_min(1e-12))
        li = torch.log1p(intensity * 1e4)
        li = li / li.max().clamp_min(1e-8)
        feats.append(dict(mz=mz.tolist(), log_intensity=li.tolist(),
                          labels=(intensity / intensity.sum()).tolist()))
    torch.manual_seed(seed)  # the collator's randperm
    return MSDeltaDataCollatorForPreTraining(mask_ratio=mask_ratio)(feats)


def load_config(name: str, pairformer_config: str, transformer_config: str,
                pair_overrides: dict | None = None) -> MSDeltaConfig:
    if name == "transformer":
        d = json.loads(Path(transformer_config).read_text())
    else:
        d = json.loads(Path(pairformer_config).read_text())
        if name == "pairformer_triattn":
            d.update(pair_use_triangle_attention=True, pair_tri_attn_impl="sdpa")
        elif name != "pairformer":
            raise ValueError(name)
        d.update(pair_overrides or {})
    return MSDeltaConfig(**d)


def build_model(cfg: MSDeltaConfig, dev: torch.device, seed: int = 0) -> MSDeltaForPreTraining:
    torch.manual_seed(seed)
    return MSDeltaForPreTraining(cfg).to(dev).train()


def run_step(model, batch, dev, autocast: bool = True):
    """One forward+backward; returns (t_forward_end, t_backward_start, t_end) around sync."""
    ctx = (torch.autocast(dev.type, dtype=torch.bfloat16) if autocast
           else contextlib.nullcontext())
    with ctx:
        out = model(**batch)
    loss = out.loss
    sync(dev)
    t_bwd = time.perf_counter()
    loss.backward()
    sync(dev)
    t_end = time.perf_counter()
    model.zero_grad(set_to_none=True)
    return t_bwd, t_end


def is_oom(e: BaseException) -> bool:
    s = str(e).lower()
    return (isinstance(e, torch.OutOfMemoryError) or "out of memory" in s
            or "out_of_device_memory" in s or "ur_result_error_out_of_resources" in s)


def reset_peak(dev):
    gc.collect()
    if dev.type == "xpu":
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()
        return torch.xpu.memory_allocated()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated()
    return 0


def peak_since(dev, base):
    if dev.type == "xpu":
        return (torch.xpu.max_memory_allocated() - base) / 1e9
    if dev.type == "cuda":
        return (torch.cuda.max_memory_allocated() - base) / 1e9
    return None


def plain_mode(model, batch, dev, gc_on: bool, warmup: int, reps: int) -> dict:
    if gc_on:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    try:
        for _ in range(warmup):
            run_step(model, batch, dev)
        base = reset_peak(dev)
        times = []
        for _ in range(reps):
            sync(dev)
            t0 = time.perf_counter()
            _, t_end = run_step(model, batch, dev)
            times.append(t_end - t0)
        return dict(status="ok", step_ms=1e3 * statistics.median(times),
                    steps_ms=[1e3 * t for t in times], peak_gb=peak_since(dev, base),
                    weights_gb=base / 1e9 if dev.type != "cpu" else None)
    finally:
        model.gradient_checkpointing_disable()


def profiled_mode(model, batch, dev, prof: BlockProfiler, warmup: int, reps: int) -> dict:
    model.gradient_checkpointing_disable()
    for _ in range(warmup):
        run_step(model, batch, dev)
    per_step = []
    base = reset_peak(dev)
    try:
        prof.active = True
        for _ in range(reps):
            prof.reset()
            sync(dev)
            t0 = time.perf_counter()
            t_bwd, t_end = run_step(model, batch, dev)
            fwd = prof.exclusive_forward(t_bwd - t0)
            bwd = prof.backward_attribution(t_bwd, t_end)
            per_step.append(dict(fwd=fwd, bwd=bwd, fwd_total=t_bwd - t0,
                                 bwd_total=t_end - t_bwd, fwd_mem=dict(prof.fwd_mem),
                                 calls=dict(prof.calls)))
    finally:
        prof.active = False
    labels = sorted({k for s in per_step for k in (*s["fwd"], *s["bwd"])})
    med = lambda xs: 1e3 * statistics.median(xs)  # noqa: E731
    blocks = {}
    for lab in labels:
        f = med([s["fwd"].get(lab, 0.0) for s in per_step])
        bw = med([s["bwd"].get(lab, 0.0) for s in per_step])
        blocks[lab] = dict(fwd_ms=f, bwd_ms=bw, total_ms=f + bw,
                           calls=per_step[-1]["calls"].get(lab, 0),
                           fwd_mem_gb=per_step[-1]["fwd_mem"].get(lab, 0) / 1e9)
    fwd_total = med([s["fwd_total"] for s in per_step])
    bwd_total = med([s["bwd_total"] for s in per_step])
    step = fwd_total + bwd_total
    for lab, v in blocks.items():
        v["frac_of_step"] = v["total_ms"] / step if step else None
        v["frac_of_fwd"] = v["fwd_ms"] / fwd_total if fwd_total else None
        v["frac_of_bwd"] = v["bwd_ms"] / bwd_total if bwd_total else None
    return dict(status="ok", fwd_ms=fwd_total, bwd_ms=bwd_total, step_ms=step,
                peak_gb=peak_since(dev, base), blocks=blocks)


def calibration(prof_rec: dict, n_layers: int, n_pair_updates: int | None = None) -> dict:
    """Per-layer costs and the ratio the user asked for.

    ``n_pair_updates`` (K114-P): with ``pair_update_every > 1`` only some layers run the pair
    update (blocks a-f); their cost is also reported per update rather than averaged per layer.
    """
    b = prof_rec["blocks"]
    get = lambda labs, key: sum(b[k][key] for k in labs if k in b)  # noqa: E731
    out = {}
    for key in ("fwd_ms", "bwd_ms", "total_ms"):
        pair = get(PAIR_BLOCKS, key) / n_layers
        single = get(SINGLE_BLOCKS, key) / n_layers
        tblock = get(TRANSFORMER_BLOCKS, key) / n_layers
        tag = key.replace("_ms", "")
        if pair:
            out[f"pair_update_{tag}_ms_per_layer"] = pair
            out[f"single_block_{tag}_ms_per_layer"] = single
            out[f"single_blocks_per_pair_update_{tag}"] = pair / single if single else None
        if tblock:
            out[f"transformer_block_{tag}_ms_per_layer"] = tblock
        if pair and n_pair_updates:
            ops = get(PAIR_BLOCKS[:6], key) / n_pair_updates
            out[f"pair_ops_{tag}_ms_per_update"] = ops
            out[f"single_blocks_per_pair_ops_update_{tag}"] = ops / single if single else None
    if n_pair_updates:
        out["n_pair_updates"] = n_pair_updates
    return out


def run_config(model_name, model, cfg, prof, b, n, dev, a) -> dict:
    rec = dict(model=model_name, B=b, N=n)
    batch = {k: v.to(dev) for k, v in make_batch(b, n, a.seed, a.mask_ratio).items()}
    rec["lengths"] = batch["attention_mask"].sum(1).tolist()
    rec["masked"] = int(batch["mask_positions"].sum())
    modes = [("plain_gc_off", lambda: plain_mode(model, batch, dev, False, a.warmup, a.reps)),
             ("plain_gc_on", lambda: plain_mode(model, batch, dev, True, a.warmup, a.reps)),
             ("profiled", lambda: profiled_mode(model, batch, dev, prof, 1, a.reps))]
    for mode, fn in modes:
        if mode == "profiled" and rec["plain_gc_off"]["status"] != "ok":
            rec[mode] = dict(status="skipped", reason="plain_gc_off did not run")
            continue
        try:
            rec[mode] = fn()
        except Exception as e:  # noqa: BLE001 -- recorded, never fatal
            rec[mode] = dict(status="oom" if is_oom(e) else "error",
                             error=f"{type(e).__name__}: {e}"[:1000])
            if not is_oom(e):
                rec[mode]["traceback"] = traceback.format_exc()[-4000:]
                traceback.print_exc()
        finally:
            prof.active = False
            model.zero_grad(set_to_none=True)
            reset_peak(dev)
        print(f"[profile] {model_name} B={b} N={n} {mode}: {rec[mode]['status']} "
              f"{rec[mode].get('step_ms', '')} ms peak {rec[mode].get('peak_gb', '')} GB",
              flush=True)
    if rec["profiled"].get("status") == "ok":
        n_upd = (sum(getattr(layer, "updates", True) for layer in model.msdelta.bias_module.layers)
                 if getattr(cfg, "architecture", "transformer") == "pairformer" else None)
        rec["calibration"] = calibration(rec["profiled"], cfg.num_hidden_layers, n_upd)
    del batch
    return rec


def print_tables(rows):
    for r in rows:
        p = r.get("profiled", {})
        head = (f"\n== {r['model']}  B={r['B']}  N={r['N']}  "
                f"gc_off {_fmt(r.get('plain_gc_off'))}  gc_on {_fmt(r.get('plain_gc_on'))}")
        print(head)
        if p.get("status") != "ok":
            print(f"   profiled: {p.get('status')} {p.get('error', p.get('reason', ''))[:100]}")
            continue
        print(f"   profiled step {p['step_ms']:.1f} ms (fwd {p['fwd_ms']:.1f}, bwd {p['bwd_ms']:.1f})"
              f", peak {p['peak_gb'] if p['peak_gb'] is None else round(p['peak_gb'], 2)} GB")
        print(f"   {'block':<22}{'calls':>6}{'fwd ms':>10}{'bwd ms':>10}{'total':>10}"
              f"{'% step':>8}{'fwd GB':>9}")
        for lab, v in sorted(p["blocks"].items(), key=lambda kv: -kv[1]["total_ms"]):
            print(f"   {lab:<22}{v['calls']:>6}{v['fwd_ms']:>10.2f}{v['bwd_ms']:>10.2f}"
                  f"{v['total_ms']:>10.2f}{100 * (v['frac_of_step'] or 0):>7.1f}%"
                  f"{v['fwd_mem_gb']:>9.3f}")
        c = r.get("calibration", {})
        if "single_blocks_per_pair_update_total" in c:
            print(f"   per layer: pair update {c['pair_update_total_ms_per_layer']:.2f} ms, single "
                  f"block {c['single_block_total_ms_per_layer']:.2f} ms -> "
                  f"{c['single_blocks_per_pair_update_total']:.2f} single blocks per pair update "
                  f"(fwd {c['single_blocks_per_pair_update_fwd']:.2f}, "
                  f"bwd {c['single_blocks_per_pair_update_bwd']:.2f})")
        if "transformer_block_total_ms_per_layer" in c:
            print(f"   per layer: transformer block {c['transformer_block_total_ms_per_layer']:.2f} ms")
    # Summary grid.
    print("\n== summary: single blocks per pair update (fwd+bwd, per layer), and transformer "
          "blocks per pair update at the same B, N")
    tb = {(r["B"], r["N"]): r["calibration"]["transformer_block_total_ms_per_layer"]
          for r in rows if r["model"] == "transformer"
          and "transformer_block_total_ms_per_layer" in r.get("calibration", {})}
    print(f"   {'model':<20}{'B':>4}{'N':>5}{'pair ms':>10}{'single ms':>11}{'ratio':>8}"
          f"{'tfm ms':>9}{'pair/tfm':>10}")
    for r in rows:
        c = r.get("calibration", {})
        if "pair_update_total_ms_per_layer" not in c:
            continue
        t = tb.get((r["B"], r["N"]))
        pr = c["pair_update_total_ms_per_layer"]
        print(f"   {r['model']:<20}{r['B']:>4}{r['N']:>5}{pr:>10.2f}"
              f"{c['single_block_total_ms_per_layer']:>11.2f}"
              f"{c['single_blocks_per_pair_update_total'] or float('nan'):>8.2f}"
              f"{t if t else float('nan'):>9.2f}{pr / t if t else float('nan'):>10.2f}")


def _fmt(m):
    if not m:
        return "-"
    if m.get("status") != "ok":
        return m.get("status", "-")
    pk = m.get("peak_gb")
    return f"{m['step_ms']:.0f} ms/{'-' if pk is None else f'{pk:.1f}'} GB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="pairformer,pairformer_triattn,transformer")
    ap.add_argument("--batches", default="8,32")
    ap.add_argument("--peaks", default="100,150,256,512")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--pairformer-config", default=PAIRFORMER_CONFIG)
    ap.add_argument("--transformer-config", default=TRANSFORMER_CONFIG)
    ap.add_argument("--pair-config-overrides", default="{}",
                    help="JSON applied on top of both Pairformer configs (K114-P), e.g. "
                         "'{\"pair_update_every\": 3, \"pair_bias_lag\": 1}'")
    a = ap.parse_args()
    pair_overrides = json.loads(a.pair_config_overrides)

    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    dev = torch.device("xpu" if xpu else "cpu")
    env = dict(torch=torch.__version__, device=str(dev),
               device_name=torch.xpu.get_device_name(0) if xpu else None,
               device_total_gb=torch.xpu.get_device_properties(0).total_memory / 1e9 if xpu else None,
               ZE_AFFINITY_MASK=os.environ.get("ZE_AFFINITY_MASK"),
               job=os.environ.get("PBS_JOBID"), code_dir=os.environ.get("MSDELTA_CODE_DIR"),
               autocast="bf16", pairformer_config=a.pairformer_config,
               pairformer_config_source="stage0-prep:configs/stage0/pairformer/config.json @ cde0905",
               transformer_config=a.transformer_config, warmup=a.warmup, reps=a.reps,
               mask_ratio=a.mask_ratio, pair_config_overrides=pair_overrides)
    print(f"[env] {env}", flush=True)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(env=env, configs={}, results=[])

    batches = [int(x) for x in a.batches.split(",")]
    peaks = sorted(int(x) for x in a.peaks.split(","))
    for name in a.models.split(","):
        try:
            cfg = load_config(name, a.pairformer_config, a.transformer_config, pair_overrides)
            model = build_model(cfg, dev, a.seed)
        except Exception as e:  # noqa: BLE001
            report["configs"][name] = dict(error=f"{type(e).__name__}: {e}",
                                           traceback=traceback.format_exc())
            traceback.print_exc()
            continue
        report["configs"][name] = dict(config=cfg.to_diff_dict(),
                                       n_params=sum(p.numel() for p in model.parameters()))
        prof = BlockProfiler(dev)
        instrument(model, prof)
        oom_at: dict[int, int] = {}  # B -> smallest N where both plain modes OOMed
        for b in batches:
            for n in peaks:
                smaller_b_oom = any(bb < b and nn <= n for bb, nn in oom_at.items())
                if b in oom_at or smaller_b_oom:
                    report["results"].append(dict(model=name, B=b, N=n, skipped=(
                        "a smaller N (or batch) already OOMed in both plain modes")))
                    print(f"[profile] {name} B={b} N={n}: skipped", flush=True)
                    continue
                rec = run_config(name, model, cfg, prof, b, n, dev, a)
                report["results"].append(rec)
                if (rec["plain_gc_off"]["status"] == "oom"
                        and rec["plain_gc_on"]["status"] == "oom"):
                    oom_at[b] = n
                out_path.write_text(json.dumps(report, indent=1, default=str))
        prof.remove()
        del model, prof
        reset_peak(dev)
    out_path.write_text(json.dumps(report, indent=1, default=str))
    print(f"[profile] wrote {out_path}", flush=True)
    print_tables([r for r in report["results"] if "skipped" not in r])


if __name__ == "__main__":
    main()
