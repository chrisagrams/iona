"""K172-P / K176-P: do the pair and single streams of the Pairformer actually overlap?

    python pbs/diag/k172_overlap.py --out results/raw/diag/k172_overlap/<jobid>.json

The P2 Pairformer (configs/p2/p2-cz32-k5/config.json: 512 wide, c_z 32, outgoing triangle
multiplication, factored write-back, no triangle attention), with k = ``pair_update_every`` in
--ks and num_hidden_layers in --layers, in these modes (--modes):

  lag1_seq    pair_bias_lag=1                         (the lag-1 schedule, one device, one stream)
  lag1_conc   pair_bias_lag=1, pair_concurrent=True   (update m on a side stream, K172-P)
  lag1_dev    pair_bias_lag=1, pair_device_offset=d   (update m on device main + d, K176-P;
              --pair-device-offset, default 1; skipped with a clear record if that device
              does not exist)
  lag0_seq    pair_bias_lag=0 (reference; one pair update more than lag 1 at the same k)

Whole pretraining steps (MSDeltaForPreTraining, masked-intensity loss, bf16 autocast over
fp32 weights, train mode so dropout is on, no optimizer step), timed forward+backward with
a sync of every device in use around each step: median over --reps steps after --warmup,
plus peak memory above the weights on EACH device the model uses. Batches come from
``pairformer_profile.make_batch`` (random spectra padded and masked by the pretraining
collator). For lag1_dev the cross-device copies of one forward are counted (wrapping
``pairformer.transfer``): bytes per round (the s entering the round to the pair device, z(m+1)
back) and the one-off copies (masks, z_init); backward moves the same volume of gradients.

Numerics: lag1_conc and lag1_dev get lag1_seq's weights (load_state_dict) and each runs one
step from the same RNG seed against lag1_seq, compared twice: in fp32 (no autocast) and under
bf16 autocast, as max abs / relative norm difference of the loss, the logits and every
gradient. (P2 has pair_dropout 0, so the pair device's RNG plays no part.)
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # PYTHONSAFEPATH=1 in PBS jobs
from pairformer_profile import is_oom, make_batch  # noqa: E402

import msdelta.models.pairformer as pf  # noqa: E402
from msdelta.models.configuration_msdelta import MSDeltaConfig  # noqa: E402
from msdelta.models.modeling_msdelta import MSDeltaForPreTraining  # noqa: E402

P2_CONFIG = "configs/p2/p2-cz32-k5/config.json"
ALL_MODES = ("lag1_seq", "lag1_conc", "lag1_dev", "lag0_seq")


def mode_overrides(mode: str, offset: int) -> dict:
    return {
        "lag1_seq": dict(pair_bias_lag=1),
        "lag1_conc": dict(pair_bias_lag=1, pair_concurrent=True),
        "lag1_dev": dict(pair_bias_lag=1, pair_device_offset=offset),
        "lag0_seq": dict(pair_bias_lag=0),
    }[mode]


def backend(dev: torch.device):
    return {"xpu": getattr(torch, "xpu", None), "cuda": torch.cuda}.get(dev.type)


def pick_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu", 0)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def device_count(dev: torch.device) -> int:
    be = backend(dev)
    return be.device_count() if be is not None else 1


def model_devices(model) -> list[torch.device]:
    return sorted({p.device for p in model.parameters()}, key=str)


def sync_all(devs) -> None:
    for d in devs:
        be = backend(d)
        if be is not None:
            be.synchronize(d)


def reset_peaks(devs) -> dict[str, int]:
    gc.collect()
    base = {}
    for d in devs:
        be = backend(d)
        if be is not None:
            be.empty_cache()
            be.reset_peak_memory_stats(d)
            base[str(d)] = be.memory_allocated(d)
    return base


def peaks_since(devs, base) -> dict[str, float]:
    out = {}
    for d in devs:
        be = backend(d)
        if be is not None:
            out[str(d)] = (be.max_memory_allocated(d) - base[str(d)]) / 1e9
    return out


def make_config(path: str, layers: int, k: int, mode: str, offset: int) -> MSDeltaConfig:
    d = json.loads(Path(path).read_text())
    d.update(num_hidden_layers=layers, pair_update_every=k, **mode_overrides(mode, offset))
    return MSDeltaConfig(**d)


def build(cfg: MSDeltaConfig, dev: torch.device, seed: int = 0) -> MSDeltaForPreTraining:
    torch.manual_seed(seed)
    model = MSDeltaForPreTraining(cfg).to(dev).train()
    if getattr(cfg, "pair_device_offset", 0):
        model.msdelta.bias_module.place_pair_stream(dev)  # after .to (K176-P)
    return model


def step(model, batch, dev, autocast: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ctx = (torch.autocast(dev.type, dtype=torch.bfloat16) if autocast
           else contextlib.nullcontext())
    with ctx:
        out = model(**batch)
    out.loss.backward()
    return out.loss.detach(), out.logits.detach()


def timed(model, batch, dev, warmup: int, reps: int) -> dict:
    devs = model_devices(model)
    try:
        for _ in range(warmup):
            step(model, batch, dev, True)
            model.zero_grad(set_to_none=True)
        sync_all(devs)
        base = reset_peaks(devs)
        times = []
        for _ in range(reps):
            sync_all(devs)
            t0 = time.perf_counter()
            step(model, batch, dev, True)
            sync_all(devs)
            times.append(time.perf_counter() - t0)
            model.zero_grad(set_to_none=True)
        peaks = peaks_since(devs, base)
        return dict(status="ok", step_ms=1e3 * statistics.median(times),
                    steps_ms=[1e3 * t for t in times], devices=[str(d) for d in devs],
                    peak_gb_per_device=peaks,
                    peak_gb=max(peaks.values()) if peaks else None)
    except Exception as e:  # noqa: BLE001 -- recorded, the sweep goes on
        if not is_oom(e):
            traceback.print_exc()
        return dict(status="oom" if is_oom(e) else "error",
                    error=f"{type(e).__name__}: {e}"[:500])
    finally:
        model.zero_grad(set_to_none=True)


def copy_volume(model, batch, dev) -> dict:
    """Count the cross-device copies of one forward (bf16 autocast, no grad)."""
    calls = []
    real = pf.transfer

    def counting(t, device):
        if t.device != torch.device(device):
            calls.append((tuple(t.shape), str(t.dtype), t.numel() * t.element_size()))
        return real(t, device)

    pf.transfer = counting
    try:
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
            model(**batch)
    finally:
        pf.transfer = real
    n_updates = sum(layer.updates for layer in model.msdelta.bias_module.layers)
    total = sum(c[2] for c in calls)
    # Masks (2 bool copies) and z_init cross once; then per update one s over, one z back.
    once = sum(c[2] for c in calls[:3])
    return dict(calls=[list(c) for c in calls], fwd_bytes_total=total, n_updates=n_updates,
                fwd_bytes_once=once,
                fwd_bytes_per_round=(total - once) / n_updates if n_updates else None,
                note="backward moves the same volume of gradients (masks excepted)")


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float().to(b.device), b.float()
    d = (a - b).norm().item()
    return dict(max_abs=(a - b).abs().max().item(), rel_norm=d / max(b.norm().item(), 1e-30))


def compare(ref, other, batch, dev, autocast: bool) -> dict:
    """One step each from the same seed; loss / logits / per-parameter gradient differences."""
    res = {}
    for name, model in (("ref", ref), ("other", other)):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1234)
        loss, logits = step(model, batch, dev, autocast)
        sync_all(model_devices(model))
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        res[name] = (loss, logits, grads)
        model.zero_grad(set_to_none=True)
    (lr, xr, gr), (lo, xo, go) = res["ref"], res["other"]
    grad_diffs = {n: _diff(go[n], gr[n]) for n in gr if n in go}
    worst = max(grad_diffs.items(), key=lambda kv: kv[1]["rel_norm"]) if grad_diffs else None
    tol = 2e-2 if autocast else 1e-5
    out = dict(
        autocast="bf16" if autocast else "fp32",
        loss_ref=lr.item(), loss_other=lo.item(), loss=_diff(lo, lr), logits=_diff(xo, xr),
        grads_same_keys=gr.keys() == go.keys(), n_grads=len(gr),
        grad_max_rel_norm=worst[1]["rel_norm"] if worst else None,
        grad_worst_param=worst[0] if worst else None,
        bitwise_equal=bool(torch.equal(lo.to(lr.device), lr) and torch.equal(xo.to(xr.device), xr)
                           and all(torch.equal(go[n].to(gr[n].device), gr[n])
                                   for n in gr if n in go)),
    )
    out["match"] = bool(out["grads_same_keys"] and out["loss"]["rel_norm"] <= tol
                        and out["logits"]["rel_norm"] <= tol
                        and (out["grad_max_rel_norm"] or 0.0) <= tol)
    return out


def run_setting(a, dev, layers: int, k: int, batch, modes: list[str], dev_ok: str | None) -> dict:
    row = dict(layers=layers, k=k, modes={}, numerics={})
    cfg = lambda mode: make_config(a.config, layers, k, mode, a.pair_device_offset)  # noqa: E731
    # Numerics first: lag1_conc / lag1_dev on lag1_seq's weights.
    seq = None
    try:
        seq = build(cfg("lag1_seq"), dev)
        for mode in ("lag1_conc", "lag1_dev"):
            if mode not in modes or (mode == "lag1_dev" and dev_ok):
                continue
            other = build(cfg(mode), dev)
            other.load_state_dict(seq.state_dict())  # copies in place, keeps the placement
            for autocast in (False, True):
                key = f"{mode}_{'bf16' if autocast else 'fp32'}"
                row["numerics"][key] = compare(seq, other, batch, dev, autocast)
            if mode == "lag1_dev":
                row["copy_volume"] = copy_volume(other, batch, dev)
                row["pair_device"] = str(other.msdelta.bias_module.pair_device())
            other = None
            gc.collect()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        row["numerics"]["error"] = f"{type(e).__name__}: {e}"[:500]
    finally:
        seq = None
        gc.collect()
    for mode in modes:
        if mode == "lag1_dev" and dev_ok:
            row["modes"][mode] = dict(status="skipped", error=dev_ok)
            continue
        model = build(cfg(mode), dev)
        rec = timed(model, batch, dev, a.warmup, a.reps)
        rec["n_pair_updates"] = sum(layer.updates for layer in model.msdelta.bias_module.layers)
        row["modes"][mode] = rec
        print(f"[k172] L={layers} k={k} {mode:9s} {rec['status']} "
              f"{rec.get('step_ms', float('nan')):8.1f} ms  peak {rec.get('peak_gb_per_device')} "
              f"GB  updates {rec['n_pair_updates']}", flush=True)
        model = None
        gc.collect()
    m = row["modes"]
    ok = lambda x: m.get(x, {}).get("status") == "ok"  # noqa: E731
    for fast in ("lag1_conc", "lag1_dev"):
        if ok("lag1_seq") and ok(fast):
            row[f"speedup_{fast}_vs_lag1_seq"] = m["lag1_seq"]["step_ms"] / m[fast]["step_ms"]
        if ok("lag0_seq") and ok(fast):
            row[f"speedup_{fast}_vs_lag0"] = m["lag0_seq"]["step_ms"] / m[fast]["step_ms"]
    for key, num in row["numerics"].items():
        if isinstance(num, dict):
            print(f"[k172] L={layers} k={k} numerics {key}: match={num['match']} "
                  f"bitwise={num['bitwise_equal']} loss_rel={num['loss']['rel_norm']:.2e} "
                  f"grad_max_rel={num['grad_max_rel_norm']:.2e}", flush=True)
    if "copy_volume" in row:
        cv = row["copy_volume"]
        per_round = (cv["fwd_bytes_per_round"] or 0.0) / 1e6
        print(f"[k172] L={layers} k={k} lag1_dev copies/forward: "
              f"{cv['fwd_bytes_total'] / 1e6:.1f} MB ({per_round:.1f} MB per round, "
              f"{cv['fwd_bytes_once'] / 1e6:.1f} MB once) to {row.get('pair_device')}", flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=P2_CONFIG)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--peaks", type=int, default=150)
    p.add_argument("--ks", default="5,7")
    p.add_argument("--layers", default="10,14")
    p.add_argument("--modes", default=",".join(ALL_MODES))
    p.add_argument("--pair-device-offset", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=20)
    a = p.parse_args()

    dev = pick_device()
    modes = [m.strip() for m in a.modes.replace("+", ",").split(",") if m.strip()]
    n_dev = device_count(dev)
    names = ([backend(dev).get_device_name(i) for i in range(n_dev)]
             if dev.type != "cpu" else ["cpu"])
    dev_ok = None  # None = lag1_dev can run, else the reason it cannot
    pair_index = (dev.index or 0) + a.pair_device_offset
    if "lag1_dev" in modes and dev.type != "cpu" and pair_index >= n_dev:
        dev_ok = (f"pair device index {pair_index} does not exist "
                  f"({n_dev} {dev.type} device(s) visible)")
        print(f"[k172] lag1_dev SKIPPED: {dev_ok}", flush=True)
    batch = {k: v.to(dev) for k, v in make_batch(a.batch, a.peaks).items()}
    meta = dict(
        torch=torch.__version__, device=str(dev), device_count=n_dev, device_names=names,
        config=a.config, batch=a.batch, peaks=a.peaks, warmup=a.warmup, reps=a.reps,
        modes=modes, pair_device_offset=a.pair_device_offset, lag1_dev_skipped=dev_ok,
        autocast="bf16", job=os.environ.get("PBS_JOBID", "local"),
        frameworks=os.environ.get("FRAMEWORKS_MODULE", "frameworks/2025.3.1"),
        env={k: os.environ.get(k) for k in ("ZE_AFFINITY_MASK", "ZEX_NUMBER_OF_CCS",
                                             "ONEAPI_DEVICE_SELECTOR", "ZE_FLAT_DEVICE_HIERARCHY",
                                             "K176_MODE")},
        python=sys.executable,
    )
    print(f"[k172] {meta}", flush=True)
    rows = []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for layers in (int(x) for x in a.layers.replace("+", ",").split(",")):
        for k in (int(x) for x in a.ks.replace("+", ",").split(",")):
            rows.append(run_setting(a, dev, layers, k, batch, modes, dev_ok))
            out.write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))  # partial results
    print(f"[k172] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
