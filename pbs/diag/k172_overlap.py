"""K172-P: does ``pair_concurrent`` (pair update on a side stream) overlap the two streams?

    python pbs/diag/k172_overlap.py --out results/raw/diag/k172_overlap/<jobid>.json

One tile. The P2 Pairformer (configs/p2/p2-cz32-k5/config.json: 512 x 10, c_z 32, outgoing
triangle multiplication, factored write-back, no triangle attention), with k =
``pair_update_every`` in --ks and num_hidden_layers in --layers, three modes each:

  lag1_seq    pair_bias_lag=1, pair_concurrent=False  (the lag-1 schedule, one stream)
  lag1_conc   pair_bias_lag=1, pair_concurrent=True   (update m on a side stream, K172-P)
  lag0_seq    pair_bias_lag=0 (reference; one pair update more than lag 1 at the same k)

Whole pretraining steps (MSDeltaForPreTraining, masked-intensity loss, bf16 autocast over
fp32 weights, train mode so dropout is on, no optimizer step), timed forward+backward with
a device sync around each step: median over --reps steps after --warmup, plus peak memory
above the weights. Batches come from ``pairformer_profile.make_batch`` (random spectra padded
and masked by the pretraining collator).

Numerics: lag1_conc gets lag1_seq's weights (load_state_dict) and both run one step from the
same RNG seed (the dropout draws are issued in the same order), compared twice: in fp32 (no
autocast; kernels are the same, so the difference should be ~0 unless a kernel is
nondeterministic) and under bf16 autocast (tolerance, reported as max abs / relative norm
difference of the loss, the logits and every gradient).
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
from pairformer_profile import is_oom, make_batch, peak_since, reset_peak, sync  # noqa: E402

import msdelta.models.pairformer as pf  # noqa: E402
from msdelta.models.configuration_msdelta import MSDeltaConfig  # noqa: E402
from msdelta.models.modeling_msdelta import MSDeltaForPreTraining  # noqa: E402

P2_CONFIG = "configs/p2/p2-cz32-k5/config.json"
MODES = {
    "lag1_seq": dict(pair_bias_lag=1, pair_concurrent=False),
    "lag1_conc": dict(pair_bias_lag=1, pair_concurrent=True),
    "lag0_seq": dict(pair_bias_lag=0, pair_concurrent=False),
}


def pick_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu", 0)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def make_config(path: str, layers: int, k: int, mode: str) -> MSDeltaConfig:
    d = json.loads(Path(path).read_text())
    d.update(num_hidden_layers=layers, pair_update_every=k, **MODES[mode])
    return MSDeltaConfig(**d)


def build(cfg: MSDeltaConfig, dev: torch.device, seed: int = 0) -> MSDeltaForPreTraining:
    torch.manual_seed(seed)
    return MSDeltaForPreTraining(cfg).to(dev).train()


def step(model, batch, dev, autocast: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ctx = (torch.autocast(dev.type, dtype=torch.bfloat16) if autocast
           else contextlib.nullcontext())
    with ctx:
        out = model(**batch)
    out.loss.backward()
    return out.loss.detach(), out.logits.detach()


def timed(model, batch, dev, warmup: int, reps: int) -> dict:
    try:
        for _ in range(warmup):
            step(model, batch, dev, True)
            model.zero_grad(set_to_none=True)
        base = reset_peak(dev)
        times = []
        for _ in range(reps):
            sync(dev)
            t0 = time.perf_counter()
            step(model, batch, dev, True)
            sync(dev)
            times.append(time.perf_counter() - t0)
            model.zero_grad(set_to_none=True)
        return dict(status="ok", step_ms=1e3 * statistics.median(times),
                    steps_ms=[1e3 * t for t in times], peak_gb=peak_since(dev, base))
    except Exception as e:  # noqa: BLE001 -- recorded, the sweep goes on
        if not is_oom(e):
            traceback.print_exc()
        return dict(status="oom" if is_oom(e) else "error", error=f"{type(e).__name__}: {e}"[:500])
    finally:
        model.zero_grad(set_to_none=True)


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    d = (a - b).norm().item()
    return dict(max_abs=(a - b).abs().max().item(), rel_norm=d / max(b.norm().item(), 1e-30))


def compare(seq, conc, batch, dev, autocast: bool) -> dict:
    """One step each from the same seed; loss / logits / per-parameter gradient differences."""
    res = {}
    for name, model in (("seq", seq), ("conc", conc)):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1234)
        loss, logits = step(model, batch, dev, autocast)
        sync(dev)
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        res[name] = (loss, logits, grads)
        model.zero_grad(set_to_none=True)
    (ls, gs_logits, gs), (lc, gc_logits, gcn) = res["seq"], res["conc"]
    grad_diffs = {n: _diff(gcn[n], gs[n]) for n in gs if n in gcn}
    worst = max(grad_diffs.items(), key=lambda kv: kv[1]["rel_norm"]) if grad_diffs else None
    tol = 2e-2 if autocast else 1e-5
    out = dict(
        autocast="bf16" if autocast else "fp32",
        loss_seq=ls.item(), loss_conc=lc.item(), loss=_diff(lc, ls),
        logits=_diff(gc_logits, gs_logits),
        grads_same_keys=gs.keys() == gcn.keys(), n_grads=len(gs),
        grad_max_rel_norm=worst[1]["rel_norm"] if worst else None,
        grad_worst_param=worst[0] if worst else None,
        bitwise_equal=bool(torch.equal(lc, ls) and torch.equal(gc_logits, gs_logits)
                           and all(torch.equal(gcn[n], gs[n]) for n in gs if n in gcn)),
    )
    out["match"] = bool(out["grads_same_keys"] and out["loss"]["rel_norm"] <= tol
                        and out["logits"]["rel_norm"] <= tol
                        and (out["grad_max_rel_norm"] or 0.0) <= tol)
    return out


def run_setting(a, dev, layers: int, k: int, batch) -> dict:
    row = dict(layers=layers, k=k, modes={}, numerics={})
    # Numerics first (lag1_conc on lag1_seq's weights).
    try:
        seq = build(make_config(a.config, layers, k, "lag1_seq"), dev)
        conc = build(make_config(a.config, layers, k, "lag1_conc"), dev)
        conc.load_state_dict(seq.state_dict())
        row["streams"] = type(pf.device_streams(dev)).__name__
        for autocast in (False, True):
            key = "bf16" if autocast else "fp32"
            row["numerics"][key] = compare(seq, conc, batch, dev, autocast)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        row["numerics"]["error"] = f"{type(e).__name__}: {e}"[:500]
    finally:
        seq = conc = None
        gc.collect()
    for mode in a.modes.split(","):
        model = build(make_config(a.config, layers, k, mode), dev)
        rec = timed(model, batch, dev, a.warmup, a.reps)
        rec["n_pair_updates"] = sum(layer.updates for layer in model.msdelta.bias_module.layers)
        row["modes"][mode] = rec
        print(f"[k172] L={layers} k={k} {mode:9s} {rec['status']} "
              f"{rec.get('step_ms', float('nan')):8.1f} ms  peak {rec.get('peak_gb')} GB  "
              f"updates {rec['n_pair_updates']}", flush=True)
        model = None
        gc.collect()
    m = row["modes"]
    if all(m.get(x, {}).get("status") == "ok" for x in ("lag1_seq", "lag1_conc")):
        row["speedup_conc_vs_seq"] = m["lag1_seq"]["step_ms"] / m["lag1_conc"]["step_ms"]
    if all(m.get(x, {}).get("status") == "ok" for x in ("lag0_seq", "lag1_conc")):
        row["speedup_conc_vs_lag0"] = m["lag0_seq"]["step_ms"] / m["lag1_conc"]["step_ms"]
    for key in ("fp32", "bf16"):
        num = row["numerics"].get(key)
        if num:
            print(f"[k172] L={layers} k={k} numerics {key}: match={num['match']} "
                  f"bitwise={num['bitwise_equal']} loss_rel={num['loss']['rel_norm']:.2e} "
                  f"grad_max_rel={num['grad_max_rel_norm']:.2e}", flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=P2_CONFIG)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--peaks", type=int, default=150)
    p.add_argument("--ks", default="5,7")
    p.add_argument("--layers", default="10,14")
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=20)
    a = p.parse_args()

    dev = pick_device()
    batch = {k: v.to(dev) for k, v in make_batch(a.batch, a.peaks).items()}
    meta = dict(
        torch=torch.__version__, device=str(dev),
        device_name=(torch.xpu.get_device_name(dev) if dev.type == "xpu"
                     else torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"),
        config=a.config, batch=a.batch, peaks=a.peaks, warmup=a.warmup, reps=a.reps,
        autocast="bf16", job=os.environ.get("PBS_JOBID", "local"),
        frameworks=os.environ.get("FRAMEWORKS_MODULE", "frameworks/2025.3.1"),
        python=sys.executable,
    )
    print(f"[k172] {meta}", flush=True)
    rows = []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for layers in (int(x) for x in a.layers.split(",")):
        for k in (int(x) for x in a.ks.split(",")):
            rows.append(run_setting(a, dev, layers, k, batch))
            out.write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))  # partial results
    print(f"[k172] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
