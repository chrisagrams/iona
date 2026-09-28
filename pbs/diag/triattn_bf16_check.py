"""K115: numerical accuracy of triangle attention under bf16 (naive vs SDPA vs all-bf16).

    python pbs/diag/triattn_bf16_check.py --out results/raw/diag/triattn_bf16/<jobid>.json

Stage 0 card dims (c_z = 64, 4 heads x 16, chunk 32), one starting-node and one ending-node
module. Per (B, N): the SAME inputs and weights for every variant --

  - weights: the model's init (MSDeltaPreTrainedModel._init_weights: Linear ~ N(0, 0.02),
    biases 0, LayerNorm 1/0), fixed seed; optionally also a "sharp" init (q/k/bias weights
    ~ N(0, 0.1)) that gives peaked attention, closer to a trained model than the near-uniform
    softmax of the 0.02 init;
  - z: N(0, 1) normalised per pair vector (LayerNorm without affine), fixed seed;
  - padding: lengths spread uniformly over 50..N (one spectrum at N), keys masked as in the
    model; the upstream gradient G ~ N(0, 1) is zeroed on padded pairs (i or j padded), as the
    model's masked losses would, and the loss is sum(out * G).

Variants (vs REF, fp32, no autocast, naive):
  A      naive, bf16 autocast (the current training setting)
  B      sdpa 4-D (flattened), bf16 autocast         B+ckpt  same with per-chunk checkpoint
  C      sdpa 5-D (sdpa_flatten=False), bf16 autocast C+ckpt
  D      all bf16 (module + z cast, no autocast), naive -- the naive chunk still does its
         logits + bias + mask and softmax in fp32 (that upcast is in the code)
  Dpure  as D but the chunk stays bf16 too (logits, mask, softmax): no fp32 anywhere in our code
  E      all bf16, sdpa 4-D                           E+ckpt
Also REF64 (fp64, naive) vs REF, to show the fp32 reference itself is accurate.

Metrics per variant and module: max abs error and relative L2 error ||x - ref|| / ||ref|| of
the output (valid pairs only), dL/dz, and every parameter gradient (summary: worst and median
rel-L2 over parameters, plus which parameter is worst). Timing / peak memory as in
triattn_bench.py (1 warm-up, median of 3 fwd+bwd steps, peak above the pre-step allocation).
The SDPA backend is read from a profiler trace of one module step per SDPA variant, plus the
kernel probe of triattn_bench.py.
"""
import argparse
import copy
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triattn_bench import _relevant, is_oom, sdpa_kernel_probe, sync  # noqa: E402

from msdelta.models.configuration_msdelta import MSDeltaConfig  # noqa: E402
from msdelta.models.pairformer import TriangleAttention  # noqa: E402

# name: (impl, ckpt, flatten, mode); mode in fp32 | fp64 | autocast | bf16 | bf16pure
VARIANTS = {
    "REF": ("naive", False, True, "fp32"),
    "REF64": ("naive", False, True, "fp64"),
    "A": ("naive", False, True, "autocast"),
    "B": ("sdpa", False, True, "autocast"),
    "B+ckpt": ("sdpa", True, True, "autocast"),
    "C": ("sdpa", False, False, "autocast"),
    "C+ckpt": ("sdpa", True, False, "autocast"),
    "D": ("naive", False, True, "bf16"),
    "Dpure": ("naive", False, True, "bf16pure"),
    "E": ("sdpa", False, True, "bf16"),
    "E+ckpt": ("sdpa", True, True, "bf16"),
}


def _chunk_naive_pure(self, q, k, v, bias, key_mask):
    """_chunk_naive without the fp32 upcast: logits, bias, mask and softmax in q's dtype."""
    scale = 1.0 / (self.d ** 0.5)
    logits = torch.einsum("bcjhd,bckhd->bcjkh", q, k) * scale
    km = key_mask.clamp(min=torch.finfo(q.dtype).min).to(q.dtype)
    logits = logits + bias[:, None] + km
    attn = torch.softmax(logits, dim=3)
    return torch.einsum("bcjkh,bckhd->bcjhd", attn, v)


def base_module(starting, a, init, seed):
    cfg = MSDeltaConfig(architecture="pairformer", pair_channels=a.c_z,
                        pair_use_triangle_attention=True, pair_tri_attn_heads=a.heads,
                        pair_tri_attn_dim=a.dim, pair_tri_attn_chunk=a.chunk)
    torch.manual_seed(seed)
    m = TriangleAttention(cfg, starting=starting)
    g = torch.Generator().manual_seed(seed)
    std = cfg.initializer_range
    with torch.no_grad():  # MSDeltaPreTrainedModel._init_weights
        for name, lin in (("q", m.q), ("k", m.k), ("v", m.v), ("bias", m.bias),
                          ("gate", m.gate), ("out", m.out)):
            s = 0.1 if (init == "sharp" and name in ("q", "k", "bias")) else std
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * s)
            if lin.bias is not None:
                lin.bias.zero_()
        m.norm.weight.fill_(1.0)
        m.norm.bias.zero_()
    return m


def make_variant(base, name, dev):
    impl, ckpt, flatten, mode = VARIANTS[name]
    m = copy.deepcopy(base)
    m.impl, m.checkpoint_chunks, m.sdpa_flatten = impl, ckpt, flatten
    if mode == "fp64":
        m = m.double()
    elif mode in ("bf16", "bf16pure"):
        m = m.to(torch.bfloat16)
    if mode == "bf16pure":
        m._chunk_naive = _chunk_naive_pure.__get__(m)
    return m.to(dev), mode


def inputs(b, n, c_z, seed):
    g = torch.Generator().manual_seed(seed)
    z = F.layer_norm(torch.randn(b, n, n, c_z, generator=g), (c_z,))
    lengths = torch.randint(50, n + 1, (b,), generator=g)
    lengths[0] = n
    mask = torch.arange(n)[None, :] < lengths[:, None]
    pair = mask[:, :, None] & mask[:, None, :]
    grad_out = torch.randn(b, n, n, c_z, generator=g) * pair[..., None]
    return z, mask, pair, grad_out, lengths.tolist()


def run_step(m, mode, z, mask, grad_out, dev, want_grads):
    dt = {"fp64": torch.float64, "bf16": torch.bfloat16, "bf16pure": torch.bfloat16}.get(
        mode, torch.float32)
    zz = z.to(dt).detach().requires_grad_(want_grads)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=(mode == "autocast")):
        out = m(zz, mask)
    loss = (out.to(grad_out.dtype) * grad_out).sum()
    loss.backward()
    res = None
    if want_grads:
        res = dict(out=out.detach().double(), dz=zz.grad.detach().double(),
                   params={k: p.grad.detach().double() for k, p in m.named_parameters()})
    m.zero_grad(set_to_none=True)
    return res


def err(x, ref, sel=None):
    if sel is not None:
        x, ref = x[sel], ref[sel]
    d = x - ref
    return dict(max_abs=d.abs().max().item(), rel_l2=(d.norm() / ref.norm().clamp_min(1e-300)).item(),
                ref_max_abs=ref.abs().max().item(),
                finite=bool(torch.isfinite(x).all().item()))


def compare(res, ref, pair):
    out = err(res["out"], ref["out"], pair)
    dz = err(res["dz"], ref["dz"])
    per = {k: err(res["params"][k], ref["params"][k]) for k in ref["params"]}
    rels = sorted(v["rel_l2"] for v in per.values())
    worst = max(per, key=lambda k: per[k]["rel_l2"])
    return dict(out=out, dz=dz, param_worst=dict(name=worst, **per[worst]),
                param_median_rel_l2=statistics.median(rels), params=per)


def time_mem(m, mode, z, mask, grad_out, dev, reps):
    run_step(m, mode, z, mask, grad_out, dev, want_grads=True)  # warm-up
    sync(dev)
    times, peak = [], None
    for _ in range(reps):
        if dev.type == "xpu":
            torch.xpu.empty_cache()
            base = torch.xpu.memory_allocated()
            torch.xpu.reset_peak_memory_stats()
        t0 = time.perf_counter()
        run_step(m, mode, z, mask, grad_out, dev, want_grads=True)
        sync(dev)
        times.append(time.perf_counter() - t0)
        if dev.type == "xpu":
            p = torch.xpu.max_memory_allocated() - base
            peak = p if peak is None else max(peak, p)
    return dict(time_ms=1e3 * statistics.median(times), times_ms=[1e3 * t for t in times],
                peak_gb=None if peak is None else peak / 1e9)


def module_backend(m, mode, z, mask, grad_out, dev):
    """Profiler op names (attention/softmax/bmm/...) of one module step."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(2):  # Kineto can fail once on XPU
        try:
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                run_step(m, mode, z, mask, grad_out, dev, want_grads=False)
                sync(dev)
            return sorted({e.key for e in prof.key_averages() if _relevant(e.key)})
        except RuntimeError as e:
            last = f"error: {e}"[:300]
            m.zero_grad(set_to_none=True)
    return [last]


def run_size(b, n, a, dev, init, bases):
    z, mask, pair, grad_out, lengths = inputs(b, n, a.c_z, seed=1000 + 7 * b + n)
    rows = []
    for mod_name, base in bases.items():
        zd, maskd, paird, god = z.to(dev), mask.to(dev), pair.to(dev), grad_out.to(dev)
        ref = None
        for vname in a.variants.split(","):
            rec = dict(init=init, module=mod_name, variant=vname, B=b, N=n, lengths=lengths)
            m = None
            try:
                m, mode = make_variant(base, vname, dev)
                go = god.double() if mode == "fp64" else god
                res = run_step(m, mode, zd, maskd, go, dev, want_grads=True)
                if vname == "REF":
                    ref = res
                else:
                    rec["vs_ref"] = compare(res, ref, paird)
                del res
                if a.reps > 0 and init == "model":
                    rec.update(time_mem(m, mode, zd, maskd, go, dev, a.reps))
                if (VARIANTS[vname][0] == "sdpa" and init == "model" and mod_name == "start"
                        and b == 8 and n == 100):
                    rec["profiler_ops"] = module_backend(m, mode, zd, maskd, go, dev)
                rec["status"] = "ok"
            except Exception as e:  # noqa: BLE001 -- record and continue
                rec.update(status="oom" if is_oom(e) else "error",
                           error=f"{type(e).__name__}: {e}"[:500])
                if not is_oom(e):
                    traceback.print_exc()
            finally:
                del m
                if dev.type == "xpu":
                    torch.xpu.empty_cache()
            short = {k: v for k, v in rec.items() if k not in ("lengths",)}
            if "vs_ref" in short:
                v = short["vs_ref"]
                short["vs_ref"] = dict(out=v["out"]["rel_l2"], dz=v["dz"]["rel_l2"],
                                       pw=(v["param_worst"]["name"], v["param_worst"]["rel_l2"]),
                                       pmed=v["param_median_rel_l2"])
            print(f"[check] {json.dumps(short)}", flush=True)
            rows.append(rec)
        del ref
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="8,32")
    ap.add_argument("--peaks", default="100,150")
    ap.add_argument("--inits", default="model,sharp")
    ap.add_argument("--c-z", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    assert a.variants.split(",")[0] == "REF", "REF must come first"
    torch.set_float32_matmul_precision("highest")
    dev = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    env = dict(torch=torch.__version__, device=str(dev),
               device_name=torch.xpu.get_device_name(0) if dev.type == "xpu" else None,
               ZE_AFFINITY_MASK=os.environ.get("ZE_AFFINITY_MASK"),
               job=os.environ.get("PBS_JOBID"),
               float32_matmul_precision=torch.get_float32_matmul_precision())
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        env["ipex"] = ipex.__version__
    except Exception as e:  # noqa: BLE001
        env["ipex"] = f"not imported ({type(e).__name__})"
    if dev.type == "xpu":
        env["device_total_gb"] = torch.xpu.get_device_properties(0).total_memory / 1e9
    print(f"[env] {env}", flush=True)
    probe = None if a.no_probe else sdpa_kernel_probe(dev, a)
    rows = []
    for init in a.inits.split(","):
        bases = {"start": base_module(True, a, init, seed=11),
                 "end": base_module(False, a, init, seed=12)}
        for b in map(int, a.batches.split(",")):
            for n in map(int, a.peaks.split(",")):
                rows.extend(run_size(b, n, a, dev, init, bases))
    out = dict(env=env, dims=dict(c_z=a.c_z, heads=a.heads, dim=a.dim, chunk=a.chunk,
                                  reps=a.reps),
               variants={k: dict(zip(("impl", "ckpt", "flatten", "mode"), v))
                         for k, v in VARIANTS.items()},
               sdpa_kernel_probe=probe, results=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[check] wrote {a.out}", flush=True)
    print(f"{'init':<6}{'mod':<6}{'var':<8}{'B':>3}{'N':>5}{'out rel':>10}{'dz rel':>10}"
          f"{'pworst':>10}{'pmed':>10}  {'worst param':<14}{'ms':>8}{'GB':>7}")
    for r in rows:
        v = r.get("vs_ref")
        f = (lambda x: f"{x:10.2e}")
        cols = (f(v["out"]["rel_l2"]) + f(v["dz"]["rel_l2"]) + f(v["param_worst"]["rel_l2"])
                + f(v["param_median_rel_l2"]) + f"  {v['param_worst']['name']:<14}") if v \
            else f"{'-':>10}" * 4 + f"  {'-':<14}"
        ms = f"{r['time_ms']:8.1f}" if r.get("time_ms") is not None else f"{'-':>8}"
        gb = f"{r['peak_gb']:7.2f}" if r.get("peak_gb") is not None else f"{'-':>7}"
        print(f"{r['init']:<6}{r['module']:<6}{r['variant']:<8}{r['B']:>3}{r['N']:>5}{cols}"
              f"{ms}{gb}  {r['status']}")


if __name__ == "__main__":
    main()
