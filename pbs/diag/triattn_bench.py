"""K102: memory and time of ONE triangle-attention module, naive vs checkpointed vs SDPA.

    python pbs/diag/triattn_bench.py --out results/raw/diag/triattn_bench/<jobid>.json

Stage 0 card dims (notes/P1_stage0_card.md): c_z = 64, 4 heads x 16, chunk 32. One module
(starting node), fp32 weights and input under bf16 autocast as in training, padded keys
(every other spectrum has 15% padding). Per configuration: 1 warm-up step, then the median
of 3 timed forward+backward steps, and the peak device memory of one step above what was
allocated before it (weights, z). Out-of-memory is recorded, not fatal.

Variants: naive / naive+ckpt / sdpa / sdpa+ckpt (the 4-D flattened SDPA call used by the
model) and sdpa5d / sdpa5d+ckpt (the 5-D broadcast-mask call). Which SDPA kernel ran is read
from a profiler trace of one step (aten op names), with and without a grad-requiring mask.
"""
import argparse
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch

from msdelta.models.configuration_msdelta import MSDeltaConfig
from msdelta.models.pairformer import TriangleAttention

VARIANTS = {
    "naive": dict(impl="naive", ckpt=False, flatten=True),
    "naive+ckpt": dict(impl="naive", ckpt=True, flatten=True),
    "sdpa": dict(impl="sdpa", ckpt=False, flatten=True),
    "sdpa+ckpt": dict(impl="sdpa", ckpt=True, flatten=True),
    "sdpa5d": dict(impl="sdpa", ckpt=False, flatten=False),
    "sdpa5d+ckpt": dict(impl="sdpa", ckpt=True, flatten=False),
}


def build(variant, c_z, heads, dim, chunk, dev):
    v = VARIANTS[variant]
    cfg = MSDeltaConfig(architecture="pairformer", pair_channels=c_z,
                        pair_use_triangle_attention=True, pair_tri_attn_heads=heads,
                        pair_tri_attn_dim=dim, pair_tri_attn_chunk=chunk,
                        pair_tri_attn_impl=v["impl"],
                        pair_tri_attn_checkpoint_chunks=v["ckpt"])
    torch.manual_seed(0)
    m = TriangleAttention(cfg, starting=True).to(dev)
    m.sdpa_flatten = v["flatten"]
    return m


def inputs(b, n, c_z, dev):
    g = torch.Generator().manual_seed(0)
    z = torch.randn(b, n, n, c_z, generator=g).to(dev)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[1::2, n - int(0.15 * n):] = False
    return z, mask.to(dev)


def step(m, z, mask, dev):
    zz = z.detach().requires_grad_(True)
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        out = m(zz, mask)
    out.float().sum().backward()
    m.zero_grad(set_to_none=True)


def sync(dev):
    if dev.type == "xpu":
        torch.xpu.synchronize()


def is_oom(e):
    s = str(e).lower()
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in s or "out_of_device_memory" in s


def run_config(variant, b, n, a, dev):
    rec = dict(variant=variant, B=b, N=n)
    m = z = mask = None
    try:
        m = build(variant, a.c_z, a.heads, a.dim, a.chunk, dev)
        z, mask = inputs(b, n, a.c_z, dev)
        step(m, z, mask, dev)  # warm-up
        sync(dev)
        times = []
        peak = None
        for _ in range(a.reps):
            if dev.type == "xpu":
                torch.xpu.empty_cache()
                base = torch.xpu.memory_allocated()
                torch.xpu.reset_peak_memory_stats()
            t0 = time.perf_counter()
            step(m, z, mask, dev)
            sync(dev)
            times.append(time.perf_counter() - t0)
            if dev.type == "xpu":
                p = torch.xpu.max_memory_allocated() - base
                peak = p if peak is None else max(peak, p)
        rec.update(status="ok", time_ms=1e3 * statistics.median(times),
                   times_ms=[1e3 * t for t in times],
                   peak_gb=None if peak is None else peak / 1e9)
    except Exception as e:  # noqa: BLE001 -- record and continue
        rec.update(status="oom" if is_oom(e) else "error", error=f"{type(e).__name__}: {e}"[:500])
        if not is_oom(e):
            traceback.print_exc()
    finally:
        del m, z, mask
        if dev.type == "xpu":
            torch.xpu.empty_cache()
    print(f"[bench] {json.dumps(rec)}", flush=True)
    return rec


def _relevant(name):
    return any(t in name for t in ("attention", "softmax", "bmm", "matmul", "flash", "sdp"))


def sdpa_kernel_probe(dev, a):
    """Which aten SDPA op the XPU dispatcher picked, for the model's call shapes."""
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile
    from torch.utils._python_dispatch import TorchDispatchMode
    res = {}
    b, c, n, h, d = 8, a.chunk, 100, a.heads, a.dim
    for layout in ("4d", "5d"):
        for mask_grad in (True, False):
            key = f"{layout}_mask_grad={mask_grad}"
            try:
                shape = (b * c, h, n, d) if layout == "4d" else (b, c, h, n, d)
                mshape = (b * c, h, n, n) if layout == "4d" else (b, 1, h, n, n)
                q, k, v = (torch.randn(shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
                           for _ in range(3))
                am = torch.randn(mshape, device=dev, dtype=torch.bfloat16,
                                 requires_grad=mask_grad)
                names = None
                for _ in range(2):  # Kineto can fail once on XPU (PTI_ERROR_INTERNAL)
                    try:
                        with profile(activities=[ProfilerActivity.CPU]) as prof:
                            out = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
                            out.float().sum().backward()
                            sync(dev)
                        names = sorted({e.key for e in prof.key_averages() if _relevant(e.key)})
                        break
                    except RuntimeError:
                        q.grad = k.grad = v.grad = am.grad = None
                # Dispatcher view as well (the aten ops actually called, fwd + bwd).
                seen = set()

                class Log(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        seen.add(str(func.overloadpacket.__name__))
                        return func(*args, **(kwargs or {}))

                with Log():
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
                    out.float().sum().backward()
                sync(dev)
                names = dict(profiler=names, dispatch=sorted(n for n in seen if _relevant(n)))
                res[key] = names
            except Exception as e:  # noqa: BLE001
                res[key] = [f"error: {type(e).__name__}: {str(e)[:300]}"]
            print(f"[probe] {key}: {res[key]}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="8,32")
    ap.add_argument("--peaks", default="100,150")
    ap.add_argument("--c-z", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    dev = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    env = dict(torch=torch.__version__, device=str(dev),
               device_name=torch.xpu.get_device_name(0) if dev.type == "xpu" else None,
               ZE_AFFINITY_MASK=os.environ.get("ZE_AFFINITY_MASK"),
               job=os.environ.get("PBS_JOBID"))
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        env["ipex"] = ipex.__version__
    except Exception as e:  # noqa: BLE001
        env["ipex"] = f"not imported ({type(e).__name__})"
    if dev.type == "xpu":
        env["device_total_gb"] = torch.xpu.get_device_properties(0).total_memory / 1e9
    print(f"[env] {env}", flush=True)
    probe = sdpa_kernel_probe(dev, a)
    rows = []
    for b in map(int, a.batches.split(",")):
        for n in map(int, a.peaks.split(",")):
            for variant in a.variants.split(","):
                rows.append(run_config(variant, b, n, a, dev))
    out = dict(env=env, dims=dict(c_z=a.c_z, heads=a.heads, dim=a.dim, chunk=a.chunk,
                                  reps=a.reps, autocast="bf16"),
               sdpa_kernel_probe=probe, results=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[bench] wrote {a.out}", flush=True)
    print(f"{'variant':<13}{'B':>4}{'N':>5}{'ms':>10}{'peak GB':>10}  status")
    for r in rows:
        ms = f"{r['time_ms']:.1f}" if r.get("time_ms") is not None else "-"
        pk = f"{r['peak_gb']:.2f}" if r.get("peak_gb") is not None else "-"
        print(f"{r['variant']:<13}{r['B']:>4}{r['N']:>5}{ms:>10}{pk:>10}  {r['status']}")


if __name__ == "__main__":
    main()
