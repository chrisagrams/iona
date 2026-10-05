"""K119: does FlexAttention (torch.nn.attention.flex_attention) work on Aurora's XPU for
triangle attention, and how does it compare with the SDPA path the model uses?

    python pbs/diag/flexattn_test.py --out $MSDELTA_DIAG/flexattn/<jobid>.json

Intel Triton. The project .venv has upstream triton 3.8.0 (no ``backends/intel``), which
shadows the frameworks module's Intel Triton 3.6.0 in
``/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/lib/python3.12/site-packages``.
When this file runs as a script (never on import, so tests are unaffected) it first builds
a per-process shim directory holding only symlinks to that site-packages' ``triton`` package
and its ``triton-*.dist-info``, puts the shim at ``sys.path[0]`` and at the front of
``PYTHONPATH`` (so inductor's compile subprocesses see the same triton), and only then
imports torch. Only ``triton`` is redirected; every other package still comes from where the
venv puts it. Nothing in the venv is changed. ``MSDELTA_SYSTEM_TRITON=0`` turns it off,
``MSDELTA_SYSTEM_TRITON_SITE`` overrides the directory. The imported triton (version, file,
backends) is printed and recorded.

Steps, each wrapped so that any failure is recorded with its traceback, never a crash:
  a  imports (triton, flex_attention), the active triton driver, a trivial triton kernel.
  b  flex_attention compiled with torch.compile, triangle bias through score_mod, padded keys
     through block_mask (mask_mod) -- and, as a fallback variant, the padding inside score_mod.
     Two layouts for the "bias shared across rows i" pattern:
       rowbatch  rows i folded into the batch dim, (B*N, H, N, d): bias[b // N, h, q, kv]
                 (what the SDPA path does with its expanded mask, without the expansion);
       rowhead   rows i folded into the head dim, (B, N*H, N, d): bias[b, h % H, q, kv].
  c  outputs AND gradients (input z and every parameter) of the full TriangleAttention module
     (starting and ending node) with the flex core vs the current SDPA path
     (msdelta/models/pairformer.py, pair_tri_attn_impl="sdpa") on the same weights and
     inputs, fp32 and bf16 autocast; both also against the naive fp32 path as ground truth.
  d  forward+backward time (1 warm-up/compile step, median of --reps) and peak memory at
     B 8/32 x N 100/150 (Stage 0 card dims: c_z 64, 4 heads x 16, SDPA chunk 32), like
     pbs/diag/triattn_bench.py.
"""
from __future__ import annotations

import os
import sys
import tempfile

SYSTEM_SITE = ("/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/"
               "lib/python3.12/site-packages")


def prefer_system_triton(site: str | None = None) -> dict:
    """Shadow the venv's triton with the system (Intel) one, for this process and its children."""
    site = site or os.environ.get("MSDELTA_SYSTEM_TRITON_SITE", SYSTEM_SITE)
    info = dict(requested=True, site=site)
    if "triton" in sys.modules:
        info["warning"] = "triton already imported before the shim; shim has no effect"
    if not os.path.isdir(os.path.join(site, "triton")):
        info["error"] = f"no triton package under {site}"
        return info
    shim = tempfile.mkdtemp(prefix="msdelta-systriton-")
    linked = []
    for name in sorted(os.listdir(site)):
        if name == "triton" or (name.startswith("triton-") and name.endswith(".dist-info")):
            os.symlink(os.path.join(site, name), os.path.join(shim, name))
            linked.append(name)
    sys.path.insert(0, shim)
    os.environ["PYTHONPATH"] = shim + (":" + os.environ["PYTHONPATH"]
                                       if os.environ.get("PYTHONPATH") else "")
    info.update(shim=shim, linked=linked)
    return info


if __name__ == "__main__" and os.environ.get("MSDELTA_SYSTEM_TRITON", "1") != "0":
    TRITON_SHIM = prefer_system_triton()
else:
    TRITON_SHIM = dict(requested=False)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from msdelta.models.configuration_msdelta import MSDeltaConfig  # noqa: E402
from msdelta.models.pairformer import TriangleAttention  # noqa: E402


# ----------------------------------------------------------------------------------------
# Triangle attention with a pluggable attention core (CPU-testable).
# ----------------------------------------------------------------------------------------

def reference_flex(q, k, v, score_mod=None, mask_mod=None, block_mask=None, scale=None):
    """flex_attention semantics in plain torch: score_mod / mask_mod evaluated on index grids.

    q/k/v ``(B, H, Q, d)``. The index tensors broadcast, so a score_mod written for flex
    (``bias[b // N, h, q_idx, kv_idx]``) runs unchanged. ``block_mask`` is ignored (mask_mod
    carries the same information)."""
    b, h, nq, d = q.shape
    nk = k.shape[2]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    dev = q.device
    bi = torch.arange(b, device=dev).view(b, 1, 1, 1)
    hi = torch.arange(h, device=dev).view(1, h, 1, 1)
    qi = torch.arange(nq, device=dev).view(1, 1, nq, 1)
    ki = torch.arange(nk, device=dev).view(1, 1, 1, nk)
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    if score_mod is not None:
        scores = score_mod(scores, bi, hi, qi, ki)
    if mask_mod is not None:
        scores = scores.masked_fill(~mask_mod(bi, hi, qi, ki), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", attn, v.float()).to(q.dtype)


def make_mods(bias: torch.Tensor, key_valid: torch.Tensor, n: int, h: int, layout: str,
              masking: str):
    """score_mod / mask_mod for the triangle pattern.

    bias ``(B, H, N(j), N(k))`` shared over the query row i; key_valid ``(B, N)`` bool.
    layout "rowbatch": flex batch index bb = b * N + i. "rowhead": flex head hh = i * H + h.
    masking "blockmask": padding via mask_mod (block_mask); "scoremask": inside score_mod."""
    if layout == "rowbatch":
        def bidx(bb, hh):
            return bb // n, hh
    elif layout == "rowhead":
        def bidx(bb, hh):
            return bb, hh % h
    else:
        raise ValueError(layout)

    def mask_mod(bb, hh, q_idx, kv_idx):
        b, _ = bidx(bb, hh)
        return key_valid[b, kv_idx]

    if masking == "blockmask":
        def score_mod(score, bb, hh, q_idx, kv_idx):
            b, hd = bidx(bb, hh)
            return score + bias[b, hd, q_idx, kv_idx]
    elif masking == "scoremask":
        def score_mod(score, bb, hh, q_idx, kv_idx):
            b, hd = bidx(bb, hh)
            return torch.where(key_valid[b, kv_idx], score + bias[b, hd, q_idx, kv_idx],
                               -float("inf"))
    else:
        raise ValueError(masking)
    return score_mod, mask_mod


def flex_triangle_attention(module: TriangleAttention, z: torch.Tensor, mask: torch.Tensor,
                            attn_fn, layout: str = "rowbatch", masking: str = "blockmask",
                            block_mask_fn=None):
    """``TriangleAttention.forward`` with its chunked core replaced by one attention call.

    Same weights, same projections, gate and output as the module; ``attn_fn(q, k, v,
    score_mod=, mask_mod=, block_mask=, scale=)`` is flex_attention (compiled) or
    ``reference_flex``. ``block_mask_fn(mask_mod, B, H, Q, KV)`` builds the BlockMask (only
    with masking="blockmask"; None means no block mask)."""
    if not module.starting:
        z = z.transpose(1, 2)
    z = module.norm(z)
    b, n = z.shape[:2]
    h, d = module.h, module.d
    q = module.q(z).view(b, n, n, h, d)
    k = module.k(z).view(b, n, n, h, d)
    v = module.v(z).view(b, n, n, h, d)
    bias = module.bias(z).to(q.dtype).permute(0, 3, 1, 2)  # (B, H, N(j), N(k))
    # (B, N_i, N_j, H, d) -> (B, N_i, H, N_j, d)
    q, k, v = (t.permute(0, 1, 3, 2, 4) for t in (q, k, v))
    if layout == "rowbatch":
        shape, fb, fh = (b * n, h, n, d), b * n, h
    else:
        shape, fb, fh = (b, n * h, n, d), b, n * h
    q, k, v = (t.reshape(shape) for t in (q, k, v))
    score_mod, mask_mod = make_mods(bias, mask.bool(), n, h, layout, masking)
    block_mask = None
    if masking == "blockmask" and block_mask_fn is not None:
        block_mask = block_mask_fn(mask_mod, fb, None, n, n)
    out = attn_fn(q, k, v, score_mod=score_mod, mask_mod=mask_mod, block_mask=block_mask,
                  scale=1.0 / math.sqrt(d))
    out = out.reshape(b, n, h, n, d).permute(0, 1, 3, 2, 4).reshape(b, n, n, h * d)
    out = torch.sigmoid(module.gate(z)) * out
    out = module.out(out)
    return out if module.starting else out.transpose(1, 2)


def build_module(impl: str, starting: bool, c_z=64, heads=4, dim=16, chunk=32, seed=0):
    cfg = MSDeltaConfig(architecture="pairformer", pair_channels=c_z,
                        pair_use_triangle_attention=True, pair_tri_attn_heads=heads,
                        pair_tri_attn_dim=dim, pair_tri_attn_chunk=chunk,
                        pair_tri_attn_impl=impl)
    torch.manual_seed(seed)
    m = TriangleAttention(cfg, starting=starting)
    # Non-trivial gate/out biases (default init is fine; perturb LayerNorm so it matters).
    with torch.no_grad():
        m.norm.weight.add_(0.1 * torch.randn_like(m.norm.weight))
        m.norm.bias.add_(0.1 * torch.randn_like(m.norm.bias))
    return m


def make_inputs(b, n, c_z, dev, seed=0, pad_frac=0.15):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(b, n, n, c_z, generator=g)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[1::2, n - max(1, int(pad_frac * n)):] = False
    return z.to(dev), mask.to(dev)


def fwd_bwd(fn, module, z, mask, dtype):
    """Run fn(module, z, mask) (optionally under bf16 autocast); return out and grads."""
    module.zero_grad(set_to_none=True)
    zz = z.detach().clone().requires_grad_(True)
    if dtype == "bf16":
        with torch.autocast(z.device.type, dtype=torch.bfloat16):
            out = fn(module, zz, mask)
    else:
        out = fn(module, zz, mask)
    g = torch.Generator().manual_seed(123)
    w = torch.randn(out.shape, generator=g).to(out.device)
    (out.float() * w).sum().backward()
    grads = {"z": zz.grad.detach().float().clone()}
    for name, p in module.named_parameters():
        grads[name] = (p.grad.detach().float().clone() if p.grad is not None
                       else torch.zeros_like(p, dtype=torch.float32))
    module.zero_grad(set_to_none=True)
    return out.detach().float(), grads


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a - b).float()
    ref = b.float()
    return dict(max_abs=diff.abs().max().item(),
                rel_l2=(diff.norm() / ref.norm().clamp_min(1e-30)).item(),
                finite=bool(torch.isfinite(a).all().item()))


def compare_all(out_a, grads_a, out_b, grads_b) -> dict:
    res = {"out": compare(out_a, out_b)}
    for name in grads_b:
        res[f"grad.{name}"] = compare(grads_a[name], grads_b[name])
    res["worst_rel_l2"] = max(v["rel_l2"] for v in res.values() if isinstance(v, dict))
    res["all_finite"] = all(v["finite"] for v in res.values() if isinstance(v, dict))
    return res


def module_call(module, z, mask):
    return module(z, mask)


# ----------------------------------------------------------------------------------------
# GPU steps.
# ----------------------------------------------------------------------------------------

def sync(dev):
    if dev.type == "xpu":
        torch.xpu.synchronize()


def is_oom(e):
    s = str(e).lower()
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in s or "out_of_device_memory" in s


def guarded(record: dict, key: str, fn, *args, **kwargs):
    """Run fn; store its result or the failure (type, message, traceback) under record[key]."""
    t0 = time.perf_counter()
    try:
        res = fn(*args, **kwargs)
        record[key] = dict(status="ok", result=res, seconds=time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001 -- everything is recorded, nothing is fatal
        record[key] = dict(status="oom" if is_oom(e) else "error",
                           error=f"{type(e).__name__}: {e}"[:2000],
                           traceback=traceback.format_exc()[-6000:],
                           seconds=time.perf_counter() - t0)
        print(f"[flex] {key} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
    print(f"[flex] {key}: {record[key]['status']} ({record[key]['seconds']:.1f}s)", flush=True)
    return record[key]


def step_imports(dev):
    import triton
    info = dict(triton_version=triton.__version__, triton_file=triton.__file__,
                triton_realpath=os.path.realpath(triton.__file__))
    bdir = Path(triton.__file__).parent / "backends"
    info["triton_backends"] = sorted(p.name for p in bdir.iterdir()) if bdir.is_dir() else []
    info["intel_backend_present"] = (bdir / "intel").is_dir()
    try:
        from triton.runtime import driver
        drv = driver.active
        info["active_driver"] = type(drv).__module__ + "." + type(drv).__name__
        try:
            info["active_target"] = str(drv.get_current_target())
        except Exception as e:  # noqa: BLE001
            info["active_target"] = f"error: {type(e).__name__}: {e}"[:300]
    except Exception as e:  # noqa: BLE001
        info["active_driver"] = f"error: {type(e).__name__}: {e}"[:500]
    from torch.nn.attention import flex_attention as fa_mod
    info["flex_attention_module"] = fa_mod.__file__
    info["torch_version"] = torch.__version__
    info["device"] = str(dev)
    print(f"[flex] triton {info['triton_version']} from {info['triton_realpath']}; "
          f"backends {info['triton_backends']}; driver {info.get('active_driver')}", flush=True)
    return info


def step_triton_kernel(dev):
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m),
                 mask=m)

    n = 10_000
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    o = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
    sync(dev)
    err = (o - (x + y)).abs().max().item()
    if err > 1e-6:
        raise AssertionError(f"triton add kernel wrong: max err {err}")
    return dict(max_err=err)


def make_flex(dev):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    compiled = torch.compile(flex_attention, dynamic=False)

    def attn_fn(q, k, v, score_mod=None, mask_mod=None, block_mask=None, scale=None):
        return compiled(q, k, v, score_mod=score_mod, block_mask=block_mask, scale=scale)

    def block_mask_fn(mask_mod, B, H, Q, KV):
        return create_block_mask(mask_mod, B, H, Q, KV, device=dev)

    return attn_fn, block_mask_fn


def step_trivial_flex(dev, attn_fn, block_mask_fn):
    """Plain compiled flex_attention (no score_mod, full mask) vs SDPA."""
    import torch.nn.functional as F
    q, k, v = (torch.randn(4, 2, 128, 16, device=dev) for _ in range(3))
    out = attn_fn(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    sync(dev)
    return compare(out, ref)


def step_correctness(dev, attn_fn, block_mask_fn, layout, masking, starting, dtype, b, n):
    sdpa = build_module("sdpa", starting, chunk=16).to(dev)
    naive = build_module("naive", starting, chunk=16).to(dev)
    naive.load_state_dict(sdpa.state_dict())
    z, mask = make_inputs(b, n, 64, dev)

    def flex_call(module, zz, mm):
        return flex_triangle_attention(module, zz, mm, attn_fn, layout, masking, block_mask_fn)

    out_f, g_f = fwd_bwd(flex_call, sdpa, z, mask, dtype)
    out_s, g_s = fwd_bwd(module_call, sdpa, z, mask, dtype)
    out_n, g_n = fwd_bwd(module_call, naive, z, mask, "fp32")
    sync(dev)
    res = dict(flex_vs_sdpa=compare_all(out_f, g_f, out_s, g_s),
               flex_vs_naive_fp32=compare_all(out_f, g_f, out_n, g_n),
               sdpa_vs_naive_fp32=compare_all(out_s, g_s, out_n, g_n))
    print(f"[flex]   rel_l2 worst: flex/sdpa {res['flex_vs_sdpa']['worst_rel_l2']:.2e}  "
          f"flex/naive {res['flex_vs_naive_fp32']['worst_rel_l2']:.2e}  "
          f"sdpa/naive {res['sdpa_vs_naive_fp32']['worst_rel_l2']:.2e}", flush=True)
    return res


def bench_one(dev, variant, attn_fn, block_mask_fn, b, n, reps):
    module = build_module("sdpa", True, chunk=32).to(dev)
    z, mask = make_inputs(b, n, 64, dev)
    if variant == "sdpa":
        fn = module_call
    else:
        _, layout, masking = variant.split("_")

        def fn(m, zz, mm):
            return flex_triangle_attention(m, zz, mm, attn_fn, layout, masking, block_mask_fn)

    def step():
        zz = z.detach().requires_grad_(True)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            out = fn(module, zz, mask)
        out.float().sum().backward()
        module.zero_grad(set_to_none=True)

    rec = {}
    try:
        t0 = time.perf_counter()
        step()  # warm-up (includes compilation for flex)
        sync(dev)
        rec["first_step_s"] = time.perf_counter() - t0
        times, peak = [], None
        for _ in range(reps):
            if dev.type == "xpu":
                torch.xpu.empty_cache()
                base = torch.xpu.memory_allocated()
                torch.xpu.reset_peak_memory_stats()
            t0 = time.perf_counter()
            step()
            sync(dev)
            times.append(time.perf_counter() - t0)
            if dev.type == "xpu":
                p = torch.xpu.max_memory_allocated() - base
                peak = p if peak is None else max(peak, p)
        rec.update(status="ok", time_ms=1e3 * statistics.median(times),
                   times_ms=[1e3 * t for t in times],
                   peak_gb=None if peak is None else peak / 1e9)
    except Exception as e:  # noqa: BLE001
        rec.update(status="oom" if is_oom(e) else "error", error=f"{type(e).__name__}: {e}"[:2000],
                   traceback=traceback.format_exc()[-6000:])
    finally:
        del module, z, mask
        if dev.type == "xpu":
            torch.xpu.empty_cache()
    rec.update(variant=variant, B=b, N=n)
    print(f"[bench] {json.dumps({k: v for k, v in rec.items() if k != 'traceback'})}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="8,32")
    ap.add_argument("--peaks", default="100,150")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--check-b", type=int, default=2)
    ap.add_argument("--check-n", default="37,100")
    ap.add_argument("--bench-variants", default="sdpa,flex_rowbatch_blockmask,flex_rowhead_blockmask")
    a = ap.parse_args()

    report = dict(triton_shim=TRITON_SHIM, env={}, steps={}, correctness={}, bench=[])
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out_path.write_text(json.dumps(report, indent=1, default=str))

    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    dev = torch.device("xpu" if xpu else "cpu")
    report["env"] = dict(torch=torch.__version__, device=str(dev),
                         device_name=torch.xpu.get_device_name(0) if xpu else None,
                         ZE_AFFINITY_MASK=os.environ.get("ZE_AFFINITY_MASK"),
                         job=os.environ.get("PBS_JOBID"),
                         TRITON_CACHE_DIR=os.environ.get("TRITON_CACHE_DIR"),
                         TORCHINDUCTOR_CACHE_DIR=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                         sys_path_head=sys.path[:4])
    print(f"[env] {report['env']}\n[env] triton shim: {TRITON_SHIM}", flush=True)
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)

    s = report["steps"]
    guarded(s, "a_imports", step_imports, dev)
    guarded(s, "a_triton_kernel", step_triton_kernel, dev)
    save()
    fl = guarded(s, "b_make_flex", make_flex, dev)
    if fl["status"] != "ok":
        save()
        print("[flex] flex_attention unavailable; stopping.", flush=True)
        return
    attn_fn, block_mask_fn = fl["result"]
    fl["result"] = "compiled flex_attention + create_block_mask"
    guarded(s, "b_trivial_flex", step_trivial_flex, dev, attn_fn, block_mask_fn)
    save()

    # c: correctness. Full matrix at the first check size, default variant only at the rest.
    sizes = [int(x) for x in a.check_n.split(",")]
    combos = []
    for dtype in ("fp32", "bf16"):
        for layout in ("rowbatch", "rowhead"):
            for masking in ("blockmask", "scoremask"):
                combos.append((layout, masking, True, dtype, sizes[0]))
        combos.append(("rowbatch", "blockmask", False, dtype, sizes[0]))
        for n in sizes[1:]:
            combos.append(("rowbatch", "blockmask", True, dtype, n))
            combos.append(("rowhead", "blockmask", True, dtype, n))
    for layout, masking, starting, dtype, n in combos:
        key = f"{layout}_{masking}_{'start' if starting else 'end'}_{dtype}_B{a.check_b}_N{n}"
        guarded(report["correctness"], key, step_correctness, dev, attn_fn, block_mask_fn,
                layout, masking, starting, dtype, a.check_b, n)
        save()

    # d: bench.
    for b in map(int, a.batches.split(",")):
        for n in map(int, a.peaks.split(",")):
            for variant in a.bench_variants.split(","):
                report["bench"].append(bench_one(dev, variant, attn_fn, block_mask_fn, b, n,
                                                 a.reps))
                save()

    save()
    print(f"[flex] wrote {out_path}", flush=True)
    print("\n== steps")
    for k, v in s.items():
        print(f"  {k:<20} {v['status']}  {v.get('error', '')[:120]}")
    print("== correctness (worst rel_l2 over out + all grads)")
    print(f"  {'case':<44}{'flex/sdpa':>11}{'flex/naive':>12}{'sdpa/naive':>12}")
    for k, v in report["correctness"].items():
        if v["status"] != "ok":
            print(f"  {k:<44}  {v['status']}: {v.get('error', '')[:80]}")
            continue
        r = v["result"]
        print(f"  {k:<44}{r['flex_vs_sdpa']['worst_rel_l2']:>11.2e}"
              f"{r['flex_vs_naive_fp32']['worst_rel_l2']:>12.2e}"
              f"{r['sdpa_vs_naive_fp32']['worst_rel_l2']:>12.2e}")
    print("== bench (bf16 autocast, fwd+bwd of one module)")
    print(f"  {'variant':<26}{'B':>4}{'N':>5}{'ms':>10}{'peak GB':>10}{'1st s':>8}  status")
    for r in report["bench"]:
        ms = f"{r['time_ms']:.1f}" if r.get("time_ms") is not None else "-"
        pk = f"{r['peak_gb']:.2f}" if r.get("peak_gb") is not None else "-"
        fs = f"{r['first_step_s']:.1f}" if r.get("first_step_s") is not None else "-"
        print(f"  {r['variant']:<26}{r['B']:>4}{r['N']:>5}{ms:>10}{pk:>10}{fs:>8}  {r['status']}")


if __name__ == "__main__":
    main()
