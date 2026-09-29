"""K117-P: how much of the SDPA triangle-attention path is copies, and what do copy-free variants give?

    python pbs/diag/triattn_copy_bench.py --out results/raw/diag/triattn_copy_bench/<jobid>.json

Context (notes/OPEN_QUESTIONS.md K116-P / K117-P): the model's SDPA path
(msdelta/models/pairformer.py, TriangleAttention, pair_tri_attn_impl="sdpa", sdpa_flatten) folds
(B, chunk) into one batch dim, so the triangle bias beta_jk -- shared by every row i -- is COPIED
once per row of each chunk (``attn_mask.expand(B, c, H, N, N).reshape(B*c, H, N, N)``) and, in
backward, the copies' gradients are summed back (ExpandBackward). q/k/v and the output also go
through permute + reshape copies per chunk.

Dims: Stage 0 card (configs/stage0/pairformer/config.json and its verbatim copy
configs/diag/k114-pairformer-stage0/config.json): c_z = 64, 4 heads x 16, chunk 32. One module.
Weights are TriangleAttention's with the model's init (triattn_bf16_check.base_module), fp32
weights and input under bf16 autocast as in training (--precision fp32 for the CPU unit test).

Variants (all on the SAME weights and inputs):
  current  the model's module as is (impl "sdpa", sdpa_flatten=True). When profiled, a
           line-for-line reimplementation with record_function labels is run instead (its output
           is checked against the module: ``reimpl_max_abs_diff``, expected 0).
  sdpa5d   the module with sdpa_flatten=False (5-D call, broadcast mask, no copy; K116 found
           Intel's fused kernel rejects 5-D, so this runs the math path). Reference point only.
  hview    copy-free 4-D mask: rows of a chunk go into SDPA's HEAD dim, (B*H, c, N, d), and the
           mask (B, H, N, N) (made contiguous ONCE per call) is passed as the strided view
           ``mask[:, :, None].expand(B, H, c, N, N).reshape(B*H, c, N, N)`` -- stride 0 over the
           rows, no materialised copy (checked: ``hview_mask_view_check``).
  cached   the current 4-D layout with the expanded (B*c, H, N, N) masks prebuilt once per input
           (outside the timed region) and reused by every chunk and every call: isolates the cost
           of the bias projection + mask build + per-chunk expand copies + their backward. The
           cached masks are grad-requiring leaves (so SDPA still computes dmask as in the real
           path); one leaf per chunk size is shared by all chunks, so backward adds an
           AccumulateGrad sum over chunks (reported separately by the profile). The accuracy
           check routes the leaves' gradients back through the bias projection, so its dz and
           bias.weight gradients are complete.
  naive    (not default; --variants) the model's naive path, for the K115 "SDPA 0.47x naive" anchor
           (its einsum/softmax are not SDPA ops, so its profile puts them under "other").

Per (B, N):
  1. accuracy (--inits, starting AND ending module): REF = naive path in fp32 without autocast
     (per-chunk checkpoint, numerically identical, to fit N = 512), float32 matmul precision
     "highest", as in triattn_bf16_check.py. Each variant vs REF and vs current: max abs error and
     rel-L2 of the output (valid pairs), dL/dz and every parameter gradient, with the masked
     upstream gradient of triattn_bf16_check. A variant AGREES if, for the output, dz and EVERY
     parameter, its rel-L2 vs REF <= max(--tol-ratio x current's rel-L2 vs REF, --tol-floor):
     "no less accurate than the current path", the K115 acceptance rule (DECISIONS 2026-09-28).
     A faster variant that fails this is flagged (``agrees: false``, summary FLAG).
  2. timing (starting module, --inits' first entry), modes fwd (grad enabled, graph built then
     dropped: the training forward) and fwdbwd (loss = sum(out * G)): --warmup steps, then the
     median of --reps; peak device memory above the pre-step allocation, as triattn_bench.py.
  3. profile (torch.profiler, CPU + XPU activities, --profile-steps steps after timing): the
     chrome trace is parsed and every device kernel is charged to the op that launched it
     (runtime-call correlation), backward kernels to the forward op that created their autograd
     node (sequence numbers). Categories: attention (the SDPA op and its backward; sub
     ``internal_copy`` if the kernel is a copy inside it), copy (mask_expand, qkv_layout,
     out_layout, layout_bwd:<op>, cast, copy:<op>), other (mask_build, proj_in, gate_out, ...).
     Per step, us. If no device kernels are found (CPU run), self CPU time is used instead
     (``basis``). The raw traces of --trace-sizes are saved (gzip) next to the JSON.
Out-of-memory is recorded and the run continues. Results are rewritten after every (B, N), so
a killed job leaves the finished sizes. Exit code 1 if any accuracy/timing record is an error
(not OOM) or nothing ran; profiler failures are recorded but not fatal (Kineto can flake on XPU).
"""
from __future__ import annotations

import argparse
import bisect
import contextlib
import copy
import gzip
import json
import os
import re
import statistics
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triattn_bench import _relevant, is_oom, sync  # noqa: E402
from triattn_bf16_check import base_module, compare  # noqa: E402

VARIANTS = {
    "current": "model module, impl sdpa, sdpa_flatten=True (4-D, per-chunk mask expand copy)",
    "sdpa5d": "model module, sdpa_flatten=False (5-D broadcast mask, math path)",
    "hview": "4-D (B*H, c, N, d), mask as stride-0 expand() view over rows, no copy",
    "cached": "current 4-D layout, expanded masks prebuilt once and reused (grad-requiring leaves)",
    "naive": "model module, impl naive (einsum + fp32 softmax)",
}
DEFAULT_VARIANTS = "current,sdpa5d,hview,cached"
LABEL = "k117:"


# ---------------------------------------------------------------- the variants ----------

def _rf(label: bool, name: str):
    if not label:
        return contextlib.nullcontext()
    return torch.profiler.record_function(LABEL + name)


def chunk_sizes(n: int, chunk: int) -> list[int]:
    return sorted({min(chunk, n - s) for s in range(0, n, chunk)})


def build_cache(m, z, mask) -> dict:
    """Expanded (B*c, H, N, N) masks for each chunk size c, as grad-requiring leaves.

    Call under the same autocast context as the forward (the mask is in the compute dtype).
    """
    with torch.no_grad():
        zt = z if m.starting else z.transpose(1, 2)
        bias = m.bias(m.norm(zt))
        b, n = bias.shape[:2]
        am = bias.permute(0, 3, 1, 2)[:, None]
        am = am.masked_fill(~mask[:, None, None, None, :], torch.finfo(bias.dtype).min)
        return {c: am.expand(b, c, m.h, n, n).reshape(b * c, m.h, n, n).detach().requires_grad_(True)
                for c in chunk_sizes(n, m.chunk)}


def forward_variant(m, z, mask, variant: str, cache: dict | None = None, label: bool = False):
    """TriangleAttention forward for one variant (m: a TriangleAttention, left unchanged)."""
    if variant in ("sdpa5d", "naive") or (variant == "current" and not label):
        saved = (m.impl, m.sdpa_flatten)
        m.impl = "naive" if variant == "naive" else "sdpa"
        m.sdpa_flatten = variant != "sdpa5d"
        try:
            return m(z, mask)
        finally:
            m.impl, m.sdpa_flatten = saved
    if variant not in ("current", "hview", "cached"):
        raise ValueError(f"unknown variant {variant!r}")
    h, d = m.h, m.d
    with _rf(label, "proj_in"):
        if not m.starting:
            z = z.transpose(1, 2)
        z = m.norm(z)
        b, n = z.shape[:2]
        q = m.q(z).view(b, n, n, h, d)
        k = m.k(z).view(b, n, n, h, d)
        v = m.v(z).view(b, n, n, h, d)
    if variant == "current":  # exactly TriangleAttention.forward / _chunk_sdpa
        with _rf(label, "mask_build"):
            bias = m.bias(z)
            am = bias.to(q.dtype).permute(0, 3, 1, 2)[:, None]
            am = am.masked_fill(~mask[:, None, None, None, :], torch.finfo(q.dtype).min)
    elif variant == "hview":
        with _rf(label, "mask_build"):
            bias = m.bias(z)
            am = bias.to(q.dtype).permute(0, 3, 1, 2)
            am = am.masked_fill(~mask[:, None, None, :], torch.finfo(q.dtype).min).contiguous()
    out = torch.empty_like(q)
    for s in range(0, n, m.chunk):
        e = min(s + m.chunk, n)
        c = e - s
        qc, kc, vc = q[:, s:e], k[:, s:e], v[:, s:e]
        if variant in ("current", "cached"):
            with _rf(label, "qkv_layout"):
                qc, kc, vc = (t.permute(0, 1, 3, 2, 4).reshape(b * c, h, n, d) for t in (qc, kc, vc))
            if variant == "current":
                with _rf(label, "mask_expand_copy"):
                    amc = am.expand(b, c, h, n, n).reshape(b * c, h, n, n)
            else:
                amc = cache[c]
            with _rf(label, "sdpa"):
                o = F.scaled_dot_product_attention(qc, kc, vc, attn_mask=amc)
            with _rf(label, "out_layout"):
                out[:, s:e] = o.view(b, c, h, n, d).permute(0, 1, 3, 2, 4)
        else:  # hview
            with _rf(label, "qkv_layout"):
                qc, kc, vc = (t.permute(0, 3, 1, 2, 4).reshape(b * h, c, n, d) for t in (qc, kc, vc))
            with _rf(label, "mask_expand_view"):
                amc = am[:, :, None].expand(b, h, c, n, n).reshape(b * h, c, n, n)
            with _rf(label, "sdpa"):
                o = F.scaled_dot_product_attention(qc, kc, vc, attn_mask=amc)
            with _rf(label, "out_layout"):
                out[:, s:e] = o.view(b, h, c, n, d).permute(0, 2, 3, 1, 4)
    with _rf(label, "gate_out"):
        out = torch.sigmoid(m.gate(z)) * out.reshape(b, n, n, -1)
        out = m.out(out)
    return out if m.starting else out.transpose(1, 2)


def hview_mask_view_check(dev) -> dict:
    """The hview mask really is a view: same storage, stride 0 over the rows."""
    b, h, c, n = 2, 4, 8, 16
    am = torch.randn(b, h, n, n, device=dev)
    amc = am[:, :, None].expand(b, h, c, n, n).reshape(b * h, c, n, n)
    return dict(same_storage=amc.data_ptr() == am.data_ptr(), row_stride=amc.stride(1),
                strides=list(amc.stride()), is_view=bool(amc.data_ptr() == am.data_ptr()
                                                         and amc.stride(1) == 0))


# ---------------------------------------------------------------- inputs / steps --------

def inputs(b, n, c_z, seed):
    """triattn_bf16_check.inputs, with the minimum length capped for small test sizes."""
    g = torch.Generator().manual_seed(seed)
    z = F.layer_norm(torch.randn(b, n, n, c_z, generator=g), (c_z,))
    lo = min(50, max(1, n // 2))
    lengths = torch.randint(lo, n + 1, (b,), generator=g)
    lengths[0] = n
    mask = torch.arange(n)[None, :] < lengths[:, None]
    pair = mask[:, :, None] & mask[:, None, :]
    grad_out = torch.randn(b, n, n, c_z, generator=g) * pair[..., None]
    return z, mask, pair, grad_out, lengths.tolist()


def _autocast(dev, precision):
    return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=(precision == "bf16"))


def make_step(m, variant, z, mask, grad_out, dev, precision, mode, cache=None, label=False):
    def step():
        zz = z.detach().requires_grad_(True)
        if cache:
            for t in cache.values():
                t.grad = None
        with _autocast(dev, precision):
            out = forward_variant(m, zz, mask, variant, cache, label)
        if mode == "fwdbwd":
            (out.float() * grad_out).sum().backward()
            m.zero_grad(set_to_none=True)
    return step


def grads_of(m, variant, z, mask, grad_out, dev, precision, cache=None):
    """Output, dL/dz and parameter gradients of one step (the accuracy check)."""
    m.zero_grad(set_to_none=True)
    zz = z.detach().clone().requires_grad_(True)
    if cache:
        for t in cache.values():
            t.grad = None
    with _autocast(dev, precision):
        out = forward_variant(m, zz, mask, variant, cache)
    (out.float() * grad_out).sum().backward()
    if variant == "cached":  # route the cached masks' gradients back through the bias projection
        b, n = z.shape[:2]
        dmask = sum(t.grad.float().reshape(b, c, m.h, n, n).sum(1) for c, t in cache.items()
                    if t.grad is not None)
        dmask = dmask.masked_fill(~mask[:, None, None, :], 0.0)
        with _autocast(dev, precision):
            zt = zz if m.starting else zz.transpose(1, 2)
            bias = m.bias(m.norm(zt))
        bias.backward(dmask.permute(0, 2, 3, 1).to(bias.dtype))
    res = dict(out=out.detach().double(), dz=zz.grad.detach().double(),
               params={k: p.grad.detach().double() for k, p in m.named_parameters()
                       if p.grad is not None})
    m.zero_grad(set_to_none=True)
    return res


def agreement(v_vs_ref: dict, cur_vs_ref: dict, ratio: float, floor: float) -> dict:
    """Variant no less accurate than current (vs REF) on out, dz and every parameter."""
    checks = {}
    for key in ("out", "dz"):
        ev, ec = v_vs_ref[key]["rel_l2"], cur_vs_ref[key]["rel_l2"]
        lim = max(ratio * ec, floor)
        checks[key] = dict(rel_l2=ev, current_rel_l2=ec, limit=lim,
                           ok=bool(v_vs_ref[key]["finite"] and ev <= lim))
    worst, worst_frac = None, -1.0
    params_ok = True
    for name, e in v_vs_ref["params"].items():
        ec = cur_vs_ref["params"].get(name, {}).get("rel_l2", 0.0)
        lim = max(ratio * ec, floor)
        ok = bool(e["finite"] and e["rel_l2"] <= lim)
        params_ok &= ok
        if e["rel_l2"] / lim > worst_frac:
            worst_frac = e["rel_l2"] / lim
            worst = dict(name=name, rel_l2=e["rel_l2"], current_rel_l2=ec, limit=lim, ok=ok)
    missing = sorted(set(cur_vs_ref["params"]) - set(v_vs_ref["params"]))
    checks["params"] = dict(worst=worst, missing=missing, ok=params_ok and not missing)
    return dict(ok=all(c["ok"] for c in checks.values()), checks=checks)


def _brief(cmp: dict) -> dict:
    return dict(out_max_abs=cmp["out"]["max_abs"], out_rel_l2=cmp["out"]["rel_l2"],
                dz_max_abs=cmp["dz"]["max_abs"], dz_rel_l2=cmp["dz"]["rel_l2"],
                param_worst=cmp["param_worst"]["name"],
                param_worst_max_abs=cmp["param_worst"]["max_abs"],
                param_worst_rel_l2=cmp["param_worst"]["rel_l2"],
                param_median_rel_l2=cmp["param_median_rel_l2"])


def check_size(b, n, a, dev, init, variants, starting, seed):
    """Accuracy of every variant vs REF (fp32 naive) and vs current, one module, one size."""
    z, mask, pair, grad_out, lengths = inputs(b, n, a.c_z, seed=1000 + 7 * b + n)
    z, mask, pair, grad_out = z.to(dev), mask.to(dev), pair.to(dev), grad_out.to(dev)
    base = base_module(starting, a, init, seed=seed)
    base.impl = "sdpa"
    module = "start" if starting else "end"
    rows = []
    ref = cur = None
    try:
        m_ref = copy.deepcopy(base).to(dev)
        m_ref.impl, m_ref.checkpoint_chunks = "naive", True
        ref = grads_of(m_ref, "naive", z, mask, grad_out, dev, "fp32")
        del m_ref
        m = copy.deepcopy(base).to(dev)
        cur = grads_of(m, "current", z, mask, grad_out, dev, a.precision)
        cur_vs_ref = compare(cur, ref, pair)
        with torch.no_grad(), _autocast(dev, a.precision):  # labelled reimplementation == module
            o_mod = forward_variant(m, z, mask, "current")
            o_re = forward_variant(m, z, mask, "current", label=True)
            reimpl_diff = (o_mod.float() - o_re.float()).abs().max().item()
        del o_mod, o_re
        rows.append(dict(kind="accuracy", init=init, module=module, variant="current", B=b, N=n,
                         lengths=lengths, status="ok", vs_ref=_brief(cur_vs_ref),
                         reimpl_max_abs_diff=reimpl_diff, agrees=True))
    except Exception as e:  # noqa: BLE001 -- record and continue
        rows.append(dict(kind="accuracy", init=init, module=module, variant="current", B=b, N=n,
                         status="oom" if is_oom(e) else "error",
                         error=f"{type(e).__name__}: {e}"[:500]))
        if not is_oom(e):
            traceback.print_exc()
        _free(dev)
        return rows
    for variant in variants:
        if variant == "current":
            continue
        rec = dict(kind="accuracy", init=init, module=module, variant=variant, B=b, N=n)
        cache = res = None
        try:
            with _autocast(dev, a.precision):
                cache = build_cache(m, z, mask) if variant == "cached" else None
            res = grads_of(m, variant, z, mask, grad_out, dev, a.precision, cache)
            v_ref, v_cur = compare(res, ref, pair), compare(res, cur, pair)
            rec.update(status="ok", vs_ref=_brief(v_ref), vs_current=_brief(v_cur),
                       **agreement(v_ref, cur_vs_ref, a.tol_ratio, a.tol_floor))
            rec["agrees"] = rec.pop("ok")
        except Exception as e:  # noqa: BLE001
            rec.update(status="oom" if is_oom(e) else "error", error=f"{type(e).__name__}: {e}"[:500])
            if not is_oom(e):
                traceback.print_exc()
        finally:
            del cache, res
            _free(dev)
        rows.append(rec)
    del m, ref, cur
    _free(dev)
    for r in rows:
        short = {k: r.get(k) for k in ("init", "module", "variant", "B", "N", "status", "agrees")}
        if "vs_ref" in r:
            short["out_rel_ref"] = r["vs_ref"]["out_rel_l2"]
            short["dz_rel_ref"] = r["vs_ref"]["dz_rel_l2"]
        if "vs_current" in r:
            short["out_maxabs_cur"] = r["vs_current"]["out_max_abs"]
        print(f"[acc] {json.dumps(short)}", flush=True)
    return rows


def _free(dev):
    if dev.type == "xpu":
        torch.xpu.empty_cache()


# ---------------------------------------------------------------- timing ----------------

def time_config(m, variant, z, mask, grad_out, dev, a, mode):
    rec = dict(kind="timing", variant=variant, mode=mode, B=z.shape[0], N=z.shape[1])
    cache = None
    try:
        if variant == "cached":
            with _autocast(dev, a.precision):
                cache = build_cache(m, z, mask)
        step = make_step(m, variant, z, mask, grad_out, dev, a.precision, mode, cache)
        for _ in range(a.warmup):
            step()
        sync(dev)
        times, peak = [], None
        for _ in range(a.reps):
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
    except Exception as e:  # noqa: BLE001 -- record and continue
        rec.update(status="oom" if is_oom(e) else "error", error=f"{type(e).__name__}: {e}"[:500])
        if not is_oom(e):
            traceback.print_exc()
    finally:
        del cache
        m.zero_grad(set_to_none=True)
        _free(dev)
    print(f"[time] {json.dumps(rec)}", flush=True)
    return rec


# ---------------------------------------------------------------- profile breakdown -----

ATTN_RE = re.compile(r"scaled_dot_product|attention|flash|sdp", re.I)
COPY_OPS = {"aten::copy_", "aten::clone", "aten::contiguous", "aten::_to_copy"}
CAST_OPS = {"aten::to", "aten::_to_copy"}
LAYOUT_OPS = {"aten::expand", "aten::reshape", "aten::_reshape_alias", "aten::view",
              "aten::permute", "aten::transpose", "aten::slice", "aten::select",
              "aten::as_strided", "aten::clone", "aten::contiguous", "aten::copy_",
              "aten::_unsafe_view", "aten::unsqueeze", "aten::squeeze", "aten::narrow",
              "aten::t", "aten::expand_as", "aten::view_as"}
LABEL_CAT = {"mask_build": ("other", "mask_build"),
             "mask_expand_copy": ("copy", "mask_expand"),
             "mask_expand_view": ("copy", "mask_expand"),
             "qkv_layout": ("copy", "qkv_layout"),
             "out_layout": ("copy", "out_layout"),
             "proj_in": ("other", "proj_in"),
             "gate_out": ("other", "gate_out")}
LAYOUT_NODES = {"CopySlices", "SliceBackward0", "ExpandBackward0", "CloneBackward0",
                "PermuteBackward0", "ViewBackward0", "UnsafeViewBackward0",
                "ReshapeAliasBackward0", "TransposeBackward0", "SelectBackward0",
                "AsStridedBackward0", "SqueezeBackward0", "UnsqueezeBackward0"}
BWD_PREFIX = "autograd::engine::evaluate_function: "


class _Node:
    __slots__ = ("name", "ts", "end", "parent", "seq", "ext", "tid", "child_dur", "bwd_root")

    def __init__(self, e):
        self.name = e.get("name", "")
        self.ts = float(e["ts"])
        self.end = self.ts + float(e.get("dur") or 0.0)
        args = e.get("args") or {}
        self.seq = args.get("Sequence number")
        self.ext = args.get("External id")
        self.tid = e.get("tid")
        self.parent = None
        self.child_dur = 0.0
        self.bwd_root = None

    def chain(self):
        out, n = [], self
        while n is not None:
            out.append(n)
            n = n.parent
        return out[::-1]  # outer -> inner


def _is_device(e):
    c = str(e.get("cat", "")).lower()
    return (e.get("ph") == "X" and any(t in c for t in ("kernel", "memcpy", "memset"))
            and "runtime" not in c and "annotation" not in c)


def _is_runtime(e):
    """Host-side launch record (runtime or driver API call) carrying the kernel's correlation."""
    c = str(e.get("cat", "")).lower()
    return e.get("ph") == "X" and ("runtime" in c or "driver" in c)


def _is_host_op(e):
    return e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation") and "ts" in e


def classify(ctx: list[str], bwd_node: str | None, inner_op: str | None) -> tuple[str, str]:
    """(category, subcategory) of one kernel / op.

    ctx: forward-context names, outer -> inner (for a backward kernel: the chain of the forward
    op that created its autograd node); bwd_node: autograd node name for backward kernels;
    inner_op: innermost host op that launched the kernel (for internal-copy detection).
    """
    labels = [x[len(LABEL):] for x in ctx if x.startswith(LABEL)]
    label = labels[-1] if labels else None
    suffix = ":bwd" if bwd_node is not None else ""
    if label == "sdpa" or any(ATTN_RE.search(x) for x in ctx if not x.startswith(LABEL)) or (
            bwd_node is not None and ATTN_RE.search(bwd_node)):
        if inner_op in COPY_OPS:
            return "attention", "internal_copy" + suffix
        return "attention", ("bwd" if bwd_node is not None else "fwd")
    if label in LABEL_CAT:
        cat, sub = LABEL_CAT[label]
        return cat, sub + suffix
    ops = [x for x in ctx if not x.startswith(LABEL)]
    origin = ops[-1] if ops else None
    if bwd_node is not None:
        if origin in CAST_OPS or any(o in CAST_OPS for o in ops):
            return "copy", "cast:bwd"
        if origin in LAYOUT_OPS:
            return "copy", f"layout_bwd:{origin}"
        node = bwd_node.split("::")[-1]
        if node in LAYOUT_NODES:
            return "copy", f"layout_bwd:{node}"
        return "other", f"bwd:{bwd_node}"
    if inner_op in COPY_OPS:
        if any(o in CAST_OPS for o in ops):
            return "copy", "cast"
        outer = next((o for o in ops if o.startswith("aten::")), inner_op)
        return "copy", f"copy:{outer}"
    return "other", (origin or inner_op or "unknown")


def breakdown(events: list[dict], steps: int, top: int = 40) -> dict:
    """Per-category time per step from a chrome trace's events (see module docstring)."""
    by_tid = defaultdict(list)
    for e in events:
        if _is_host_op(e):
            by_tid[e.get("tid")].append(_Node(e))
    by_ext, by_seq = {}, {}
    starts = {}
    for tid, nodes in by_tid.items():
        nodes.sort(key=lambda x: (x.ts, -(x.end - x.ts)))
        stack = []
        for nd in nodes:
            while stack and not (stack[-1].ts <= nd.ts + 1e-3 and stack[-1].end >= nd.end - 1e-3):
                stack.pop()
            nd.parent = stack[-1] if stack else None
            if nd.parent is not None:
                nd.parent.child_dur += nd.end - nd.ts
                nd.bwd_root = nd.parent.bwd_root
            if nd.name.startswith(BWD_PREFIX):
                nd.bwd_root = nd
            stack.append(nd)
            if nd.ext is not None:
                by_ext.setdefault(nd.ext, nd)
            if nd.seq is not None and nd.bwd_root is None and not nd.name.startswith("autograd::"):
                by_seq.setdefault(nd.seq, nd)  # outermost forward op carrying this seq nr
        starts[tid] = [nd.ts for nd in nodes]

    def innermost(tid, t):
        nodes = by_tid.get(tid)
        if not nodes:
            return None
        i = bisect.bisect_right(starts[tid], t + 1e-3) - 1
        nd = nodes[i] if i >= 0 else None
        while nd is not None and nd.end < t - 1e-3:
            nd = nd.parent
        return nd

    def context(nd):
        """(ctx names, bwd node name, innermost op name) for a launching host op."""
        if nd is None:
            return [], None, None
        chain = nd.chain()
        inner = next((x.name for x in reversed(chain) if not x.name.startswith(LABEL)), None)
        if nd.bwd_root is None:
            return [x.name for x in chain], None, inner
        root = nd.bwd_root
        node_name = root.name[len(BWD_PREFIX):]
        fwd = by_seq.get(root.seq)
        return ([x.name for x in fwd.chain()] if fwd is not None else []), node_name, inner

    runtime_by_corr = {}
    for e in events:
        if _is_runtime(e):
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                runtime_by_corr.setdefault(corr, e)
    items = []  # (dur_us, kernel name, launching host node)
    unattributed = 0.0
    n_dev = 0
    for e in events:
        if not _is_device(e):
            continue
        n_dev += 1
        args = e.get("args") or {}
        nd = None
        r = runtime_by_corr.get(args.get("correlation"))
        if r is not None:
            nd = innermost(r.get("tid"), float(r["ts"]))
        if nd is None and args.get("External id") is not None:
            nd = by_ext.get(args["External id"])
        dur = float(e.get("dur") or 0.0)
        if nd is None:
            unattributed += dur
        items.append((dur, e.get("name", ""), nd))
    basis = "device"
    if n_dev == 0:
        basis = "cpu_self"
        for nodes in by_tid.values():
            for nd in nodes:
                if nd.name.startswith(LABEL):
                    continue
                items.append((max(0.0, (nd.end - nd.ts) - nd.child_dur), "", nd))
    by_cat, by_sub = defaultdict(float), defaultdict(float)
    rows = defaultdict(lambda: [0.0, 0])
    sdpa_ops = set()
    for dur, kname, nd in items:
        if nd is None:
            cat, sub, inner = "unattributed", "-", None
        else:
            ctx, bwd, inner = context(nd)
            cat, sub = classify(ctx, bwd, inner)
            for x in ctx + [c.name for c in nd.chain()]:
                if ATTN_RE.search(x) and not x.startswith(LABEL):
                    sdpa_ops.add(x)
        by_cat[cat] += dur
        by_sub[f"{cat}/{sub}"] += dur
        r = rows[(cat, sub, inner or "", kname[:160])]
        r[0] += dur
        r[1] += 1
    total = sum(by_cat.values())
    s = max(steps, 1)
    return dict(
        basis=basis, steps=steps, total_us_per_step=total / s,
        by_category_us={k: v / s for k, v in sorted(by_cat.items())},
        fraction={k: (v / total if total else 0.0) for k, v in sorted(by_cat.items())},
        by_subcategory_us={k: v / s for k, v in sorted(by_sub.items(), key=lambda kv: -kv[1])},
        top=[dict(category=c, sub=sb, op=op, kernel=kn, us_per_step=v[0] / s, count=v[1])
             for (c, sb, op, kn), v in sorted(rows.items(), key=lambda kv: -kv[1][0])[:top]],
        n_device_events=n_dev, unattributed_us_per_step=unattributed / s,
        sdpa_ops=sorted(sdpa_ops))


def profile_config(m, variant, z, mask, grad_out, dev, a, mode, trace_path=None):
    """Profiler breakdown of --profile-steps steps (labels on for current/hview/cached)."""
    from torch.profiler import ProfilerActivity, profile
    rec = dict(kind="profile", variant=variant, mode=mode, B=z.shape[0], N=z.shape[1])
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.XPU] if dev.type == "xpu" else [])
    cache = None
    try:
        if variant == "cached":
            with _autocast(dev, a.precision):
                cache = build_cache(m, z, mask)
        step = make_step(m, variant, z, mask, grad_out, dev, a.precision, mode, cache, label=True)
        step()  # warm-up with labels (same kernels)
        sync(dev)
        prof, errors = None, []
        for attempt in range(3):  # Kineto can fail once on XPU (PTI_ERROR_INTERNAL)
            try:
                with profile(activities=acts) as prof:
                    for _ in range(a.profile_steps):
                        step()
                    sync(dev)
                break
            except RuntimeError as e:
                errors.append(f"{[str(x) for x in acts]}: {e}"[:300])
                prof = None
                m.zero_grad(set_to_none=True)
                if attempt == 1 and len(acts) > 1:
                    acts = acts[:1]  # last try: CPU only (no device times)
        if prof is None:
            rec.update(status="error", error="; ".join(errors))
        else:
            fd, tmp = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            try:
                prof.export_chrome_trace(tmp)
                trace = json.loads(Path(tmp).read_text())
            finally:
                os.unlink(tmp)
            if trace_path is not None:
                Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(trace_path, "wt") as f:
                    json.dump(trace, f)
                rec["trace"] = str(trace_path)
            rec.update(status="ok", activities=[str(x) for x in acts], retries=errors,
                       **breakdown(trace.get("traceEvents", []), a.profile_steps))
            try:  # second opinion: the profiler's own per-op device totals
                ka = prof.key_averages()
                rec["key_averages_top"] = sorted(
                    ({"op": e.key, "self_device_us": e.self_device_time_total / a.profile_steps,
                      "count": e.count} for e in ka if e.self_device_time_total > 0),
                    key=lambda x: -x["self_device_us"])[:25]
            except Exception as e:  # noqa: BLE001
                rec["key_averages_top"] = f"error: {type(e).__name__}: {e}"[:300]
    except Exception as e:  # noqa: BLE001 -- record and continue
        rec.update(status="oom" if is_oom(e) else "error", error=f"{type(e).__name__}: {e}"[:500])
        if not is_oom(e):
            traceback.print_exc()
    finally:
        del cache
        m.zero_grad(set_to_none=True)
        _free(dev)
    short = {k: rec.get(k) for k in ("variant", "mode", "B", "N", "status", "basis",
                                      "total_us_per_step", "by_category_us")}
    print(f"[prof] {json.dumps(short)}", flush=True)
    return rec


def dispatch_probe(dev, a, variants) -> dict:
    """aten attention ops each variant dispatches to (fwd+bwd, small size, --precision)."""
    from torch.utils._python_dispatch import TorchDispatchMode
    res = {}
    b, n = 2, min(64, a.chunk * 2)
    z, mask, _, grad_out, _ = inputs(b, n, a.c_z, seed=5)
    z, mask, grad_out = z.to(dev), mask.to(dev), grad_out.to(dev)
    m = base_module(True, a, "model", seed=11).to(dev)
    for variant in variants:
        seen = set()

        class Log(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                seen.add(str(func.overloadpacket.__name__))
                return func(*args, **(kwargs or {}))
        try:
            with _autocast(dev, a.precision):
                cache = build_cache(m, z, mask) if variant == "cached" else None
            step = make_step(m, variant, z, mask, grad_out, dev, a.precision, "fwdbwd", cache)
            with Log():
                step()
            sync(dev)
            res[variant] = sorted(x for x in seen if _relevant(x))
        except Exception as e:  # noqa: BLE001
            res[variant] = [f"error: {type(e).__name__}: {str(e)[:300]}"]
        print(f"[probe] {variant}: {res[variant]}", flush=True)
    return res


# ---------------------------------------------------------------- main ------------------

def summarize(results: list[dict], variants: list[str] = ()) -> list[dict]:
    order = {v: i for i, v in enumerate(variants)}
    rows = []
    timing = {(r["variant"], r["mode"], r["B"], r["N"]): r for r in results if r["kind"] == "timing"}
    prof = {(r["variant"], r["mode"], r["B"], r["N"]): r for r in results if r["kind"] == "profile"}
    acc = defaultdict(list)
    for r in results:
        if r["kind"] == "accuracy":
            acc[(r["variant"], r["B"], r["N"])].append(r)
    for (variant, mode, b, n), t in sorted(timing.items(), key=lambda kv: (
            kv[0][3], kv[0][2], kv[0][1], order.get(kv[0][0], len(order)), kv[0][0])):
        cur = timing.get(("current", mode, b, n), {})
        p = prof.get((variant, mode, b, n), {})
        accs = acc.get((variant, b, n), [])
        agrees = None
        if accs:
            agrees = all(r.get("status") == "ok" and r.get("agrees") for r in accs)
        rows.append(dict(
            variant=variant, mode=mode, B=b, N=n, status=t["status"], time_ms=t.get("time_ms"),
            peak_gb=t.get("peak_gb"),
            speedup_vs_current=(cur["time_ms"] / t["time_ms"]
                                if t.get("time_ms") and cur.get("time_ms") else None),
            copy_fraction=(p.get("fraction") or {}).get("copy"),
            attention_fraction=(p.get("fraction") or {}).get("attention"),
            agrees=agrees))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="2,8")
    ap.add_argument("--peaks", default="200,256,512")
    ap.add_argument("--c-z", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--modes", default="fwd,fwdbwd")
    ap.add_argument("--inits", default="model,sharp",
                    help="accuracy check inits (triattn_bf16_check); timing uses the first")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"),
                    help="bf16 = fp32 weights under bf16 autocast (training); fp32 = CPU tests")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--profile-steps", type=int, default=2)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--no-accuracy", action="store_true")
    ap.add_argument("--trace-sizes", default="2x200",
                    help="BxN sizes whose raw chrome traces are saved (gzip); '' for none")
    ap.add_argument("--tol-ratio", type=float, default=1.25)
    ap.add_argument("--tol-floor", type=float, default=1e-4)
    ap.add_argument("--time-budget-min", type=float, default=40.0)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "xpu"))
    a = ap.parse_args(argv)
    t_start = time.perf_counter()
    variants = a.variants.split(",")
    for v in variants:
        if v not in VARIANTS:
            ap.error(f"unknown variant {v!r}; choose from {list(VARIANTS)}")
    if variants[0] != "current":
        ap.error("the first variant must be 'current' (the reference for speed and accuracy)")
    modes = a.modes.split(",")
    inits = a.inits.split(",")
    torch.set_float32_matmul_precision("highest")
    if a.device == "auto":
        dev = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    else:
        dev = torch.device(a.device)
    env = dict(torch=torch.__version__, device=str(dev),
               device_name=torch.xpu.get_device_name(0) if dev.type == "xpu" else None,
               ZE_AFFINITY_MASK=os.environ.get("ZE_AFFINITY_MASK"),
               job=os.environ.get("PBS_JOBID"), code_dir=os.environ.get("MSDELTA_CODE_DIR"),
               float32_matmul_precision=torch.get_float32_matmul_precision())
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        env["ipex"] = ipex.__version__
    except Exception as e:  # noqa: BLE001
        env["ipex"] = f"not imported ({type(e).__name__})"
    if dev.type == "xpu":
        env["device_total_gb"] = torch.xpu.get_device_properties(0).total_memory / 1e9
    print(f"[env] {env}", flush=True)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_sizes = {tuple(map(int, s.split("x"))) for s in a.trace_sizes.split(",") if s}
    view_check = hview_mask_view_check(dev)
    print(f"[view] {view_check}", flush=True)
    probe = dispatch_probe(dev, a, variants)
    results: list[dict] = []
    doc = dict(task="K117-P", env=env, args=vars(a),
               dims=dict(c_z=a.c_z, heads=a.heads, dim=a.dim, chunk=a.chunk,
                         precision=a.precision, warmup=a.warmup, reps=a.reps,
                         profile_steps=a.profile_steps),
               variants={v: VARIANTS[v] for v in variants},
               tolerance=dict(ratio=a.tol_ratio, floor=a.tol_floor,
                              rule="agrees iff for out, dz and every parameter gradient: "
                                   "rel_l2(variant vs REF) <= max(ratio * rel_l2(current vs REF), "
                                   "floor); REF = naive fp32"),
               hview_mask_view_check=view_check, sdpa_dispatch=probe, results=results)

    def write(status):
        doc["status"] = status
        doc["elapsed_s"] = time.perf_counter() - t_start
        doc["summary"] = summarize(results, variants)
        out_path.write_text(json.dumps(doc, indent=1))

    def over_budget():
        return (time.perf_counter() - t_start) / 60.0 > a.time_budget_min

    for n in map(int, a.peaks.split(",")):
        for b in map(int, a.batches.split(",")):
            if over_budget():
                results.append(dict(kind="skipped", B=b, N=n, status="skipped_budget"))
                print(f"[skip] B={b} N={n}: time budget ({a.time_budget_min} min) used", flush=True)
                continue
            if not a.no_accuracy:
                for init in inits:
                    for starting, seed in ((True, 11), (False, 12)):
                        results.extend(check_size(b, n, a, dev, init, variants, starting, seed))
            z, mask, _, grad_out, _ = inputs(b, n, a.c_z, seed=1000 + 7 * b + n)
            try:
                z, mask, grad_out = z.to(dev), mask.to(dev), grad_out.to(dev)
                m = base_module(True, a, inits[0], seed=11).to(dev)
                m.impl = "sdpa"
            except Exception as e:  # noqa: BLE001
                results.append(dict(kind="timing", variant="*", mode="*", B=b, N=n,
                                    status="oom" if is_oom(e) else "error",
                                    error=f"{type(e).__name__}: {e}"[:500]))
                continue
            for mode in modes:
                for variant in variants:
                    results.append(time_config(m, variant, z, mask, grad_out, dev, a, mode))
            if not a.no_profile:
                for mode in modes:
                    for variant in variants:
                        tp = None
                        if (b, n) in trace_sizes:
                            tp = out_path.parent / f"{out_path.stem}_traces" / \
                                f"B{b}_N{n}_{variant}_{mode}.json.gz"
                        results.append(profile_config(m, variant, z, mask, grad_out, dev, a,
                                                      mode, tp))
            del m, z, mask, grad_out
            _free(dev)
            write("running")
    bad = [r for r in results if r["kind"] in ("accuracy", "timing") and r["status"] == "error"]
    ran = [r for r in results if r["kind"] == "timing" and r["status"] == "ok"]
    rc = 1 if (bad or not ran) else 0
    write("done" if rc == 0 else "failed")
    print(f"[bench] wrote {out_path} (rc={rc}, {len(bad)} errors, {len(ran)} timings ok)", flush=True)
    print(f"{'variant':<9}{'mode':<8}{'B':>3}{'N':>5}{'ms':>10}{'x cur':>7}{'GB':>7}"
          f"{'copy%':>7}{'attn%':>7}  agrees  status")
    for r in doc["summary"]:
        f = (lambda x, fmt: format(x, fmt) if x is not None else "-")
        flag = "FLAG" if r["agrees"] is False else ("yes" if r["agrees"] else "-")
        print(f"{r['variant']:<9}{r['mode']:<8}{r['B']:>3}{r['N']:>5}{f(r['time_ms'], '10.2f'):>10}"
              f"{f(r['speedup_vs_current'], '7.2f'):>7}{f(r['peak_gb'], '7.2f'):>7}"
              f"{f(r['copy_fraction'] and 100 * r['copy_fraction'], '7.1f'):>7}"
              f"{f(r['attention_fraction'] and 100 * r['attention_fraction'], '7.1f'):>7}"
              f"  {flag:<6}  {r['status']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
