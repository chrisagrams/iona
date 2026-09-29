"""CPU checks for the K117 triangle-attention copy benchmark (pbs/diag/triattn_copy_bench.py).

Every variant must compute the same function as the model's SDPA path (fp32, small sizes, a
tail chunk shorter than the chunk size, padded keys, starting and ending node): outputs,
dL/dz and every parameter gradient. A wrong variant must be flagged. The hview mask must be a
real view (no copy). The profiler-trace parser must charge kernels to the right categories
(synthetic device trace), and a full tiny run must write the documented JSON schema.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pbs/diag"))  # the bench imports triattn_bench / triattn_bf16_check


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "pbs/diag" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load("triattn_copy_bench")
DEV = torch.device("cpu")
B, N = 2, 40  # chunk 16 -> chunks 16, 16, 8 (a tail chunk)


class _Args:
    c_z, heads, dim, chunk = 16, 2, 8, 16
    precision = "fp32"
    tol_ratio, tol_floor = 1.25, 1e-4


def _module(starting, init="sharp"):
    m = bench.base_module(starting, _Args, init, seed=11 if starting else 12)
    m.impl = "sdpa"
    return m


@pytest.mark.parametrize("starting", [True, False], ids=["start", "end"])
@pytest.mark.parametrize("variant", ["current", "sdpa5d", "hview", "cached", "naive"])
def test_variant_matches_model_sdpa_path(variant, starting):
    torch.manual_seed(0)
    z, mask, _, grad_out, _ = bench.inputs(B, N, _Args.c_z, seed=3)
    assert not mask.all(), "test needs padded keys"
    m = _module(starting)
    ref = copy.deepcopy(m)
    ref_out = bench.grads_of(ref, "current", z, mask, grad_out, DEV, "fp32")
    with torch.no_grad():  # the bench's "current" branch really is the module
        direct = m(z, mask)  # (no_grad may pick another CPU SDPA kernel: close, not bitwise)
    torch.testing.assert_close(ref_out["out"], direct.double(), rtol=1e-5, atol=1e-7)
    cache = bench.build_cache(m, z, mask) if variant == "cached" else None
    got = bench.grads_of(m, variant, z, mask, grad_out, DEV, "fp32", cache)
    torch.testing.assert_close(got["out"], ref_out["out"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got["dz"], ref_out["dz"], rtol=1e-5, atol=1e-6)
    assert set(got["params"]) == set(ref_out["params"])
    for k in ref_out["params"]:
        torch.testing.assert_close(got["params"][k], ref_out["params"][k], rtol=1e-4, atol=1e-6,
                                   msg=f"{variant}: grad of {k}")


@pytest.mark.parametrize("starting", [True, False], ids=["start", "end"])
def test_labelled_reimplementation_is_the_module_exactly(starting):
    z, mask, _, _, _ = bench.inputs(B, N, _Args.c_z, seed=4)
    m = _module(starting)
    with torch.no_grad():
        a = bench.forward_variant(m, z, mask, "current")
        b = bench.forward_variant(m, z, mask, "current", label=True)
    assert torch.equal(a, b)
    assert m.impl == "sdpa" and m.sdpa_flatten  # forward_variant leaves the module unchanged
    bench.forward_variant(m, z, mask, "sdpa5d")
    assert m.impl == "sdpa" and m.sdpa_flatten


def test_hview_mask_is_a_view():
    chk = bench.hview_mask_view_check(DEV)
    assert chk["is_view"] and chk["same_storage"] and chk["row_stride"] == 0


def test_cached_masks_are_leaves_per_chunk_size():
    z, mask, _, _, _ = bench.inputs(B, N, _Args.c_z, seed=5)
    cache = bench.build_cache(_module(True), z, mask)
    assert sorted(cache) == [8, 16]
    for c, t in cache.items():
        assert t.is_leaf and t.requires_grad and t.shape == (B * c, _Args.heads, N, N)


def test_check_size_agrees_and_flags_a_wrong_variant(monkeypatch):
    rows = bench.check_size(B, N, _Args, DEV, "sharp", ["current", "hview", "cached"], True, 11)
    assert [r["variant"] for r in rows] == ["current", "hview", "cached"]
    for r in rows:
        assert r["status"] == "ok" and r["agrees"] is True, r
    assert rows[0]["reimpl_max_abs_diff"] == 0.0
    for r in rows[1:]:
        assert r["vs_current"]["out_max_abs"] < 1e-5
        assert r["checks"]["params"]["missing"] == []

    real = bench.forward_variant

    def broken(m, z, mask, variant, cache=None, label=False):
        out = real(m, z, mask, variant, cache, label)
        return out * 1.01 if variant == "hview" else out

    monkeypatch.setattr(bench, "forward_variant", broken)
    rows = bench.check_size(B, N, _Args, DEV, "sharp", ["current", "hview"], True, 11)
    assert rows[1]["status"] == "ok" and rows[1]["agrees"] is False
    assert rows[1]["checks"]["out"]["ok"] is False


def test_agreement_rule():
    def cmp(out, dz, params):
        e = lambda x: dict(rel_l2=x, max_abs=x, finite=True)  # noqa: E731
        return dict(out=e(out), dz=e(dz), params={k: e(v) for k, v in params.items()})
    cur = cmp(6e-3, 7e-3, {"w": 7e-3, "b": 1e-3})
    ok = bench.agreement(cmp(7e-3, 8e-3, {"w": 8e-3, "b": 1.2e-3}), cur, 1.25, 1e-4)
    assert ok["ok"]
    bad = bench.agreement(cmp(7e-3, 8e-3, {"w": 8e-3, "b": 5e-3}), cur, 1.25, 1e-4)
    assert not bad["ok"] and bad["checks"]["params"]["worst"]["name"] == "b"
    tiny = bench.agreement(cmp(5e-5, 5e-5, {"w": 5e-5, "b": 5e-5}),
                           cmp(1e-7, 1e-7, {"w": 1e-7, "b": 1e-7}), 1.25, 1e-4)
    assert tiny["ok"]  # the floor: fp32 round-off is not a disagreement


def _synthetic_trace():
    """Two forward ops (labelled mask copy, SDPA), their backward, one parameter-grad kernel."""
    ev = []
    X = lambda cat, name, ts, dur, tid=1, **args: ev.append(  # noqa: E731
        dict(ph="X", cat=cat, name=name, ts=ts, dur=dur, tid=tid, pid=0, args=args))
    X("user_annotation", "k117:mask_expand_copy", 0, 10)
    X("cpu_op", "aten::reshape", 1, 8, **{"Sequence number": 5, "External id": 1})
    X("cpu_op", "aten::clone", 2, 6, **{"External id": 2})
    X("cpu_op", "aten::copy_", 3, 4, **{"External id": 3})
    X("xpu_runtime", "zeCommandListAppendLaunchKernel", 4, 1, correlation=100)
    X("kernel", "copy_kernel", 50, 30, tid=9, correlation=100)
    X("user_annotation", "k117:sdpa", 20, 10)
    X("cpu_op", "aten::scaled_dot_product_attention", 21, 8,
      **{"Sequence number": 6, "External id": 4})
    X("xpu_runtime", "zeCommandListAppendLaunchKernel", 22, 1, correlation=101)
    X("kernel", "fused_attn_fwd", 90, 100, tid=9, correlation=101)
    # backward (autograd thread 2)
    X("cpu_op", "autograd::engine::evaluate_function: FusedAttnBackward0", 200, 10, tid=2,
      **{"Sequence number": 6})
    X("xpu_runtime", "zeCommandListAppendLaunchKernel", 201, 1, tid=2, correlation=102)
    X("kernel", "fused_attn_bwd", 300, 200, tid=9, correlation=102)
    X("cpu_op", "autograd::engine::evaluate_function: ExpandBackward0", 220, 10, tid=2,
      **{"Sequence number": 5})
    X("cpu_op", "aten::sum", 221, 5, tid=2, **{"External id": 9})
    X("xpu_runtime", "zeCommandListAppendLaunchKernel", 222, 1, tid=2, correlation=103)
    X("kernel", "reduce_kernel", 600, 40, tid=9, correlation=103)
    X("cpu_op", "autograd::engine::evaluate_function: torch::autograd::AccumulateGrad", 240,
      5, tid=2)
    X("xpu_runtime", "zeCommandListAppendLaunchKernel", 241, 1, tid=2, correlation=104)
    X("kernel", "add_kernel", 700, 6, tid=9, correlation=104)
    X("kernel", "orphan_kernel", 800, 4, tid=9, correlation=999)
    return ev


def test_breakdown_on_a_synthetic_device_trace():
    r = bench.breakdown(_synthetic_trace(), steps=2)
    assert r["basis"] == "device" and r["n_device_events"] == 6
    sub = r["by_subcategory_us"]
    assert sub["copy/mask_expand"] == pytest.approx(15)       # 30 us / 2 steps
    assert sub["copy/mask_expand:bwd"] == pytest.approx(20)   # ExpandBackward's sum
    assert sub["attention/fwd"] == pytest.approx(50)
    assert sub["attention/bwd"] == pytest.approx(100)
    assert sub["other/bwd:torch::autograd::AccumulateGrad"] == pytest.approx(3)
    assert r["unattributed_us_per_step"] == pytest.approx(2)
    assert r["total_us_per_step"] == pytest.approx(190)
    assert r["fraction"]["attention"] == pytest.approx(150 / 190)
    assert "aten::scaled_dot_product_attention" in r["sdpa_ops"]


@pytest.mark.parametrize("ctx,bwd,inner,expected", [
    (["k117:qkv_layout", "aten::reshape", "aten::clone"], None, "aten::copy_", ("copy", "qkv_layout")),
    (["aten::linear", "aten::to", "aten::_to_copy"], None, "aten::copy_", ("copy", "cast")),
    (["aten::reshape", "aten::clone"], None, "aten::copy_", ("copy", "copy:aten::reshape")),
    (["aten::slice"], "SliceBackward0", "aten::add", ("copy", "layout_bwd:aten::slice")),
    ([], "torch::autograd::CopySlices", "aten::copy_", ("copy", "layout_bwd:CopySlices")),
    (["aten::scaled_dot_product_attention"], None, "aten::contiguous",
     ("attention", "internal_copy")),
    (["aten::linear"], "AddmmBackward0", "aten::mm", ("other", "bwd:AddmmBackward0")),
    (["k117:gate_out", "aten::sigmoid"], None, "aten::sigmoid", ("other", "gate_out")),
])
def test_classify(ctx, bwd, inner, expected):
    assert bench.classify(ctx, bwd, inner) == expected


def test_full_run_writes_the_json_schema(tmp_path):
    out = tmp_path / "res" / "local.json"
    rc = bench.main(["--out", str(out), "--batches", "2", "--peaks", "24,40", "--c-z", "16",
                     "--heads", "2", "--dim", "8", "--chunk", "16", "--precision", "fp32",
                     "--reps", "1", "--profile-steps", "1", "--inits", "sharp",
                     "--trace-sizes", "2x24", "--device", "cpu"])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["status"] == "done" and doc["task"] == "K117-P"
    for key in ("env", "args", "dims", "variants", "tolerance", "hview_mask_view_check",
                "sdpa_dispatch", "results", "summary", "elapsed_s"):
        assert key in doc, key
    assert list(doc["variants"]) == ["current", "sdpa5d", "hview", "cached"]
    kinds = {r["kind"] for r in doc["results"]}
    assert {"accuracy", "timing", "profile"} <= kinds
    acc = [r for r in doc["results"] if r["kind"] == "accuracy"]
    assert len(acc) == 2 * 2 * 4  # sizes x modules x variants
    assert all(r["status"] == "ok" and r["agrees"] for r in acc)
    timing = [r for r in doc["results"] if r["kind"] == "timing"]
    assert len(timing) == 2 * 2 * 4  # sizes x modes x variants
    for r in timing:
        assert r["status"] == "ok" and r["time_ms"] > 0 and r["mode"] in ("fwd", "fwdbwd")
    assert len(doc["summary"]) == len(timing)
    for s in doc["summary"]:
        assert s["speedup_vs_current"] is not None and s["agrees"] is True
        if s["variant"] == "current":
            assert s["speedup_vs_current"] == pytest.approx(1.0)
    prof = [r for r in doc["results"] if r["kind"] == "profile"]
    assert len(prof) == 2 * 2 * 4
    ok = [r for r in prof if r["status"] == "ok"]
    if not ok:
        pytest.skip("Kineto unavailable here: " + prof[0].get("error", "")[:200])
    for r in ok:
        assert r["basis"] in ("device", "cpu_self") and r["total_us_per_step"] > 0
        assert set(r["by_category_us"]) <= {"attention", "copy", "other", "unattributed"}
        assert "attention" in r["by_category_us"]
    cur = next(r for r in ok if r["variant"] == "current" and r["mode"] == "fwdbwd")
    assert "copy/mask_expand" in cur["by_subcategory_us"]
    assert "copy/mask_expand:bwd" in cur["by_subcategory_us"]
    cached = next((r for r in ok if r["variant"] == "cached" and r["mode"] == "fwd"), None)
    if cached is not None:
        assert "copy/mask_expand" not in cached["by_subcategory_us"]
        assert "other/mask_build" not in cached["by_subcategory_us"]
    traces = sorted((out.parent / "local_traces").glob("*.json.gz"))
    assert len(traces) == len([r for r in ok if r["N"] == 24])
