"""CPU checks for the K114 / K119 diagnostics (pbs/diag/pairformer_profile.py,
pbs/diag/flexattn_test.py).

K114: the block profiler's wrappers must be identities on values -- same loss, same gradients
with the profiler active, inactive, and removed -- and its backward attribution must tile the
measured backward. K119: everything in the FlexAttention test that does not need a GPU -- the
score_mod / mask_mod indexing of the triangle bias, both layouts, both masking modes, the full
module rebuilt around the pluggable core -- against the model's own SDPA and naive paths.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from msdelta.models.configuration_msdelta import MSDeltaConfig
from msdelta.models.modeling_msdelta import MSDeltaForPreTraining

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "pbs/diag" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prof_mod = _load("pairformer_profile")
flex_mod = _load("flexattn_test")


# ---------------------------------------------------------------------------- K114 ----

def _tiny(arch: str, **kw) -> MSDeltaConfig:
    base = dict(hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                intermediate_size=64, hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0, delta_bias_n_freqs=8,
                delta_bias_per_head_hidden=4)
    if arch == "pairformer":
        base.update(architecture="pairformer", pair_channels=8, pair_tri_channels=8,
                    pair_opm_channels=4, pair_tri_attn_heads=2, pair_tri_attn_dim=4,
                    pair_tri_attn_chunk=5, pair_use_triangle_attention=True,
                    pair_tri_attn_impl="sdpa")
    base.update(kw)
    return MSDeltaConfig(**base)


def _loss_and_grads(model, batch):
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return out.loss.detach(), out.logits.detach(), grads


@pytest.mark.parametrize("arch", ["pairformer", "transformer"])
def test_profiler_is_identity_on_outputs_and_grads(arch):
    cfg = _tiny(arch)
    torch.manual_seed(0)
    model = MSDeltaForPreTraining(cfg).train()
    # The write-back output projection is zero-initialised; give it weights so its
    # gradient path is exercised too.
    with torch.no_grad():
        for p in model.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.02)
    batch = prof_mod.make_batch(3, 14, seed=1, mask_ratio=0.5)
    assert batch["attention_mask"].shape == (3, 14) and (batch["attention_mask"] == 0).any()
    ref = _loss_and_grads(model, batch)

    prof = prof_mod.BlockProfiler(torch.device("cpu"))
    labels = prof_mod.instrument(model, prof)
    inactive = _loss_and_grads(model, batch)
    prof.active = True
    prof.reset()
    active = _loss_and_grads(model, batch)
    prof.active = False
    assert prof.bwd_events, "no backward markers fired"
    prof.remove()
    removed = _loss_and_grads(model, batch)
    assert not any("forward" in vars(m) for m in model.modules())

    for other in (inactive, active, removed):
        assert torch.equal(other[0], ref[0])
        assert torch.equal(other[1], ref[1])
        assert other[2].keys() == ref[2].keys()
        # Values are identical; the extra marker nodes can change the order in which the
        # engine accumulates a parameter's gradient contributions (float rounding only).
        for name in ref[2]:
            a, r = other[2][name], ref[2][name]
            assert (a - r).norm() <= 1e-4 * r.norm() + 1e-12, name  # seen: 1.6e-5

    expected = {"token_embed", "final_ln", "head"}
    expected |= ({"z_init", "pair_features", "a_writeback", "b_trimul_out", "c_trimul_in",
                  "d_triattn_start", "e_triattn_end", "f_pair_transition", "g_bias_readout",
                  "pair_glue", "h_single_attention", "i_single_transition"}
                 if arch == "pairformer" else {"dmz_bias", "attention", "ffn", "block_glue"})
    assert expected <= set(labels)


@pytest.mark.parametrize("arch", ["pairformer", "transformer"])
def test_profiled_mode_tiles_the_step(arch):
    cfg = _tiny(arch)
    model = prof_mod.build_model(cfg, torch.device("cpu"))
    prof = prof_mod.BlockProfiler(torch.device("cpu"))
    prof_mod.instrument(model, prof)
    batch = prof_mod.make_batch(2, 10, seed=0)
    rec = prof_mod.profiled_mode(model, batch, torch.device("cpu"), prof, warmup=1, reps=1)
    blocks = rec["blocks"]
    assert sum(v["bwd_ms"] for v in blocks.values()) == pytest.approx(rec["bwd_ms"], rel=1e-6)
    assert sum(v["fwd_ms"] for v in blocks.values()) == pytest.approx(rec["fwd_ms"], rel=1e-6)
    per_layer = {k: v["calls"] for k, v in blocks.items()}
    if arch == "pairformer":
        assert per_layer["b_trimul_out"] == cfg.num_hidden_layers
        assert per_layer["h_single_attention"] == cfg.num_hidden_layers
        assert blocks["b_trimul_out"]["bwd_ms"] > 0 and blocks["h_single_attention"]["bwd_ms"] > 0
        cal = prof_mod.calibration(rec, cfg.num_hidden_layers)
        assert cal["single_blocks_per_pair_update_total"] > 0
    else:
        assert per_layer["attention"] == cfg.num_hidden_layers
        assert blocks["attention"]["bwd_ms"] > 0


def test_make_batch_masks_like_pretraining():
    b = prof_mod.make_batch(8, 100, seed=0, mask_ratio=0.5)
    lengths = b["attention_mask"].sum(1)
    assert lengths.max() == 100 and lengths.min() >= 30
    valid = b["attention_mask"].bool()
    assert (b["mz"][valid] >= 100).all() and (b["mz"][valid] <= 2000).all()
    assert (b["mask_positions"] & ~valid).sum() == 0
    for row in range(8):
        assert b["mask_positions"][row].sum() == round(0.5 * int(lengths[row]))


def test_stage0_config_copy_matches_the_branch():
    """The profiling config is a verbatim copy of stage0-prep's (cde0905), when git has it."""
    import subprocess
    local = json.loads((REPO / prof_mod.PAIRFORMER_CONFIG).read_text())
    assert local["architecture"] == "pairformer" and local["pair_use_triangle_attention"] is False
    out = subprocess.run(["git", "show", "cde0905:configs/stage0/pairformer/config.json"],
                         cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("commit cde0905 not available")
    assert json.loads(out.stdout) == local


def test_load_config_variants():
    p = prof_mod.load_config("pairformer", str(REPO / prof_mod.PAIRFORMER_CONFIG),
                             str(REPO / prof_mod.TRANSFORMER_CONFIG))
    t = prof_mod.load_config("pairformer_triattn", str(REPO / prof_mod.PAIRFORMER_CONFIG),
                             str(REPO / prof_mod.TRANSFORMER_CONFIG))
    x = prof_mod.load_config("transformer", str(REPO / prof_mod.PAIRFORMER_CONFIG),
                             str(REPO / prof_mod.TRANSFORMER_CONFIG))
    assert not p.pair_use_triangle_attention
    assert t.pair_use_triangle_attention and t.pair_tri_attn_impl == "sdpa"
    assert getattr(x, "architecture", "transformer") == "transformer" and x.hidden_size == 640


# ---------------------------------------------------------------------------- K119 ----

def _module(impl, starting, chunk=3):
    return flex_mod.build_module(impl, starting, c_z=8, heads=2, dim=4, chunk=chunk)


def _ref_attn(q, k, v, score_mod=None, mask_mod=None, block_mask=None, scale=None):
    return flex_mod.reference_flex(q, k, v, score_mod=score_mod, mask_mod=mask_mod, scale=scale)


def test_score_mod_indexing_matches_expanded_bias():
    """bias[b // N] (rowbatch) and bias[b, h % H] (rowhead) pick the right bias entries."""
    torch.manual_seed(0)
    b, h, n = 2, 3, 5
    bias = torch.randn(b, h, n, n)
    valid = torch.ones(b, n, dtype=torch.bool)
    valid[1, 3:] = False
    scores = torch.zeros(b * n, h, n, n)
    for layout in ("rowbatch", "rowhead"):
        shape = (b * n, h, n, n) if layout == "rowbatch" else (b, n * h, n, n)
        sm, mm = flex_mod.make_mods(bias, valid, n, h, layout, "blockmask")
        bi = torch.arange(shape[0]).view(-1, 1, 1, 1)
        hi = torch.arange(shape[1]).view(1, -1, 1, 1)
        qi = torch.arange(n).view(1, 1, n, 1)
        ki = torch.arange(n).view(1, 1, 1, n)
        got = sm(scores.reshape(shape), bi, hi, qi, ki)
        mask = mm(bi, hi, qi, ki).expand(shape)
        for bb in range(shape[0]):
            for hh in range(shape[1]):
                if layout == "rowbatch":
                    sb, sh = bb // n, hh
                else:
                    sb, sh = bb, hh % h
                assert torch.equal(got[bb, hh], bias[sb, sh])
                assert torch.equal(mask[bb, hh, 0], valid[sb])


@pytest.mark.parametrize("layout", ["rowbatch", "rowhead"])
@pytest.mark.parametrize("masking", ["blockmask", "scoremask"])
@pytest.mark.parametrize("starting", [True, False])
def test_flex_core_reference_matches_module(layout, masking, starting):
    """The rebuilt module with the flex-semantics core == the model's SDPA and naive paths,
    outputs and gradients (z and every parameter), fp32, with padded keys."""
    sdpa = _module("sdpa", starting)
    naive = _module("naive", starting)
    naive.load_state_dict(sdpa.state_dict())
    z, mask = flex_mod.make_inputs(3, 7, 8, torch.device("cpu"), pad_frac=0.3)
    assert (~mask).any()

    def flex_call(m, zz, mm):
        return flex_mod.flex_triangle_attention(m, zz, mm, _ref_attn, layout, masking, None)

    out_f, g_f = flex_mod.fwd_bwd(flex_call, sdpa, z, mask, "fp32")
    for module in (sdpa, naive):
        out_m, g_m = flex_mod.fwd_bwd(flex_mod.module_call, module, z, mask, "fp32")
        res = flex_mod.compare_all(out_f, g_f, out_m, g_m)
        assert res["all_finite"]
        assert res["worst_rel_l2"] < 1e-5, res


def test_eager_flex_attention_on_cpu_matches_module():
    """The real torch flex_attention (eager, CPU) with our score_mod / block_mask, if usable."""
    fa = pytest.importorskip("torch.nn.attention.flex_attention")
    sdpa = _module("sdpa", True)
    z, mask = flex_mod.make_inputs(2, 6, 8, torch.device("cpu"), pad_frac=0.34)

    def attn_fn(q, k, v, score_mod=None, mask_mod=None, block_mask=None, scale=None):
        return fa.flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask, scale=scale)

    def bm_fn(mask_mod, B, H, Q, KV):
        return fa.create_block_mask(mask_mod, B, H, Q, KV, device="cpu")

    try:
        for layout in ("rowbatch", "rowhead"):
            out_f, g_f = flex_mod.fwd_bwd(
                lambda m, zz, mm: flex_mod.flex_triangle_attention(m, zz, mm, attn_fn, layout,
                                                                   "blockmask", bm_fn),
                sdpa, z, mask, "fp32")
            out_s, g_s = flex_mod.fwd_bwd(flex_mod.module_call, sdpa, z, mask, "fp32")
            res = flex_mod.compare_all(out_f, g_f, out_s, g_s)
            assert res["worst_rel_l2"] < 1e-4, (layout, res)
    except (NotImplementedError, RuntimeError) as e:  # pragma: no cover - torch-version dependent
        pytest.skip(f"eager flex_attention unavailable on CPU here: {e}")


def test_prefer_system_triton_shims_only_triton(tmp_path, monkeypatch):
    site = tmp_path / "site"
    (site / "triton" / "backends" / "intel").mkdir(parents=True)
    (site / "triton-3.6.0+git.dist-info").mkdir()
    (site / "transformers").mkdir()
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    info = flex_mod.prefer_system_triton(str(site))
    shim = Path(info["shim"])
    assert sys.path[0] == str(shim)
    assert os.environ["PYTHONPATH"] == f"{shim}:/somewhere"
    assert sorted(p.name for p in shim.iterdir()) == ["triton", "triton-3.6.0+git.dist-info"]
    assert (shim / "triton").resolve() == (site / "triton").resolve()
    assert "error" not in info
    assert flex_mod.prefer_system_triton(str(tmp_path / "nope"))["error"]
