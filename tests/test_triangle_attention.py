"""Triangle-attention memory options (K102): per-chunk checkpointing and the SDPA paths
("sdpa", and "sdpa_view" from K117: rows in SDPA's head dim, stride-0 mask view, no mask copy).

Both must be numerically equivalent to the naive chunked path (outputs and gradients, fp32 on
CPU, with padded keys and padded queries, starting and ending node), and checkpointing must
drop the per-chunk attention weights from what autograd keeps.
"""

from __future__ import annotations

import math

import pytest
import torch

from msdelta.models.configuration_msdelta import MSDeltaConfig
from msdelta.models.modeling_msdelta import MSDeltaModel
from msdelta.models.pairformer import TriangleAttention

B, N, H, D, C_Z, CHUNK = 3, 11, 2, 4, 8, 4  # 11 = 4 + 4 + 3: a ragged last chunk


def _config(**overrides) -> MSDeltaConfig:
    base = dict(
        architecture="pairformer", hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
        intermediate_size=64, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        delta_bias_n_freqs=8, delta_bias_per_head_hidden=4, pair_channels=C_Z,
        pair_tri_channels=8, pair_use_triangle_attention=True, pair_tri_attn_heads=H,
        pair_tri_attn_dim=D, pair_tri_attn_chunk=CHUNK, pair_opm_channels=4,
    )
    base.update(overrides)
    return MSDeltaConfig(**base)


def _reference_forward(m: TriangleAttention, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The pre-K102 forward, verbatim (the equivalence target)."""
    if not m.starting:
        z = z.transpose(1, 2)
    z = m.norm(z)
    b, n = z.shape[:2]
    q = m.q(z).view(b, n, n, m.h, m.d)
    k = m.k(z).view(b, n, n, m.h, m.d)
    v = m.v(z).view(b, n, n, m.h, m.d)
    bias = m.bias(z)
    key_mask = torch.zeros(mask.shape, dtype=torch.float32, device=z.device)
    key_mask = key_mask.masked_fill(~mask, torch.finfo(torch.float32).min)
    key_mask = key_mask[:, None, None, :, None]
    scale = 1.0 / math.sqrt(m.d)
    out = torch.empty_like(q)
    for s in range(0, n, m.chunk):
        e = min(s + m.chunk, n)
        logits = torch.einsum("bcjhd,bckhd->bcjkh", q[:, s:e], k[:, s:e]) * scale
        logits = logits.float() + bias[:, None].float() + key_mask
        attn = torch.softmax(logits, dim=3).to(v.dtype)
        out[:, s:e] = torch.einsum("bcjkh,bckhd->bcjhd", attn, v[:, s:e])
    out = torch.sigmoid(m.gate(z)) * out.reshape(b, n, n, -1)
    out = m.out(out)
    return out if m.starting else out.transpose(1, 2)


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, N, N, C_Z, generator=g)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, N - 4:] = False  # 4 padded peaks: padded keys AND padded query rows
    mask[2, N - 1:] = False
    return z, mask


def _module(starting: bool, **overrides) -> TriangleAttention:
    torch.manual_seed(0)
    m = TriangleAttention(_config(**overrides), starting=starting)
    with torch.no_grad():  # non-trivial bias / gate so every term matters
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return m


def _run(m, z, mask, fn=None):
    z = z.clone().requires_grad_(True)
    out = m(z, mask) if fn is None else fn(m, z, mask)
    g = torch.Generator().manual_seed(1)
    w = torch.randn(out.shape, generator=g)
    (out * w).sum().backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    return out.detach(), z.grad.clone(), grads


VARIANTS = [
    dict(),
    dict(pair_tri_attn_checkpoint_chunks=True),
    dict(pair_tri_attn_impl="sdpa"),
    dict(pair_tri_attn_impl="sdpa", pair_tri_attn_checkpoint_chunks=True),
    dict(pair_tri_attn_impl="naive"),
    dict(pair_tri_attn_impl="sdpa_view"),
    dict(pair_tri_attn_impl="sdpa_view", pair_tri_attn_checkpoint_chunks=True),
]
VARIANT_IDS = ["default", "default+ckpt", "sdpa", "sdpa+ckpt", "naive", "sdpa_view",
               "sdpa_view+ckpt"]


@pytest.mark.parametrize("starting", [True, False], ids=["starting", "ending"])
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize("flatten", [True, False], ids=["sdpa4d", "sdpa5d"])
def test_matches_reference(starting, variant, flatten):
    if not flatten and variant.get("pair_tri_attn_impl", "sdpa") != "sdpa":
        pytest.skip("sdpa_flatten only affects the sdpa path")
    z, mask = _inputs()
    m = _module(starting, **variant)
    m.sdpa_flatten = flatten
    ref_out, ref_dz, ref_grads = _run(m, z, mask, _reference_forward)
    out, dz, grads = _run(m, z, mask)
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dz, ref_dz, rtol=1e-5, atol=1e-5)
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol=1e-5, atol=1e-5,
                                   msg=lambda s, n=name: f"{n}: {s}")


@pytest.mark.parametrize("variant", VARIANTS[1:], ids=VARIANT_IDS[1:])
def test_no_grad_matches(variant):
    z, mask = _inputs(seed=3)
    m = _module(True, **variant)
    with torch.no_grad():
        torch.testing.assert_close(m(z, mask), _reference_forward(m, z, mask),
                                   rtol=1e-5, atol=1e-5)


def _saved_bytes(m, z, mask) -> int:
    """Bytes autograd keeps for backward (unique storages of the saved tensors)."""
    storages: dict[int, int] = {}

    def pack(t):
        s = t.untyped_storage()
        storages[s.data_ptr()] = s.nbytes()
        return t

    z = z.clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = m(z, mask)
    out.sum().backward()
    return sum(storages.values())


def test_checkpoint_drops_saved_attention_weights():
    n, chunk, h, b = 32, 4, 2, 2
    g = torch.Generator().manual_seed(0)
    z = torch.randn(b, n, n, C_Z, generator=g)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[1, n - 6:] = False
    naive = _saved_bytes(_module(True, pair_tri_attn_chunk=chunk, pair_tri_attn_impl="naive"), z, mask)
    ckpt = _saved_bytes(_module(True, pair_tri_attn_chunk=chunk, pair_tri_attn_impl="naive",
                                pair_tri_attn_checkpoint_chunks=True), z, mask)
    all_weights = b * n ** 3 * h * 4        # fp32 softmax output over every chunk
    one_chunk = b * chunk * n * n * h * 4
    # Naive keeps every chunk's attention weights; checkpointing keeps none of them (only the
    # chunk inputs, which are views of the per-module q/k/v/bias tensors).
    assert naive >= all_weights
    assert naive - ckpt >= all_weights
    assert ckpt < one_chunk + 12 * b * n * n * C_Z * 4  # a dozen pair-sized tensors at most


def test_model_level_equivalence():
    g = torch.Generator().manual_seed(0)
    mz = torch.sort(torch.rand(2, 13, generator=g) * 1500 + 100, dim=1).values
    li = torch.rand(2, 13, generator=g)
    am = torch.ones(2, 13, dtype=torch.long)
    am[1, 9:] = 0
    outs = []
    for variant in VARIANTS:
        torch.manual_seed(0)
        model = MSDeltaModel(_config(**variant)).eval()
        outs.append(model(mz=mz, log_intensity=li, attention_mask=am).last_hidden_state)
    for o in outs[1:]:
        torch.testing.assert_close(o, outs[0], rtol=1e-5, atol=1e-5)


def test_config_validation():
    with pytest.raises(ValueError, match="pair_tri_attn_impl"):
        _config(pair_tri_attn_impl="flash")
    assert _config(pair_tri_attn_impl="sdpa_view").pair_tri_attn_impl == "sdpa_view"
    assert MSDeltaConfig().pair_tri_attn_impl == "sdpa"  # default since K115
    assert MSDeltaConfig().pair_tri_attn_checkpoint_chunks is False


# ---------------------------------------------------------------- sdpa_view (K117) --------

def _pair(starting, impl_a, impl_b, ckpt=False, **overrides):
    """Two modules with identical weights, differing only in the impl."""
    a = _module(starting, pair_tri_attn_impl=impl_a, pair_tri_attn_checkpoint_chunks=ckpt,
                **overrides)
    b = _module(starting, pair_tri_attn_impl=impl_b, pair_tri_attn_checkpoint_chunks=ckpt,
                **overrides)
    b.load_state_dict(a.state_dict())
    return a, b


def _inputs_n(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, n, n, C_Z, generator=g)
    mask = torch.ones(B, n, dtype=torch.bool)
    if n > 1:
        mask[1, max(1, n - 4):] = False  # padded keys and padded query rows
        mask[2, n - 1:] = False
    return z, mask


@pytest.mark.parametrize("starting", [True, False], ids=["starting", "ending"])
@pytest.mark.parametrize("ckpt", [False, True], ids=["plain", "ckpt"])
@pytest.mark.parametrize("n,chunk", [(11, 4), (13, 5), (7, 32), (9, 1), (16, 16), (1, 4)],
                         ids=["n11c4", "n13c5", "n7c32", "n9c1", "n16c16", "n1c4"])
@pytest.mark.parametrize("other", ["sdpa", "naive"])
def test_sdpa_view_matches_fp32(starting, ckpt, n, chunk, other):
    """Outputs, dL/dz and every parameter gradient match "sdpa" to ~1e-6 (fp32; forward is
    bit-identical on CPU) and "naive" to the suite's 1e-5 (sdpa itself differs from naive by up
    to ~2e-5 abs on the larger parameter gradients: summation order)."""
    view, ref = _pair(starting, "sdpa_view", other, ckpt, pair_tri_attn_chunk=chunk)
    z, mask = _inputs_n(n)
    out_r, dz_r, g_r = _run(ref, z, mask)
    out_v, dz_v, g_v = _run(view, z, mask)
    tol = dict(rtol=1e-6, atol=2e-6) if other == "sdpa" else dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_v, out_r, **tol)
    torch.testing.assert_close(dz_v, dz_r, **tol)
    for name in g_r:
        torch.testing.assert_close(g_v[name], g_r[name], **tol,
                                   msg=lambda s, n=name: f"{n}: {s}")


def test_sdpa_view_fully_padded_sample_is_finite():
    view, ref = _pair(True, "sdpa_view", "sdpa")
    z, mask = _inputs_n(N)
    mask[2] = False  # every key padded: the dtype-min mask keeps the softmax finite
    out_r, dz_r, _ = _run(ref, z, mask)
    out_v, dz_v, _ = _run(view, z, mask)
    assert torch.isfinite(out_v).all() and torch.isfinite(dz_v).all()
    torch.testing.assert_close(out_v, out_r, rtol=0, atol=2e-6)
    torch.testing.assert_close(dz_v, dz_r, rtol=0, atol=2e-6)


def _rel_l2(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("starting", [True, False], ids=["starting", "ending"])
def test_sdpa_view_bf16_no_less_accurate_than_sdpa(starting):
    """bf16 autocast vs the fp32 naive reference: sdpa_view is no less accurate than sdpa (the
    K115 acceptance rule, rel-L2 <= max(2 x sdpa's, 1e-3)) for the output, dz and every grad."""
    z, mask = _inputs_n(N, seed=5)
    out_ref, dz_ref, g_ref = _run(_module(starting, pair_tri_attn_impl="naive"), z, mask)
    res = {}
    for impl in ("sdpa", "sdpa_view"):
        m = _module(starting, pair_tri_attn_impl=impl)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            res[impl] = _run(m, z, mask, lambda mm, zz, mk: mm(zz, mk).float())
    names = ["out", "dz"] + list(g_ref)

    def pick(r, name):
        return r[0] if name == "out" else r[1] if name == "dz" else r[2][name]

    ref = (out_ref, dz_ref, g_ref)
    for name in names:
        r, s, v = pick(ref, name), pick(res["sdpa"], name), pick(res["sdpa_view"], name)
        err_s, err_v = _rel_l2(s, r), _rel_l2(v, r)
        assert err_v <= max(2 * err_s, 1e-3), f"{name}: view {err_v:.2e} vs sdpa {err_s:.2e}"
        assert _rel_l2(v, s) <= max(2 * err_s, 1e-3), f"{name}: view-sdpa {_rel_l2(v, s):.2e}"


def test_sdpa_view_mask_is_a_view(monkeypatch):
    """Each chunk's SDPA mask shares one (B, H, N, N) storage, stride 0 over the rows: no copy."""
    import msdelta.models.pairformer as pf

    seen = []
    orig = pf.F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, **kw):
        seen.append(attn_mask)
        return orig(q, k, v, attn_mask=attn_mask, **kw)

    monkeypatch.setattr(pf.F, "scaled_dot_product_attention", spy)
    m = _module(True, pair_tri_attn_impl="sdpa_view")
    z, mask = _inputs_n(N)
    with torch.no_grad():
        m(z, mask)
    assert len(seen) == math.ceil(N / CHUNK)
    assert len({am.untyped_storage().data_ptr() for am in seen}) == 1
    for am in seen:
        assert am.shape[0] == B * H and am.stride(1) == 0


def test_impls_share_state_dict():
    """Switching impl changes no parameters: checkpoints load across impls unchanged."""
    sd = {}
    for impl in ("naive", "sdpa", "sdpa_view"):
        torch.manual_seed(0)
        sd[impl] = MSDeltaModel(_config(pair_tri_attn_impl=impl)).state_dict()
    assert list(sd["sdpa_view"]) == list(sd["sdpa"]) == list(sd["naive"])
    for k in sd["sdpa"]:
        assert sd["sdpa_view"][k].shape == sd["sdpa"][k].shape
