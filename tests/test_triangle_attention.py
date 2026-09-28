"""Triangle-attention memory options (K102): per-chunk checkpointing and the SDPA path.

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
]


@pytest.mark.parametrize("starting", [True, False], ids=["starting", "ending"])
@pytest.mark.parametrize("variant", VARIANTS, ids=["naive", "naive+ckpt", "sdpa", "sdpa+ckpt"])
@pytest.mark.parametrize("flatten", [True, False], ids=["sdpa4d", "sdpa5d"])
def test_matches_reference(starting, variant, flatten):
    if not flatten and variant.get("pair_tri_attn_impl") != "sdpa":
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


@pytest.mark.parametrize("variant", VARIANTS[1:], ids=["naive+ckpt", "sdpa", "sdpa+ckpt"])
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
    naive = _saved_bytes(_module(True, pair_tri_attn_chunk=chunk), z, mask)
    ckpt = _saved_bytes(_module(True, pair_tri_attn_chunk=chunk,
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
    assert MSDeltaConfig().pair_tri_attn_impl == "naive"
    assert MSDeltaConfig().pair_tri_attn_checkpoint_chunks is False
