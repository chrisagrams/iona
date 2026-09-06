"""Correctness tests for the Pairformer encoder (spec section 11).

Runnable two ways:

    python -m pytest tests/test_pairformer.py -q      # if pytest is installed
    python tests/test_pairformer.py                   # plain script, no extra deps

These are the day-one invariants that catch the whole class of indexing / masking bugs the
triangle operations invite: permutation equivariance (T1), mask invariance (T2), pair
asymmetry (T3), and a shape/dtype smoke test (T5). They run on CPU in float32 in a few seconds.
"""

from __future__ import annotations

import sys

import torch

from msdelta.model.configuration import MSDeltaConfig
from msdelta.model.experiments.pairformer import (
    MSDeltaPairformerConfig,
    MSDeltaPairformerForPreTraining,
    MSDeltaPairformerModel,
)
from msdelta.model.modeling import MSDeltaForPreTraining


def _config(**overrides) -> MSDeltaPairformerConfig:
    base = dict(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        fourier_int_n_freqs=4,
        delta_bias_n_freqs=8,
        mass_defect_n_freqs=4,
        pair_channels=16,
        tri_channels=16,
        tri_attn_heads=2,
        tri_attn_dim=8,
        tri_attn_chunk=3,
        opm_channels=4,
        global_cond_dim=16,
        pair_update="triangle",
        use_triangle_attention=True,
        use_writeback=True,
        use_global_cond=True,
    )
    base.update(overrides)
    return MSDeltaPairformerConfig(**base)


def _inputs(n: int, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    mz = torch.sort(torch.rand(1, n, generator=g) * 1800 + 100, dim=1).values
    log_intensity = torch.rand(1, n, generator=g)
    attention_mask = torch.ones(1, n, dtype=torch.long)
    precursor_mz = torch.tensor([1650.0])
    charge = torch.tensor([2])
    return mz, log_intensity, attention_mask, precursor_mz, charge


def _encode(model, mz, log_intensity, attention_mask, precursor_mz, charge):
    with torch.no_grad():
        return model(
            mz=mz,
            log_intensity=log_intensity,
            attention_mask=attention_mask,
            precursor_mz=precursor_mz,
            charge=charge,
        ).last_hidden_state


def test_t1_permutation_equivariance():
    """Peaks are a set: permuting the input permutes the output identically."""
    torch.manual_seed(0)
    model = MSDeltaPairformerModel(_config()).eval()
    n = 9
    mz, li, am, prec, charge = _inputs(n)
    base = _encode(model, mz, li, am, prec, charge)

    perm = torch.randperm(n)
    permd = _encode(model, mz[:, perm], li[:, perm], am[:, perm], prec, charge)
    err = (permd - base[:, perm]).abs().max().item()
    assert err < 1e-4, f"permutation equivariance broken: max abs diff {err:.2e}"


def test_t2_mask_invariance():
    """Appending padded peaks must not change the outputs on the real peaks."""
    torch.manual_seed(0)
    model = MSDeltaPairformerModel(_config()).eval()
    n, pad = 7, 5
    mz, li, am, prec, charge = _inputs(n)
    base = _encode(model, mz, li, am, prec, charge)

    mz_p = torch.cat([mz, torch.rand(1, pad) * 1800 + 100], dim=1)
    li_p = torch.cat([li, torch.rand(1, pad)], dim=1)
    am_p = torch.cat([am, torch.zeros(1, pad, dtype=torch.long)], dim=1)
    padded = _encode(model, mz_p, li_p, am_p, prec, charge)
    err = (padded[:, :n] - base).abs().max().item()
    assert err < 1e-4, f"mask invariance broken: max abs diff {err:.2e} (missing mask in a sum?)"


def test_t3_pair_asymmetry():
    """z_ij must differ from z_ji (W_a and W_b are separate matrices; loss has a direction)."""
    torch.manual_seed(0)
    model = MSDeltaPairformerModel(_config()).eval()
    n = 6
    mz, li, _, prec, charge = _inputs(n)
    with torch.no_grad():
        s = model.embed(mz, li, None)
        z = model.bias_module.init_state(s, mz, li, prec, charge)
    asym = (z - z.transpose(1, 2)).abs().max().item()
    assert asym > 1e-3, f"pair state is symmetric (W_a == W_b?): max |z_ij - z_ji| {asym:.2e}"


def test_t5_smoke_shapes_and_loss():
    """End-to-end forward with the pretraining head produces finite loss and correct shapes."""
    torch.manual_seed(0)
    config = _config(num_hidden_layers=2)
    model = MSDeltaPairformerForPreTraining(config).train()
    b, n = 2, 8
    g = torch.Generator().manual_seed(1)
    mz = torch.sort(torch.rand(b, n, generator=g) * 1800 + 100, dim=1).values
    li = torch.rand(b, n, generator=g)
    am = torch.ones(b, n, dtype=torch.long)
    mask_positions = torch.zeros(b, n, dtype=torch.bool)
    mask_positions[:, 0] = True
    labels = torch.rand(b, n, generator=g)
    out = model(
        mz=mz,
        log_intensity=li,
        attention_mask=am,
        mask_positions=mask_positions,
        labels=labels,
        precursor_mz=torch.tensor([1600.0, 900.0]),
        charge=torch.tensor([2, 1]),
    )
    assert out.logits.shape == (b, n)
    assert torch.isfinite(out.loss), "loss is not finite"
    out.loss.backward()
    assert any(p.grad is not None for p in model.parameters()), "no gradients flowed"


def test_ablation_phases_construct_and_run():
    """Every point on the B1..B5 ladder builds and runs a forward pass."""
    torch.manual_seed(0)
    mz, li, am, prec, charge = _inputs(6)
    ladder = [
        dict(pair_update="static", use_triangle_attention=False, use_writeback=False),  # B1
        dict(pair_update="transition", use_triangle_attention=False, use_writeback=False),  # B2
        dict(pair_update="triangle", use_triangle_attention=False, use_writeback=False),  # B3
        dict(pair_update="triangle", use_triangle_attention=True, use_writeback=False),  # B4
        dict(pair_update="triangle", use_triangle_attention=True, use_writeback=True),  # B5
    ]
    for flags in ladder:
        model = MSDeltaPairformerModel(_config(**flags)).eval()
        out = _encode(model, mz, li, am, prec, charge)
        assert out.shape == (1, 6, 32) and torch.isfinite(out).all(), f"phase failed: {flags}"


def test_evaluate_diagnostic_surface():
    """bias_module.evaluate(grid) returns (grid, heads) for eval/viz.py and alignment.py."""
    torch.manual_seed(0)
    config = _config()
    model = MSDeltaPairformerModel(config).eval()
    grid = torch.linspace(-200.0, 200.0, 401)
    curves = model.bias_module.evaluate(grid)
    assert curves.shape == (grid.numel(), config.num_attention_heads)
    assert torch.isfinite(curves).all()


def test_baseline_accepts_conditioning_keys():
    """The baseline pretraining model ignores the collator's charge/precursor keys."""
    torch.manual_seed(0)
    model = MSDeltaForPreTraining(
        MSDeltaConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2, intermediate_size=64)
    ).eval()
    b, n = 2, 6
    out = model(
        mz=torch.rand(b, n) * 1000 + 100,
        log_intensity=torch.rand(b, n),
        attention_mask=torch.ones(b, n, dtype=torch.long),
        charge=torch.tensor([2, 1]),
        precursor_mz=torch.tensor([800.0, 500.0]),
    )
    assert out.logits.shape == (b, n)


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL {test.__name__}: {error}")
        except Exception as error:  # noqa: BLE001 - surface any error as a test failure
            failures += 1
            print(f"ERROR {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
