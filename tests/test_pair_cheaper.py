"""K151-P / K152-P: one-direction triangle multiplication and the write-back forms.

Defaults must keep the original model (both triangle multiplications, outer-product write-back);
the factored outer product must equal the materialised one (same parameters, float rounding only).
"""
from __future__ import annotations

import pytest
import torch

from msdelta.models.pairformer import OuterProductMean
from tests.test_pairformer import _batch, _config, _model


def _randomize_writeback(model):
    for layer in model.bias_module.layers:
        if hasattr(layer, "opm"):
            torch.nn.init.normal_(layer.opm.out.weight, std=0.1)
            torch.nn.init.normal_(layer.opm.out.bias, std=0.1)


def test_defaults():
    cfg = _config()
    assert (cfg.pair_tri_mul, cfg.pair_writeback, cfg.pair_writeback_impl) == \
        ("both", "outer", "factored")
    layer = _model(cfg).bias_module.layers[0]
    assert hasattr(layer, "tri_out") and hasattr(layer, "tri_in")


@pytest.mark.parametrize("bad", [dict(pair_tri_mul="none"), dict(pair_writeback="sum"),
                                 dict(pair_writeback_impl="fast")])
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        _config(**bad)


def test_factored_equals_materialized():
    torch.manual_seed(0)
    ref = OuterProductMean(_config(pair_writeback_impl="materialize")).double()
    fac = OuterProductMean(_config(pair_writeback_impl="factored")).double()
    torch.nn.init.normal_(ref.out.weight)
    torch.nn.init.normal_(ref.out.bias)
    fac.load_state_dict(ref.state_dict())
    s = torch.randn(3, 7, 32, dtype=torch.float64)
    mask = torch.rand(3, 7) > 0.3
    torch.testing.assert_close(fac(s, mask), ref(s, mask))


def test_factored_model_matches_materialized_model_and_loads_its_weights():
    ref = _model(_config(pair_writeback_impl="materialize"))
    _randomize_writeback(ref)
    fac = _model(_config(pair_writeback_impl="factored"))
    fac.load_state_dict(ref.state_dict())  # same parameters: checkpoints are interchangeable
    batch = _batch()
    with torch.no_grad():
        torch.testing.assert_close(fac(**batch).last_hidden_state, ref(**batch).last_hidden_state,
                                   rtol=1e-4, atol=1e-5)


def test_pointwise_shapes_and_gradients():
    model = _model(_config(pair_writeback="pointwise")).train()
    opm = model.bias_module.layers[0].opm
    assert opm.out.in_features == 4  # pair_opm_channels, not its square
    _randomize_writeback(model)
    model(**_batch()).last_hidden_state.sum().backward()
    assert opm.left.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("direction,kept,dropped", [("outgoing", "tri_out", "tri_in"),
                                                    ("incoming", "tri_in", "tri_out")])
def test_one_triangle_multiplication(direction, kept, dropped):
    model = _model(_config(pair_tri_mul=direction))
    for layer in model.bias_module.layers:
        assert hasattr(layer, kept) and not hasattr(layer, dropped)
    both = _model(_config())
    assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in both.parameters())
    with torch.no_grad():
        out = model(**_batch()).last_hidden_state
    assert torch.isfinite(out).all()
