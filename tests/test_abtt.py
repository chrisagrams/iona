"""All-but-the-top post-processing for the zero-shot eval (Mu & Viswanath 2018)."""

import numpy as np
import pytest
import torch

import msdelta  # noqa: F401  (FT33: collecting a lone test file segfaults without it)
from msdelta.eval_zeroshot_layers import all_but_top, apply_abtt, fit_abtt


def _anisotropic(n=400, d=32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) * np.linspace(10, 0.5, d)   # a few dominant directions
    return (x + 5.0).astype(np.float64)                     # and a large common mean


def test_fit_apply_matches_reference():
    x = _anisotropic()
    for d in (1, 3, 8):
        ref = all_but_top(x, d)
        mean, dirs = fit_abtt(torch.from_numpy(x), 8)
        got = apply_abtt(torch.from_numpy(x).float(), mean, dirs, d).double().numpy()
        # directions are sign-ambiguous, the projection is not
        assert np.allclose(got, ref, atol=1e-3)


def test_removed_directions_are_gone_and_rest_kept():
    x = torch.from_numpy(_anisotropic()).float()
    mean, dirs = fit_abtt(x, 4)
    out = apply_abtt(x, mean, dirs, 4)
    assert torch.allclose(out @ dirs.T, torch.zeros(len(x), 4), atol=1e-2)
    assert torch.allclose(out.mean(0), torch.zeros(x.shape[1]), atol=1e-3)
    centered = apply_abtt(x, mean, dirs, 0)
    assert torch.allclose(centered, x - x.mean(0), atol=1e-4)


def test_fit_on_one_set_applied_to_another():
    a = torch.from_numpy(_anisotropic(seed=0)).float()
    b = torch.from_numpy(_anisotropic(seed=1)).float()
    mean, dirs = fit_abtt(a, 2)
    out = apply_abtt(b, mean, dirs, 2)
    assert out.shape == b.shape
    assert torch.allclose(out, (b - mean) - ((b - mean) @ dirs.T) @ dirs)


def test_reference_contract():
    x = _anisotropic(n=10, d=4)
    assert all_but_top(x, 0) is x
    with pytest.raises(ValueError):
        all_but_top(x, 4)
    with pytest.raises(ValueError):
        fit_abtt(torch.from_numpy(x), 4)
