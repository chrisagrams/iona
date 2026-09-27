"""Golden-output regression: released checkpoints on frozen inputs, against references
the code produced when they were written (tests/golden/reference/, see regenerate.py).

Opt-in (marker `golden`): `pytest tests/golden --golden`, or `pbs/run_e2e.pbs` on a
debug node. Skips, saying why, where /flare or the local HF cache is not available.

A failure here means the code now computes something different for the same weights
and inputs. That is either a bug, or an intended change -- in which case regenerate the
references deliberately (regenerate.py's docstring) and commit them with the change.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.golden import common

pytestmark = pytest.mark.golden

FLOAT16_ULP = 2.0 ** -11          # half-ulp relative rounding of a float16 reference


@pytest.fixture(scope="module")
def reference():
    missing = common.missing_inputs()
    if missing:
        pytest.skip(f"golden inputs not readable here (needs /flare and the HF cache): "
                    f"{missing}")
    manifest = json.loads((common.REFERENCE / "MANIFEST.json").read_text())
    with np.load(common.REFERENCE / "golden.npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    return manifest, arrays


@pytest.fixture(scope="module")
def cpu_fp32(reference):
    return common.compute("cpu", "fp32")


def compare(manifest, ref: dict, new: dict, dtype: str) -> None:
    tol = manifest["tolerances"][dtype]
    stored16 = set(manifest["stored_float16"])
    failures = []
    for key, expected in ref.items():
        if key.endswith("_keys") or key.endswith("_lengths"):
            assert new[key].tolist() == expected.tolist(), key
            continue
        kind = key.split("/", 1)[1]
        atol = tol["embeddings" if kind == "embeddings" else kind]
        expected = expected.astype(np.float64)
        got = np.asarray(new[key], np.float64)
        assert got.shape == expected.shape, (key, got.shape, expected.shape)
        bound = atol + (np.abs(expected) * FLOAT16_ULP if key in stored16 else 0.0)
        excess = np.abs(got - expected) - bound
        if (excess > 0).any():
            failures.append(f"{key}: max |diff| {np.abs(got - expected).max():.3g} "
                            f"(atol {atol:g}), {int((excess > 0).sum())} elements over")
    assert not failures, "golden outputs moved:\n  " + "\n  ".join(failures)


def test_weights_are_the_frozen_ones(reference):
    """If a checkpoint was replaced, every other failure here is meaningless."""
    manifest, _ = reference
    assert common.weight_hashes() == manifest["sha256"], (
        "a frozen checkpoint changed on disk; the references no longer describe it")


def test_inputs_are_the_frozen_ones(reference):
    manifest, _ = reference
    _, peptides, charges = common.inputs()
    assert [[p, c] for p, c in zip(peptides, charges)] == manifest["inputs"]["peptides"]


def test_cpu_fp32_matches_reference(reference, cpu_fp32):
    """Pooled mean+max embeddings, pretraining-head log-probs, grouped-retrieval metrics
    and peptide embeddings, recomputed on CPU in fp32."""
    manifest, arrays = reference
    compare(manifest, arrays, cpu_fp32, "fp32")


def test_xpu_bf16_matches_reference(reference):
    """bf16 autocast on XPU against the fp32 reference, at bf16 tolerance -- the path the
    fused-kernel dtype bug took (FT-dtype)."""
    import torch
    if not torch.xpu.is_available():
        pytest.skip("no XPU; run via pbs/run_e2e.pbs")
    manifest, arrays = reference
    compare(manifest, arrays, common.compute("xpu", "bf16"), "bf16")
