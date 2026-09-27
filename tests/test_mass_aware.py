"""A8 mass-aware student training: mass, same-mass negative pool, mass-bucketed batches."""

import numpy as np
import pytest

import msdelta.reranking  # noqa: F401  -- module-level msdelta import, see FT33
from msdelta.reranking import (AlignmentCollator, MassBatchSampler, MassNegativePool,
                               peptide_neutral_mass)

# Opt-in (A8: mass-aware student, not in the recipe). Run with --legacy.
# test_peptide_neutral_mass lives in test_contrastive.py::TestSameMassBatches.
pytestmark = pytest.mark.legacy




def test_mass_pool_negatives_within_ppm_and_never_self_or_il_twin():
    peps = ["PEPTIDE", "PEPTLDE", "EPPTIDE", "DEPTIPE", "PEPTIDEK", "AAAAAAAK"]
    pool = MassNegativePool(peps)
    rng = np.random.default_rng(0)
    neg = pool.negatives("PEPTIDE", rng, k=10, ppm=20.0)
    # permutations of PEPTIDE have the same mass; the I->L twin must be excluded
    assert set(neg) == {"EPPTIDE", "DEPTIPE"}
    base = peptide_neutral_mass("PEPTIDE")
    assert all(abs(peptide_neutral_mass(n) - base) <= base * 20e-6 for n in neg)
    assert len(pool.negatives("PEPTIDE", rng, k=1)) == 1
    assert pool.negatives("AAAAAAAK", rng, k=5) == []


def test_mass_batch_sampler_contiguous_complete_and_reshuffles():
    rng = np.random.default_rng(1)
    masses = rng.uniform(500, 3000, size=1000)
    s = MassBatchSampler(masses, batch_size=64, jitter=0.5, seed=0)
    e1, e2 = list(s), list(s)
    assert len(e1) == len(s) == 16
    flat = sorted(i for b in e1 for i in b)
    assert flat == list(range(1000))                      # every row exactly once
    order = np.sort(masses)
    typical = np.median(np.diff(order)) * 64
    spans = [masses[b].max() - masses[b].min() for b in e1 if len(b) == 64]
    assert np.median(spans) < 3 * typical + 1.0           # neighbours in mass
    assert e1 != e2                                       # reshuffled without set_epoch


def test_collator_mass_negatives():
    peps = ["PEPTIDEK", "EPPTIDEK", "DEPTIPEK", "AAAAAAAK"]
    c = AlignmentCollator(hard_negatives=2, neg_source="mass", neg_pool=MassNegativePool(peps))
    feats = [{"peptide": p, "charge": 2, "mz": [1.0], "log_intensity": [1.0],
              "target": [0.0] * 8} for p in ("PEPTIDEK", "AAAAAAAK")]
    b = c(feats)
    assert b["neg_valid"].shape == (2, 2)
    assert b["neg_valid"][0].tolist() == [True, True]     # two same-mass permutations
    assert b["neg_valid"][1].tolist() == [False, False]   # nothing at that mass
