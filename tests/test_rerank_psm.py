"""PSM reranking eval: target-decoy q-values and modification notation."""

import numpy as np
import pytest

import msdelta.rerank_psm_fdr  # noqa: F401  -- module-level msdelta import, see FT33
from msdelta.rerank_psm_embed import to_notation
from msdelta.rerank_psm_fdr import accepted, qvalues, top_per_spectrum


def test_qvalues_hand_computed():
    # best first: T T D T D ;  FDR=(D+1)/T  -> 1/1, 1/2, 2/2, 2/3, 3/3
    scores = np.array([5, 4, 3, 2, 1], float)
    decoy = np.array([0, 0, 1, 0, 1], bool)
    np.testing.assert_allclose(qvalues(scores, decoy), [0.5, 0.5, 2 / 3, 2 / 3, 1.0])


def test_qvalues_order_invariant():
    rng = np.random.default_rng(0)
    s = rng.normal(size=200); d = rng.random(200) < 0.3
    perm = rng.permutation(200)
    np.testing.assert_allclose(qvalues(s, d)[perm], qvalues(s[perm], d[perm]))


def test_all_targets_perfect_separation_counts_all():
    s = np.r_[np.linspace(10, 5, 300), np.linspace(1, 0, 3)]
    d = np.r_[np.zeros(300, bool), np.ones(3, bool)]
    r = accepted(s, d, np.array([f"P{i}" for i in range(303)]))
    assert r["psms_1pct"] == 300


def test_peptide_level_collapses_duplicates():
    """300 target PSMs over 150 distinct peptides (each seen twice) -> 150 peptides.
    (With the +1 correction a set needs ~100 targets before 1% is reachable at all.)"""
    s = np.r_[np.linspace(10, 5, 300), [0.0]]
    d = np.r_[np.zeros(300, bool), [True]]
    peps = np.array([f"P{i // 2}" for i in range(300)] + ["DECOY"])
    r = accepted(s, d, peps)
    assert r["psms_1pct"] == 300 and r["peptides_1pct"] == 150


def test_top_per_spectrum():
    codes = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.9, 0.5, 0.3, 0.2])
    assert sorted(top_per_spectrum(codes, score).tolist()) == [1, 3]


@pytest.mark.parametrize("seq,mods,expected", [
    ("PEPTIDEK", [], "PEPTIDEK"),
    ("ACDK", [{"position": 2, "mass": 57.02146}], "AC[57.0215]DK"),
    ("MK", [{"position": 1, "mass": 15.9949}], "M[15.9949]K"),
    ("AK", [{"position": 0, "mass": 42.0106}], "[42.0106]AK"),
    ("MCK", [{"position": 0, "mass": 42.0106}, {"position": 1, "mass": 15.9949},
             {"position": 2, "mass": 57.02146}], "[42.0106]M[15.9949]C[57.0215]K"),
])
def test_to_notation(seq, mods, expected):
    assert to_notation(seq, mods) == expected


def test_notation_parses_back():
    from msdelta.reranking import RESIDUE_TO_ID, parse_peptide
    ids, masses = parse_peptide(to_notation("MCK", [{"position": 0, "mass": 42.0106},
                                                    {"position": 2, "mass": 57.02146}]))
    assert ids[0] == RESIDUE_TO_ID["n"] and masses[0] == pytest.approx(42.0106)
    assert masses[2] == pytest.approx(57.0215, abs=1e-4)


def test_da_floor_matching():
    """Ion-trap tolerance. 1000.2 vs 1000.0 is 200 ppm (inside 250 ppm); 100.04 vs 100.0
    is 400 ppm (outside 250 ppm) but only 0.04 Da, inside a 0.05 Da floor."""
    from msdelta.rescoring import _match
    obs = np.array([100.04, 1000.2])            # sorted, as _match requires
    theo = np.array([1000.0, 100.0])
    assert _match(obs, theo, 20.0).tolist() == [False, False]          # old default
    assert _match(obs, theo, 250.0).tolist() == [True, False]
    assert _match(obs, theo, 250.0, da_floor=0.05).tolist() == [True, True]
