"""Evaluation metrics on tiny hand-worked fixtures: every expected number is derived below.

Retrieval fixtures place unit vectors on a circle at the given angles (degrees). Cosine
similarity is monotone in the angular distance min(|a-b|, 360-|a-b|), so each query's
ranking is read off those distances; the angles are chosen so no query sees a tie.

Definitions (msdelta.finetuning.contrastive.contrastive.retrieval_metrics_exact):
  R           relevant items for a query = other spectra of its group; R=0 is not scored.
  Hit@1       relevant item at rank 1.
  R-Precision hits within the top R, / R.
  MAP@R       sum over hits at rank i <= R of precision@i, / R.
  MAP@100     sum over hits at rank i <= min(100, n-1) of precision@i, / R.
  R@5         hits within the top 5, / R.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from msdelta.contrastive import retrieval_metrics_exact, retrieval_metrics_topk

KEYS = ("Hit@1", "Precision@1", "R-Precision", "MAP@R", "R@5", "MAP@100")


def _circle(*degrees):
    rad = torch.tensor(degrees, dtype=torch.float64) * math.pi / 180
    return torch.stack([rad.cos(), rad.sin()], dim=1).float()


def _both(emb, groups):
    """Score with both implementations; topk with chunk=2 to exercise the chunk loop."""
    exact = retrieval_metrics_exact(emb, groups)
    topk = retrieval_metrics_topk(emb, np.asarray(groups), chunk=2)
    return exact, topk


def _check(emb, groups, expected):
    for got in _both(emb, groups):
        for key, value in expected.items():
            assert got[key] == pytest.approx(value, abs=1e-6), key
        assert got["Precision@1"] == got["Hit@1"]


def test_perfect_ranking():
    """Groups {0,10} and {90,100}: every query's only relevant item (R=1) is at rank 1.
    Every metric is 1."""
    _check(_circle(0, 10, 90, 100), [0, 0, 1, 1],
           {"Hit@1": 1.0, "R-Precision": 1.0, "MAP@R": 1.0, "MAP@100": 1.0, "R@5": 1.0})


def test_worst_ranking():
    """Group 0 = {0, 180}, group 1 = {60, 240}. Each point's groupmate is diametrically
    opposite, so for every query (R=1) it ranks LAST, at rank 3 of 3:
      0:   60 (60), 240 (120), 180 (180)       -- the other three are symmetric.
    Hit@1 = R-Precision = MAP@R = 0; MAP@100 = precision@3 / R = 1/3; R@5 = 1 (rank 3 <= 5)."""
    _check(_circle(0, 180, 60, 240), [0, 0, 1, 1],
           {"Hit@1": 0.0, "R-Precision": 0.0, "MAP@R": 0.0, "MAP@100": 1 / 3, "R@5": 1.0})


def test_mixed_with_r1_r2_and_a_singleton():
    """a=0, b=15, c=70 (group 0, R=2); d=40, e=105 (group 1, R=1); f=200 (singleton).

      query  distances, nearest first                    hits at   Hit1 RP   MAP@R  MAP@100
      a      b15  d40  c70  e105 f160                    1, 3      1    1/2  1/2    (1+2/3)/2 = 5/6
      b      a15  d25  c55  e90  f175                    1, 3      1    1/2  1/2    5/6
      c      d30  e35  b55  a70  f130                    3, 4      0    0    0      (1/3+2/4)/2 = 5/12
      d      b25  c30  a40  e65  f160                    4         0    0    0      1/4
      e      c35  d65  b90  f95  a105                    2         0    0    0      1/2
      f      singleton: R=0, not scored (5 scorable queries)

    Hit@1 = 2/5; R-Precision = MAP@R = (1/2+1/2)/5 = 1/5;
    MAP@100 = (10/12 + 10/12 + 5/12 + 3/12 + 6/12) / 5 = 34/60; every hit is in the top 5,
    so R@5 = 1."""
    _check(_circle(0, 15, 70, 40, 105, 200), [0, 0, 0, 1, 1, 2],
           {"Hit@1": 2 / 5, "R-Precision": 1 / 5, "MAP@R": 1 / 5, "MAP@100": 34 / 60,
            "R@5": 1.0})
    assert retrieval_metrics_topk(_circle(0, 15, 70, 40, 105, 200),
                                  np.array([0, 0, 0, 1, 1, 2]))["queries"] == 5.0


def test_map_at_r_differs_from_r_precision():
    """MAP@R rewards WHERE in the top R the hits sit; R-Precision only counts them.
    p0=0, p1=30, p2=75 (group 0, R=2); p3=10 (singleton, not scored).

      p0: p3 10,  p1 30,  p2 75       hits 2, 3   RP 1/2  MAP@R (1/2)/2 = 1/4  AP (1/2+2/3)/2 = 7/12
      p1: p3 20,  p0 30,  p2 45       hits 2, 3   RP 1/2  MAP@R 1/4            AP 7/12
      p2: p1 45,  p3 65,  p0 75       hits 1, 3   RP 1/2  MAP@R (1/1)/2 = 1/2  AP (1+2/3)/2 = 5/6

    Hit@1 = 1/3, R-Precision = 1/2, MAP@R = (1/4+1/4+1/2)/3 = 1/3,
    MAP@100 = (7/12+7/12+10/12)/3 = 2/3, R@5 = 1."""
    _check(_circle(0, 30, 75, 10), [0, 0, 0, 1],
           {"Hit@1": 1 / 3, "R-Precision": 1 / 2, "MAP@R": 1 / 3, "MAP@100": 2 / 3,
            "R@5": 1.0})


def test_all_singletons_score_nothing():
    """No query has a relevant item: nothing to score, and both say so with {}."""
    emb, groups = _circle(0, 90, 180), [0, 1, 2]
    assert retrieval_metrics_exact(emb, groups) == {}
    assert retrieval_metrics_topk(emb, np.asarray(groups)) == {}


@pytest.mark.parametrize("seed", [0, 1])
def test_topk_agrees_with_exact_on_a_random_case(seed):
    """Continuous random embeddings (no ties) with group sizes 1-6: identical numbers.
    (tests/test_grouped_retrieval.py sweeps more seeds and chunk sizes.)"""
    rng = np.random.default_rng(seed)
    groups = np.concatenate([[g] * rng.integers(1, 7) for g in range(30)])
    emb = torch.tensor(rng.normal(size=(len(groups), 8)), dtype=torch.float32)
    exact = retrieval_metrics_exact(emb, groups)
    topk = retrieval_metrics_topk(emb, groups, k=100, chunk=7)
    for key in KEYS:
        assert topk[key] == pytest.approx(exact[key], abs=1e-6), key


def test_grouped_eval_all_and_experimental_only_variants():
    """eval_grouped_retrieval scores one embedding pass twice: `all` (consensus + three
    replicates per analyte, R=3) and `experimental` (replicates only, R=2).

    Analyte 0: replicates at 0, 10, 25, consensus c0 at 210.
    Analyte 1: replicates at 90, 100, 115, consensus c1 at 300.

    experimental: each replicate's two groupmates are its two nearest (e.g. 25: 15, 25,
    then 65 to the other analyte), so every metric is 1.

    all (n=8, MAP@100 cutoff = 7):
      every replicate query: groupmates at ranks 1, 2, own consensus LAST (rank 7), e.g.
        0:   10, 25, c1 60, 90, 100, 115, c0 150
        115: 15, 25, 90, c0 95, 105, 115, c1 175
      -> Hit1 1, RP 2/3, MAP@R (1+1)/3 = 2/3, AP (1+1+3/7)/3 = 17/21, R@5 2/3
      each consensus query: its replicates at ranks 5, 6, 7, e.g.
        c0 (210): c1 90, 115 (95), 100 (110), 90 (120), 0 (150), 10 (160), 25 (175)
      -> Hit1 0, RP 0, MAP@R 0, AP (1/5+2/6+3/7)/3 = 101/315, R@5 1/3
    Over 6 replicate + 2 consensus queries:
      Hit@1 = 6/8, R-Precision = MAP@R = 6(2/3)/8 = 1/2,
      MAP@100 = (6(17/21) + 2(101/315))/8 = 1732/2520, R@5 = (6(2/3) + 2(1/3))/8 = 7/12.
    """
    from msdelta.eval.eval_grouped_retrieval import _variants

    emb = _circle(0, 10, 25, 210, 90, 100, 115, 300)
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    experimental = np.array([True, True, True, False] * 2)
    out = _variants(emb, groups, experimental, torch.device("cpu"), retrieval_metrics_topk)
    for key in ("Hit@1", "R-Precision", "MAP@R", "MAP@100", "R@5"):
        assert out[f"experimental/{key}"] == pytest.approx(1.0, abs=1e-6), key
    assert out["experimental/queries"] == 6.0 and out["all/queries"] == 8.0
    expected = {"Hit@1": 6 / 8, "R-Precision": 1 / 2, "MAP@R": 1 / 2,
                "MAP@100": 1732 / 2520, "R@5": 7 / 12}
    for key, value in expected.items():
        assert out[f"all/{key}"] == pytest.approx(value, abs=1e-6), key


def test_denoise_metrics_hand_computed():
    """Two spectra of three peaks plus one padded (-100) slot each; noise is class 1.

      logits  [ 2.0, -1.0,  0.5 | -3.0,  1.0, -0.5 ]
      labels  [ 1,    0,    0   |  0,    1,    1   ]
      predict [ 1,    0,    1   |  0,    1,    0   ]   (logit >= 0)

    TP 2 (2.0, 1.0), FP 1 (0.5), FN 1 (-0.5), TN 2: precision = recall = F1 = 2/3,
    accuracy 4/6, balanced accuracy (2/3 + 2/3)/2 = 2/3.
    Pooled AUROC over the 3x3 (noise, signal) pairs: 2.0 and 1.0 beat every signal logit
    (-1.0, 0.5, -3.0): 6; -0.5 beats -1.0 and -3.0 but not 0.5: 2. AUROC = 8/9.
    Within each spectrum the noise peaks outrank the signal peaks (spectrum 1: 2.0 > 0.5,
    -1.0; spectrum 2: 1.0, -0.5 > -3.0), so per-spectrum AUROC = 1 -- the pooled number
    is lower because 0.5 (spectrum 1) outranks -0.5 (spectrum 2), a cross-spectrum pair."""
    from msdelta.finetune_denoise import denoise_metrics

    class Prediction:
        predictions = np.array([[2.0, -1.0, 0.5, 9.0], [-3.0, 1.0, -0.5, 9.0]], np.float32)
        label_ids = np.array([[1, 0, 0, -100], [0, 1, 1, -100]])

    m = denoise_metrics(Prediction())
    assert m["n_peaks"] == 6 and m["label_dropped"] == 0
    for key in ("precision", "recall", "f1", "balanced_accuracy"):
        assert m[key] == pytest.approx(2 / 3), key
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["auroc"] == pytest.approx(8 / 9)
    assert m["noise_fraction"] == pytest.approx(0.5)
    assert m["auroc_per_spectrum"] == pytest.approx(1.0)
    assert m["spectra_scored"] == 2 and m["spectra_unscorable"] == 0


def test_cross_modal_peptide_to_spectrum_hand_computed():
    """Alignment eval: rank peptide candidates for each spectrum.
    Candidates s0=0, s1=40, s2=100 (groups 0, 1, 2); spectra sp0=5, sp1=15, sp2=170.

      sp0 (group 0): s0 5,  s1 35,  s2 95     truth at rank 1
      sp1 (group 1): s0 15, s1 25,  s2 85     truth at rank 2
      sp2 (group 2): s2 70, s1 130, s0 170    truth at rank 1

    hit@1 = 2/3, hit@5 = 1 (only 3 candidates), MRR = (1 + 1/2 + 1)/3 = 5/6.
    Equal counts, so paired cosine = mean(cos 5, cos 25, cos 70)."""
    from msdelta.reranking import cross_modal_metrics

    m = cross_modal_metrics(_circle(0, 40, 100), _circle(5, 15, 170), [0, 1, 2])
    assert m["crossmodal/hit@1"] == pytest.approx(2 / 3)
    assert m["crossmodal/hit@5"] == pytest.approx(1.0)
    assert m["crossmodal/mrr"] == pytest.approx(5 / 6)
    assert m["crossmodal/n_spectra"] == 3 and m["crossmodal/n_candidates"] == 3
    cos = lambda d: math.cos(math.radians(d))
    assert m["crossmodal/paired_cosine"] == pytest.approx((cos(5) + cos(25) + cos(70)) / 3,
                                                          abs=1e-6)
