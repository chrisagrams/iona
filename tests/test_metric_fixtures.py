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


def test_filtered_metrics_pass_fail_split():
    """Standard filtered evaluation (msdelta.eval.filtered_retrieval, 2026-09-27).

    4 groups x 3 spectra, well separated; spectrum 1 of group 0 has its precursor one isotope
    high (+1.00336/z, z=2), so the 20 ppm filter excludes it from queries 0 and 2 and excludes
    BOTH positives of query 1. F = {0, 1, 2}, F_all = {1}.
      open: every query perfect -> 1.0 everywhere.
      20ppm: q0, q2 find 1 of 2 positives at rank 1 -> AP@R = 1/2; q1 finds none -> 0.
             F = (0.5 + 0 + 0.5) / 3 = 1/3; Fbar = 1; full = (1 + 9) / 12 = 10/12.
             net_loss = 1/12 (q1 right unfiltered, wrong filtered); rescue = 1 (q1 open hit).
      iso20ppm: the +1 isotope step is allowed -> 1.0 everywhere.
    """
    import numpy as np
    import torch

    from msdelta.eval.eval_grouped_retrieval import _variants
    from msdelta.finetuning.contrastive.contrastive import retrieval_metrics_topk
    torch.manual_seed(0)
    g = np.repeat(np.arange(4), 3)
    emb = torch.eye(8)[:4][g] * 10 + 0.01 * torch.randn(12, 8)
    prec = np.repeat(np.array([500., 600., 700., 800.]), 3)
    prec[1] += 1.00336 / 2
    out = _variants(emb, g, np.ones(12, dtype=bool), torch.device("cpu"), retrieval_metrics_topk,
                    filter_inputs=(prec, np.full(12, 2)))
    approx = lambda k, v: abs(out[k] - v) < 1e-6  # noqa: E731
    assert approx("experimental/open/full/MAP@R", 1.0)
    assert approx("experimental/20ppm/F/MAP@R", 1 / 3)
    assert approx("experimental/20ppm/Fbar/MAP@R", 1.0)
    assert approx("experimental/20ppm/full/MAP@R", 10 / 12)
    assert approx("experimental/20ppm/net_loss", 1 / 12)
    assert approx("experimental/rescue", 1.0)
    assert approx("experimental/iso20ppm/full/MAP@R", 1.0)
    assert out["experimental/open/F/n"] == 3 and out["experimental/open/F_all/n"] == 1
    assert approx("experimental/open/full/MAP@R", out["experimental/MAP@R"])
    # without precursor inputs, no filtered keys (old behaviour)
    plain = _variants(emb, g, np.ones(12, dtype=bool), torch.device("cpu"), retrieval_metrics_topk)
    assert not any("/20ppm/" in k for k in plain)


def _mz(mass, z):
    from msdelta.eval.filtered_retrieval import PROTON
    return (mass + z * PROTON) / z


def test_crossmodal_filtered_report_hand_computed():
    """Cross-modal filtered evaluation (filtered_retrieval.crossmodal_report, K77-A).

    Candidates (peptide+charge, all z=2) on the circle, neutral masses:
      c0 0 deg 1000 Da;  c1 90 deg 1200;  c2 180 deg 1400;  c3 270 deg 1600;
      c4 30 deg 1201.00336 (a decoy exactly one 13C step above c1).
    Spectra (z=2), truth sp_i -> c_i, measured precursor m/z = theoretical of the truth,
    except sp1, recorded one isotope high: mz(1200) + 1.00336/2 = mz(c4) exactly.
      sp0  20 deg: c4 10, c0 20, c1 70, c3 110, c2 160   truth rank 2
      sp1  80 deg: c1 10, c4 50, c0 80, c2 100, c3 170   truth rank 1
      sp2 170 deg: c2 10, c1 80, c3 100, c4 140, c0 170  truth rank 1
      sp3 265 deg: c3 5, c2 85, c0 95, c4 125, c1 175    truth rank 1
    open:     hit@1 3/4, hit@5 1, MRR (1/2 + 1 + 1 + 1)/4 = 7/8 (= cross_modal_metrics).
    20ppm:    sp0 keeps only c0 -> hit (net GAIN); sp1 keeps only c4 (c1 is 0.5017 m/z away,
              tolerance 20e-6 * 601 = 0.012) -> its truth is excluded: miss (net LOSS).
              F = F_all = {sp1} (one correct candidate per query), Fbar = {sp0, sp2, sp3}.
              full hit@1 = hit@5 = MRR = 3/4; F: 0; Fbar: 1; net_loss = net_gain = 1/4.
    iso20ppm: sp1 keeps c1 (k=+1) and c4 (k=0), c1 is closer -> hit; sp0 still only c0.
              full = 1 on every metric; net_loss 0, net_gain 1/4.
    open on F: sp1 is a hit -> rescue = 1. open on Fbar: hit@1 2/3, MRR (1/2+1+1)/3 = 5/6.
    Legacy yHydra windows (neutral mass, no charge check): +-1.1 Da keeps c1 for sp1 ->
    hit@1 1, nothing outside; legacy 20 ppm excludes it -> 3/4, 1/4 outside.
    """
    from msdelta.eval.filtered_retrieval import (crossmodal_filters, crossmodal_flatten,
                                                 crossmodal_ranks, crossmodal_report)
    from msdelta.reranking import cross_modal_metrics

    cands = _circle(0, 90, 180, 270, 30).float().numpy()
    spectra = _circle(20, 80, 170, 265).float().numpy()
    truth = np.arange(4)
    mass = np.array([1000.0, 1200.0, 1400.0, 1600.0, 1200.0 + 1.00336])
    z_c = np.full(5, 2)
    z_q = np.full(4, 2)
    mz = _mz(mass[:4], 2)
    mz[1] += 1.00336 / 2
    rep = crossmodal_report(spectra, cands, truth, mz, z_q, mass, z_c)
    flat = crossmodal_flatten(rep)
    exp = {
        "open/full": (4, 3 / 4, 1.0, 7 / 8), "open/F": (1, 1.0, 1.0, 1.0),
        "open/F_all": (1, 1.0, 1.0, 1.0), "open/Fbar": (3, 2 / 3, 1.0, 5 / 6),
        "20ppm/full": (4, 3 / 4, 3 / 4, 3 / 4), "20ppm/F": (1, 0.0, 0.0, 0.0),
        "20ppm/Fbar": (3, 1.0, 1.0, 1.0), "iso20ppm/full": (4, 1.0, 1.0, 1.0),
        "iso20ppm/F": (1, 1.0, 1.0, 1.0), "iso20ppm/Fbar": (3, 1.0, 1.0, 1.0),
    }
    for cell, (n, h1, h5, mrr) in exp.items():
        assert flat[f"crossmodal/{cell}/n"] == n, cell
        for key, v in (("hit@1", h1), ("hit@5", h5), ("mrr", mrr)):
            assert flat[f"crossmodal/{cell}/{key}"] == pytest.approx(v), (cell, key)
    assert flat["crossmodal/20ppm/net_loss"] == pytest.approx(1 / 4)
    assert flat["crossmodal/20ppm/net_gain"] == pytest.approx(1 / 4)
    assert flat["crossmodal/iso20ppm/net_loss"] == 0.0
    assert flat["crossmodal/iso20ppm/net_gain"] == pytest.approx(1 / 4)
    assert flat["crossmodal/rescue"] == 1.0
    assert "crossmodal/open/net_loss" not in flat
    # unfiltered = the existing cross_modal_metrics numbers
    old = cross_modal_metrics(cands, spectra, truth, np.arange(5))
    assert flat["crossmodal/open/full/hit@1"] == pytest.approx(old["crossmodal/hit@1"])
    assert flat["crossmodal/open/full/mrr"] == pytest.approx(old["crossmodal/mrr"])
    assert flat["crossmodal/open/full/hit@5"] == pytest.approx(old["crossmodal/hit@5"])
    # legacy yHydra windows through the same ranking
    f = crossmodal_filters(mz, z_q, mass, z_c)
    rank, inside, size = crossmodal_ranks(spectra, cands, truth, f["legacy_1.1Da"])
    assert (inside & (rank == 0)).mean() == 1.0 and inside.all()
    assert list(size) == [1, 2, 1, 1]
    rank, inside, _ = crossmodal_ranks(spectra, cands, truth, f["legacy_20ppm"])
    assert (inside & (rank == 0)).mean() == pytest.approx(3 / 4)
    assert list(inside) == [True, False, True, True]
    # the plain filter requires the same charge; candidates without a charge skip that check
    z_c3 = z_c.copy()
    z_c3[2] = 3
    assert not crossmodal_filters(mz, z_q, mass, z_c3)["20ppm"](np.array([2]))[0, 2]
    assert crossmodal_filters(mz, z_q, mass)["20ppm"](np.array([2]))[0, 2]


def test_align_test_filtered_keys_and_skip():
    """eval_align_test.filtered_metrics: flat keys when rows carry a measured precursor
    (all theoretical here, so F is empty and 20 ppm changes nothing), {} when they do not."""
    from datasets import Dataset

    from msdelta.eval.eval_align_test import filtered_metrics
    from msdelta.rescoring.reranking import peptide_neutral_mass

    peps = ["PEPTIDEK", "PEPTIDEK", "ACDEFGHIK", "ACDEFGHIK", "LLLLMMR", "LLLLMMR"]
    keys = [(p, 2) for p in peps]
    cands = list(dict.fromkeys(keys))
    group = np.array([cands.index(k) for k in keys])
    spec = _circle(0, 5, 120, 125, 240, 245).float()
    seq = _circle(2, 122, 242).float()
    prec = [_mz(peptide_neutral_mass(p), 2) for p in peps]
    rows = Dataset.from_dict({"peptide": peps, "charge": [2] * 6, "precursor": prec})
    out = filtered_metrics(rows, keys, cands, group, spec, seq, group)
    assert out["crossmodal/open/full/hit@1"] == 1.0
    assert out["crossmodal/20ppm/full/hit@1"] == 1.0
    assert out["crossmodal/open/F/n"] == 0 and "crossmodal/open/F/hit@1" not in out
    assert out["teacher_spectrum/20ppm/full/MAP@R"] == pytest.approx(1.0)
    no_prec = Dataset.from_dict({"peptide": peps, "charge": [2] * 6})
    assert filtered_metrics(no_prec, keys, cands, group, spec, seq, group) == {}
    zeros = Dataset.from_dict({"peptide": peps, "charge": [2] * 6, "precursor": [0.0] * 6})
    assert filtered_metrics(zeros, keys, cands, group, spec, seq, group) == {}
def _library_fixture():
    """Library search fixture (C25): 3 groups x (2 experimental + 1 consensus), plus one
    experimental query of a 4th group with NO consensus (unscorable). Charge 2 throughout.

      library (consensus)  c0 = 0 deg (500 m/z), c1 = 120 (600), c2 = 240 (700)
      queries  group 0     a = 10  (500 + 1.00336/2: one isotope high), b = 70  (500)
               group 1     d = 125 (600),  e = 200 (600)
               group 2     f = 250 (700),  h = 50  (700)
               group 3     u = 300 (800)   no consensus -> unscorable
    """
    angles = [10, 70, 0, 125, 200, 120, 250, 50, 240, 300]
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
    source = ["e", "e", "c", "e", "e", "c", "e", "e", "c", "e"]
    prec = np.array([500 + 1.00336 / 2, 500, 500, 600, 600, 600, 700, 700, 700, 800])
    experimental = np.array([s == "e" for s in source])
    return _circle(*angles), groups, experimental, ~experimental, prec, np.full(10, 2)


def test_library_search_hand_computed():
    """Experimental queries vs the consensus-only library (msdelta.eval.library_search).
    Only c0, c1, c2 are ranked (experimental spectra are never library entries, consensus
    never queries), so each query's rank is read off its distances to c0/c1/c2:

      query  distances c0 / c1 / c2      open rank   20ppm allows   20ppm rank  iso rank
      a      10 / 110 / 130              1           nothing (+1 iso) inf       1
      b      70 /  50 / 170              2           c0 only        1           1
      d     125 /   5 / 115              1           c1 only        1           1
      e     160 /  80 /  40              2           c1 only        1           1
      f     110 / 130 /  10              1           c2 only        1           1
      h      50 /  70 / 170              3           c2 only        1           1
      u      group 3 has no consensus: unscorable, excluded (6 queries scored)

    open:  Hit@1 = 3/6, Hit@5 = 1, MRR = (1 + 1/2 + 1 + 1/2 + 1 + 1/3)/6 = 13/18.
           Ranks sorted 1,1,1,2,2,3: mean 10/6, median (1+2)/2 = 1.5, max 3;
           p90 (linear, position 0.9*5 = 4.5 between 2 and 3) = 2.5, p99 (4.95) = 2.95;
           frac_rank_le_1 = 1/2, le_5 = le_10 = le_100 = 1; excluded 0.
    F = {a} (20 ppm excludes its correct entry), Fbar = the other 5.
      open  F: Hit@1 1 (so rescue = 1);  Fbar: Hit@1 2/5, MRR (1/2+1+1/2+1+1/3)/5 = 2/3,
            ranks 1,1,2,2,3: mean 9/5, median 2, p90 (position 3.6) = 2.6, max 3.
      20ppm full: Hit@1 = MRR = 5/6; F: Hit@1 = MRR = 0; Fbar: 1.
            Rank stats over the 5 FOUND queries (all rank 1): mean = median = max = 1;
            excluded 1 (a); frac_rank_le_100 = 5/6 (over all 6). F: excluded 1, no rank
            stats (nothing found), frac_rank_le_1 = 0.
            net_loss = 1/6 (a), net_gain = 3/6 (b, e, h).
      iso20ppm: a's +1 isotope step is allowed -> every rank 1: Hit@1 = MRR = 1 on full,
            F and Fbar; net_loss 0, net_gain 1/2.
    """
    from msdelta.eval.library_search import library_report

    emb, groups, experimental, consensus, prec, z = _library_fixture()
    out = library_report(emb, groups, experimental, consensus, prec, z, chunk=4)
    expected = {
        "library/queries": 6, "library/unscorable": 1, "library/library_size": 3,
        "library/Hit@1": 1 / 2, "library/Hit@5": 1.0, "library/MRR": 13 / 18,
        "library/rank_mean": 10 / 6, "library/rank_median": 1.5, "library/rank_p90": 2.5,
        "library/rank_p99": 2.95, "library/rank_max": 3, "library/frac_rank_le_1": 1 / 2,
        "library/frac_rank_le_5": 1.0, "library/frac_rank_le_100": 1.0,
        "library/open/full/excluded": 0, "library/open/full/rank_median": 1.5,
        "library/open/Fbar/rank_mean": 9 / 5, "library/open/Fbar/rank_median": 2,
        "library/open/Fbar/rank_p90": 2.6, "library/open/Fbar/rank_max": 3,
        "library/20ppm/full/excluded": 1, "library/20ppm/full/rank_mean": 1.0,
        "library/20ppm/full/rank_median": 1.0, "library/20ppm/full/rank_max": 1.0,
        "library/20ppm/full/frac_rank_le_100": 5 / 6, "library/20ppm/F/excluded": 1,
        "library/20ppm/F/frac_rank_le_1": 0.0, "library/iso20ppm/full/excluded": 0,
        "library/open/full/n": 6, "library/open/F/n": 1, "library/open/Fbar/n": 5,
        "library/open/full/MRR": 13 / 18, "library/open/F/Hit@1": 1.0,
        "library/open/Fbar/Hit@1": 2 / 5, "library/open/Fbar/MRR": 2 / 3,
        "library/rescue": 1.0,
        "library/20ppm/full/Hit@1": 5 / 6, "library/20ppm/full/MRR": 5 / 6,
        "library/20ppm/full/Hit@5": 5 / 6,
        "library/20ppm/F/Hit@1": 0.0, "library/20ppm/F/MRR": 0.0,
        "library/20ppm/Fbar/Hit@1": 1.0, "library/20ppm/Fbar/MRR": 1.0,
        "library/20ppm/net_loss": 1 / 6, "library/20ppm/net_gain": 1 / 2,
        "library/iso20ppm/full/Hit@1": 1.0, "library/iso20ppm/full/MRR": 1.0,
        "library/iso20ppm/F/Hit@1": 1.0, "library/iso20ppm/Fbar/Hit@1": 1.0,
        "library/iso20ppm/net_loss": 0.0, "library/iso20ppm/net_gain": 1 / 2,
    }
    for key, value in expected.items():
        assert out[key] == pytest.approx(value, abs=1e-6), key
    assert "library/20ppm/F/rank_mean" not in out and "library/MAP@R" not in out
    # without precursors: only the unfiltered numbers
    plain = library_report(emb, groups, experimental, consensus)
    assert plain["library/MRR"] == pytest.approx(13 / 18, abs=1e-6)
    assert not any("/20ppm/" in k or "/open/" in k for k in plain)


def test_library_ties_count_against_the_query():
    """A library entry tied with the correct one ranks ahead of it: identical vectors
    everywhere give rank 2 of 2, Hit@1 = 0, MRR = 1/2."""
    from msdelta.eval.library_search import library_report

    emb = torch.ones(4, 3)
    out = library_report(emb, [0, 0, 1, 1], [True, False, True, False],
                         [False, True, False, True])
    assert out["library/Hit@1"] == 0.0 and out["library/MRR"] == pytest.approx(0.5)


def test_library_mode_leaves_existing_keys_unchanged():
    """_variants with the library mode on adds only `library/...` keys: every `all/` and
    `experimental/` key (filtered ones included) is identical to the run without it."""
    from msdelta.eval.eval_grouped_retrieval import _variants

    emb, groups, experimental, consensus, prec, z = _library_fixture()
    emb = emb + 0.05 * torch.randn(len(emb), 2, generator=torch.Generator().manual_seed(0))
    cpu = torch.device("cpu")
    base = _variants(emb, groups, experimental, cpu, retrieval_metrics_topk,
                     filter_inputs=(prec, z))
    lib = _variants(emb, groups, experimental, cpu, retrieval_metrics_topk,
                    filter_inputs=(prec, z), consensus=consensus)
    assert not any(k.startswith("library/") for k in base)
    assert {k: v for k, v in lib.items() if not k.startswith("library/")} == base
    assert any(k.startswith("all/") for k in base)
    assert any(k.startswith("experimental/20ppm/") for k in base)
    assert "library/iso20ppm/full/Hit@1" in lib
