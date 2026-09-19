"""Metrics. A metric that crashes loses the whole job; one that is silently inverted is worse."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msdelta.finetune_denoise import denoise_metrics
from msdelta.reranking import cross_modal_metrics


def _eval_pred(logits, labels):
    class P:
        def __init__(self, p, l):
            self.predictions, self.label_ids = p, l
    return P(np.asarray(logits, dtype=np.float32), np.asarray(labels))


class TestDenoiseMetrics:
    def test_perfect_prediction(self):
        """Noise is the POSITIVE class. If that ever flips, AUROC flips with it."""
        out = denoise_metrics(_eval_pred([5.0, -5.0, 5.0, -5.0], [1, 0, 1, 0]))
        assert out["auroc"] == pytest.approx(1.0)
        assert out["accuracy"] == pytest.approx(1.0)

    def test_inverted_prediction(self):
        out = denoise_metrics(_eval_pred([5.0, -5.0, 5.0, -5.0], [0, 1, 0, 1]))
        assert out["auroc"] == pytest.approx(0.0)

    def test_ignore_index_is_dropped(self):
        """-100 marks padding. Counting it would score the model on peaks that do not exist."""
        out = denoise_metrics(_eval_pred([5.0, -5.0, 0.0, 0.0], [1, 0, -100, -100]))
        assert out["n_peaks"] == 2
        assert out["auroc"] == pytest.approx(1.0)

    def test_survives_stray_labels(self):
        """A multiclass ValueError here killed a four-hour job.

        Twelve-rank bf16 evaluation produced 32 labels that were neither 0, 1 nor -100.
        They must be counted and discarded, never raised on.
        """
        out = denoise_metrics(_eval_pred([5.0, -5.0, 1.0, 1.0], [1, 0, 4.09, 5.22]))
        assert out["label_extra"] == 2
        assert out["n_peaks"] == 2
        assert np.isfinite(out["auroc"])

    def test_single_class_does_not_raise(self):
        """AUROC is undefined on one class; a tiny eval batch can contain only one."""
        out = denoise_metrics(_eval_pred([1.0, 2.0], [1, 1]))
        assert np.isfinite(out["accuracy"])

    def test_all_padding_does_not_raise(self):
        out = denoise_metrics(_eval_pred([1.0, 2.0], [-100, -100]))
        assert out["n_peaks"] == 0

    def test_noise_fraction_is_reported(self):
        out = denoise_metrics(_eval_pred([1.0, 1.0, 1.0, 1.0], [1, 1, 1, 0]))
        assert out["noise_fraction"] == pytest.approx(0.75)


class TestCrossModalMetrics:
    def test_aligned_embeddings_rank_perfectly(self):
        candidates = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
        truth = np.array([0, 0, 1, 1, 2, 3])
        out = cross_modal_metrics(candidates, candidates[truth], truth, np.arange(4))
        assert out["crossmodal/hit@1"] == pytest.approx(1.0)
        assert out["crossmodal/mrr"] == pytest.approx(1.0)

    def test_candidates_are_deduplicated(self):
        candidates = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
        truth = np.array([0, 0, 1, 1, 2, 3])
        out = cross_modal_metrics(candidates, candidates[truth], truth, np.arange(4))
        assert out["crossmodal/n_candidates"] == 4
        assert out["crossmodal/n_spectra"] == 6

    def test_random_embeddings_score_near_chance(self):
        """Guards against a metric that looks good because it is measuring nothing."""
        torch.manual_seed(0)
        n = 50
        candidates = torch.nn.functional.normalize(torch.randn(n, 32), dim=-1)
        spectra = torch.nn.functional.normalize(torch.randn(n, 32), dim=-1)
        out = cross_modal_metrics(candidates, spectra, np.arange(n), np.arange(n))
        assert out["crossmodal/hit@1"] < 10.0 / n

    def test_hit_at_5_is_at_least_hit_at_1(self):
        torch.manual_seed(0)
        candidates = torch.nn.functional.normalize(torch.randn(20, 16), dim=-1)
        spectra = torch.nn.functional.normalize(torch.randn(20, 16), dim=-1)
        out = cross_modal_metrics(candidates, spectra, np.arange(20), np.arange(20))
        assert out["crossmodal/hit@5"] >= out["crossmodal/hit@1"]
