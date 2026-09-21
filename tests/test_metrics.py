"""Metrics. A metric that crashes loses the whole job; one that is silently inverted is worse."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msdelta.finetune_denoise import denoise_metrics


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


class TestPerSpectrumAUROC:
    """Pooled AUROC and within-spectrum AUROC answer different questions.

    The reranking embedding scored 0.846 pooled and cost 0.109 hit@1, because hit@1
    ranks within a spectrum. These tests pin the case where the same gap would appear
    in denoise, so a headline number measured on the wrong axis cannot go unnoticed.
    """

    def _auroc(self, logits, labels):
        from msdelta.finetune_denoise import per_spectrum_auroc
        return per_spectrum_auroc(np.asarray(logits, dtype=float),
                                  np.asarray(labels, dtype=float))

    def test_perfect_within_each_spectrum_scores_one(self):
        logits = [[-2.0, -1.0, 1.0, 2.0], [-9.0, -8.0, 8.0, 9.0]]
        labels = [[0, 0, 1, 1], [0, 0, 1, 1]]
        assert self._auroc(logits, labels)["auroc_per_spectrum"] == pytest.approx(1.0)

    def test_a_per_spectrum_offset_inflates_POOLED_auroc_but_not_this(self):
        """(regression) The failure this metric exists to expose.

        Spectrum B's noise peaks are ranked BELOW its own signal peaks -- the model has
        it exactly backwards there -- but B's whole logit range sits above A's. Pooled,
        that offset makes B's noise outrank A's signal and props the number up.
        Within-spectrum, B is scored as the failure it is.
        """
        from sklearn.metrics import roc_auc_score
        # noise == label 1. Spectrum A is mostly SIGNAL and its logits sit low;
        # spectrum B is mostly NOISE and its logits sit high. Inside each spectrum the
        # ordering is exactly inverted, so the model is useless where it is used.
        logits = [[0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0, 14.0]]
        labels = [[1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
        pooled = roc_auc_score(np.asarray(labels).reshape(-1),
                               np.asarray(logits, dtype=float).reshape(-1))
        within = self._auroc(logits, labels)["auroc_per_spectrum"]
        # The offset alone -- noisy spectra scored above clean ones -- carries pooled to
        # 0.64 while every within-spectrum AUROC is 0.0.
        assert pooled == pytest.approx(0.64)
        assert within == pytest.approx(0.0)
        assert pooled > within

    def test_single_class_spectra_are_counted_not_silently_dropped(self):
        """If most spectra are unscorable the average describes a biased minority."""
        logits = [[1.0, 2.0, 3.0, 4.0], [-1.0, 1.0, -2.0, 2.0]]
        labels = [[1, 1, 1, 1], [0, 1, 0, 1]]          # first is all noise
        got = self._auroc(logits, labels)
        assert got["spectra_unscorable"] == 1.0
        assert got["spectra_scored"] == 1.0

    def test_padding_is_excluded_within_a_spectrum_too(self):
        logits = [[-1.0, 1.0, 99.0, 99.0]]
        labels = [[0, 1, -100, -100]]
        assert self._auroc(logits, labels)["auroc_per_spectrum"] == pytest.approx(1.0)

    def test_reports_spread_not_just_the_mean(self):
        """A mean of 0.9 over spectra split between 1.0 and 0.8 is not one population."""
        logits = [[-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]]
        labels = [[0, 1], [0, 1], [0, 1]]
        got = self._auroc(logits, labels)
        assert got["auroc_per_spectrum_sd"] > 0
        assert got["auroc_per_spectrum_p10"] <= got["auroc_per_spectrum"]

    def test_refuses_to_guess_when_the_shape_is_not_two_dimensional(self):
        """Already-flattened input would compute over the wrong axis; say so instead."""
        got = self._auroc([1.0, -1.0, 2.0], [1, 0, 1])
        assert np.isnan(got["auroc_per_spectrum"])
