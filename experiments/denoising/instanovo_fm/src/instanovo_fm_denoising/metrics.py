"""Peak-level denoising metrics matching ``msdelta.denoising.denoising_metrics``."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def denoising_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute peak-level metrics with noise as the positive class."""
    logits = np.asarray(logits).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    valid = labels != -100
    logits = logits[valid]
    labels = labels[valid].astype(np.int64)
    predicted = logits >= 0
    pr_precision, pr_recall, _ = precision_recall_curve(labels, logits)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(labels, logits)),
        "auprc": float(auc(pr_recall, pr_precision)),
        "noise_prevalence": float(labels.mean()),
    }


def full_spectrum_metrics(
    raw_noise: Sequence[np.ndarray],
    scored: dict[int, tuple[np.ndarray, np.ndarray]],
    max_peaks: int,
) -> dict[str, float]:
    """Score every original peak, treating peaks preprocessing removed as noise.

    ``raw_noise[i]`` holds the original labels of raw validation spectrum
    ``i``; ``scored[i]`` holds ``(kept_index, logits)`` for spectra the head
    saw. Spectra are included when ``0 < len <= max_peaks``, matching the
    population MSDelta's denoising probe evaluates. A removed peak, or every
    peak of a spectrum preprocessing skipped, gets a score above every head logit
    and above the zero threshold, i.e. a confident noise prediction.
    """
    head_logits = [logits for _, logits in scored.values() if len(logits)]
    removed_score = max(max((float(x.max()) for x in head_logits), default=0.0), 0.0) + 1.0

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    num_spectra = num_skipped = num_head_peaks = 0
    for index, noise in enumerate(raw_noise):
        noise = np.asarray(noise)
        if not 0 < len(noise) <= max_peaks:
            continue
        num_spectra += 1
        scores = np.full(len(noise), removed_score, dtype=np.float64)
        if index in scored:
            kept_index, logits = scored[index]
            scores[kept_index] = logits
            num_head_peaks += len(kept_index)
        else:
            num_skipped += 1
        all_scores.append(scores)
        all_labels.append(noise.astype(np.float32))

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return {
        **denoising_metrics(scores, labels),
        "num_spectra": float(num_spectra),
        "num_spectra_skipped_by_preprocessing": float(num_skipped),
        "head_peak_fraction": num_head_peaks / len(labels),
    }
