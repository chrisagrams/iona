"""Peak-level denoising metrics matching ``msdelta.denoising.denoising_metrics``."""

from __future__ import annotations

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
