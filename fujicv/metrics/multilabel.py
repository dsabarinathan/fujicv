"""Multi-label classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as sk

from fujicv.metrics.registry import register_metric


class _BaseMetric:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError


@register_metric("SubsetAccuracy")
class SubsetAccuracy(_BaseMetric):
    """Subset accuracy (exact match ratio)."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_bin = (y_pred >= self.threshold).astype(int)
        return float(sk.accuracy_score(y_true, y_bin))


@register_metric("HammingLoss")
class HammingLoss(_BaseMetric):
    """Hamming loss — fraction of incorrectly predicted labels.

    Args:
        threshold: Threshold for converting probabilities to binary predictions.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_bin = (y_pred >= self.threshold).astype(int)
        return float(sk.hamming_loss(y_true, y_bin))


@register_metric("mAP")
class mAP(_BaseMetric):
    """Mean average precision across all labels."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return float(sk.average_precision_score(y_true, y_pred, average="macro"))
        except ValueError:
            return float("nan")


@register_metric("PerLabelAUROC")
class PerLabelAUROC(_BaseMetric):
    """Macro-averaged AUROC computed per label.

    Returns the mean AUROC across labels that have both positive and negative
    samples; labels with only one class present are skipped.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        n_labels = y_true.shape[1]
        aucs = []
        for i in range(n_labels):
            yt = y_true[:, i]
            yp = y_pred[:, i]
            if len(np.unique(yt)) < 2:
                continue
            try:
                aucs.append(sk.roc_auc_score(yt, yp))
            except ValueError:
                pass
        return float(np.mean(aucs)) if aucs else float("nan")
