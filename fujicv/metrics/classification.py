"""Classification metrics."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn import metrics as sk

from fujicv.metrics.registry import register_metric


class _BaseMetric:
    """Callable base class for metrics."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError


@register_metric("Accuracy")
class Accuracy(_BaseMetric):
    """Top-1 accuracy."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(axis=1)
        return float(sk.accuracy_score(y_true, y_pred))


@register_metric("BalancedAccuracy")
class BalancedAccuracy(_BaseMetric):
    """Balanced accuracy (macro-averaged recall)."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(axis=1)
        return float(sk.balanced_accuracy_score(y_true, y_pred))


@register_metric("Precision")
class Precision(_BaseMetric):
    """Macro-averaged precision.

    Args:
        average: Averaging strategy (default ``'macro'``).
    """

    def __init__(self, average: str = "macro") -> None:
        self.average = average

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(axis=1)
        return float(sk.precision_score(y_true, y_pred, average=self.average, zero_division=0))


@register_metric("Recall")
class Recall(_BaseMetric):
    """Macro-averaged recall.

    Args:
        average: Averaging strategy (default ``'macro'``).
    """

    def __init__(self, average: str = "macro") -> None:
        self.average = average

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(axis=1)
        return float(sk.recall_score(y_true, y_pred, average=self.average, zero_division=0))


@register_metric("F1")
class F1(_BaseMetric):
    """Macro-averaged F1 score.

    Args:
        average: Averaging strategy (default ``'macro'``).
    """

    def __init__(self, average: str = "macro") -> None:
        self.average = average

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(axis=1)
        return float(sk.f1_score(y_true, y_pred, average=self.average, zero_division=0))


@register_metric("TopKAccuracy")
class TopKAccuracy(_BaseMetric):
    """Top-K accuracy.

    Args:
        k: Number of top predictions to consider (default 5).
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim != 2:
            raise ValueError("TopKAccuracy requires probability/logit array of shape (N, C)")
        n = len(y_true)
        top_k_indices = np.argsort(y_pred, axis=1)[:, -self.k :]
        correct = sum(y_true[i] in top_k_indices[i] for i in range(n))
        return float(correct) / n


@register_metric("CohenKappa")
class CohenKappa(_BaseMetric):
    """Cohen's kappa coefficient.

    Args:
        weights: ``None``, ``'linear'``, or ``'quadratic'`` (default ``None``).
    """

    def __init__(self, weights: Optional[str] = None) -> None:
        self.weights = weights

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(axis=1)
        return float(sk.cohen_kappa_score(y_true, y_pred, weights=self.weights))


@register_metric("MCC")
class MCC(_BaseMetric):
    """Matthews Correlation Coefficient."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(axis=1)
        return float(sk.matthews_corrcoef(y_true, y_pred))


@register_metric("AUROC")
class AUROC(_BaseMetric):
    """Area under the ROC curve.

    Args:
        multi_class: Strategy for multi-class — ``'ovr'`` (default) or ``'ovo'``.
        average: Averaging strategy (default ``'macro'``).
    """

    def __init__(self, multi_class: str = "ovr", average: str = "macro") -> None:
        self.multi_class = multi_class
        self.average = average

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        n_classes = int(y_pred.shape[1]) if y_pred.ndim == 2 else 2
        if n_classes == 2 and y_pred.ndim == 2:
            y_pred = y_pred[:, 1]
        try:
            return float(
                sk.roc_auc_score(
                    y_true,
                    y_pred,
                    multi_class=self.multi_class if n_classes > 2 else "raise",
                    average=self.average,
                )
            )
        except ValueError:
            return float("nan")
