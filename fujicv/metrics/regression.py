"""Regression metrics."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn import metrics as sk

from fujicv.metrics.registry import register_metric


class _BaseMetric:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError


@register_metric("MAE")
class MAE(_BaseMetric):
    """Mean absolute error."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(sk.mean_absolute_error(y_true, y_pred))


@register_metric("MSE")
class MSE(_BaseMetric):
    """Mean squared error."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(sk.mean_squared_error(y_true, y_pred))


@register_metric("RMSE")
class RMSE(_BaseMetric):
    """Root mean squared error."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(sk.mean_squared_error(y_true, y_pred)))


@register_metric("R2Score")
class R2Score(_BaseMetric):
    """Coefficient of determination R²."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(sk.r2_score(y_true, y_pred))


@register_metric("MAPE")
class MAPE(_BaseMetric):
    """Mean absolute percentage error.

    Samples where ``|y_true| < eps`` are excluded to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.abs(y_true) > self.eps
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


@register_metric("PearsonCorr")
class PearsonCorr(_BaseMetric):
    """Pearson product-moment correlation coefficient."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        r, _ = stats.pearsonr(y_true.ravel(), y_pred.ravel())
        return float(r)


@register_metric("SpearmanCorr")
class SpearmanCorr(_BaseMetric):
    """Spearman rank-order correlation coefficient."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        r, _ = stats.spearmanr(y_true.ravel(), y_pred.ravel())
        return float(r)
