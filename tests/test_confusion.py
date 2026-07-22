"""Tests for confusion matrix and per-class metrics."""

from __future__ import annotations

import numpy as np
import pytest


def _preds(n=50, classes=4):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, classes, n)
    y_pred = rng.integers(0, classes, n)
    return y_true, y_pred


# ── plot_confusion_matrix ─────────────────────────────────────────────────────

def test_confusion_matrix_returns_fig_ax():
    from fujicv.eval.confusion import plot_confusion_matrix
    y_true, y_pred = _preds()
    fig, ax = plot_confusion_matrix(y_true, y_pred, show=False)
    import matplotlib
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_confusion_matrix_with_class_names():
    from fujicv.eval.confusion import plot_confusion_matrix
    y_true, y_pred = _preds(classes=3)
    fig, ax = plot_confusion_matrix(
        y_true, y_pred, class_names=["cat", "dog", "bird"], show=False
    )
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_confusion_matrix_not_normalized():
    from fujicv.eval.confusion import plot_confusion_matrix
    y_true, y_pred = _preds()
    fig, ax = plot_confusion_matrix(y_true, y_pred, normalize=False, show=False)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_confusion_matrix_binary():
    from fujicv.eval.confusion import plot_confusion_matrix
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    fig, ax = plot_confusion_matrix(y_true, y_pred,
                                     class_names=["neg", "pos"], show=False)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ── per_class_metrics ─────────────────────────────────────────────────────────

def test_per_class_metrics_shape():
    from fujicv.eval.confusion import per_class_metrics
    y_true, y_pred = _preds(classes=4)
    df = per_class_metrics(y_true, y_pred)
    assert list(df.columns) == ["class", "precision", "recall", "f1", "support"]
    assert len(df) == 4


def test_per_class_metrics_with_names():
    from fujicv.eval.confusion import per_class_metrics
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0])
    df = per_class_metrics(y_true, y_pred, class_names=["a", "b", "c"])
    assert list(df["class"]) == ["a", "b", "c"]


def test_per_class_metrics_perfect():
    from fujicv.eval.confusion import per_class_metrics
    y = np.array([0, 1, 2, 0, 1, 2])
    df = per_class_metrics(y, y)
    np.testing.assert_allclose(df["precision"].values, 1.0)
    np.testing.assert_allclose(df["recall"].values,    1.0)
    np.testing.assert_allclose(df["f1"].values,        1.0)


def test_per_class_metrics_range():
    from fujicv.eval.confusion import per_class_metrics
    y_true, y_pred = _preds(classes=5)
    df = per_class_metrics(y_true, y_pred)
    assert (df["precision"] >= 0).all() and (df["precision"] <= 1).all()
    assert (df["recall"]    >= 0).all() and (df["recall"]    <= 1).all()
    assert (df["f1"]        >= 0).all() and (df["f1"]        <= 1).all()
