"""Confusion matrix visualization and per-class metrics."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np


def plot_confusion_matrix(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple = (8, 7),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    text_size: int = 10,
):
    """Plot a confusion matrix with per-cell counts or normalised fractions.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: List of class label strings.  Auto-generated (0, 1, …)
            if not provided.
        normalize: Show row-normalised fractions instead of raw counts
            (default True).
        title: Plot title.
        cmap: Matplotlib colormap name (default ``'Blues'``).
        figsize: Figure size in inches.
        save_path: If given, save the figure to this path.
        show: Display interactively (default True).
        text_size: Font size for cell annotations.

    Returns:
        ``(fig, ax)`` matplotlib objects.

    Example::

        from fujicv.eval.confusion import plot_confusion_matrix
        fig, ax = plot_confusion_matrix(y_true, y_pred,
                                         class_names=["cat", "dog", "bird"])
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError as e:
        raise ImportError("matplotlib and scikit-learn are required") from e

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n      = int(max(y_true.max(), y_pred.max())) + 1

    if class_names is None:
        class_names = [str(i) for i in range(n)]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm.astype(float) / row_sums
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap, vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    thresh = cm_display.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_display[i, j]
            text = f"{val:{fmt}}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=text_size,
                    color="white" if val > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def per_class_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Compute per-class precision, recall, F1, and support.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Optional list of class label strings.

    Returns:
        ``pandas.DataFrame`` with columns
        ``['class', 'precision', 'recall', 'f1', 'support']``.

    Example::

        from fujicv.eval.confusion import per_class_metrics
        df = per_class_metrics(y_true, y_pred, class_names=["cat", "dog"])
        print(df)
    """
    try:
        import pandas as pd
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError as e:
        raise ImportError("pandas and scikit-learn are required") from e

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n      = int(max(y_true.max(), y_pred.max())) + 1

    if class_names is None:
        class_names = [str(i) for i in range(n)]

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )
    return pd.DataFrame({
        "class":     class_names,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "support":   sup.astype(int),
    })
