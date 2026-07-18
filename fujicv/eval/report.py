"""Classification report and confusion matrix utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics as sk


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: Optional[str] = "true",
    figsize: Tuple[int, int] = (10, 8),
) -> Tuple[str, plt.Figure]:
    """Generate a text classification report and confusion matrix heatmap.

    Args:
        y_true: 1-D array of ground-truth class indices.
        y_pred: 1-D array of predicted class indices (or 2-D logit/prob array
            — argmax is applied automatically).
        class_names: List of class name strings.  ``None`` uses integer labels.
        normalize: Normalisation for the confusion matrix — ``'true'``
            (default), ``'pred'``, ``'all'``, or ``None``.
        figsize: Figure size in inches (default ``(10, 8)``).

    Returns:
        Tuple of ``(text_report, figure)`` where *figure* contains the heatmap.
    """
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(axis=1)

    labels = class_names if class_names is not None else None
    text = sk.classification_report(y_true, y_pred, target_names=labels, zero_division=0)

    cm = sk.confusion_matrix(y_true, y_pred, normalize=normalize)

    fig, ax = plt.subplots(figsize=figsize)
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names or "auto",
        yticklabels=class_names or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix" + (" (normalised)" if normalize else ""))
    fig.tight_layout()
    return text, fig
