"""ROC and precision-recall curve plots."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as sk
from sklearn.preprocessing import label_binarize


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    multi_class: str = "ovr",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot ROC curves.

    For binary tasks only one curve is drawn.  For multi-class tasks,
    per-class OvR (or OvO) curves plus a macro-average are shown.

    Args:
        y_true: 1-D integer ground-truth labels.
        y_probs: 2-D probability array of shape ``(N, C)`` or 1-D for binary.
        class_names: Class name strings.
        multi_class: ``'ovr'`` (default) or ``'ovo'``.
        figsize: Figure dimensions.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if y_probs.ndim == 1 or y_probs.shape[1] == 2:
        # Binary
        probs = y_probs if y_probs.ndim == 1 else y_probs[:, 1]
        fpr, tpr, _ = sk.roc_curve(y_true, probs)
        auc = sk.auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    else:
        n_classes = y_probs.shape[1]
        classes = list(range(n_classes))
        y_bin = label_binarize(y_true, classes=classes)

        all_fpr = np.unique(np.concatenate([
            sk.roc_curve(y_bin[:, i], y_probs[:, i])[0] for i in range(n_classes)
        ]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):
            fpr, tpr, _ = sk.roc_curve(y_bin[:, i], y_probs[:, i])
            auc = sk.auc(fpr, tpr)
            name = class_names[i] if class_names else str(i)
            ax.plot(fpr, tpr, alpha=0.5, label=f"{name} (AUC={auc:.3f})")
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= n_classes
        macro_auc = sk.auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, "k--", linewidth=2, label=f"Macro avg (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "r:", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot precision-recall curves.

    Args:
        y_true: 1-D integer ground-truth labels.
        y_probs: 2-D probability array of shape ``(N, C)`` or 1-D for binary.
        class_names: Class name strings.
        figsize: Figure dimensions.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if y_probs.ndim == 1 or y_probs.shape[1] == 2:
        probs = y_probs if y_probs.ndim == 1 else y_probs[:, 1]
        precision, recall, _ = sk.precision_recall_curve(y_true, probs)
        ap = sk.average_precision_score(y_true, probs)
        ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    else:
        n_classes = y_probs.shape[1]
        classes = list(range(n_classes))
        y_bin = label_binarize(y_true, classes=classes)

        for i in range(n_classes):
            precision, recall, _ = sk.precision_recall_curve(y_bin[:, i], y_probs[:, i])
            ap = sk.average_precision_score(y_bin[:, i], y_probs[:, i])
            name = class_names[i] if class_names else str(i)
            ax.plot(recall, precision, alpha=0.6, label=f"{name} (AP={ap:.3f})")

        mAP = sk.average_precision_score(y_bin, y_probs, average="macro")
        ax.set_title(f"Precision-Recall Curve  (mAP={mAP:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    if y_probs.ndim == 1 or y_probs.shape[1] == 2:
        ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
