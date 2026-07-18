"""Training curve visualisations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from fujicv.engine.trainer import History


def plot_loss_curves(history: "History") -> plt.Figure:
    """Plot training and validation loss curves.

    Args:
        history: A :class:`~fujicv.engine.trainer.History` object.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    train_loss = history.metrics.get("train_loss", [])
    val_loss = history.metrics.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    if train_loss:
        ax.plot(epochs, train_loss, label="Train loss", marker="o", markersize=3)
    if val_loss:
        ax.plot(epochs, val_loss, label="Val loss", marker="s", markersize=3, linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metric_curves(history: "History", metric_name: str) -> plt.Figure:
    """Plot training and validation curves for a specific metric.

    Args:
        history: A :class:`~fujicv.engine.trainer.History` object.
        metric_name: Base metric name without ``train_`` / ``val_`` prefix.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    train_key = f"train_{metric_name}"
    val_key = f"val_{metric_name}"

    train_vals = history.metrics.get(train_key, [])
    val_vals = history.metrics.get(val_key, [])
    epochs = range(1, max(len(train_vals), len(val_vals)) + 1)

    if train_vals:
        ax.plot(list(epochs)[: len(train_vals)], train_vals,
                label=f"Train {metric_name}", marker="o", markersize=3)
    if val_vals:
        ax.plot(list(epochs)[: len(val_vals)], val_vals,
                label=f"Val {metric_name}", marker="s", markersize=3, linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
