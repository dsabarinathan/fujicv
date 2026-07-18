"""t-SNE visualisation of feature embeddings."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature embeddings from a model's penultimate layer.

    The model should be an ``_AssembledModel`` or any model whose ``backbone``
    + global-pool output is exposed.  This function calls ``model.backbone``
    followed by global average pooling.  If the model does not have a
    ``backbone`` attribute the full model is used (which includes the head —
    may not be what you want).

    Args:
        model: Trained model.
        dataloader: DataLoader that yields ``(image, label)`` batches.
        device: Target device.

    Returns:
        Tuple ``(embeddings, labels)`` as numpy arrays of shape ``(N, D)``
        and ``(N,)`` respectively.
    """
    device = torch.device(device)
    model.eval()
    model.to(device)

    all_embeddings: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    pool = nn.AdaptiveAvgPool2d(1)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)

            if hasattr(model, "backbone"):
                feats = model.backbone(images)
                if isinstance(feats, (list, tuple)):
                    feats = feats[-1]
                if feats.dim() == 4:
                    feats = pool(feats).flatten(1)
                elif feats.dim() == 3:
                    feats = feats[:, 0]  # CLS token
            else:
                feats = model(images)

            all_embeddings.append(feats.cpu().numpy())
            if isinstance(labels, torch.Tensor):
                all_labels.append(labels.cpu().numpy())
            else:
                all_labels.append(np.array(labels))

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    return embeddings, labels_arr


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Reduce embeddings with t-SNE and produce a scatter plot.

    Args:
        embeddings: Feature array of shape ``(N, D)``.
        labels: Integer label array of shape ``(N,)``.
        class_names: Optional list of class name strings.
        perplexity: t-SNE perplexity parameter (default 30).
        n_iter: Number of optimisation iterations (default 1000).
        random_state: Random seed for reproducibility.
        figsize: Figure size in inches.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    from sklearn.manifold import TSNE

    logger.info("Running t-SNE on %d embeddings (dim=%d)…", len(embeddings), embeddings.shape[1])
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
    )
    reduced = tsne.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=figsize)

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = class_names[int(lbl)] if class_names is not None else str(int(lbl))
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=12,
            alpha=0.7,
            color=cmap(i % 20),
            label=name,
        )

    ax.set_title("t-SNE of Feature Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize="small", markerscale=2)
    fig.tight_layout()
    return fig
