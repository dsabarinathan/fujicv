"""Extract embeddings from a trained model for retrieval / kNN / clustering."""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Embedder:
    """Run a model over data and collect its pre-head feature embeddings.

    The embedding source is resolved in this order:

    1. ``model.forward_features(x)`` if present — the pooled, pre-head vector
       (FujiCV's :class:`~fujicv.models.builder.ModelBuilder` models expose this).
    2. Otherwise the model's plain ``forward`` output (e.g. a backbone with no
       head, or a metric-learning head returning cosine logits).

    Args:
        model: Trained ``nn.Module``.
        device: Inference device (default: CUDA if available else CPU).
        normalize: L2-normalize each embedding (default ``True`` — required for
            cosine retrieval).

    Example::

        embedder = Embedder(model)
        emb, labels = embedder.embed(gallery_loader, return_labels=True)
        # emb: (N, D) float32 numpy array
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self.model = model
        self.normalize = normalize
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()

        core = getattr(model, "_orig_mod", model)  # unwrap torch.compile
        self._feature_fn = getattr(core, "forward_features", None)

    @torch.no_grad()
    def embed_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Embed a single ``(B, C, H, W)`` tensor → ``(B, D)`` on-device tensor."""
        images = images.to(self.device, non_blocking=True)
        feats = self._feature_fn(images) if self._feature_fn is not None else self.model(images)
        if feats.dim() > 2:
            feats = feats.flatten(1)
        if self.normalize:
            feats = F.normalize(feats, dim=1)
        return feats

    @torch.no_grad()
    def embed(
        self,
        loader: DataLoader,
        return_labels: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Embed every item in *loader*.

        Args:
            loader: DataLoader yielding ``images`` or ``(images, labels)`` batches.
            return_labels: Also return the stacked labels (requires the loader
                to yield labels).

        Returns:
            ``(N, D)`` float32 embeddings, or ``(embeddings, labels)`` if
            *return_labels* is ``True``.
        """
        self.model.eval()
        chunks = []
        labels = []
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                if return_labels and len(batch) < 2:
                    raise ValueError("return_labels=True but the loader yields no labels.")
                images = batch[0]
                if return_labels:
                    lb = batch[1]
                    labels.append(lb.cpu().numpy() if torch.is_tensor(lb) else np.asarray(lb))
            else:
                if return_labels:
                    raise ValueError("return_labels=True but the loader yields no labels.")
                images = batch
            chunks.append(self.embed_batch(images).cpu().numpy())

        embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
        logger.info("Embedded %d items → dim %d", embeddings.shape[0], embeddings.shape[1])
        if return_labels:
            return embeddings, np.concatenate(labels, axis=0)
        return embeddings


__all__ = ["Embedder"]
