"""Nearest-neighbour retrieval index (cosine similarity, optional FAISS)."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RetrievalIndex:
    """Cosine-similarity nearest-neighbour index over a gallery of embeddings.

    By default uses a pure-PyTorch matmul search (exact, no extra deps). If
    ``use_faiss=True`` and ``faiss`` is installed, a FAISS inner-product index
    is used instead for large galleries.

    Embeddings are L2-normalized on add and on query, so inner product == cosine.

    Args:
        embeddings: ``(N, D)`` gallery embeddings (numpy or tensor).
        labels: Optional ``(N,)`` gallery labels, returned alongside search hits.
        use_faiss: Use a FAISS index if available (falls back to torch with a
            warning if faiss is not installed).
        device: Torch device for the matmul backend (default: CUDA if available).

    Example::

        index = RetrievalIndex(gallery_emb, gallery_labels)
        sims, idx, hit_labels = index.search(query_emb, k=5)
    """

    def __init__(
        self,
        embeddings,
        labels=None,
        use_faiss: bool = False,
        device: Optional[str] = None,
    ) -> None:
        emb = self._to_tensor(embeddings)
        self._gallery = F.normalize(emb, dim=1)
        self.labels = None if labels is None else np.asarray(labels).reshape(-1)
        self.dim = self._gallery.shape[1]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._faiss_index = None
        if use_faiss:
            self._try_build_faiss()
        if self._faiss_index is None:
            self._gallery = self._gallery.to(self.device)

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().float()
        return torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def _try_build_faiss(self) -> None:
        try:
            import faiss
        except ImportError:
            logger.warning("RetrievalIndex: faiss not installed; using torch backend.")
            return
        index = faiss.IndexFlatIP(self.dim)  # inner product on normalized = cosine
        index.add(self._gallery.cpu().numpy())
        self._faiss_index = index
        logger.info("RetrievalIndex: FAISS IndexFlatIP built over %d vectors.", self._gallery.shape[0])

    def search(self, queries, k: int = 5) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return the top-``k`` gallery matches for each query.

        Args:
            queries: ``(Q, D)`` query embeddings (or a single ``(D,)`` vector).
            k: Number of neighbours to return.

        Returns:
            ``(similarities, indices, labels)`` — each ``(Q, k)``. ``labels`` is
            ``None`` if the index was built without gallery labels.
        """
        q = self._to_tensor(queries)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = F.normalize(q, dim=1)
        k = min(k, self._gallery.shape[0])

        if self._faiss_index is not None:
            sims, idx = self._faiss_index.search(q.cpu().numpy().astype(np.float32), k)
        else:
            sim = q.to(self.device) @ self._gallery.t()  # (Q, N)
            topk = sim.topk(k, dim=1)
            sims = topk.values.cpu().numpy()
            idx = topk.indices.cpu().numpy()

        hit_labels = None if self.labels is None else self.labels[idx]
        return sims, idx, hit_labels

    def __len__(self) -> int:
        return int(self._gallery.shape[0])


__all__ = ["RetrievalIndex"]
