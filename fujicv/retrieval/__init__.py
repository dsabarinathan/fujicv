"""Image-retrieval toolkit: embedding extraction, kNN search, and eval metrics.

Completes the metric-learning stack (ArcFace/CosFace/Sub-center heads + GeM
pooling) with the tools to actually use learned embeddings:

* :class:`Embedder` — extract pre-head embeddings from a trained model.
* :class:`RetrievalIndex` — cosine nearest-neighbour search (optional FAISS).
* :func:`evaluate_retrieval` — Recall@K / Precision@K / mAP@K report.
"""

from fujicv.retrieval.embedder import Embedder
from fujicv.retrieval.index import RetrievalIndex
from fujicv.retrieval.metrics import (
    evaluate_retrieval,
    mean_average_precision_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "Embedder",
    "RetrievalIndex",
    "recall_at_k",
    "precision_at_k",
    "mean_average_precision_at_k",
    "evaluate_retrieval",
]
