"""Image-retrieval evaluation metrics (Recall@K, Precision@K, mAP@K).

These are the standard deep-metric-learning evaluation measures. They operate
on an embedding matrix and integer labels, ranking gallery items by cosine
similarity to each query.

Two modes:
  * **Single set** (default): ``query == gallery``; each item is a query against
    all others (its own row is excluded — "leave-one-out").
  * **Query vs. gallery**: pass separate ``query_embeddings``/``query_labels``.

Definitions (relevant = same label as the query):
  * **Recall@K** — fraction of queries whose top-K contains ≥1 relevant item.
  * **Precision@K** — mean fraction of the top-K that are relevant.
  * **mAP@K** — mean over queries of Average Precision within the top-K.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _ranked_relevance(
    embeddings,
    labels,
    max_k: int,
    query_embeddings=None,
    query_labels=None,
) -> torch.Tensor:
    """Return a ``(num_queries, max_k)`` boolean relevance matrix.

    Entry ``[i, j]`` is ``True`` if the j-th nearest gallery item to query ``i``
    shares its label. Embeddings are L2-normalized internally, so raw or
    pre-normalized vectors both work.
    """
    gallery = torch.nn.functional.normalize(_as_tensor(embeddings), dim=1)
    gal_labels = torch.as_tensor(np.asarray(labels)).view(-1)

    single_set = query_embeddings is None
    if single_set:
        queries = gallery
        q_labels = gal_labels
    else:
        queries = torch.nn.functional.normalize(_as_tensor(query_embeddings), dim=1)
        q_labels = torch.as_tensor(np.asarray(query_labels)).view(-1)

    sim = queries @ gallery.t()  # cosine similarity (both normalized)
    if single_set:
        # Exclude self-match by setting the diagonal to -inf.
        sim.fill_diagonal_(float("-inf"))

    k = min(max_k, gallery.shape[0] - (1 if single_set else 0))
    top_idx = sim.topk(k, dim=1).indices                 # (Q, k)
    retrieved_labels = gal_labels[top_idx]               # (Q, k)
    relevant = retrieved_labels == q_labels.view(-1, 1)  # (Q, k) bool
    return relevant


def recall_at_k(embeddings, labels, k: int = 1, **kw) -> float:
    """Fraction of queries whose top-``k`` contains at least one same-label item."""
    rel = _ranked_relevance(embeddings, labels, k, **kw)
    hit = rel[:, :k].any(dim=1).float()
    return float(hit.mean().item())


def precision_at_k(embeddings, labels, k: int = 5, **kw) -> float:
    """Mean fraction of the top-``k`` retrieved items that are same-label."""
    rel = _ranked_relevance(embeddings, labels, k, **kw)
    return float(rel[:, :k].float().mean().item())


def mean_average_precision_at_k(embeddings, labels, k: int = 10, **kw) -> float:
    """Mean Average Precision within the top-``k`` (mAP@K)."""
    rel = _ranked_relevance(embeddings, labels, k, **kw).float()  # (Q, k)
    q, kk = rel.shape
    if kk == 0:
        return 0.0
    ranks = torch.arange(1, kk + 1, dtype=torch.float32)
    cum_hits = torch.cumsum(rel, dim=1)               # relevant found up to i
    precision_at_i = cum_hits / ranks                 # P@i for each position
    # AP = sum(P@i * rel_i) / min(#relevant_in_topk, k); guard divide-by-zero.
    num_rel = rel.sum(dim=1).clamp(min=1.0)
    ap = (precision_at_i * rel).sum(dim=1) / num_rel
    return float(ap.mean().item())


def evaluate_retrieval(
    embeddings,
    labels,
    ks: Sequence[int] = (1, 5, 10),
    query_embeddings=None,
    query_labels=None,
    map_k: Optional[int] = None,
) -> Dict[str, float]:
    """Compute a standard retrieval report.

    Returns a dict with ``recall@k`` for each *k*, plus ``mAP@{map_k}``
    (defaults to the largest *k*).

    Example::

        emb, lab = embedder.embed(loader, return_labels=True)
        report = evaluate_retrieval(emb, lab, ks=(1, 5, 10))
        print(report)  # {'recall@1': .., 'recall@5': .., 'mAP@10': ..}
    """
    kw = dict(query_embeddings=query_embeddings, query_labels=query_labels)
    report: Dict[str, float] = {}
    for k in ks:
        report[f"recall@{k}"] = recall_at_k(embeddings, labels, k=k, **kw)
    mk = map_k or max(ks)
    report[f"mAP@{mk}"] = mean_average_precision_at_k(embeddings, labels, k=mk, **kw)
    return report


__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mean_average_precision_at_k",
    "evaluate_retrieval",
]
