"""Tests for the retrieval toolkit: metrics, Embedder, and RetrievalIndex."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fujicv.retrieval import (
    Embedder,
    RetrievalIndex,
    evaluate_retrieval,
    mean_average_precision_at_k,
    precision_at_k,
    recall_at_k,
)

# ── metrics: two clean clusters (perfect retrieval) ──────────────────────────
_A = np.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]])   # label 0
_B = np.array([[0.0, 1.0], [0.01, 0.99], [0.02, 0.98]])   # label 1
_EMB = np.vstack([_A, _B])
_LAB = np.array([0, 0, 0, 1, 1, 1])


def test_recall_at_1_perfect_clusters():
    # Each item's nearest neighbour (excluding self) shares its label.
    assert recall_at_k(_EMB, _LAB, k=1) == 1.0


def test_precision_at_2_perfect_clusters():
    # Top-2 of each item are the other two same-cluster points.
    assert precision_at_k(_EMB, _LAB, k=2) == 1.0


def test_map_perfect_clusters_is_one():
    assert mean_average_precision_at_k(_EMB, _LAB, k=2) == pytest.approx(1.0)


def test_recall_partial():
    # Query 0 sits between clusters; its nearest is the wrong-label point.
    emb = np.array([
        [1.0, 0.0],     # q0 label 0 — nearest will be [0.9,0.44] (label1)
        [0.9, 0.44],    # label 1, very close to q0
        [-1.0, 0.0],    # label 0, far from q0
        [-0.9, -0.44],  # label 1
    ])
    lab = np.array([0, 1, 0, 1])
    # q0's nearest is index1 (label1) → miss; others cluster correctly.
    r1 = recall_at_k(emb, lab, k=1)
    assert 0.0 <= r1 < 1.0


def test_map_at_k_known_value_query_gallery():
    """Hand-computed AP for a single query with ranking [rel, non, rel]."""
    q = np.array([[1.0, 0.0]])
    q_lab = np.array([0])
    gallery = np.array([
        [1.0, 0.0],                       # rel, sim 1.00
        [0.95, np.sqrt(1 - 0.95**2)],     # non, sim 0.95
        [0.90, np.sqrt(1 - 0.90**2)],     # rel, sim 0.90
    ])
    g_lab = np.array([0, 1, 0])
    # AP = (P@1*1 + P@3*1)/2 = (1 + 2/3)/2 = 0.8333...
    ap = mean_average_precision_at_k(
        gallery, g_lab, k=3, query_embeddings=q, query_labels=q_lab
    )
    assert ap == pytest.approx(0.8333, abs=1e-3)


def test_evaluate_retrieval_report_keys():
    report = evaluate_retrieval(_EMB, _LAB, ks=(1, 2))
    assert set(report) == {"recall@1", "recall@2", "mAP@2"}
    assert report["recall@1"] == 1.0


def test_metrics_accept_torch_tensors():
    emb = torch.tensor(_EMB, dtype=torch.float32)
    lab = torch.tensor(_LAB)
    assert recall_at_k(emb, lab, k=1) == 1.0


# ── Embedder ─────────────────────────────────────────────────────────────────
class _WithFeatures(nn.Module):
    """Model exposing forward_features (as FujiCV assembled models do)."""

    def __init__(self, d=4):
        super().__init__()
        self.head = nn.Linear(d, 3)

    def forward_features(self, x):
        return x.flatten(1)  # (B, d)

    def forward(self, x):
        return self.head(self.forward_features(x))


class _NoFeatures(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x.flatten(1))


def _emb_loader(n=12, d=4):
    X = torch.randn(n, d)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=4)


def test_embedder_uses_forward_features_and_normalizes():
    embedder = Embedder(_WithFeatures(4), device="cpu", normalize=True)
    emb, labels = embedder.embed(_emb_loader(), return_labels=True)
    assert emb.shape == (12, 4)
    assert labels.shape == (12,)
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)  # L2-normalized


def test_embedder_fallback_to_forward():
    embedder = Embedder(_NoFeatures(4), device="cpu", normalize=False)
    emb = embedder.embed(_emb_loader())
    assert emb.shape == (12, 4)


def test_embedder_return_labels_without_labels_raises():
    loader = DataLoader(torch.randn(8, 4), batch_size=4)  # no labels
    embedder = Embedder(_WithFeatures(4), device="cpu")
    with pytest.raises(ValueError, match="no labels"):
        embedder.embed(loader, return_labels=True)


# ── RetrievalIndex ───────────────────────────────────────────────────────────
def test_retrieval_index_search_topk():
    index = RetrievalIndex(_EMB, _LAB, device="cpu")
    assert len(index) == 6
    sims, idx, labels = index.search(_A, k=2)  # query with the 3 label-0 points
    assert sims.shape == (3, 2)
    assert idx.shape == (3, 2)
    # Every returned neighbour of a label-0 query should be label 0.
    assert (labels == 0).all()


def test_retrieval_index_single_vector_query():
    index = RetrievalIndex(_EMB, _LAB, device="cpu")
    sims, idx, labels = index.search(np.array([1.0, 0.0]), k=1)
    assert sims.shape == (1, 1)
    assert labels[0, 0] == 0


def test_retrieval_index_without_labels():
    index = RetrievalIndex(_EMB, device="cpu")
    sims, idx, labels = index.search(_A, k=1)
    assert labels is None
    assert idx.shape == (3, 1)
