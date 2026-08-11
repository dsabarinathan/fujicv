"""Tests for metric-learning margin heads."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fujicv.models.metric_heads import (
    AddMarginProduct,
    ArcMarginProduct,
    CosMarginProduct,
    SubCenterArcMarginProduct,
)


def test_arcface_output_shape_train_and_infer():
    head = ArcMarginProduct(in_features=8, num_classes=5, s=30.0, m=0.5)
    feats = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])

    train_logits = head(feats, labels)
    assert train_logits.shape == (4, 5)

    infer_logits = head(feats)  # labels=None
    assert infer_logits.shape == (4, 5)


def test_arcface_infer_is_scaled_cosine():
    head = ArcMarginProduct(in_features=8, num_classes=5, s=30.0, m=0.5)
    feats = torch.randn(4, 8)
    cosine = F.linear(F.normalize(feats), F.normalize(head.weight))
    out = head(feats)
    assert torch.allclose(out, cosine * head.s, atol=1e-5)


def test_arcface_margin_lowers_target_logit():
    """The additive angular margin must reduce the target-class logit vs. plain cosine."""
    torch.manual_seed(0)
    head = ArcMarginProduct(in_features=16, num_classes=10, s=30.0, m=0.5)
    feats = torch.randn(8, 16)
    labels = torch.randint(0, 10, (8,))

    with_margin = head(feats, labels)
    plain = head(feats)  # scaled cosine, no margin

    tgt_margin = with_margin.gather(1, labels.unsqueeze(1)).squeeze(1)
    tgt_plain = plain.gather(1, labels.unsqueeze(1)).squeeze(1)
    assert torch.all(tgt_margin <= tgt_plain + 1e-4)


def test_arcface_backward_produces_grads():
    head = ArcMarginProduct(in_features=8, num_classes=5)
    feats = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    loss = F.cross_entropy(head(feats, labels), labels)
    loss.backward()
    assert feats.grad is not None
    assert head.weight.grad is not None
    assert torch.isfinite(loss)


def test_cosface_margin_and_alias():
    assert CosMarginProduct is AddMarginProduct
    head = AddMarginProduct(in_features=8, num_classes=5, s=30.0, m=0.35)
    feats = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])

    with_margin = head(feats, labels)
    plain = head(feats)
    tgt_margin = with_margin.gather(1, labels.unsqueeze(1)).squeeze(1)
    tgt_plain = plain.gather(1, labels.unsqueeze(1)).squeeze(1)
    # CosFace subtracts m from the target cosine before scaling → exactly s*m lower.
    assert torch.allclose(tgt_plain - tgt_margin, torch.full_like(tgt_margin, head.s * head.m), atol=1e-4)


def test_subcenter_arcface_shapes_and_k():
    head = SubCenterArcMarginProduct(in_features=8, num_classes=5, k=3, s=30.0, m=0.5)
    assert head.weight.shape == (5 * 3, 8)
    feats = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])
    assert head(feats, labels).shape == (4, 5)
    assert head(feats).shape == (4, 5)


def test_subcenter_rejects_bad_k():
    with pytest.raises(ValueError):
        SubCenterArcMarginProduct(in_features=8, num_classes=5, k=0)


def test_subcenter_backward():
    head = SubCenterArcMarginProduct(in_features=8, num_classes=5, k=2)
    feats = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    loss = F.cross_entropy(head(feats, labels), labels)
    loss.backward()
    assert head.weight.grad is not None
    assert torch.isfinite(loss)
