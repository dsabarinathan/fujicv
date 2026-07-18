"""Unit tests for loss functions."""

from __future__ import annotations

import pytest
import torch


def _cls_batch(n=8, c=4):
    logits = torch.randn(n, c)
    targets = torch.randint(0, c, (n,))
    return logits, targets


def _reg_batch(n=8):
    preds = torch.randn(n)
    targets = torch.randn(n)
    return preds, targets


def _ml_batch(n=8, l=5):
    logits = torch.randn(n, l)
    targets = torch.randint(0, 2, (n, l)).float()
    return logits, targets


# ---- Classification losses ------------------------------------------------

def test_cross_entropy_loss():
    from fujicv.losses.classification import CrossEntropyLoss
    loss = CrossEntropyLoss()
    logits, targets = _cls_batch()
    out = loss(logits, targets)
    assert out.ndim == 0
    assert out.item() > 0


def test_weighted_cross_entropy_loss():
    from fujicv.losses.classification import WeightedCrossEntropyLoss
    counts = torch.tensor([100.0, 50.0, 25.0, 10.0])
    loss = WeightedCrossEntropyLoss(class_counts=counts)
    logits, targets = _cls_batch(c=4)
    out = loss(logits, targets)
    assert out.item() > 0


def test_label_smoothing_ce():
    from fujicv.losses.classification import LabelSmoothingCE
    loss = LabelSmoothingCE(smoothing=0.1)
    logits, targets = _cls_batch()
    out = loss(logits, targets)
    assert out.item() > 0


def test_label_smoothing_invalid():
    from fujicv.losses.classification import LabelSmoothingCE
    with pytest.raises(ValueError):
        LabelSmoothingCE(smoothing=1.5)


def test_focal_loss():
    from fujicv.losses.classification import FocalLoss
    loss = FocalLoss(alpha=0.25, gamma=2.0)
    logits, targets = _cls_batch()
    out = loss(logits, targets)
    assert out.item() >= 0


def test_class_balanced_loss():
    from fujicv.losses.classification import ClassBalancedLoss
    counts = torch.tensor([200.0, 100.0, 50.0, 10.0])
    loss = ClassBalancedLoss(class_counts=counts)
    logits, targets = _cls_batch(c=4)
    out = loss(logits, targets)
    assert out.item() > 0


# ---- Regression losses ----------------------------------------------------

def test_mse_loss():
    from fujicv.losses.regression import MSELoss
    loss = MSELoss()
    preds, targets = _reg_batch()
    out = loss(preds, targets)
    assert out.item() >= 0


def test_mae_loss():
    from fujicv.losses.regression import MAELoss
    loss = MAELoss()
    preds, targets = _reg_batch()
    out = loss(preds, targets)
    assert out.item() >= 0


def test_huber_loss():
    from fujicv.losses.regression import HuberLoss
    loss = HuberLoss(delta=1.0)
    preds, targets = _reg_batch()
    out = loss(preds, targets)
    assert out.item() >= 0


def test_log_cosh_loss():
    from fujicv.losses.regression import LogCoshLoss
    loss = LogCoshLoss()
    preds, targets = _reg_batch()
    out = loss(preds, targets)
    assert out.item() >= 0


def test_quantile_loss_median():
    from fujicv.losses.regression import QuantileLoss
    loss = QuantileLoss(quantile=0.5)
    preds = torch.zeros(8)
    targets = torch.ones(8)
    out = loss(preds, targets)
    # At q=0.5 and diff=1 everywhere: loss = 0.5 * 1 = 0.5
    assert abs(out.item() - 0.5) < 1e-5


def test_quantile_loss_invalid():
    from fujicv.losses.regression import QuantileLoss
    with pytest.raises(ValueError):
        QuantileLoss(quantile=1.5)


# ---- Multi-label losses ---------------------------------------------------

def test_bce_with_logits():
    from fujicv.losses.multilabel import BCEWithLogitsLoss
    loss = BCEWithLogitsLoss()
    logits, targets = _ml_batch()
    out = loss(logits, targets)
    assert out.item() >= 0


def test_focal_bce_loss():
    from fujicv.losses.multilabel import FocalBCELoss
    loss = FocalBCELoss(alpha=0.25, gamma=2.0)
    logits, targets = _ml_batch()
    out = loss(logits, targets)
    assert out.item() >= 0


def test_asymmetric_loss():
    from fujicv.losses.multilabel import AsymmetricLoss
    loss = AsymmetricLoss()
    logits, targets = _ml_batch()
    out = loss(logits, targets)
    assert out.item() >= 0


# ---- Registry ---------------------------------------------------------------

def test_get_loss_registry():
    from fujicv.losses import get_loss
    loss = get_loss("CrossEntropyLoss", {})
    logits, targets = _cls_batch()
    out = loss(logits, targets)
    assert out.item() > 0


def test_get_loss_unknown():
    from fujicv.losses import get_loss
    with pytest.raises(KeyError):
        get_loss("NonExistentLoss")
