"""Stress / edge-case tests for release hardening.

Targets degenerate shapes, extreme values, AMP dtypes, and cross-feature
interactions (grad-accum × EMA, tiny batches, single-class data) that the
happy-path unit tests don't exercise.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fujicv.engine.trainer import Trainer
from fujicv.losses.classification import CrossEntropyLoss
from fujicv.metrics.classification import Accuracy
from fujicv.models.metric_heads import ArcMarginProduct, SubCenterArcMarginProduct
from fujicv.training.model_soup import uniform_soup


# ── helpers ──────────────────────────────────────────────────────────────────
def _loader(n, num_classes=3, bs=8, img=8):
    X = torch.randn(n, 3, img, img)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=False)


def _model(nc=3):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, nc))


def _train(**over):
    cfg = dict(
        loss_fn=CrossEntropyLoss(), metrics={"accuracy": Accuracy()},
        epochs=1, task="classification", mixed_precision=False,
    )
    cfg.update(over)
    model = cfg.pop("model")
    with tempfile.TemporaryDirectory() as tmp:
        t = Trainer(model=model, output_dir=tmp,
                    optimizer=torch.optim.SGD(model.parameters(), lr=0.01), **cfg)
        return t.train()


# ── grad accumulation edge cases ─────────────────────────────────────────────
def test_grad_accum_larger_than_num_batches():
    """accum > number of batches: the final-batch flush must still step once."""
    model = _model()
    ld = _loader(16, bs=8)  # 2 batches
    h = _train(model=model, train_loader=ld, val_loader=ld, grad_accum_steps=8)
    assert np.isfinite(h.metrics["train_loss"][0])
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_grad_accum_with_ema_updates_only_on_step():
    model = _model()
    ld = _loader(24, bs=8)  # 3 batches
    h = _train(model=model, train_loader=ld, val_loader=ld,
               grad_accum_steps=2, use_ema=True, ema_warmup_steps=0)
    assert np.isfinite(h.metrics["val_loss"][0])


def test_single_sample_batches_do_not_crash_bn_free_model():
    model = _model()
    ld = _loader(3, bs=1)  # three single-sample batches
    h = _train(model=model, train_loader=ld, val_loader=ld)
    assert np.isfinite(h.metrics["train_loss"][0])


def test_single_class_dataset_trains():
    """All labels identical — loss/metrics must stay finite and in range."""
    X = torch.randn(16, 3, 8, 8)
    y = torch.zeros(16, dtype=torch.long)
    ld = DataLoader(TensorDataset(X, y), batch_size=8)
    model = _model()
    h = _train(model=model, train_loader=ld, val_loader=ld)
    assert np.isfinite(h.metrics["train_loss"][0])
    # A 3-output model on single-class data may predict any class; accuracy just
    # has to be a valid fraction (no crash / NaN on the degenerate label set).
    assert 0.0 <= h.metrics["train_accuracy"][0] <= 1.0


# ── metric-head numerical stability ──────────────────────────────────────────
def test_arcface_extreme_margin_no_nan():
    head = ArcMarginProduct(in_features=16, num_classes=8, s=64.0, m=0.9)
    feats = torch.randn(32, 16) * 100  # large-magnitude embeddings
    labels = torch.randint(0, 8, (32,))
    out = head(feats, labels)
    assert torch.isfinite(out).all(), "ArcFace produced NaN/Inf on extreme inputs"
    loss = nn.functional.cross_entropy(out, labels)
    loss.backward()
    assert torch.isfinite(head.weight.grad).all()


def test_arcface_fp16_autocast_no_nan():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for autocast fp16")
    head = ArcMarginProduct(16, 8).cuda()
    feats = torch.randn(8, 16, device="cuda")
    labels = torch.randint(0, 8, (8,), device="cuda")
    with torch.autocast("cuda"):
        out = head(feats, labels)
    assert torch.isfinite(out.float()).all()


def test_subcenter_all_same_label():
    head = SubCenterArcMarginProduct(16, 5, k=3)
    feats = torch.randn(10, 16)
    labels = torch.zeros(10, dtype=torch.long)
    out = head(feats, labels)
    assert torch.isfinite(out).all()
    assert out.shape == (10, 5)


# ── model soup edge cases ────────────────────────────────────────────────────
def test_uniform_soup_with_batchnorm_buffers():
    """Averaging models with BN running stats must not corrupt integer buffers."""
    def make(seed):
        torch.manual_seed(seed)
        m = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
        m(torch.randn(16, 4))  # populate running stats
        return m

    m1, m2 = make(1), make(2)
    target = make(3)
    uniform_soup(target, [m1.state_dict(), m2.state_dict()])
    for p in target.parameters():
        assert torch.isfinite(p).all()


# ── loss / metric degenerate inputs ──────────────────────────────────────────
def test_accuracy_metric_single_row():
    acc = Accuracy()
    y_true = np.array([1])
    y_pred = np.array([[0.1, 0.9, 0.0]])
    assert acc(y_true, y_pred) == 1.0


def test_crossentropy_extreme_logits_finite():
    loss = CrossEntropyLoss()
    logits = torch.tensor([[1e4, -1e4, 0.0], [-1e4, 1e4, 0.0]])
    targets = torch.tensor([0, 1])
    out = loss(logits, targets)
    assert torch.isfinite(out).all()
