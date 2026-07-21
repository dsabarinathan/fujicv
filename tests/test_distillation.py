"""Tests for Knowledge Distillation losses and DistillationTrainer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ── DistillationLoss ──────────────────────────────────────────────────────────

def _logits(n=8, c=4):
    return torch.randn(n, c), torch.randn(n, c), torch.randint(0, c, (n,))


def test_distillation_loss_scalar():
    from fujicv.losses.distillation import DistillationLoss
    loss_fn = DistillationLoss(alpha=0.7, temperature=4.0)
    s, t, y = _logits()
    loss = loss_fn(s, t, y)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_distillation_loss_alpha_one_pure_soft():
    """alpha=1.0 → pure soft loss, ignores hard labels."""
    from fujicv.losses.distillation import DistillationLoss
    loss_fn = DistillationLoss(alpha=1.0, temperature=4.0)
    s, t, y = _logits()
    loss = loss_fn(s, t, y)
    assert loss.item() >= 0


def test_distillation_loss_alpha_zero_pure_hard():
    """alpha=0.0 → pure cross-entropy, ignores teacher."""
    from fujicv.losses.distillation import DistillationLoss
    import torch.nn.functional as F
    loss_fn = DistillationLoss(alpha=0.0, temperature=4.0)
    s, t, y = _logits()
    loss = loss_fn(s, t, y)
    expected = F.cross_entropy(s, y)
    assert abs(loss.item() - expected.item()) < 1e-5


def test_distillation_loss_invalid_alpha():
    from fujicv.losses.distillation import DistillationLoss
    with pytest.raises(ValueError):
        DistillationLoss(alpha=1.5)


def test_distillation_loss_invalid_temperature():
    from fujicv.losses.distillation import DistillationLoss
    with pytest.raises(ValueError):
        DistillationLoss(temperature=0.0)


def test_distillation_loss_registered():
    from fujicv.losses import get_loss
    loss_fn = get_loss("DistillationLoss", {"alpha": 0.5, "temperature": 3.0})
    s, t, y = _logits()
    assert loss_fn(s, t, y).item() >= 0


def test_distillation_loss_backward():
    from fujicv.losses.distillation import DistillationLoss
    loss_fn = DistillationLoss()
    s = torch.randn(4, 3, requires_grad=True)
    t = torch.randn(4, 3)
    y = torch.randint(0, 3, (4,))
    loss = loss_fn(s, t, y)
    loss.backward()
    assert s.grad is not None


# ── FeatureDistillationLoss ───────────────────────────────────────────────────

def test_feature_distillation_loss_same_dim():
    from fujicv.losses.distillation import FeatureDistillationLoss
    loss_fn = FeatureDistillationLoss()
    s = torch.randn(4, 128)
    t = torch.randn(4, 128)
    loss = loss_fn(s, t)
    assert loss.item() >= 0


def test_feature_distillation_loss_with_projector():
    from fujicv.losses.distillation import FeatureDistillationLoss
    proj = nn.Linear(64, 128, bias=False)
    loss_fn = FeatureDistillationLoss(projector=proj)
    s = torch.randn(4, 64)
    t = torch.randn(4, 128)
    loss = loss_fn(s, t)
    assert loss.item() >= 0


def test_feature_distillation_registered():
    from fujicv.losses import get_loss
    loss_fn = get_loss("FeatureDistillationLoss", {})
    s = torch.randn(4, 32)
    t = torch.randn(4, 32)
    assert loss_fn(s, t).item() >= 0


# ── DistillationTrainer ───────────────────────────────────────────────────────

def _make_tiny_loader():
    """Tiny random DataLoader for smoke-testing the trainer."""
    from torch.utils.data import TensorDataset, DataLoader
    imgs    = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 3, (16,))
    ds = TensorDataset(imgs, targets)
    return DataLoader(ds, batch_size=8)


def test_distillation_trainer_smoke():
    from fujicv.engine.distillation_trainer import DistillationTrainer
    from fujicv.losses.distillation import DistillationLoss
    from fujicv.metrics.classification import Accuracy

    teacher = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))
    student = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))

    loader = _make_tiny_loader()

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        trainer = DistillationTrainer(
            teacher=teacher,
            model=student,
            train_loader=loader,
            val_loader=loader,
            loss_fn=DistillationLoss(alpha=0.7, temperature=4.0),
            metrics={"accuracy": Accuracy()},
            optimizer=torch.optim.Adam(student.parameters(), lr=1e-3),
            epochs=2,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
        )
        history = trainer.train()

    assert "train_loss" in history.metrics or "loss" in str(history.metrics)
    assert len(list(history.metrics.values())[0]) == 2


def test_distillation_trainer_rejects_wrong_loss():
    from fujicv.engine.distillation_trainer import DistillationTrainer
    from fujicv.losses.classification import CrossEntropyLoss

    teacher = nn.Linear(3, 3)
    student = nn.Linear(3, 3)
    loader = _make_tiny_loader()

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(TypeError, match="DistillationLoss"):
            DistillationTrainer(
                teacher=teacher,
                model=student,
                train_loader=loader,
                val_loader=loader,
                loss_fn=CrossEntropyLoss(),
                metrics={},
                optimizer=torch.optim.SGD(student.parameters(), lr=0.01),
                epochs=1,
                task="classification",
                output_dir=tmp,
                mixed_precision=False,
            )


def test_teacher_frozen():
    """Teacher parameters must not receive gradients during training."""
    from fujicv.engine.distillation_trainer import DistillationTrainer
    from fujicv.losses.distillation import DistillationLoss

    teacher = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))
    student = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))
    loader  = _make_tiny_loader()

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        trainer = DistillationTrainer(
            teacher=teacher,
            model=student,
            train_loader=loader,
            val_loader=loader,
            loss_fn=DistillationLoss(),
            metrics={},
            optimizer=torch.optim.Adam(student.parameters(), lr=1e-3),
            epochs=1,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
        )
        trainer.train()

    for p in teacher.parameters():
        assert not p.requires_grad
