"""Tests for EMA integration inside Trainer."""

from __future__ import annotations

import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fujicv.losses.classification import CrossEntropyLoss
from fujicv.metrics.classification import Accuracy


def _loader(n=16):
    imgs    = torch.randn(n, 3, 8, 8)
    targets = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(imgs, targets), batch_size=8)


def _model():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))


def test_trainer_use_ema_flag():
    """Trainer with use_ema=True should create a _ema attribute."""
    from fujicv.engine.trainer import Trainer
    loader = _loader()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=_model(),
            train_loader=loader,
            val_loader=loader,
            loss_fn=CrossEntropyLoss(),
            metrics={},
            optimizer=torch.optim.Adam(_model().parameters(), lr=1e-3),
            epochs=1,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
            use_ema=True,
            ema_decay=0.99,
        )
        assert trainer._ema is not None


def test_trainer_no_ema_by_default():
    from fujicv.engine.trainer import Trainer
    loader = _loader()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=_model(),
            train_loader=loader,
            val_loader=loader,
            loss_fn=CrossEntropyLoss(),
            metrics={},
            optimizer=torch.optim.Adam(_model().parameters(), lr=1e-3),
            epochs=1,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
        )
        assert trainer._ema is None


def test_trainer_ema_updates_during_training():
    """EMA shadow weights should change after training."""
    from fujicv.engine.trainer import Trainer
    from fujicv.training.ema import ModelEMA
    model  = _model()
    loader = _loader()

    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            loss_fn=CrossEntropyLoss(),
            metrics={},
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            epochs=2,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
            use_ema=True,
            ema_decay=0.9,
            ema_warmup_steps=0,
        )
        shadow_before = {n: p.clone() for n, p in trainer._ema.shadow.named_parameters()}
        trainer.train()
        shadow_after  = {n: p.clone() for n, p in trainer._ema.shadow.named_parameters()}

    changed = any(
        not torch.allclose(shadow_before[n], shadow_after[n])
        for n in shadow_before
    )
    assert changed, "EMA shadow weights did not change during training"


def test_trainer_ema_checkpoint_contains_ema_state():
    """best.pt should contain 'ema_state_dict' when use_ema=True."""
    from fujicv.engine.trainer import Trainer
    from pathlib import Path
    model  = _model()
    loader = _loader()

    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            loss_fn=CrossEntropyLoss(),
            metrics={"accuracy": Accuracy()},
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            epochs=2,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
            use_ema=True,
            monitor_metric="val_accuracy",
        )
        trainer.train()
        ckpt_path = Path(tmp) / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            assert "ema_state_dict" in ckpt


def test_trainer_history_has_train_val_keys_with_ema():
    """History keys must always have train_ / val_ prefix even with EMA."""
    from fujicv.engine.trainer import Trainer
    model  = _model()
    loader = _loader()

    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            loss_fn=CrossEntropyLoss(),
            metrics={"accuracy": Accuracy()},
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            epochs=2,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
            use_ema=True,
        )
        history = trainer.train()

    keys = list(history.metrics.keys())
    assert "train_loss"     in keys
    assert "val_loss"       in keys
    assert "train_accuracy" in keys
    assert "val_accuracy"   in keys
