"""Guard: checkpoints must load under torch>=2.6's weights_only=True default.

FujiCV checkpoints embed non-tensor objects (History, class_to_idx, task), so
every internal ``torch.load`` must pass ``weights_only=False``. These tests
simulate the strict default on ANY torch version by patching ``torch.load`` to
default ``weights_only=True`` — our code must override it explicitly.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fujicv.engine.trainer import Trainer
from fujicv.inference.predictor import Predictor
from fujicv.losses.classification import CrossEntropyLoss
from fujicv.metrics.classification import Accuracy


@pytest.fixture()
def strict_torch_load(monkeypatch):
    """Make torch.load default to weights_only=True, like torch>=2.6."""
    orig = torch.load

    def _strict(*args, **kwargs):
        kwargs.setdefault("weights_only", True)
        return orig(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _strict)
    return _strict


def _loader(n=16, nc=3):
    X = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, nc, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=8)


def _model(nc=3):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, nc))


def _train_checkpoint(tmp, epochs=1):
    model = _model()
    Trainer(
        model=model, train_loader=_loader(), val_loader=_loader(),
        loss_fn=CrossEntropyLoss(), metrics={"accuracy": Accuracy()},
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        epochs=epochs, task="classification", output_dir=tmp,
        mixed_precision=False, class_to_idx={"a": 0, "b": 1, "c": 2},
    ).train()


def test_trainer_resume_under_strict_weights_only(strict_torch_load):
    with tempfile.TemporaryDirectory() as tmp:
        _train_checkpoint(tmp, epochs=1)
        model = _model()
        # resume_from calls _load_checkpoint → must not raise under strict default.
        trainer = Trainer(
            model=model, train_loader=_loader(), val_loader=_loader(),
            loss_fn=CrossEntropyLoss(), metrics={},
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            epochs=2, task="classification", output_dir=tmp,
            mixed_precision=False, resume_from=str(Path(tmp) / "best.pt"),
        )
        assert trainer._start_epoch >= 1


def test_predictor_from_checkpoint_under_strict_weights_only(strict_torch_load):
    with tempfile.TemporaryDirectory() as tmp:
        _train_checkpoint(tmp, epochs=1)
        predictor = Predictor.from_checkpoint(
            str(Path(tmp) / "best.pt"), model=_model(), device="cpu", image_size=8
        )
        assert predictor.task == "classification"
        assert predictor.class_to_idx == {"a": 0, "b": 1, "c": 2}
