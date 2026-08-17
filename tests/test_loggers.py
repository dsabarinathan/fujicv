"""Tests for the BaseLogger interface, MLflowLogger, and generic loggers wiring."""
from __future__ import annotations

import tempfile
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fujicv.engine.base_logger import BaseLogger
from fujicv.engine.mlflow_logger import MLflowLogger
from fujicv.engine.trainer import Trainer
from fujicv.losses.classification import CrossEntropyLoss


class _SpyLogger(BaseLogger):
    """Records every call so we can assert the Trainer drives it correctly."""

    def __init__(self) -> None:
        self.epochs: List[Tuple[int, Dict[str, float]]] = []
        self.finished = False

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        self.epochs.append((epoch, dict(metrics)))

    def finish(self) -> None:
        self.finished = True


def _loader(n=16, num_classes=3):
    X = torch.randn(n, 3, 16, 16)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)


def _model(num_classes=3):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes))


def test_base_logger_defaults():
    class Minimal(BaseLogger):
        def log_epoch(self, epoch, metrics):
            pass

        def finish(self):
            pass

    lg = Minimal()
    assert lg.active is True
    # Optional methods are no-ops, must not raise.
    lg.log_scalar("x", 1.0, 0)
    lg.log_artifact("some/path", "name", "model")


def test_trainer_drives_generic_loggers():
    spy = _SpyLogger()
    model = _model()
    with tempfile.TemporaryDirectory() as tmp:
        Trainer(
            model=model,
            train_loader=_loader(),
            val_loader=_loader(),
            loss_fn=CrossEntropyLoss(),
            metrics={},
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            epochs=2,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
            loggers=[spy],
        ).train()

    assert len(spy.epochs) == 2
    assert spy.epochs[0][0] == 0 and spy.epochs[1][0] == 1
    assert "train_loss" in spy.epochs[0][1]
    assert spy.finished is True


def test_mlflow_logger_noop_when_uninstalled():
    """If mlflow is missing, the logger must degrade to a silent no-op."""
    import importlib.util

    mlf = MLflowLogger(experiment_name="unit-test")
    if importlib.util.find_spec("mlflow") is None:
        assert mlf.active is False
    # Regardless of availability, these must never raise.
    mlf.log_epoch(0, {"train_loss": 1.0})
    mlf.log_scalar("x", 0.5, 0)
    mlf.finish()
