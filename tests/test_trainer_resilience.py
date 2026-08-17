"""Tests for Trainer resilience: incremental history + DDP helper correctness."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fujicv.engine.trainer import History, Trainer
from fujicv.losses.classification import CrossEntropyLoss
from fujicv.metrics.classification import Accuracy


def _loader(n=24, num_classes=3):
    X = torch.randn(n, 3, 16, 16)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)


def _model(num_classes=3):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes))


def test_history_to_json_roundtrip():
    h = History()
    h.update({"train_loss": 1.0, "val_loss": 1.2})
    h.update({"train_loss": 0.8, "val_loss": 1.1})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "history.json"
        h.to_json(path)
        loaded = json.loads(path.read_text())
    assert loaded["train_loss"] == [1.0, 0.8]
    assert loaded["val_loss"] == [1.2, 1.1]


def test_history_written_every_epoch():
    """history.csv and history.json must exist and grow during training."""
    model = _model()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model,
            train_loader=_loader(),
            val_loader=_loader(),
            loss_fn=CrossEntropyLoss(),
            metrics={"accuracy": Accuracy()},
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            epochs=3,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
        )
        trainer.train()
        hist_json = json.loads((Path(tmp) / "history.json").read_text())
        assert len(hist_json["train_loss"]) == 3
        assert (Path(tmp) / "history.csv").exists()


def test_main_process_true_without_ddp():
    model = _model()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model,
            train_loader=_loader(),
            val_loader=_loader(),
            loss_fn=CrossEntropyLoss(),
            metrics={},
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            output_dir=tmp,
            mixed_precision=False,
        )
        assert trainer._is_main_process is True
        # Gather/reduce helpers must be no-ops outside DDP.
        import numpy as np
        arr = np.arange(6).reshape(3, 2)
        assert np.array_equal(trainer._gather_concat(arr), arr)
        assert trainer._reduce_sum(2.0, 3.0) == [2.0, 3.0]
