"""Tests for the TensorBoard logger."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fujicv.engine.tensorboard_logger import TensorBoardLogger

pytest.importorskip("torch.utils.tensorboard")
pytest.importorskip("tensorboard")


def test_logger_writes_event_files():
    with tempfile.TemporaryDirectory() as tmp:
        tb = TensorBoardLogger(log_dir=tmp, config={"lr": 1e-3, "epochs": 2})
        assert tb.active
        tb.log_epoch(0, {"train_loss": 1.2, "val_loss": 1.5, "val_accuracy": 0.4})
        tb.log_epoch(1, {"train_loss": 0.8, "val_loss": 1.1, "val_accuracy": 0.6})
        tb.finish()
        assert not tb.active
        # SummaryWriter creates at least one tfevents file.
        events = list(Path(tmp).rglob("*tfevents*"))
        assert events, "no TensorBoard event file was written"


def test_grouped_tag():
    assert TensorBoardLogger._grouped_tag("train_loss") == "loss/train"
    assert TensorBoardLogger._grouped_tag("val_accuracy") == "accuracy/val"
    assert TensorBoardLogger._grouped_tag("test_f1") == "f1/test"
    assert TensorBoardLogger._grouped_tag("lr") == "lr"


def test_log_after_finish_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        tb = TensorBoardLogger(log_dir=tmp)
        tb.finish()
        # Must not raise even though the writer is closed.
        tb.log_epoch(0, {"train_loss": 1.0})
        tb.log_scalar("custom", 0.5, step=0)
