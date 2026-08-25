"""End-to-end integration test: dummy dataset → Trainer → artifacts on disk."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fujicv.engine.trainer import Trainer
from fujicv.losses.classification import CrossEntropyLoss
from fujicv.metrics.classification import Accuracy


def _make_loader(n: int = 40, num_classes: int = 3, img_size: int = 16) -> DataLoader:
    X = torch.randn(n, 3, img_size, img_size)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)


def _tiny_model(num_classes: int = 3) -> nn.Module:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, num_classes),
    )


def test_full_training_loop_creates_artifacts():
    """2-epoch run must produce best.pt, last.pt, and history.csv."""
    loader = _make_loader()
    model  = _tiny_model()
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
        )
        history = trainer.train()

        tmp_path = Path(tmp)
        assert (tmp_path / "best.pt").exists(),    "best.pt not created"
        assert (tmp_path / "last.pt").exists(),    "last.pt not created"
        assert (tmp_path / "history.csv").exists(), "history.csv not created"

        # History should contain exactly 2 rows
        assert len(history.metrics["train_loss"]) == 2
        assert len(history.metrics["val_loss"])   == 2


def test_best_checkpoint_is_loadable():
    """best.pt must be loadable and contain expected keys."""
    loader = _make_loader()
    model  = _tiny_model()
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
        )
        trainer.train()

        ckpt = torch.load(Path(tmp) / "best.pt", map_location="cpu", weights_only=False)
        assert "model_state_dict"     in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "epoch"                in ckpt


def test_history_csv_has_correct_columns():
    """history.csv must include train_loss, val_loss, train_accuracy, val_accuracy."""
    import csv as csv_mod

    loader = _make_loader()
    model  = _tiny_model()
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
        )
        trainer.train()

        with open(Path(tmp) / "history.csv") as fh:
            reader = csv_mod.DictReader(fh)
            rows   = list(reader)

        assert len(rows) == 2
        for col in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
            assert col in rows[0], f"Column '{col}' missing from history.csv"


def test_resume_from_checkpoint():
    """Resuming from last.pt should pick up from the correct epoch."""
    loader = _make_loader()
    model  = _tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        # First run: 1 epoch
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            loss_fn=CrossEntropyLoss(),
            metrics={},
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            epochs=1,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
        )
        trainer.train()

        # Resume: run 1 more epoch (total 2)
        model2 = _tiny_model()
        trainer2 = Trainer(
            model=model2,
            train_loader=loader,
            val_loader=loader,
            loss_fn=CrossEntropyLoss(),
            metrics={},
            optimizer=torch.optim.Adam(model2.parameters(), lr=1e-3),
            epochs=2,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
            resume_from=Path(tmp) / "last.pt",
        )
        history2 = trainer2.train()
        # history is cumulative: 1 epoch from the first run + 1 from the resumed run
        assert len(history2.metrics["train_loss"]) == 2


def test_early_stopping_halts_training():
    """With lr=0 (frozen model) and patience=1, training stops after 2 epochs."""
    loader = _make_loader()
    model  = _tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            loss_fn=CrossEntropyLoss(),
            metrics={},
            # lr=0 → weights never update → val_loss flat → early stop triggers after epoch 2
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            epochs=10,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
            early_stopping_patience=1,
        )
        history = trainer.train()
        # Should stop well before 10 epochs since val_loss never improves
        assert len(history.metrics["train_loss"]) < 10


def test_multi_gpu_warning_single_gpu_env():
    """On single-GPU or CPU, use_ddp=False must not raise."""
    loader = _make_loader()
    model  = _tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            loss_fn=CrossEntropyLoss(),
            metrics={},
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            epochs=1,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
            use_ddp=False,
        )
        trainer.train()   # must not raise
