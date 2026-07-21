"""Tests for KFoldTrainer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n=60, num_classes=3):
    labels = [str(i % num_classes) for i in range(n)]
    return pd.DataFrame({"img": [f"img_{i}.jpg" for i in range(n)], "label": labels})


def _tiny_model():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))


def _tiny_loader(df, transform):
    """Ignore df/transform; return random tensors to avoid disk I/O in tests."""
    imgs    = torch.randn(len(df), 3, 32, 32)
    targets = torch.randint(0, 3, (len(df),))
    ds = TensorDataset(imgs, targets)
    return ds


def _make_trainer(output_dir):
    from fujicv.losses.classification import CrossEntropyLoss
    from fujicv.metrics.classification import Accuracy

    def factory(model, train_loader, val_loader):
        from fujicv.engine.trainer import Trainer
        return Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=CrossEntropyLoss(),
            metrics={"accuracy": Accuracy()},
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            epochs=2,
            task="classification",
            output_dir=output_dir,
            mixed_precision=False,
        )
    return factory


# ── KFoldTrainer ──────────────────────────────────────────────────────────────

def test_kfold_runs_correct_number_of_folds():
    from fujicv.training.kfold import KFoldTrainer

    df = _make_df(60)

    with tempfile.TemporaryDirectory() as tmp:
        kfold = KFoldTrainer(
            model_factory=_tiny_model,
            train_df=df,
            dataset_factory=_tiny_loader,
            train_transform=None,
            val_transform=None,
            trainer_factory=_make_trainer(tmp),
            n_splits=3,
            stratify_col="label",
            output_dir=tmp,
            dataloader_kwargs={"batch_size": 16},
        )
        results = kfold.run()

    assert len(results["fold_histories"]) == 3
    assert len(results["fold_metrics"]) == 3


def test_kfold_summary_shape():
    from fujicv.training.kfold import KFoldTrainer

    df = _make_df(60)
    with tempfile.TemporaryDirectory() as tmp:
        kfold = KFoldTrainer(
            model_factory=_tiny_model,
            train_df=df,
            dataset_factory=_tiny_loader,
            train_transform=None,
            val_transform=None,
            trainer_factory=_make_trainer(tmp),
            n_splits=3,
            stratify_col="label",
            output_dir=tmp,
            dataloader_kwargs={"batch_size": 16},
        )
        results = kfold.run()

    summary = results["summary"]
    assert isinstance(summary, pd.DataFrame)
    assert "mean" in summary.columns
    assert "std" in summary.columns
    assert len(summary) > 0


def test_kfold_oof_covers_full_dataset():
    from fujicv.training.kfold import KFoldTrainer

    n = 60
    df = _make_df(n)
    with tempfile.TemporaryDirectory() as tmp:
        kfold = KFoldTrainer(
            model_factory=_tiny_model,
            train_df=df,
            dataset_factory=_tiny_loader,
            train_transform=None,
            val_transform=None,
            trainer_factory=_make_trainer(tmp),
            n_splits=3,
            stratify_col="label",
            output_dir=tmp,
            dataloader_kwargs={"batch_size": 16},
        )
        results = kfold.run()

    oof_preds   = results["oof_preds"]
    oof_targets = results["oof_targets"]

    assert oof_preds.shape[0] == n
    assert oof_targets.shape[0] == n
    # No NaN values — every sample was in exactly one val fold
    assert not np.any(np.isnan(oof_targets))


def test_kfold_fold_dirs_created():
    from fujicv.training.kfold import KFoldTrainer

    df = _make_df(30)
    with tempfile.TemporaryDirectory() as tmp:
        kfold = KFoldTrainer(
            model_factory=_tiny_model,
            train_df=df,
            dataset_factory=_tiny_loader,
            train_transform=None,
            val_transform=None,
            trainer_factory=_make_trainer(tmp),
            n_splits=3,
            stratify_col=None,  # plain KFold
            output_dir=tmp,
            dataloader_kwargs={"batch_size": 10},
        )
        kfold.run()

        for fold_i in range(3):
            fold_dir = Path(tmp) / f"fold_{fold_i}"
            assert fold_dir.exists()


def test_kfold_missing_sklearn(monkeypatch):
    from fujicv.training.kfold import KFoldTrainer
    import builtins, importlib

    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "sklearn.model_selection":
            raise ImportError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    df = _make_df(30)
    with tempfile.TemporaryDirectory() as tmp:
        kfold = KFoldTrainer(
            model_factory=_tiny_model,
            train_df=df,
            dataset_factory=_tiny_loader,
            train_transform=None,
            val_transform=None,
            trainer_factory=_make_trainer(tmp),
            n_splits=3,
            output_dir=tmp,
            dataloader_kwargs={"batch_size": 10},
        )
        with pytest.raises(ImportError, match="scikit-learn"):
            kfold.run()
