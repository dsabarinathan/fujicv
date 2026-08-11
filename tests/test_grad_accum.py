"""Tests for gradient accumulation in the Trainer."""
from __future__ import annotations

import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fujicv.engine.trainer import Trainer
from fujicv.losses.classification import CrossEntropyLoss


def _loader(n: int = 32, num_classes: int = 3, batch_size: int = 8) -> DataLoader:
    X = torch.randn(n, 3, 16, 16)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


def _model(num_classes: int = 3) -> nn.Module:
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes))


def _train(grad_accum_steps: int, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    model = _model()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model,
            train_loader=_loader(),
            val_loader=_loader(),
            loss_fn=CrossEntropyLoss(),
            metrics={},
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            epochs=1,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
            grad_clip=None,
            grad_accum_steps=grad_accum_steps,
        )
        trainer.train()
    return model


def test_grad_accum_rejects_invalid():
    model = _model()
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(ValueError):
        Trainer(
            model=model,
            train_loader=_loader(),
            val_loader=_loader(),
            loss_fn=CrossEntropyLoss(),
            metrics={},
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            output_dir=tmp,
            grad_accum_steps=0,
        )


def test_grad_accum_default_runs():
    """grad_accum_steps=1 must behave exactly like the normal loop."""
    model = _train(grad_accum_steps=1)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_grad_accum_equivalent_to_large_batch():
    """Accumulating over K micro-batches ≈ one optimizer step on the full batch.

    With a full-batch loader (batch == whole dataset) and no shuffling, doing
    4 micro-batches with grad_accum_steps=4 must produce the same weights as a
    single big batch with grad_accum_steps=1.
    """
    torch.manual_seed(0)
    X = torch.randn(32, 3, 16, 16)
    y = torch.randint(0, 3, (32,))

    # Reference: one big batch, one optimizer step.
    torch.manual_seed(1)
    ref = _model()
    with tempfile.TemporaryDirectory() as tmp:
        Trainer(
            model=ref,
            train_loader=DataLoader(TensorDataset(X, y), batch_size=32, shuffle=False),
            val_loader=DataLoader(TensorDataset(X, y), batch_size=32, shuffle=False),
            loss_fn=CrossEntropyLoss(),
            metrics={},
            optimizer=torch.optim.SGD(ref.parameters(), lr=0.1),
            epochs=1, task="classification", output_dir=tmp,
            mixed_precision=False, grad_clip=None, grad_accum_steps=1,
        ).train()

    # Accumulated: 4 micro-batches of 8, one effective step.
    torch.manual_seed(1)
    acc = _model()
    with tempfile.TemporaryDirectory() as tmp:
        Trainer(
            model=acc,
            train_loader=DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False),
            val_loader=DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False),
            loss_fn=CrossEntropyLoss(),
            metrics={},
            optimizer=torch.optim.SGD(acc.parameters(), lr=0.1),
            epochs=1, task="classification", output_dir=tmp,
            mixed_precision=False, grad_clip=None, grad_accum_steps=4,
        ).train()

    for p_ref, p_acc in zip(ref.parameters(), acc.parameters()):
        assert torch.allclose(p_ref, p_acc, atol=1e-5), "accumulated step diverged from big-batch step"
