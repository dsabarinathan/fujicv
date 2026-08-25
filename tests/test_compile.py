"""Tests for the torch.compile Trainer flag and robust _model_core unwrapping."""
from __future__ import annotations

import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fujicv.engine.trainer import Trainer
from fujicv.losses.classification import CrossEntropyLoss


def _loader(n=16, num_classes=3):
    X = torch.randn(n, 3, 16, 16)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)


def _model(num_classes=3):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes))


def _make_trainer(tmp, compile_model):
    model = _model()
    return Trainer(
        model=model,
        train_loader=_loader(),
        val_loader=_loader(),
        loss_fn=CrossEntropyLoss(),
        metrics={},
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        epochs=1,
        task="classification",
        output_dir=tmp,
        mixed_precision=False,
        compile_model=compile_model,
    )


def test_compile_flag_off_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, compile_model=False)
        assert trainer._compiled is False
        # _model_core is the plain module.
        assert isinstance(trainer._model_core, nn.Sequential)


class _FakeCompiled(nn.Module):
    """Mimics torch.compile's OptimizedModule, which exposes ._orig_mod."""

    def __init__(self, orig: nn.Module) -> None:
        super().__init__()
        self._orig_mod = orig

    def forward(self, x):
        return self._orig_mod(x)


def test_model_core_unwraps_orig_mod():
    """_model_core must peel off a torch.compile wrapper on any platform."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, compile_model=False)
        original = trainer.model
        # Simulate a compiled model wrapping the original.
        trainer.model = _FakeCompiled(original)
        assert trainer._model_core is original


def test_model_core_unwraps_compiled_and_ddp_nesting():
    """Even compiled-over-DataParallel must resolve to the plain module."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, compile_model=False)
        original = trainer.model
        # compile( DataParallel( original ) )
        trainer.model = _FakeCompiled(nn.DataParallel(original))
        assert trainer._model_core is original


def test_model_core_clean_state_dict_when_compiled():
    """Whether compile succeeds or falls back, checkpoint keys stay clean."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, compile_model=True)
        core = trainer._model_core
        assert isinstance(core, nn.Sequential)
        keys = list(core.state_dict().keys())
        assert all("_orig_mod" not in k and "module." not in k for k in keys)


def test_compiled_trainer_trains_and_checkpoints():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, compile_model=True)
        trainer.train()
        import os
        assert os.path.exists(os.path.join(tmp, "last.pt"))
        # The saved state_dict must load into a fresh plain model.
        ckpt = torch.load(os.path.join(tmp, "last.pt"), weights_only=False)
        fresh = _model()
        fresh.load_state_dict(ckpt["model_state_dict"])
