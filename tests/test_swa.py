"""Tests for Stochastic Weight Averaging (SWA)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _model():
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3)
    )


def _loader(n: int = 32):
    X = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=8)


def test_swa_creates_averaged_model():
    from fujicv.training.swa import SWA
    model = _model()
    swa   = SWA(model, swa_lr=1e-4)
    assert swa.averaged_model is not None


def test_swa_n_averaged_increments():
    from fujicv.training.swa import SWA
    model = _model()
    swa   = SWA(model, swa_lr=1e-4)
    assert swa.n_averaged == 0
    swa.update()
    assert swa.n_averaged == 1
    swa.update()
    assert swa.n_averaged == 2


def test_swa_averaged_weights_differ_after_update():
    """Averaged weights should differ from the original after multiple updates."""
    from fujicv.training.swa import SWA
    torch.manual_seed(0)
    model     = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loader    = _loader()
    swa       = SWA(model, swa_lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Capture initial averaged params before any training
    swa.update()
    swa_before = {n: p.clone() for n, p in swa.averaged_model.named_parameters()}

    for _ in range(3):
        for X, y in loader:
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()
        swa.update()

    swa_after = {n: p.clone() for n, p in swa.averaged_model.named_parameters()}

    # At least some averaged params should have changed across updates
    diffs = [
        not torch.allclose(swa_before[k], swa_after[k])
        for k in swa_before
    ]
    assert any(diffs), "SWA averaged weights should change across multiple updates"


def test_swa_finalize_runs_without_error():
    from fujicv.training.swa import SWA
    model  = _model()
    loader = _loader()
    swa    = SWA(model, swa_lr=1e-4)
    swa.update()
    swa.finalize(loader)   # should not raise


def test_swa_get_scheduler():
    from fujicv.training.swa import SWA
    import torch.optim.swa_utils as swa_utils

    model     = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    swa       = SWA(model, swa_lr=1e-3)
    scheduler = swa.get_scheduler(optimizer)
    assert isinstance(scheduler, swa_utils.SWALR)


def test_swa_get_scheduler_no_swa_lr_raises():
    from fujicv.training.swa import SWA
    import pytest
    model     = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    swa       = SWA(model)    # no swa_lr at init
    with pytest.raises(ValueError, match="swa_lr must be provided"):
        swa.get_scheduler(optimizer)


def test_swa_state_dict_roundtrip():
    from fujicv.training.swa import SWA
    model = _model()
    swa   = SWA(model, swa_lr=1e-4)
    swa.update()
    sd = swa.state_dict()

    swa2 = SWA(_model(), swa_lr=1e-4)
    swa2.load_state_dict(sd)
    assert swa2.n_averaged == 1
