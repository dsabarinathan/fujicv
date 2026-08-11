"""Tests for LR Finder."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _simple_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))


def _loader(n: int = 32):
    X = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, 4, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=8)


def test_lr_finder_history_populated():
    from fujicv.training.lr_finder import LRFinder
    model     = _simple_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    criterion = nn.CrossEntropyLoss()
    finder    = LRFinder(model, optimizer, criterion)
    finder.range_test(_loader(), start_lr=1e-7, end_lr=1.0, num_iter=20)
    assert len(finder.history["lr"])   > 0
    assert len(finder.history["loss"]) > 0


def test_lr_finder_lr_increases():
    from fujicv.training.lr_finder import LRFinder
    model     = _simple_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    finder    = LRFinder(model, optimizer, nn.CrossEntropyLoss())
    finder.range_test(_loader(), start_lr=1e-7, end_lr=1.0, num_iter=20)
    lrs = finder.history["lr"]
    assert lrs[-1] > lrs[0], "LR should increase monotonically"


def test_lr_finder_suggestion_in_range():
    from fujicv.training.lr_finder import LRFinder
    model     = _simple_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    finder    = LRFinder(model, optimizer, nn.CrossEntropyLoss())
    finder.range_test(_loader(), start_lr=1e-7, end_lr=1.0, num_iter=30)
    best_lr = finder.suggestion()
    assert 1e-8 < best_lr < 10.0


def test_lr_finder_reset_restores_model():
    from fujicv.training.lr_finder import LRFinder
    model     = _simple_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    finder    = LRFinder(model, optimizer, nn.CrossEntropyLoss())

    params_before = {n: p.clone() for n, p in model.named_parameters()}
    finder.range_test(_loader(), start_lr=1e-7, end_lr=1.0, num_iter=20)
    # After range_test the model is already restored
    params_after = {n: p.clone() for n, p in model.named_parameters()}
    for name in params_before:
        assert torch.allclose(params_before[name], params_after[name]), (
            f"Parameter {name} changed after range_test (should be restored)"
        )


def test_lr_finder_optimizer_lr_restored():
    from fujicv.training.lr_finder import LRFinder
    model     = _simple_model()
    init_lr   = 5e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
    finder    = LRFinder(model, optimizer, nn.CrossEntropyLoss())
    finder.range_test(_loader(), start_lr=1e-7, end_lr=1.0, num_iter=20)
    restored_lr = optimizer.param_groups[0]["lr"]
    assert abs(restored_lr - init_lr) < 1e-9, (
        f"Optimizer LR not restored: expected {init_lr}, got {restored_lr}"
    )


def test_lr_finder_plot_returns_figure():
    import matplotlib

    from fujicv.training.lr_finder import LRFinder
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model     = _simple_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    finder    = LRFinder(model, optimizer, nn.CrossEntropyLoss())
    finder.range_test(_loader(), start_lr=1e-7, end_lr=1.0, num_iter=30)
    fig = finder.plot()
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_lr_finder_cycles_short_loader():
    """range_test must cycle a short loader to reach num_iter."""
    from fujicv.training.lr_finder import LRFinder
    # Only 2 batches in the loader
    X = torch.randn(16, 3, 8, 8)
    y = torch.randint(0, 4, (16,))
    tiny_loader = DataLoader(TensorDataset(X, y), batch_size=8)

    model     = _simple_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    finder    = LRFinder(model, optimizer, nn.CrossEntropyLoss())
    finder.range_test(tiny_loader, start_lr=1e-7, end_lr=1.0, num_iter=10)
    assert len(finder.history["lr"]) > 0
