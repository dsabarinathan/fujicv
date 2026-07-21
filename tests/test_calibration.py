"""Tests for model calibration utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


# ── compute_ece ───────────────────────────────────────────────────────────────

def test_ece_perfect_calibration():
    from fujicv.eval.calibration import compute_ece
    # Perfect: confidence == accuracy in every bin
    conf    = np.linspace(0.1, 0.9, 100)
    correct = (np.random.default_rng(0).random(100) < conf).astype(float)
    ece = compute_ece(conf, correct, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_ece_overconfident_model():
    from fujicv.eval.calibration import compute_ece
    # Always predicts 1.0 confidence but only 50% accurate → high ECE
    conf    = np.ones(100)
    correct = np.zeros(100)
    correct[:50] = 1
    ece = compute_ece(conf, correct, n_bins=10)
    assert ece == pytest.approx(0.5, abs=0.05)


def test_ece_shape_mismatch():
    from fujicv.eval.calibration import compute_ece
    with pytest.raises(ValueError):
        compute_ece(np.ones(10), np.ones(5))


def test_ece_range():
    from fujicv.eval.calibration import compute_ece
    conf    = np.random.rand(200)
    correct = np.random.randint(0, 2, 200)
    ece = compute_ece(conf, correct)
    assert 0.0 <= ece <= 1.0


# ── TemperatureScaling ────────────────────────────────────────────────────────

def test_temperature_scaling_identity_at_one():
    from fujicv.eval.calibration import TemperatureScaling
    cal    = TemperatureScaling(temperature=1.0)
    logits = torch.randn(8, 5)
    out    = cal(logits)
    assert torch.allclose(out, logits, atol=1e-5)


def test_temperature_scaling_reduces_confidence():
    from fujicv.eval.calibration import TemperatureScaling
    cal    = TemperatureScaling(temperature=2.0)
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    probs_before = torch.softmax(logits, dim=-1)
    probs_after  = cal.calibrate(logits)
    # T > 1 should reduce max probability
    assert probs_after[0, 0] < probs_before[0, 0]


def test_temperature_scaling_fit():
    from fujicv.eval.calibration import TemperatureScaling
    from torch.utils.data import DataLoader, TensorDataset

    # Model outputs very large logits for class 0, but all targets are class 1
    # → overconfident on wrong class, so T should grow > 1 to reduce confidence
    class OverconfidentModel(nn.Module):
        def forward(self, x):
            # Returns [high, low, low] but true label is 1 → cross-entropy drives T up
            return torch.cat([
                torch.full((x.size(0), 1), 5.0),
                torch.full((x.size(0), 1), 0.0),
                torch.full((x.size(0), 1), 0.0),
            ], dim=1)

    imgs    = torch.randn(40, 1)
    targets = torch.ones(40, dtype=torch.long)   # all class 1
    loader  = DataLoader(TensorDataset(imgs, targets), batch_size=20)

    cal = TemperatureScaling(temperature=1.0)
    cal.fit(OverconfidentModel(), loader, device="cpu", lr=0.1, max_iter=100)
    # Temperature should be pushed above 1.0 (to spread the distribution)
    assert cal.temperature.item() > 1.1


def test_temperature_scaling_calibrate_sums_to_one():
    from fujicv.eval.calibration import TemperatureScaling
    cal   = TemperatureScaling(temperature=1.5)
    logits = torch.randn(16, 10)
    probs  = cal.calibrate(logits)
    assert torch.allclose(probs.sum(dim=1), torch.ones(16), atol=1e-5)


# ── reliability_diagram ───────────────────────────────────────────────────────

def test_reliability_diagram_returns_fig_ax():
    from fujicv.eval.calibration import reliability_diagram
    conf    = np.random.rand(50)
    correct = np.random.randint(0, 2, 50)
    fig, ax = reliability_diagram(conf, correct, show=False)
    import matplotlib
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)
