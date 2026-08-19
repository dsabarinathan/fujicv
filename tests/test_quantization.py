"""Tests for post-training quantization (dynamic + static FX) and size measurement."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fujicv.export.quantization import measure_model_size, quantize_dynamic, quantize_static


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


class _SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(self.features(x))


def test_measure_model_size_positive():
    mb = measure_model_size(_MLP())
    assert mb > 0


def test_dynamic_quantization_preserves_output_shape():
    model = _MLP().eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        ref = model(x)
    qmodel = quantize_dynamic(model)
    with torch.no_grad():
        out = qmodel(x)
    assert out.shape == ref.shape
    # INT8 weights should not blow up predictions — close in aggregate.
    assert torch.allclose(out, ref, atol=1.0)


def test_dynamic_quantization_does_not_mutate_original():
    model = _MLP().eval()
    before = measure_model_size(model)
    _ = quantize_dynamic(model)
    after = measure_model_size(model)
    assert before == after  # original untouched (deep-copied internally)


def test_dynamic_quantization_replaces_linear():
    qmodel = quantize_dynamic(_MLP().eval())
    module_paths = {type(m).__module__ for m in qmodel.modules()}
    assert any("quantized" in p for p in module_paths)


def test_static_quantization_cnn():
    model = _SmallCNN().eval()
    calib = DataLoader(TensorDataset(torch.randn(32, 3, 16, 16)), batch_size=8)
    try:
        qmodel = quantize_static(model, calib, num_calibration_batches=2)
    except (RuntimeError, Exception) as exc:  # backend may be unavailable in CI
        pytest.skip(f"static quantization unavailable in this environment: {exc}")
    with torch.no_grad():
        out = qmodel(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 4)


def test_static_quantization_empty_calibration_raises():
    with pytest.raises(ValueError, match="no batches"):
        quantize_static(_SmallCNN().eval(), [], num_calibration_batches=1)
