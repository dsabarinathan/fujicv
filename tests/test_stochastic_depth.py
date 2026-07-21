"""Tests for DropPath / Stochastic Depth."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ── DropPath ──────────────────────────────────────────────────────────────────

def test_droppath_identity_at_eval():
    from fujicv.models.stochastic_depth import DropPath
    dp = DropPath(drop_prob=0.5)
    dp.eval()
    x = torch.randn(4, 16, 7, 7)
    assert torch.allclose(dp(x), x)


def test_droppath_zero_prob_identity_train():
    from fujicv.models.stochastic_depth import DropPath
    dp = DropPath(drop_prob=0.0)
    dp.train()
    x = torch.randn(4, 16)
    assert torch.allclose(dp(x), x)


def test_droppath_drops_samples_train():
    """With high drop_prob, some outputs should differ from input."""
    from fujicv.models.stochastic_depth import DropPath
    torch.manual_seed(0)
    dp = DropPath(drop_prob=0.9)
    dp.train()
    x = torch.ones(64, 8)
    out = dp(x)
    # Not all rows should be identical to input (some dropped, some scaled)
    assert not torch.allclose(out, x)


def test_droppath_output_shape_preserved():
    from fujicv.models.stochastic_depth import DropPath
    dp = DropPath(drop_prob=0.2)
    dp.train()
    x = torch.randn(8, 32, 14, 14)
    assert dp(x).shape == x.shape


def test_droppath_invalid_prob():
    from fujicv.models.stochastic_depth import DropPath
    with pytest.raises(ValueError):
        DropPath(drop_prob=1.0)
    with pytest.raises(ValueError):
        DropPath(drop_prob=-0.1)


def test_droppath_extra_repr():
    from fujicv.models.stochastic_depth import DropPath
    dp = DropPath(0.15)
    assert "0.15" in repr(dp)


# ── Schedule ──────────────────────────────────────────────────────────────────

def test_stochastic_depth_schedule_length():
    from fujicv.models.stochastic_depth import build_stochastic_depth_schedule
    rates = build_stochastic_depth_schedule(12, max_drop_rate=0.3)
    assert len(rates) == 12


def test_stochastic_depth_schedule_starts_at_zero():
    from fujicv.models.stochastic_depth import build_stochastic_depth_schedule
    rates = build_stochastic_depth_schedule(6, max_drop_rate=0.2)
    assert rates[0] == pytest.approx(0.0)
    assert rates[-1] == pytest.approx(0.2)


def test_stochastic_depth_schedule_single():
    from fujicv.models.stochastic_depth import build_stochastic_depth_schedule
    rates = build_stochastic_depth_schedule(1, max_drop_rate=0.1)
    assert rates == [0.1]


def test_stochastic_depth_schedule_empty():
    from fujicv.models.stochastic_depth import build_stochastic_depth_schedule
    assert build_stochastic_depth_schedule(0) == []


# ── ModelBuilder integration ──────────────────────────────────────────────────

@pytest.mark.skip(reason="onnxruntime DLL crash in this env when timm/torchvision load together")
def test_model_builder_with_drop_path_rate():
    """drop_path_rate flows through ModelBuilder → build_backbone → timm."""
    from fujicv.models.builder import ModelBuilder

    model = ModelBuilder(
        backbone_name="resnet18",
        backbone_source="timm",
        pretrained=False,
        task="classification",
        num_outputs=3,
        image_size=64,
        drop_path_rate=0.1,
    ).build()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 3)


# ── DropPath re-exported from custom_layers ───────────────────────────────────

def test_droppath_accessible_from_custom_layers():
    from fujicv.models.custom_layers import DropPath
    dp = DropPath(0.1)
    assert isinstance(dp, nn.Module)
