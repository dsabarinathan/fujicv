"""Tests for expanded ModelBuilder custom head blocks and pooling options."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fujicv.models.builder import _AssembledModel
from fujicv.models.custom_layers import AttentionPool, GeM
from fujicv.models.head import ClassificationHead


class _CNNStub(nn.Module):
    """Backbone stub returning a 4-D feature map (B, C, H, W)."""

    def __init__(self, c: int = 16) -> None:
        super().__init__()
        self.c = c
        self.conv = nn.Conv2d(3, c, 3, padding=1)

    def forward(self, x):
        return self.conv(x)  # (B, c, H, W)


class _TokenStub(nn.Module):
    """Backbone stub returning 3-D transformer tokens (B, N, C)."""

    def __init__(self, c: int = 16, n: int = 5) -> None:
        super().__init__()
        self.c, self.n = c, n
        self.proj = nn.Linear(3, c)

    def forward(self, x):
        b = x.size(0)
        # Fake N tokens of width c.
        return self.proj(torch.randn(b, self.n, 3))


def _assemble(custom_seq, pool, pool_type, in_features=16, backbone=None):
    head = ClassificationHead(in_features=in_features, num_classes=4)
    bb = backbone or _CNNStub(in_features)
    return _AssembledModel(bb, "cnn", custom_seq, head, pool=pool, pool_type=pool_type)


def test_avg_pool_default_shape():
    m = _assemble(nn.Sequential(), nn.AdaptiveAvgPool2d(1), "avg").eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 4)


def test_gem_pool_shape():
    m = _assemble(nn.Sequential(), GeM(), "gem").eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 4)


def test_attention_pool_spatial_and_tokens():
    # Spatial (4-D)
    m = _assemble(nn.Sequential(), AttentionPool(16), "attention").eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, 16, 16)).shape == (2, 4)
    # Tokens (3-D)
    m2 = _assemble(nn.Sequential(), AttentionPool(16), "attention", backbone=_TokenStub(16)).eval()
    with torch.no_grad():
        assert m2(torch.randn(2, 3, 16, 16)).shape == (2, 4)


# ── Full ModelBuilder integration (uses a real tiny timm backbone) ──────────

pytest.importorskip("timm")


def _build(**kwargs):
    from fujicv.models.builder import ModelBuilder

    return ModelBuilder(
        backbone_name="resnet18",
        backbone_source="timm",
        pretrained=False,
        task="classification",
        num_outputs=5,
        image_size=32,
        **kwargs,
    ).build()


def test_builder_rejects_bad_pooling():
    from fujicv.models.builder import ModelBuilder

    with pytest.raises(ValueError, match="pooling must be"):
        ModelBuilder("resnet18", pooling="banana")


def test_builder_gem_pooling_builds_and_runs():
    model = _build(pooling="gem").eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, 5)
    assert any(isinstance(m, GeM) for m in model.modules())


def test_builder_custom_head_blocks():
    model = _build(
        custom_layers=[
            {"type": "Linear", "out_features": 128},
            {"type": "LayerNorm"},
            {"type": "Activation", "fn": "gelu"},
            {"type": "Dropout", "p": 0.2},
        ]
    ).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, 5)
    mods = {type(m).__name__ for m in model.custom_layers}
    assert {"Linear", "LayerNorm", "GELU", "Dropout"}.issubset(mods)


def test_builder_unknown_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        _build(custom_layers=[{"type": "Activation", "fn": "nope"}])


def test_builder_unknown_block_raises():
    with pytest.raises(ValueError, match="Unknown custom layer type"):
        _build(custom_layers=[{"type": "Nonexistent"}])
