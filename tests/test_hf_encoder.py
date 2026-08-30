"""Tests for generic HF image-encoder support: vision-tower resolution,
output-attribute extraction, and processor size parsing.

These exercise the pure-Python routing logic with stubs, so they run without
`transformers` installed (which mirrors CI)."""
from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from fujicv.data.transforms import _resolve_processor_size, get_hf_transforms
from fujicv.models.backbone import _HFBackboneWrapper, _resolve_vision_encoder


# ── vision-tower resolution (CLIP / SigLIP style) ────────────────────────────
class _VisionTower(nn.Module):
    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        # (B, N, C) tokens with a pooled output, like CLIP/SigLIP vision towers.
        return types.SimpleNamespace(
            last_hidden_state=torch.zeros(b, 5, 16),
            pooler_output=torch.zeros(b, 16),
        )


class _MultimodalModel(nn.Module):
    """Mimics CLIPModel/SiglipModel: forward needs text too; vision in submodule."""

    def __init__(self):
        super().__init__()
        self.vision_model = _VisionTower()

    def forward(self, **kwargs):
        raise RuntimeError("multimodal forward needs input_ids + pixel_values")


class _PlainViT(nn.Module):
    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        return types.SimpleNamespace(last_hidden_state=torch.zeros(b, 5, 16))


def test_resolve_vision_encoder_unwraps_multimodal():
    model = _MultimodalModel()
    enc = _resolve_vision_encoder(model)
    assert enc is model.vision_model


def test_resolve_vision_encoder_passthrough_plain():
    model = _PlainViT()
    assert _resolve_vision_encoder(model) is model


def test_wrapper_extracts_last_hidden_state():
    w = _HFBackboneWrapper(_PlainViT())
    out = w(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 5, 16)  # (B, N, C) tokens


def test_wrapper_on_resolved_vision_tower():
    enc = _resolve_vision_encoder(_MultimodalModel())
    out = _HFBackboneWrapper(enc)(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 5, 16)


class _PoolerOnly(nn.Module):
    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        return types.SimpleNamespace(last_hidden_state=None, pooler_output=torch.zeros(b, 16))


def test_wrapper_falls_back_to_pooler_output():
    out = _HFBackboneWrapper(_PoolerOnly())(torch.zeros(3, 3, 224, 224))
    assert out.shape == (3, 16)


class _TupleOut(nn.Module):
    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        return (torch.zeros(b, 8, 32),)  # bare tuple, no attributes


def test_wrapper_falls_back_to_tuple():
    out = _HFBackboneWrapper(_TupleOut())(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 8, 32)


# ── processor size parsing ───────────────────────────────────────────────────
@pytest.mark.parametrize("size,expected", [
    ({"height": 384, "width": 384}, 384),
    ({"shortest_edge": 256}, 256),
    (224, 224),
    (None, 224),          # default
    ({"unexpected": 1}, 224),
])
def test_resolve_processor_size(size, expected):
    assert _resolve_processor_size(size) == expected


def test_get_hf_transforms_requires_transformers():
    import importlib.util
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers installed; ImportError path not exercised")
    with pytest.raises(ImportError, match="transformers is required"):
        get_hf_transforms("facebook/dinov2-base")


# ── custom mean/std threading into the base transforms ───────────────────────
def test_custom_mean_std_in_transforms():
    import numpy as np

    from fujicv.data.transforms import get_val_transforms

    # SigLIP-style 0.5/0.5 normalization → a mid-gray pixel maps to ~0.
    tf = get_val_transforms(32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    out = tf(image=img)["image"]
    assert out.shape == (3, 32, 32)
    assert abs(float(out.mean())) < 0.05  # ~0 after 0.5/0.5 normalization
