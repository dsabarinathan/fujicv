"""Tests for Hugging Face backbone support in build_backbone / ModelBuilder."""
from __future__ import annotations

import importlib.util

import pytest

from fujicv.models.backbone import build_backbone

_HAS_HF = importlib.util.find_spec("transformers") is not None


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown source"):
        build_backbone("resnet50", source="not-a-zoo")


def test_hf_rejects_non_rgb_input():
    # in_chans validation happens before any network download, so this is
    # exercised even without transformers installed.
    if not _HAS_HF:
        with pytest.raises(ImportError, match="transformers is required"):
            build_backbone("google/vit-base-patch16-224", source="hf", in_chans=1)
    else:
        with pytest.raises(ValueError, match="3-channel"):
            build_backbone("google/vit-base-patch16-224", source="hf", in_chans=1)


def test_hf_features_only_unsupported():
    with pytest.raises(ValueError, match="features_only is not supported"):
        build_backbone("google/vit-base-patch16-224", source="hf", features_only=True)


def test_hf_missing_dependency_message():
    if _HAS_HF:
        pytest.skip("transformers is installed; dependency-error path not exercised")
    with pytest.raises(ImportError, match="fujicv\\[hf-models\\]"):
        build_backbone("google/vit-base-patch16-224", source="hf")


@pytest.mark.skipif(not _HAS_HF, reason="transformers not installed")
def test_hf_vit_backbone_builds():
    from fujicv.models.builder import ModelBuilder

    model = ModelBuilder(
        backbone_name="hf-internal-testing/tiny-random-ViTModel",
        backbone_source="hf",
        pretrained=True,
        task="classification",
        num_outputs=4,
        image_size=224,
    ).build()

    import torch

    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 4)
