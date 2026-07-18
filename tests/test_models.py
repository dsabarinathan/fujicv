"""Unit tests for models module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ---- Backbone tests -------------------------------------------------------

def test_build_resnet50_cnn():
    """Build a ResNet-50 backbone (CNN family) and run a dummy forward pass."""
    from fujicv.models.backbone import build_backbone

    result = build_backbone("resnet50", source="timm", pretrained=False)
    assert "model" in result
    assert result["arch_family"] == "cnn"
    assert isinstance(result["out_features"], int)
    assert result["out_features"] > 0

    model = result["model"]
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    # timm num_classes=0 → global pooled output (B, C)
    assert out.shape == (2, result["out_features"])


def test_build_vit_tiny():
    """Build a ViT-tiny backbone (ViT family) and run a dummy forward pass."""
    from fujicv.models.backbone import build_backbone

    result = build_backbone("vit_tiny_patch16_224", source="timm", pretrained=False)
    assert result["arch_family"] == "vit"
    assert result["out_features"] > 0

    model = result["model"]
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, result["out_features"])


def test_arch_family_inference():
    """Arch family is correctly inferred from model names."""
    from fujicv.models.backbone import _infer_arch_family

    assert _infer_arch_family("resnet50") == "cnn"
    assert _infer_arch_family("efficientnet_b0") == "cnn"
    assert _infer_arch_family("vit_base_patch16_224") == "vit"
    assert _infer_arch_family("swin_tiny_patch4_window7_224") == "vit"
    assert _infer_arch_family("deit_small_patch16_224") == "vit"


# ---- Head tests -----------------------------------------------------------

def test_classification_head():
    from fujicv.models.head import ClassificationHead

    head = ClassificationHead(512, 10, dropout=0.0)
    out = head(torch.zeros(4, 512))
    assert out.shape == (4, 10)


def test_regression_head_single():
    from fujicv.models.head import RegressionHead

    head = RegressionHead(256, num_outputs=1, dropout=0.0)
    out = head(torch.zeros(4, 256))
    assert out.shape == (4,)  # squeezed


def test_regression_head_multi():
    from fujicv.models.head import RegressionHead

    head = RegressionHead(256, num_outputs=3, dropout=0.0)
    out = head(torch.zeros(4, 256))
    assert out.shape == (4, 3)


def test_multilabel_head():
    from fujicv.models.head import MultiLabelHead

    head = MultiLabelHead(128, num_labels=5)
    out = head(torch.zeros(2, 128))
    assert out.shape == (2, 5)


# ---- Custom layer tests ---------------------------------------------------

def test_linear_bn_dropout():
    from fujicv.models.custom_layers import LinearBNDropout

    layer = LinearBNDropout(64, 32, dropout=0.3)
    layer.train()
    out = layer(torch.randn(8, 64))
    assert out.shape == (8, 32)


def test_gem_pool():
    from fujicv.models.custom_layers import GeM

    gem = GeM()
    x = torch.rand(2, 512, 7, 7)
    out = gem(x)
    assert out.shape == (2, 512)


# ---- ModelBuilder tests ---------------------------------------------------

def test_model_builder_classification():
    from fujicv.models.builder import ModelBuilder

    builder = ModelBuilder(
        backbone_name="resnet18",
        backbone_source="timm",
        pretrained=False,
        task="classification",
        num_outputs=5,
        image_size=224,
    )
    model = builder.build()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 5)


def test_model_builder_regression():
    from fujicv.models.builder import ModelBuilder

    builder = ModelBuilder(
        backbone_name="resnet18",
        backbone_source="timm",
        pretrained=False,
        task="regression",
        num_outputs=1,
        image_size=224,
    )
    model = builder.build()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2,)


def test_model_builder_with_custom_layers():
    from fujicv.models.builder import ModelBuilder

    builder = ModelBuilder(
        backbone_name="resnet18",
        backbone_source="timm",
        pretrained=False,
        task="multilabel",
        num_outputs=4,
        custom_layers=[{"type": "LinearBNDropout", "out_features": 256, "dropout": 0.2}],
        image_size=224,
    )
    model = builder.build()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 4)
