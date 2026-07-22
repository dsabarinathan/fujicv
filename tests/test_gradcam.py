"""Tests for Grad-CAM and Grad-CAM++."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


def _tiny_cnn():
    """Minimal CNN with a named conv layer for hook testing."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 3),
    )


def test_gradcam_heatmap_shape_from_tensor():
    from fujicv.eval.gradcam import GradCAM
    model  = _tiny_cnn()
    target = model[0]   # first conv layer
    cam    = GradCAM(model, target_layer=target, use_cuda=False)

    image  = torch.randn(1, 3, 32, 32)
    heatmap = cam.generate(image, target_class=0)
    cam.remove_hooks()

    assert heatmap.shape == (1, 32, 32) or heatmap.ndim == 2


def test_gradcam_heatmap_range():
    from fujicv.eval.gradcam import GradCAM
    model   = _tiny_cnn()
    cam_gen = GradCAM(model, target_layer=model[0], use_cuda=False)
    image   = torch.randn(1, 3, 32, 32)
    heatmap = cam_gen.generate(image, target_class=1)
    cam_gen.remove_hooks()

    assert heatmap.min() >= -1e-5
    assert heatmap.max() <= 1.0 + 1e-5


def test_gradcam_default_target_class():
    """target_class=None should pick argmax class without error."""
    from fujicv.eval.gradcam import GradCAM
    model   = _tiny_cnn()
    cam_gen = GradCAM(model, target_layer=model[0], use_cuda=False)
    image   = torch.randn(1, 3, 32, 32)
    heatmap = cam_gen.generate(image)   # no target_class
    cam_gen.remove_hooks()
    assert heatmap is not None


def test_gradcam_hook_removed():
    from fujicv.eval.gradcam import GradCAM
    model   = _tiny_cnn()
    cam_gen = GradCAM(model, target_layer=model[0], use_cuda=False)
    cam_gen.remove_hooks()
    assert len(cam_gen._hooks) == 0


def test_gradcampp_heatmap_shape():
    from fujicv.eval.gradcam import GradCAMPlusPlus
    model   = _tiny_cnn()
    cam_gen = GradCAMPlusPlus(model, target_layer=model[0], use_cuda=False)
    image   = torch.randn(1, 3, 32, 32)
    heatmap = cam_gen.generate(image, target_class=2)
    cam_gen.remove_hooks()
    assert heatmap.ndim == 2


def test_overlay_heatmap_shape():
    """overlay_heatmap requires opencv; skip if not installed."""
    pytest.importorskip("cv2", reason="opencv-python not installed")
    from fujicv.eval.gradcam import overlay_heatmap
    image   = np.ones((64, 64, 3), dtype=np.uint8) * 128
    heatmap = np.random.rand(64, 64).astype(np.float32)
    result  = overlay_heatmap(image, heatmap, alpha=0.5)
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8
