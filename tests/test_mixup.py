"""Tests for Mixup / CutMix collators."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


NUM_CLASSES = 5


def _batch(n=8, c=3, h=32, w=32):
    imgs    = [torch.randn(c, h, w) for _ in range(n)]
    targets = [i % NUM_CLASSES for i in range(n)]
    return list(zip(imgs, targets))


# ── MixupCollator ──────────────────────────────────────────────────────────────

def test_mixup_output_shape():
    from fujicv.data.mixup import MixupCollator
    col = MixupCollator(alpha=0.4, num_classes=NUM_CLASSES)
    imgs, soft = col(_batch())
    assert imgs.shape  == (8, 3, 32, 32)
    assert soft.shape  == (8, NUM_CLASSES)


def test_mixup_soft_targets_sum_to_one():
    from fujicv.data.mixup import MixupCollator
    col = MixupCollator(alpha=0.4, num_classes=NUM_CLASSES)
    _, soft = col(_batch())
    assert torch.allclose(soft.sum(dim=1), torch.ones(8), atol=1e-5)


def test_mixup_alpha_zero_returns_onehot():
    """alpha=0 → no mixing → pure one-hot targets."""
    from fujicv.data.mixup import MixupCollator
    col = MixupCollator(alpha=0.0, num_classes=NUM_CLASSES)
    _, soft = col(_batch())
    # Each row should be a one-hot vector
    assert torch.all((soft == 0) | (soft == 1))


def test_mixup_prob_zero_no_mixing():
    from fujicv.data.mixup import MixupCollator
    col = MixupCollator(alpha=0.4, num_classes=NUM_CLASSES, prob=0.0)
    batch = _batch()
    imgs, _ = col(batch)
    # Images should be the stack of originals unchanged
    originals = torch.stack([b[0] for b in batch])
    assert torch.allclose(imgs, originals)


def test_mixup_invalid_alpha():
    from fujicv.data.mixup import MixupCollator
    with pytest.raises(ValueError):
        MixupCollator(alpha=-0.1, num_classes=NUM_CLASSES)


# ── CutMixCollator ─────────────────────────────────────────────────────────────

def test_cutmix_output_shape():
    from fujicv.data.mixup import CutMixCollator
    col = CutMixCollator(alpha=1.0, num_classes=NUM_CLASSES)
    imgs, soft = col(_batch())
    assert imgs.shape == (8, 3, 32, 32)
    assert soft.shape == (8, NUM_CLASSES)


def test_cutmix_soft_targets_sum_to_one():
    from fujicv.data.mixup import CutMixCollator
    col = CutMixCollator(alpha=1.0, num_classes=NUM_CLASSES)
    _, soft = col(_batch())
    assert torch.allclose(soft.sum(dim=1), torch.ones(8), atol=1e-5)


def test_cutmix_invalid_alpha():
    from fujicv.data.mixup import CutMixCollator
    with pytest.raises(ValueError):
        CutMixCollator(alpha=0.0, num_classes=NUM_CLASSES)


def test_cutmix_prob_zero_no_mixing():
    from fujicv.data.mixup import CutMixCollator
    col = CutMixCollator(alpha=1.0, num_classes=NUM_CLASSES, prob=0.0)
    batch = _batch()
    imgs, soft = col(batch)
    originals = torch.stack([b[0] for b in batch])
    assert torch.allclose(imgs, originals)
    assert torch.all((soft == 0) | (soft == 1))


# ── MixupCutMixCollator ───────────────────────────────────────────────────────

def test_mixupcutmix_output_shape():
    from fujicv.data.mixup import MixupCutMixCollator
    col = MixupCutMixCollator(num_classes=NUM_CLASSES)
    imgs, soft = col(_batch())
    assert imgs.shape == (8, 3, 32, 32)
    assert soft.shape == (8, NUM_CLASSES)


def test_mixupcutmix_soft_targets_sum_to_one():
    from fujicv.data.mixup import MixupCutMixCollator
    col = MixupCutMixCollator(num_classes=NUM_CLASSES, mixup_prob=0.5, cutmix_prob=0.5)
    _, soft = col(_batch())
    assert torch.allclose(soft.sum(dim=1), torch.ones(8), atol=1e-5)
