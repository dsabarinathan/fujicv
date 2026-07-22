"""Tests for RandAugment and RandAugmentTransform."""

from __future__ import annotations

import numpy as np
import pytest


def _img(h=64, w=64):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


# ── RandAugment ───────────────────────────────────────────────────────────────

def test_randaugment_output_shape():
    from fujicv.data.autoaugment import RandAugment
    aug = RandAugment(n=2, magnitude=9)
    out = aug(_img())
    assert out.shape == (64, 64, 3)


def test_randaugment_output_dtype():
    from fujicv.data.autoaugment import RandAugment
    aug = RandAugment(n=2, magnitude=5)
    out = aug(_img())
    assert out.dtype == np.uint8


def test_randaugment_prob_zero_returns_original():
    from fujicv.data.autoaugment import RandAugment
    aug = RandAugment(n=3, magnitude=9, prob=0.0)
    img = _img()
    out = aug(img)
    np.testing.assert_array_equal(out, img)


def test_randaugment_different_n_values():
    from fujicv.data.autoaugment import RandAugment
    for n in [1, 2, 3, 5]:
        aug = RandAugment(n=n, magnitude=7)
        out = aug(_img())
        assert out.shape == (64, 64, 3)


def test_randaugment_invalid_magnitude():
    from fujicv.data.autoaugment import RandAugment
    with pytest.raises(ValueError):
        RandAugment(n=2, magnitude=11)
    with pytest.raises(ValueError):
        RandAugment(n=2, magnitude=-1)


def test_randaugment_invalid_n():
    from fujicv.data.autoaugment import RandAugment
    with pytest.raises(ValueError):
        RandAugment(n=0, magnitude=9)


def test_randaugment_repr():
    from fujicv.data.autoaugment import RandAugment
    aug = RandAugment(n=2, magnitude=9)
    assert "2" in repr(aug) and "9" in repr(aug)


def test_randaugment_deterministic_with_seed():
    from fujicv.data.autoaugment import RandAugment
    import random
    aug = RandAugment(n=2, magnitude=9, magnitude_std=0)
    img = _img()
    random.seed(42)
    np.random.seed(42)
    out1 = aug(img.copy())
    random.seed(42)
    np.random.seed(42)
    out2 = aug(img.copy())
    np.testing.assert_array_equal(out1, out2)


# ── RandAugmentTransform (albumentations interface) ───────────────────────────

def test_randaugment_transform_interface():
    from fujicv.data.autoaugment import RandAugmentTransform
    t = RandAugmentTransform(n=2, magnitude=9)
    img = _img()
    result = t(image=img)
    assert "image" in result
    assert result["image"].shape == (64, 64, 3)


def test_randaugment_transform_passes_extra_keys():
    from fujicv.data.autoaugment import RandAugmentTransform
    t   = RandAugmentTransform(n=1, magnitude=5)
    img = _img()
    result = t(image=img, mask=np.zeros((64, 64), dtype=np.uint8))
    assert "mask" in result
