"""Unit tests for Test-Time Augmentation (fujicv.inference.tta)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_pil(h=64, w=64):
    return Image.fromarray(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))

def _make_np(h=64, w=64):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


class _IdentityModel(nn.Module):
    """Tiny model that returns fixed logits regardless of input."""
    def __init__(self, num_outputs=3):
        super().__init__()
        self.num_outputs = num_outputs
        self.w = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        B = x.size(0)
        return torch.zeros(B, self.num_outputs) + self.w


class _RegressionModel(nn.Module):
    def forward(self, x):
        return x.mean(dim=[1, 2, 3]).unsqueeze(1)  # (B, 1)


def _val_transform(size=32):
    from fujicv.data.transforms import get_val_transforms
    return get_val_transforms(size)


# ── TTAPredictor — classification ─────────────────────────────────────────────

def test_tta_predict_classification_returns_label_and_conf():
    from fujicv.inference.tta import TTAPredictor

    model = _IdentityModel(num_outputs=3)
    tta = TTAPredictor(
        model=model,
        transform=_val_transform(32),
        task="classification",
        augments="hflip",
        class_to_idx={"cat": 0, "dog": 1, "bird": 2},
        device="cpu",
    )
    label, conf = tta.predict(_make_np())
    assert isinstance(label, str)
    assert 0.0 <= conf <= 1.0


def test_tta_predict_confidence_in_range():
    from fujicv.inference.tta import TTAPredictor

    model = nn.Linear(3 * 32 * 32, 4)
    tta = TTAPredictor(
        model=nn.Sequential(nn.Flatten(), model),
        transform=_val_transform(32),
        task="classification",
        augments="hflip",
        device="cpu",
    )
    _, conf = tta.predict(_make_np())
    assert 0.0 <= conf <= 1.0


def test_tta_pil_input():
    from fujicv.inference.tta import TTAPredictor

    model = _IdentityModel(num_outputs=2)
    tta = TTAPredictor(
        model=model, transform=_val_transform(32), task="classification",
        augments="hflip", class_to_idx={"yes": 0, "no": 1}, device="cpu",
    )
    label, conf = tta.predict(_make_pil())
    assert label in ("yes", "no")


# ── TTAPredictor — regression ─────────────────────────────────────────────────

def test_tta_regression():
    from fujicv.inference.tta import TTAPredictor

    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1))
    tta = TTAPredictor(
        model=model, transform=_val_transform(32), task="regression",
        augments="hflip", device="cpu",
    )
    val, conf = tta.predict(_make_np())
    assert isinstance(val, float)
    assert conf == 1.0


# ── TTAPredictor — multilabel ─────────────────────────────────────────────────

def test_tta_multilabel():
    from fujicv.inference.tta import TTAPredictor

    model = _IdentityModel(num_outputs=4)
    tta = TTAPredictor(
        model=model, transform=_val_transform(32), task="multilabel",
        augments="hflip",
        class_to_idx={"a": 0, "b": 1, "c": 2, "d": 3},
        device="cpu",
    )
    labels, conf = tta.predict(_make_np())
    assert isinstance(labels, list)
    assert 0.0 <= conf <= 1.0


# ── Augment presets ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("preset,expected_views", [
    ("hflip", 2),
    ("hflip_vflip", 3),
    ("rotate", 4),
    ("hflip_rotate", 5),
    ("brightness", 3),
    ("standard", 6),
    ("full", 8),
])
def test_augment_presets_correct_view_count(preset, expected_views):
    from fujicv.inference.tta import TTAPredictor, _PRESET_AUGMENTS

    assert len(_PRESET_AUGMENTS[preset]) == expected_views

    model = _IdentityModel(num_outputs=2)
    tta = TTAPredictor(
        model=model, transform=_val_transform(32), task="classification",
        augments=preset, device="cpu",
    )
    assert len(tta.augments) == expected_views
    label, conf = tta.predict(_make_np())
    assert label in ("0", "1")


def test_invalid_preset_raises():
    from fujicv.inference.tta import TTAPredictor

    model = _IdentityModel()
    with pytest.raises(ValueError, match="Unknown augments preset"):
        TTAPredictor(model=model, transform=_val_transform(), task="classification",
                     augments="nonexistent")


# ── predict_proba ─────────────────────────────────────────────────────────────

def test_predict_proba_shape():
    from fujicv.inference.tta import TTAPredictor

    n_classes = 5
    model = _IdentityModel(num_outputs=n_classes)
    tta = TTAPredictor(
        model=model, transform=_val_transform(32), task="classification",
        augments="hflip", device="cpu",
    )
    proba = tta.predict_proba(_make_np())
    assert proba.shape == (n_classes,)
    assert abs(proba.sum() - 1.0) < 1e-5  # softmax sums to 1


# ── predict_dataset ───────────────────────────────────────────────────────────

def test_predict_dataset(tmp_path):
    from fujicv.inference.tta import TTAPredictor

    paths = []
    for i in range(4):
        p = tmp_path / f"img_{i}.jpg"
        _make_pil().save(p)
        paths.append(p)

    model = _IdentityModel(num_outputs=3)
    tta = TTAPredictor(
        model=model, transform=_val_transform(32), task="classification",
        augments="hflip", class_to_idx={"a": 0, "b": 1, "c": 2}, device="cpu",
    )
    df = tta.predict_dataset(paths)
    assert len(df) == 4
    assert set(df.columns) >= {"path", "prediction", "confidence"}


def test_predict_dataset_with_proba(tmp_path):
    from fujicv.inference.tta import TTAPredictor

    paths = [tmp_path / f"img_{i}.jpg" for i in range(3)]
    for p in paths:
        _make_pil().save(p)

    model = _IdentityModel(num_outputs=2)
    tta = TTAPredictor(
        model=model, transform=_val_transform(32), task="classification",
        augments="hflip", device="cpu",
    )
    df = tta.predict_dataset(paths, return_proba=True)
    assert "proba" in df.columns
    assert df["proba"].iloc[0].shape == (2,)


# ── tta_predict convenience function ─────────────────────────────────────────

def test_tta_predict_function():
    from fujicv.inference.tta import tta_predict

    model = _IdentityModel(num_outputs=3)
    label, conf = tta_predict(
        model=model,
        image_or_path=_make_np(),
        transform=_val_transform(32),
        task="classification",
        augments="hflip",
        class_to_idx={"x": 0, "y": 1, "z": 2},
        device="cpu",
    )
    assert isinstance(label, str)
    assert 0.0 <= conf <= 1.0


# ── merge strategies ─────────────────────────────────────────────────────────

def test_merge_mean_vs_max():
    from fujicv.inference.tta import TTAPredictor

    model = _IdentityModel(num_outputs=3)
    kwargs = dict(model=model, transform=_val_transform(32), task="classification",
                  augments="hflip", device="cpu")

    tta_mean = TTAPredictor(merge="mean", **kwargs)
    tta_max  = TTAPredictor(merge="max",  **kwargs)

    img = _make_np()
    _, conf_mean = tta_mean.predict(img)
    _, conf_max  = tta_max.predict(img)
    # Both should produce valid confidences
    assert 0.0 <= conf_mean <= 1.0
    assert 0.0 <= conf_max  <= 1.0
