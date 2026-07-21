"""Unit tests for fujicv.data.hf_dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_pil(h=32, w=32):
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class _FakeHFDataset:
    """Minimal stand-in for datasets.Dataset used in unit tests.

    Supports both integer indexing (ds[0]) and column access (ds["col"]),
    matching the real HuggingFace datasets.Dataset API.
    """

    def __init__(self, records, features=None):
        self._records = records
        self.features = features or {}

    def __len__(self):
        return len(self._records)

    def __getitem__(self, key):
        if isinstance(key, str):
            # Column access: return list of values for that column
            return [r[key] for r in self._records]
        return self._records[key]

    def get(self, *args):  # noqa: ARG002
        return None

    def train_test_split(self, test_size=0.2, seed=42):  # noqa: ARG002
        n_test = max(1, int(len(self._records) * test_size))
        return {
            "train": _FakeHFDataset(self._records[n_test:], self.features),
            "test":  _FakeHFDataset(self._records[:n_test], self.features),
        }


class _FakeClassLabel:
    def __init__(self, names):
        self.names = names


# ── HFImageDataset ────────────────────────────────────────────────────────────

def test_hf_image_dataset_classification_pil():
    """Returns (image_tensor, long_label) for classification with PIL images."""
    from fujicv.data.hf_dataset import HFImageDataset

    records = [{"image": _make_pil(), "label": i % 3} for i in range(12)]
    features = {"label": _FakeClassLabel(["angular_leaf_spot", "bean_rust", "healthy"])}
    hf_ds = _FakeHFDataset(records, features)

    ds = HFImageDataset(hf_ds, image_col="image", label_col="label", task="classification")
    assert len(ds) == 12

    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 32, 32)
    assert label.dtype == torch.long
    assert 0 <= label.item() < 3


def test_hf_image_dataset_regression():
    """Returns float label for regression task."""
    from fujicv.data.hf_dataset import HFImageDataset

    records = [{"image": _make_pil(), "score": float(i) * 0.1} for i in range(8)]
    hf_ds = _FakeHFDataset(records)

    ds = HFImageDataset(hf_ds, image_col="image", label_col="score", task="regression")
    _, label = ds[0]
    assert label.dtype == torch.float32


def test_hf_image_dataset_multilabel():
    """Returns float vector for multilabel task via label_names."""
    from fujicv.data.hf_dataset import HFImageDataset

    records = [
        {"image": _make_pil(), "a": 1, "b": 0, "c": 1}
        for _ in range(6)
    ]
    hf_ds = _FakeHFDataset(records)

    ds = HFImageDataset(
        hf_ds, image_col="image", label_col="a",
        task="multilabel", label_names=["a", "b", "c"]
    )
    _, label = ds[0]
    assert label.shape == (3,)
    assert label.dtype == torch.float32


def test_hf_image_dataset_with_transform():
    """Transform is applied when provided."""
    from fujicv.data.hf_dataset import HFImageDataset
    from fujicv.data.transforms import get_val_transforms

    records = [{"image": _make_pil(64, 64), "label": 0}]
    hf_ds = _FakeHFDataset(records, {"label": _FakeClassLabel(["cat"])})

    tfm = get_val_transforms(32)
    ds = HFImageDataset(hf_ds, transform=tfm, task="classification")
    img, _ = ds[0]
    assert img.shape == (3, 32, 32)


def test_hf_image_dataset_invalid_task():
    """Raises ValueError for unrecognised task."""
    from fujicv.data.hf_dataset import HFImageDataset

    records = [{"image": _make_pil(), "label": 0}]
    hf_ds = _FakeHFDataset(records)

    with pytest.raises(ValueError, match="task"):
        HFImageDataset(hf_ds, task="ordinal")


def test_hf_image_dataset_string_labels_no_class_label_feature():
    """Builds class_to_idx automatically when feature has no .names."""
    from fujicv.data.hf_dataset import HFImageDataset

    records = [
        {"image": _make_pil(), "label": "cat"},
        {"image": _make_pil(), "label": "dog"},
        {"image": _make_pil(), "label": "cat"},
    ]
    hf_ds = _FakeHFDataset(records)

    ds = HFImageDataset(hf_ds, task="classification")
    assert "cat" in ds.class_to_idx
    assert "dog" in ds.class_to_idx
    _, label = ds[0]
    assert label.dtype == torch.long


# ── load_hf_dataset (ImportError path) ──────────────────────────────────────

def test_load_hf_dataset_missing_package(monkeypatch):
    """Raises ImportError with install instructions when datasets is absent."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    import importlib
    from fujicv.data import hf_dataset as _mod
    importlib.reload(_mod)

    with pytest.raises(ImportError, match="datasets"):
        _mod.load_hf_dataset("beans")
