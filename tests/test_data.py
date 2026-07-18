"""Unit tests for data module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def _make_toy_dataset(tmp_dir: Path, n: int = 20):
    """Create a tiny toy image dataset with a CSV."""
    img_dir = tmp_dir / "images"
    img_dir.mkdir()
    rows = []
    for i in range(n):
        label = "cat" if i % 2 == 0 else "dog"
        fname = f"img_{i:04d}.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(img_dir / fname)
        rows.append({"filepath": fname, "label": label})
    df = pd.DataFrame(rows)
    csv_path = tmp_dir / "data.csv"
    df.to_csv(csv_path, index=False)
    return df, img_dir, csv_path


# ---- CSVImageDataset tests ------------------------------------------------

def test_csv_image_dataset_classification():
    from fujicv.data.datasets import CSVImageDataset
    from fujicv.data.transforms import get_val_transforms

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df, img_dir, _ = _make_toy_dataset(tmp)
        transform = get_val_transforms(64)
        ds = CSVImageDataset(df, img_dir, "filepath", "label", "classification", transform)
        assert len(ds) == len(df)
        img, label = ds[0]
        assert img.shape == (3, 64, 64)
        assert label.dtype == __import__("torch").long


def test_csv_image_dataset_skips_missing():
    from fujicv.data.datasets import CSVImageDataset

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df, img_dir, _ = _make_toy_dataset(tmp, n=5)
        # Add a row with a non-existent image
        bad_row = pd.DataFrame([{"filepath": "nonexistent.jpg", "label": "cat"}])
        df_with_bad = pd.concat([df, bad_row], ignore_index=True)
        ds = CSVImageDataset(df_with_bad, img_dir, "filepath", "label", "classification")
        # Should exclude the missing file
        assert len(ds) == len(df)


def test_csv_image_dataset_regression():
    from fujicv.data.datasets import CSVImageDataset
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_dir = tmp / "images"
        img_dir.mkdir()
        rows = []
        for i in range(10):
            fname = f"img_{i}.jpg"
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(img_dir / fname)
            rows.append({"filepath": fname, "score": float(i) * 0.1})
        df = pd.DataFrame(rows)
        ds = CSVImageDataset(df, img_dir, "filepath", "score", "regression")
        _, label = ds[0]
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.float32


# ---- Transform tests ------------------------------------------------------

def test_get_train_transforms_levels():
    from fujicv.data.transforms import get_train_transforms

    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    for level in ("light", "medium", "heavy"):
        tfm = get_train_transforms(64, level=level)
        out = tfm(image=img)
        assert out["image"].shape == (3, 64, 64), f"Failed for level={level}"


def test_get_train_transforms_invalid_level():
    from fujicv.data.transforms import get_train_transforms

    with pytest.raises(ValueError, match="level"):
        get_train_transforms(224, level="extreme")


def test_get_val_transforms():
    from fujicv.data.transforms import get_val_transforms

    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    tfm = get_val_transforms(224)
    out = tfm(image=img)
    assert out["image"].shape == (3, 224, 224)


# ---- build_splits tests ---------------------------------------------------

def test_build_splits_random():
    from fujicv.data.datasets import build_splits

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df, img_dir, csv_path = _make_toy_dataset(tmp, n=40)
        cfg = {
            "csv_path": str(csv_path),
            "label_col": "label",
            "task": "classification",
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "random_seed": 0,
            "output_dir": str(tmp),
        }
        train_df, val_df, test_df = build_splits(cfg)
        assert len(train_df) + len(val_df) + len(test_df) == 40
        assert len(train_df) > 0
        assert (tmp / "split_assignment.csv").exists()


def test_build_splits_with_split_col():
    from fujicv.data.datasets import build_splits

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df, _, csv_path = _make_toy_dataset(tmp, n=30)
        df["split"] = ["train"] * 20 + ["val"] * 5 + ["test"] * 5
        df.to_csv(csv_path, index=False)
        cfg = {
            "csv_path": str(csv_path),
            "label_col": "label",
            "task": "classification",
            "split_col": "split",
        }
        train_df, val_df, test_df = build_splits(cfg)
        assert len(train_df) == 20
        assert len(val_df) == 5
        assert len(test_df) == 5
