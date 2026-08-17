"""Tests for Predictor batch inference, ID preservation, and TTA."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fujicv.inference.predictor import Predictor


class _ConstModel(nn.Module):
    """Returns fixed logits so predictions are deterministic."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lin = nn.Linear(1, 1)  # unused; keeps it a real nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        logits = torch.zeros(b, self.num_classes)
        logits[:, 1] = 5.0  # always predict class 1
        return logits


def _predictor(task="classification"):
    return Predictor(
        model=_ConstModel(),
        class_to_idx={"cat": 0, "dog": 1, "fox": 2},
        task=task,
        device="cpu",
    )


def _loader(n=10, with_ids=False):
    X = torch.randn(n, 3, 8, 8)
    if with_ids:
        y = torch.arange(n)  # use as ids
    else:
        y = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)


def test_predict_batch_default_ids_are_running_index():
    df = _predictor().predict_batch(_loader(10))
    assert list(df.columns) == ["image", "prediction", "confidence"]
    assert len(df) == 10
    assert df["image"].tolist() == [f"sample_{i}" for i in range(10)]
    assert (df["prediction"] == "dog").all()  # class 1


def test_predict_batch_explicit_ids_preserved():
    ids = [f"img_{i}.jpg" for i in range(10)]
    df = _predictor().predict_batch(_loader(10), ids=ids)
    assert df["image"].tolist() == ids


def test_predict_batch_yields_ids_from_loader():
    df = _predictor().predict_batch(_loader(8, with_ids=True), yields_ids=True)
    # ids came from the second tuple element (0..7)
    assert df["image"].tolist() == list(range(8))


def test_predict_batch_confidence_is_softmax_prob():
    df = _predictor().predict_batch(_loader(4))
    # Vectorized softmax of [0,5,0] → class-1 prob ≈ 0.987
    assert np.allclose(df["confidence"].values, 0.9866, atol=1e-3)


def test_predict_batch_tta_runs():
    df = _predictor().predict_batch(_loader(6), use_tta=True)
    assert len(df) == 6
    assert (df["prediction"] == "dog").all()


def test_single_predict_matches_batch_decode():
    p = _predictor()
    img = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    label, conf = p.predict(img)
    assert label == "dog"
    assert 0.9 < conf <= 1.0


def test_predict_tta_single_image():
    p = _predictor()
    img = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    label, conf = p.predict(img, use_tta=True)
    assert label == "dog"


def test_regression_batch_decode_scalar():
    class _Reg(nn.Module):
        def forward(self, x):
            return torch.full((x.size(0),), 2.5)

    p = Predictor(model=_Reg(), task="regression", device="cpu")
    df = p.predict_batch(_loader(5))
    assert np.allclose(df["prediction"].values.astype(float), 2.5)
    assert (df["confidence"] == 1.0).all()
