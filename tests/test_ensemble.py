"""Tests for EnsemblePredictor."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _net(out=3):
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, out))


def _loader(n=16, out=3):
    imgs    = torch.randn(n, 3, 8, 8)
    targets = torch.randint(0, out, (n,))
    return DataLoader(TensorDataset(imgs, targets), batch_size=8)


# ── Basic API ──────────────────────────────────────────────────────────────────

def test_ensemble_predict_returns_int():
    from fujicv.inference.ensemble import EnsemblePredictor
    models  = [_net() for _ in range(3)]
    ens     = EnsemblePredictor(models, merge="mean", task="classification")
    result  = ens.predict(torch.randn(1, 3, 8, 8))
    assert isinstance(result, int)


def test_ensemble_predict_proba_shape():
    from fujicv.inference.ensemble import EnsemblePredictor
    models = [_net(5) for _ in range(2)]
    ens    = EnsemblePredictor(models, merge="mean", task="classification")
    proba  = ens.predict_proba(torch.randn(1, 3, 8, 8))
    assert proba.shape == (5,)
    assert abs(proba.sum() - 1.0) < 1e-4


def test_ensemble_predict_proba_sums_to_one():
    from fujicv.inference.ensemble import EnsemblePredictor
    models = [_net(4) for _ in range(3)]
    ens    = EnsemblePredictor(models, merge="mean", task="classification")
    proba  = ens.predict_proba(torch.randn(3, 8, 8))
    assert abs(proba.sum() - 1.0) < 1e-4


# ── Merge strategies ──────────────────────────────────────────────────────────

def test_ensemble_vote_merge():
    from fujicv.inference.ensemble import EnsemblePredictor
    models = [_net() for _ in range(3)]
    ens    = EnsemblePredictor(models, merge="vote", task="classification")
    result = ens.predict(torch.randn(1, 3, 8, 8))
    assert 0 <= result < 3


def test_ensemble_max_merge_multilabel():
    from fujicv.inference.ensemble import EnsemblePredictor
    models = [_net(4) for _ in range(2)]
    ens    = EnsemblePredictor(models, merge="max", task="multilabel")
    result = ens.predict(torch.randn(1, 3, 8, 8))
    assert result.shape == (4,)


def test_ensemble_weighted_mean():
    from fujicv.inference.ensemble import EnsemblePredictor
    models  = [_net() for _ in range(3)]
    weights = [0.5, 0.3, 0.2]
    ens     = EnsemblePredictor(models, merge="weighted_mean",
                                 task="classification", weights=weights)
    result = ens.predict(torch.randn(1, 3, 8, 8))
    assert isinstance(result, int)


def test_ensemble_weighted_mean_requires_weights():
    from fujicv.inference.ensemble import EnsemblePredictor
    with pytest.raises(ValueError):
        EnsemblePredictor([_net()], merge="weighted_mean", task="classification")


def test_ensemble_predict_batch():
    from fujicv.inference.ensemble import EnsemblePredictor
    models = [_net() for _ in range(2)]
    ens    = EnsemblePredictor(models, merge="mean", task="classification")
    preds, tgts = ens.predict_batch(_loader(), return_targets=True)
    assert preds.shape == (16,)
    assert tgts.shape  == (16,)


def test_ensemble_regression():
    from fujicv.inference.ensemble import EnsemblePredictor
    models = [_net(1) for _ in range(2)]
    ens    = EnsemblePredictor(models, merge="mean", task="regression")
    result = ens.predict(torch.randn(1, 3, 8, 8))
    assert isinstance(result, float)


def test_ensemble_empty_models_raises():
    from fujicv.inference.ensemble import EnsemblePredictor
    with pytest.raises(ValueError):
        EnsemblePredictor([], merge="mean", task="classification")


def test_ensemble_invalid_merge_raises():
    from fujicv.inference.ensemble import EnsemblePredictor
    with pytest.raises(ValueError):
        EnsemblePredictor([_net()], merge="bad", task="classification")
