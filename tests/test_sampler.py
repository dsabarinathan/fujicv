"""Tests for WeightedRandomSampler utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def test_make_weighted_sampler_returns_sampler():
    from fujicv.data.sampler import make_weighted_sampler
    labels  = [0, 0, 0, 1, 1, 2]
    sampler = make_weighted_sampler(labels)
    from torch.utils.data import WeightedRandomSampler
    assert isinstance(sampler, WeightedRandomSampler)


def test_make_weighted_sampler_length():
    from fujicv.data.sampler import make_weighted_sampler
    labels  = [0, 0, 1, 1, 2, 2]
    sampler = make_weighted_sampler(labels, num_samples=20)
    assert len(sampler) == 20


def test_make_weighted_sampler_default_length():
    from fujicv.data.sampler import make_weighted_sampler
    labels  = list(range(50))
    sampler = make_weighted_sampler(labels)
    assert len(sampler) == 50


def test_make_weighted_sampler_integrates_with_dataloader():
    from fujicv.data.sampler import make_weighted_sampler
    labels  = [0] * 90 + [1] * 10   # 90% class 0, 10% class 1
    imgs    = torch.randn(100, 3, 8, 8)
    targets = torch.tensor(labels)
    ds      = TensorDataset(imgs, targets)
    sampler = make_weighted_sampler(labels)
    loader  = DataLoader(ds, batch_size=16, sampler=sampler)
    # Just verify it iterates without error
    batch = next(iter(loader))
    assert batch[0].shape[0] == 16


def test_make_weighted_sampler_minority_class_upsampled():
    """With balanced weights, minority class should appear more than without sampler."""
    from fujicv.data.sampler import make_weighted_sampler
    torch.manual_seed(0)
    labels  = [0] * 950 + [1] * 50   # severely imbalanced
    imgs    = torch.randn(1000, 1)
    targets = torch.tensor(labels)
    ds      = TensorDataset(imgs, targets)
    sampler = make_weighted_sampler(labels, num_samples=200)
    loader  = DataLoader(ds, batch_size=200, sampler=sampler)
    batch_targets = next(iter(loader))[1]
    minority_count = (batch_targets == 1).sum().item()
    # Without sampler, ~10/200 = 5% minority. With sampler should be ~50%.
    assert minority_count > 30, f"Expected ~100 minority samples, got {minority_count}"


def test_class_weights_from_labels_shape():
    from fujicv.data.sampler import class_weights_from_labels
    labels  = [0, 0, 1, 1, 1, 2]
    weights = class_weights_from_labels(labels)
    assert weights.shape == (3,)
    assert weights.dtype == torch.float32


def test_class_weights_from_labels_normalized():
    from fujicv.data.sampler import class_weights_from_labels
    labels  = [0, 0, 1, 1, 1, 2]
    weights = class_weights_from_labels(labels, normalize=True)
    # Normalized weights sum to num_classes
    assert abs(weights.sum().item() - 3.0) < 1e-4


def test_class_weights_minority_higher():
    from fujicv.data.sampler import class_weights_from_labels
    labels  = [0] * 90 + [1] * 10
    weights = class_weights_from_labels(labels)
    # Rare class 1 should get higher weight
    assert weights[1] > weights[0]


def test_class_weights_explicit_num_classes():
    from fujicv.data.sampler import class_weights_from_labels
    labels  = [0, 1]
    weights = class_weights_from_labels(labels, num_classes=5)
    assert weights.shape == (5,)


def test_class_weights_usable_in_cross_entropy():
    from fujicv.data.sampler import class_weights_from_labels
    labels  = [0, 0, 1, 2]
    weights = class_weights_from_labels(labels)
    logits  = torch.randn(4, 3)
    targets = torch.tensor(labels)
    loss    = torch.nn.CrossEntropyLoss(weight=weights)(logits, targets)
    assert loss.item() > 0
