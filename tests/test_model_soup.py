"""Tests for model soups (uniform + greedy)."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from fujicv.training.model_soup import greedy_soup, uniform_soup


def _linear(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    m = nn.Linear(4, 2)
    return m


def test_uniform_soup_averages_weights():
    m1 = _linear(1)
    m2 = _linear(2)
    target = _linear(99)

    uniform_soup(target, [m1.state_dict(), m2.state_dict()])

    expected_w = (m1.weight + m2.weight) / 2
    assert torch.allclose(target.weight, expected_w, atol=1e-6)
    expected_b = (m1.bias + m2.bias) / 2
    assert torch.allclose(target.bias, expected_b, atol=1e-6)


def test_uniform_soup_single_ingredient_is_identity():
    m1 = _linear(1)
    target = _linear(99)
    uniform_soup(target, [m1.state_dict()])
    assert torch.allclose(target.weight, m1.weight, atol=1e-6)


def test_uniform_soup_empty_raises():
    with pytest.raises(ValueError):
        uniform_soup(_linear(1), [])


def test_uniform_soup_mismatched_keys_raise():
    a = nn.Linear(4, 2).state_dict()
    b = nn.Conv2d(3, 4, 3).state_dict()
    with pytest.raises(ValueError):
        uniform_soup(nn.Linear(4, 2), [a, b])


def test_uniform_soup_preserves_integer_buffers():
    # BatchNorm has an integer num_batches_tracked buffer.
    m1 = nn.BatchNorm1d(4)
    m2 = copy.deepcopy(m1)
    m1.num_batches_tracked += 5
    m2.num_batches_tracked += 9
    target = nn.BatchNorm1d(4)
    uniform_soup(target, [m1.state_dict(), m2.state_dict()])
    # Integer buffer kept from the first ingredient (not averaged to a float).
    assert target.num_batches_tracked.dtype == m1.num_batches_tracked.dtype


def test_greedy_soup_keeps_only_helpful_ingredients():
    states = [_linear(s).state_dict() for s in range(4)]

    # eval_fn rewards the first ingredient's weights; others hurt.
    good = states[0]

    def eval_fn(model: nn.Module) -> float:
        # Higher when close to `good`.
        diff = sum((model.state_dict()[k] - good[k]).pow(2).sum() for k in good)
        return float(-diff)

    kept = greedy_soup(_linear(99), states, eval_fn, higher_is_better=True)
    assert 0 in kept  # best seed must be kept
    assert len(kept) >= 1


def test_greedy_soup_empty_raises():
    with pytest.raises(ValueError):
        greedy_soup(_linear(1), [], lambda m: 0.0)


def test_greedy_soup_loads_final_weights():
    states = [_linear(s).state_dict() for s in range(3)]
    target = _linear(99)
    kept = greedy_soup(target, states, lambda m: 1.0, higher_is_better=True)
    # With a constant eval, all ingredients tie and are kept (>= comparison).
    expected = {k: sum(s[k] for s in states) / len(states) for k in states[0]}
    for k in expected:
        assert torch.allclose(target.state_dict()[k], expected[k], atol=1e-5)
    assert len(kept) == 3
