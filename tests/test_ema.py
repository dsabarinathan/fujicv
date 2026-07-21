"""Tests for ModelEMA."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn


def _net():
    return nn.Linear(4, 2)


def test_ema_shadow_copy_on_init():
    from fujicv.training.ema import ModelEMA
    model = _net()
    ema   = ModelEMA(model, decay=0.99)
    # Shadow should start as a copy of the model
    for p_s, p_m in zip(ema.shadow.parameters(), model.parameters()):
        assert torch.allclose(p_s, p_m)


def test_ema_shadow_frozen():
    from fujicv.training.ema import ModelEMA
    ema = ModelEMA(_net(), decay=0.99)
    for p in ema.shadow.parameters():
        assert not p.requires_grad


def test_ema_update_tracks_model():
    from fujicv.training.ema import ModelEMA
    model = _net()
    ema   = ModelEMA(model, decay=0.9, warmup_steps=0)

    # Change model weights drastically
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(10.0)

    ema.update(model)

    # After one update with decay=0.9: shadow = 0.9 * orig + 0.1 * 10
    # orig was small randn; shadow should have shifted toward 10
    shadow_mean = torch.cat([p.flatten() for p in ema.shadow.parameters()]).mean().item()
    assert shadow_mean > 0.5   # clearly moved toward 10


def test_ema_update_count_increments():
    from fujicv.training.ema import ModelEMA
    model = _net()
    ema   = ModelEMA(model, decay=0.99)
    ema.update(model)
    ema.update(model)
    assert ema._num_updates == 2


def test_ema_invalid_decay():
    from fujicv.training.ema import ModelEMA
    with pytest.raises(ValueError):
        ModelEMA(_net(), decay=1.0)
    with pytest.raises(ValueError):
        ModelEMA(_net(), decay=0.0)


def test_ema_apply_to():
    from fujicv.training.ema import ModelEMA
    model  = _net()
    ema    = ModelEMA(model, decay=0.99)
    target = _net()
    ema.apply_to(target)
    for p_s, p_t in zip(ema.shadow.parameters(), target.parameters()):
        assert torch.allclose(p_s, p_t)


def test_ema_context_manager_restores():
    from fujicv.training.ema import ModelEMA
    model = _net()
    ema   = ModelEMA(model, decay=0.99)

    # Record original weights
    orig = {n: p.clone() for n, p in model.named_parameters()}

    # EMA weights differ (shadow was initialised from model but model has since changed)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(99.0)
    ema.update(model)   # push 99s into shadow

    # Restore model to originals
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(orig[n])

    with ema.average_parameters(model):
        # Inside: model has shadow (≈99) weights
        inside_mean = torch.cat([p.flatten() for p in model.parameters()]).mean().item()
        assert inside_mean > 1.0   # shadow has high values

    # Outside: model restored
    outside = {n: p.clone() for n, p in model.named_parameters()}
    for n in orig:
        assert torch.allclose(orig[n], outside[n])


def test_ema_state_dict_round_trip():
    from fujicv.training.ema import ModelEMA
    model = _net()
    ema   = ModelEMA(model, decay=0.9999, warmup_steps=500)
    ema.update(model)

    sd  = ema.state_dict()
    ema2 = ModelEMA(_net(), decay=0.5)
    ema2.load_state_dict(sd)

    assert ema2.decay         == ema.decay
    assert ema2._num_updates  == ema._num_updates
    assert ema2.warmup_steps  == ema.warmup_steps
