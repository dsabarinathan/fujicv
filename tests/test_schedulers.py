"""Tests for LR warmup and advanced schedulers."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn


def _opt(lr=1e-3):
    return torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=lr)


# ── linear_warmup_schedule ─────────────────────────────────────────────────────

def test_linear_warmup_starts_near_zero():
    from fujicv.training.schedulers import linear_warmup_schedule
    opt  = _opt(lr=1.0)
    sched = linear_warmup_schedule(opt, warmup_steps=10)
    # After 0 steps, LR should be 0 / 10 = 0
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-6)


def test_linear_warmup_reaches_base_lr():
    from fujicv.training.schedulers import linear_warmup_schedule
    opt  = _opt(lr=1.0)
    sched = linear_warmup_schedule(opt, warmup_steps=5)
    for _ in range(5):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0, rel=1e-3)


def test_linear_warmup_invalid():
    from fujicv.training.schedulers import linear_warmup_schedule
    with pytest.raises(ValueError):
        linear_warmup_schedule(_opt(), warmup_steps=0)


# ── cosine_with_warmup ─────────────────────────────────────────────────────────

def test_cosine_warmup_peaks_then_decays():
    from fujicv.training.schedulers import cosine_with_warmup
    opt   = _opt(lr=1.0)
    sched = cosine_with_warmup(opt, warmup_steps=5, total_steps=20)
    lrs   = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    # LR should increase during warmup
    assert lrs[4] > lrs[0]
    # LR should decrease after warmup
    assert lrs[10] < lrs[5]
    # Final LR ≈ 0 (min_lr_ratio=0)
    assert lrs[-1] < 0.1


def test_cosine_warmup_respects_min_lr_ratio():
    from fujicv.training.schedulers import cosine_with_warmup
    opt   = _opt(lr=1.0)
    sched = cosine_with_warmup(opt, warmup_steps=2, total_steps=10, min_lr_ratio=0.1)
    for _ in range(10):
        sched.step()
    assert opt.param_groups[0]["lr"] >= 0.09


def test_cosine_warmup_invalid():
    from fujicv.training.schedulers import cosine_with_warmup
    with pytest.raises(ValueError):
        cosine_with_warmup(_opt(), warmup_steps=10, total_steps=10)


# ── get_scheduler factory ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["cosine", "step", "cosine_warmup", "linear_warmup", "onecycle"])
def test_get_scheduler_valid_names(name):
    from fujicv.training.schedulers import get_scheduler
    opt = _opt(lr=1e-3)
    sched = get_scheduler(
        name, opt,
        warmup_steps=5,
        total_steps=20,
        epochs=5,
        steps_per_epoch=4,
    )
    assert sched is not None


def test_get_scheduler_plateau():
    from fujicv.training.schedulers import get_scheduler
    sched = get_scheduler("plateau", _opt())
    assert hasattr(sched, "step")


def test_get_scheduler_unknown():
    from fujicv.training.schedulers import get_scheduler
    with pytest.raises(ValueError):
        get_scheduler("unknown_scheduler", _opt())


def test_get_scheduler_cosine_warmup_requires_total_steps():
    from fujicv.training.schedulers import get_scheduler
    with pytest.raises(ValueError):
        get_scheduler("cosine_warmup", _opt())
