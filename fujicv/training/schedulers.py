"""LR warmup and advanced scheduler utilities."""

from __future__ import annotations

import math
from typing import List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)


def linear_warmup_schedule(
    optimizer: Optimizer,
    warmup_steps: int,
    after_scheduler: Optional[_LRScheduler] = None,
) -> _LRScheduler:
    """Linear warmup from 0 to base LR over ``warmup_steps`` steps.

    After warmup, delegates to ``after_scheduler`` (step-level) if provided,
    otherwise holds the base LR flat.

    Args:
        optimizer: The optimizer whose LR to warm up.
        warmup_steps: Number of steps over which to ramp the LR.
        after_scheduler: Optional scheduler to chain after warmup.

    Returns:
        A ``LambdaLR`` (or ``SequentialLR`` when ``after_scheduler`` is given).

    Example::

        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup)
        scheduler = linear_warmup_schedule(optimizer, warmup_steps=500,
                                           after_scheduler=cosine)
    """
    if warmup_steps <= 0:
        raise ValueError(f"warmup_steps must be > 0, got {warmup_steps}")

    def _warmup_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    warmup = LambdaLR(optimizer, lr_lambda=_warmup_lambda)

    if after_scheduler is None:
        return warmup

    # Chain: warmup for `warmup_steps`, then hand off to after_scheduler
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, after_scheduler],
        milestones=[warmup_steps],
    )


def cosine_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> _LRScheduler:
    """Cosine decay with linear warmup — the de-facto ViT/Swin recipe.

    LR ramps linearly from 0 to base_lr over ``warmup_steps``, then follows
    a cosine curve down to ``base_lr * min_lr_ratio``.

    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps (warmup + decay).
        min_lr_ratio: Final LR as a fraction of base LR (default 0.0).

    Example::

        scheduler = cosine_with_warmup(optimizer,
                                        warmup_steps=500,
                                        total_steps=10_000)
        for step in range(total_steps):
            train_step(...)
            scheduler.step()
    """
    if warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be < total_steps ({total_steps})"
        )

    decay_steps = total_steps - warmup_steps

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, decay_steps))
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    *,
    warmup_steps: int = 0,
    total_steps: Optional[int] = None,
    epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    # StepLR
    step_size: int = 10,
    gamma: float = 0.1,
    # CosineAnnealingLR
    eta_min: float = 0.0,
    # OneCycleLR
    max_lr: Optional[float] = None,
    # ReduceLROnPlateau
    patience: int = 5,
    factor: float = 0.5,
    # cosine_with_warmup
    min_lr_ratio: float = 0.0,
) -> Union[_LRScheduler, ReduceLROnPlateau]:
    """Factory function for common LR schedules, all with optional warmup.

    Args:
        name: One of ``'cosine'``, ``'cosine_warmup'``, ``'step'``,
              ``'onecycle'``, ``'plateau'``, ``'linear_warmup'``.
        optimizer: The optimizer.
        warmup_steps: Linear warmup steps (only used where applicable).
        total_steps: Total training steps (required for ``'cosine_warmup'``
            and ``'onecycle'``).
        epochs: Training epochs (used to derive ``total_steps`` when
            ``steps_per_epoch`` is also given).
        steps_per_epoch: Steps per epoch (used with ``epochs``).
        step_size: Step decay period for ``'step'`` (default 10).
        gamma: Decay factor for ``'step'`` (default 0.1).
        eta_min: Min LR for ``'cosine'`` (default 0.0).
        max_lr: Peak LR for ``'onecycle'`` (defaults to 10x base LR).
        patience: ReduceLROnPlateau patience (default 5).
        factor: ReduceLROnPlateau factor (default 0.5).
        min_lr_ratio: Final / base LR ratio for ``'cosine_warmup'`` (default 0.0).

    Returns:
        A scheduler instance.

    Example::

        scheduler = get_scheduler('cosine_warmup', optimizer,
                                   warmup_steps=500, total_steps=10_000)
    """
    # Derive total_steps if not given
    if total_steps is None and epochs is not None and steps_per_epoch is not None:
        total_steps = epochs * steps_per_epoch

    name = name.lower()

    if name == "cosine_warmup":
        if total_steps is None:
            raise ValueError("'cosine_warmup' requires total_steps (or epochs + steps_per_epoch)")
        ws = warmup_steps if warmup_steps > 0 else max(1, total_steps // 20)
        return cosine_with_warmup(optimizer, ws, total_steps, min_lr_ratio)

    if name == "linear_warmup":
        if warmup_steps <= 0:
            raise ValueError("'linear_warmup' requires warmup_steps > 0")
        return linear_warmup_schedule(optimizer, warmup_steps)

    if name == "cosine":
        T = total_steps if total_steps is not None else (epochs or 10)
        sched = CosineAnnealingLR(optimizer, T_max=T, eta_min=eta_min)
        if warmup_steps > 0:
            return linear_warmup_schedule(optimizer, warmup_steps, after_scheduler=sched)
        return sched

    if name == "step":
        sched = StepLR(optimizer, step_size=step_size, gamma=gamma)
        if warmup_steps > 0:
            return linear_warmup_schedule(optimizer, warmup_steps, after_scheduler=sched)
        return sched

    if name == "onecycle":
        if total_steps is None:
            raise ValueError("'onecycle' requires total_steps (or epochs + steps_per_epoch)")
        base_lr = optimizer.param_groups[0]["lr"]
        return OneCycleLR(
            optimizer,
            max_lr=max_lr or base_lr * 10,
            total_steps=total_steps,
        )

    if name == "plateau":
        return ReduceLROnPlateau(optimizer, patience=patience, factor=factor)

    raise ValueError(
        f"Unknown scheduler '{name}'. Choose from: "
        "cosine, cosine_warmup, step, onecycle, plateau, linear_warmup"
    )
