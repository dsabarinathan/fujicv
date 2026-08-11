"""Tests for layer freezing / gradual unfreezing utilities."""
from __future__ import annotations

import torch.nn as nn

from fujicv.training.freezing import (
    GradualUnfreezing,
    count_frozen_parameters,
    count_trainable_parameters,
    freeze,
    freeze_backbone,
    freeze_bn_stats,
    unfreeze,
    unfreeze_backbone,
)


class _FakeModel(nn.Module):
    """Mimics a FujiCV assembled model: .backbone + .head."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3),
        )
        self.head = nn.Linear(16, 10)


def test_freeze_and_unfreeze():
    m = nn.Linear(4, 4)
    freeze(m)
    assert count_trainable_parameters(m) == 0
    unfreeze(m)
    assert count_frozen_parameters(m) == 0


def test_freeze_backbone_keeps_head_trainable():
    m = _FakeModel()
    freeze_backbone(m)
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.head.parameters())


def test_unfreeze_backbone():
    m = _FakeModel()
    freeze_backbone(m)
    unfreeze_backbone(m)
    assert all(p.requires_grad for p in m.backbone.parameters())


def test_freeze_backbone_without_backbone_attr_warns_and_freezes_all():
    m = nn.Linear(4, 4)  # no .backbone
    freeze_backbone(m)
    assert count_trainable_parameters(m) == 0


def test_gradual_unfreezing_head_only_at_start():
    m = _FakeModel()
    unfreezer = GradualUnfreezing(m, unfreeze_epoch=2, layers_per_epoch=1)
    # Backbone frozen on init; head still trainable.
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.head.parameters())
    # Before unfreeze_epoch nothing changes.
    unfreezer.step(0)
    unfreezer.step(1)
    assert unfreezer._num_unfrozen == 0


def test_gradual_unfreezing_progressively_unfreezes():
    m = _FakeModel()
    unfreezer = GradualUnfreezing(m, unfreeze_epoch=0, layers_per_epoch=1)
    total_blocks = unfreezer.num_blocks
    assert total_blocks >= 1

    seen = 0
    for epoch in range(total_blocks + 2):
        n = unfreezer.step(epoch)
        assert n >= seen  # monotonic
        seen = n
    assert unfreezer.fully_unfrozen
    assert all(p.requires_grad for p in m.backbone.parameters())


def test_gradual_unfreezing_top_down_order():
    """First unfrozen block should be the one nearest the head (last child)."""
    m = _FakeModel()
    unfreezer = GradualUnfreezing(m, unfreeze_epoch=0, layers_per_epoch=1)
    unfreezer.step(0)
    children = list(m.backbone.children())
    last_with_params = [c for c in children if any(True for _ in c.parameters())][-1]
    assert all(p.requires_grad for p in last_with_params.parameters())


def test_freeze_bn_stats_sets_eval_mode():
    m = _FakeModel()
    m.train()
    freeze_bn_stats(m, freeze=True)
    bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)]
    assert bns and all(not bn.training for bn in bns)
    freeze_bn_stats(m, freeze=False)
    assert all(bn.training for bn in bns)
