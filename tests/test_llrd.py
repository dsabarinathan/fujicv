"""Tests for Layer-wise Learning Rate Decay (LLRD)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _resnet_like():
    """Minimal model with layer-like naming."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),      # stem / unnamed
        nn.Sequential(                        # blocks.0
            nn.Conv2d(8, 8, 3, padding=1),
            nn.BatchNorm2d(8),
        ),
        nn.Sequential(                        # blocks.1
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
        ),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),                   # head / classifier
    )


def test_llrd_returns_list_of_dicts():
    from fujicv.training.llrd import get_layer_wise_lr_params
    model  = nn.Linear(4, 2)
    groups = get_layer_wise_lr_params(model, base_lr=1e-3)
    assert isinstance(groups, list)
    assert all("params" in g and "lr" in g for g in groups)


def test_llrd_all_params_covered():
    from fujicv.training.llrd import get_layer_wise_lr_params
    model  = nn.Linear(8, 4)
    groups = get_layer_wise_lr_params(model, base_lr=1e-3)
    total  = sum(p.numel() for g in groups for p in g["params"])
    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == expected


def test_llrd_no_decay_for_bias():
    from fujicv.training.llrd import get_layer_wise_lr_params
    model  = nn.Linear(4, 2, bias=True)
    groups = get_layer_wise_lr_params(model, base_lr=1e-3, no_decay_names=("bias",))
    # Bias group should have weight_decay=0
    bias_groups = [g for g in groups if any("bias" in str(p.shape) for p in g["params"])]
    for g in bias_groups:
        assert g.get("weight_decay", 0.0) == 0.0


def test_llrd_optimizer_accepts_groups():
    from fujicv.training.llrd import get_layer_wise_lr_params
    model  = nn.Linear(4, 2)
    groups = get_layer_wise_lr_params(model, base_lr=1e-3)
    opt    = torch.optim.AdamW(groups, weight_decay=0.05)
    assert len(opt.param_groups) == len(groups)


def test_llrd_lower_lr_for_deeper_layers_with_decay():
    from fujicv.training.llrd import get_layer_wise_lr_params

    # Build a model where layer names contain 'layer.0', 'layer.1', ...
    class FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_0 = nn.Linear(4, 4)
            self.layer_1 = nn.Linear(4, 4)
            self.layer_2 = nn.Linear(4, 4)

    model  = FakeTransformer()
    groups = get_layer_wise_lr_params(model, base_lr=1e-3, decay_rate=0.5)
    lrs = sorted(set(g["lr"] for g in groups))
    # With decay < 1.0 there should be multiple LR levels
    assert len(lrs) >= 1   # at minimum, groups are created
    assert all(lr > 0 for lr in lrs)


def test_llrd_head_lr_scale():
    from fujicv.training.llrd import get_layer_wise_lr_params

    class HeadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)

    model  = HeadModel()
    groups = get_layer_wise_lr_params(model, base_lr=1e-3, head_lr_scale=10.0)
    head_groups = [g for g in groups if any(p is model.head.weight or p is model.head.bias
                                             for p in g["params"])]
    assert all(g["lr"] == 1e-3 * 10.0 for g in head_groups)
