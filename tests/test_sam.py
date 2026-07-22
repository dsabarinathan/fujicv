"""Tests for SAM optimizer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _model_and_data():
    model  = nn.Linear(4, 2)
    imgs   = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,))
    return model, imgs, labels


def test_sam_first_second_step():
    from fujicv.training.sam import SAM
    model, imgs, labels = _model_and_data()
    opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.01)

    # First pass
    loss = F.cross_entropy(model(imgs), labels)
    loss.backward()
    opt.first_step(zero_grad=True)

    # Second pass
    F.cross_entropy(model(imgs), labels).backward()
    opt.second_step(zero_grad=True)


def test_sam_weights_restored_after_second_step():
    from fujicv.training.sam import SAM
    model, imgs, labels = _model_and_data()
    w_before = model.weight.data.clone()

    opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.0)  # lr=0 → no update

    loss = F.cross_entropy(model(imgs), labels)
    loss.backward()
    opt.first_step(zero_grad=True)

    # After first_step weights are perturbed
    assert not torch.allclose(model.weight.data, w_before)

    F.cross_entropy(model(imgs), labels).backward()
    opt.second_step(zero_grad=True)

    # After second_step original weights restored then SGD update applied
    # With lr=0 the update is zero, so weights should match
    assert torch.allclose(model.weight.data, w_before, atol=1e-6)


def test_sam_reduces_loss_over_epochs():
    from fujicv.training.sam import SAM
    torch.manual_seed(0)
    model, imgs, labels = _model_and_data()
    opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.1, momentum=0.9)

    losses = []
    for _ in range(20):
        loss = F.cross_entropy(model(imgs), labels)
        loss.backward()
        opt.first_step(zero_grad=True)
        F.cross_entropy(model(imgs), labels).backward()
        opt.second_step(zero_grad=True)
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_sam_invalid_rho():
    from fujicv.training.sam import SAM
    with pytest.raises(ValueError):
        SAM(nn.Linear(2, 2).parameters(), torch.optim.SGD, rho=-0.1, lr=0.01)


def test_sam_step_without_closure_raises():
    from fujicv.training.sam import SAM
    model, imgs, labels = _model_and_data()
    opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.01)
    with pytest.raises(RuntimeError):
        opt.step()


def test_sam_adaptive_mode():
    from fujicv.training.sam import SAM
    model, imgs, labels = _model_and_data()
    opt = SAM(model.parameters(), torch.optim.AdamW, rho=0.05,
              adaptive=True, lr=1e-3)

    loss = F.cross_entropy(model(imgs), labels)
    loss.backward()
    opt.first_step(zero_grad=True)
    F.cross_entropy(model(imgs), labels).backward()
    opt.second_step(zero_grad=True)


def test_sam_works_with_adamw():
    from fujicv.training.sam import SAM
    model, imgs, labels = _model_and_data()
    opt = SAM(model.parameters(), torch.optim.AdamW, rho=0.05,
              lr=1e-3, weight_decay=0.01)

    for _ in range(5):
        F.cross_entropy(model(imgs), labels).backward()
        opt.first_step(zero_grad=True)
        F.cross_entropy(model(imgs), labels).backward()
        opt.second_step(zero_grad=True)
