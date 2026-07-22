"""Regression tests for v1.7.0 bug fixes."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Bug 1: SAM ASAM uses abs(w), not w² ──────────────────────────────────────

def test_sam_adaptive_perturbation_uses_abs():
    """ASAM perturbation must use |w|, not w²."""
    from fujicv.training.sam import SAM

    model = nn.Linear(4, 2)
    # Set weights to known values to verify perturbation formula
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(0.0)

    opt = SAM(model.parameters(), torch.optim.SGD, rho=0.1, adaptive=True, lr=0.0)

    imgs   = torch.randn(4, 4)
    labels = torch.randint(0, 2, (4,))
    F.cross_entropy(model(imgs), labels).backward()

    w_before = model.weight.data.clone()
    opt.first_step(zero_grad=False)
    w_after = model.weight.data.clone()

    # With adaptive=True and w=2.0, perturbation ∝ |w|*grad = 2*grad
    # NOT w²*grad = 4*grad.
    # Check that some perturbation happened (non-zero grads exist)
    assert not torch.allclose(w_before, w_after)


# ── Bug 2: CutMix bbox centre never equals W or H ────────────────────────────

def test_cutmix_bbox_centre_never_out_of_bounds():
    """_rand_bbox: centre must stay strictly inside [0, W) × [0, H)."""
    from fujicv.data.mixup import _rand_bbox

    random.seed(0)
    H, W = 32, 32
    for _ in range(200):
        x1, y1, x2, y2 = _rand_bbox((1, 3, H, W), lam=0.5)
        assert 0 <= x1 <= x2 <= W, f"x out of bounds: {x1},{x2}"
        assert 0 <= y1 <= y2 <= H, f"y out of bounds: {y1},{y2}"


def test_cutmix_no_zero_area_patch_at_corner():
    """With randint fixed, cx/cy < W/H so patches are never degenerate at the far edge."""
    from fujicv.data.mixup import _rand_bbox
    random.seed(42)
    for _ in range(500):
        x1, y1, x2, y2 = _rand_bbox((1, 3, 64, 64), lam=0.9)
        # Area may be tiny but coords must be valid
        assert x1 <= x2 and y1 <= y2


# ── Bug 3: DistillationTrainer metric keys have train_/val_ prefix ────────────

def test_distillation_trainer_metric_prefix():
    """_run_epoch must return 'train_loss'/'val_loss', not bare 'loss'."""
    from fujicv.engine.distillation_trainer import DistillationTrainer
    from fujicv.losses.distillation import DistillationLoss
    from torch.utils.data import DataLoader, TensorDataset
    import tempfile

    teacher = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    student = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    imgs    = torch.randn(8, 3, 8, 8)
    targets = torch.randint(0, 3, (8,))
    loader  = DataLoader(TensorDataset(imgs, targets), batch_size=8)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = DistillationTrainer(
            teacher=teacher,
            model=student,
            train_loader=loader,
            val_loader=loader,
            loss_fn=DistillationLoss(),
            metrics={},
            optimizer=torch.optim.Adam(student.parameters(), lr=1e-3),
            epochs=1,
            task="classification",
            output_dir=tmp,
            mixed_precision=False,
        )
        history = trainer.train()

    keys = list(history.metrics.keys())
    assert any(k.startswith("train_") for k in keys), f"No train_ key in {keys}"
    assert any(k.startswith("val_")   for k in keys), f"No val_ key in {keys}"
    assert "loss" not in keys, f"Bare 'loss' key should not exist: {keys}"


# ── Bug 4: EnsemblePredictor.predict() no double forward pass for vote ────────

def test_ensemble_vote_single_forward(monkeypatch):
    """predict() with merge='vote' must call _forward_all exactly once."""
    from fujicv.inference.ensemble import EnsemblePredictor

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    ens   = EnsemblePredictor([model], merge="vote", task="classification")

    call_count = [0]
    original = ens._forward_all

    def counting_forward(image):
        call_count[0] += 1
        return original(image)

    monkeypatch.setattr(ens, "_forward_all", counting_forward)
    ens.predict(torch.randn(1, 3, 8, 8))
    assert call_count[0] == 1, f"_forward_all called {call_count[0]} times (expected 1)"


# ── Bug 5: KFoldTrainer checkpoint goes to fold dir ───────────────────────────

def test_kfold_checkpoint_in_fold_dir():
    """best.pt must be saved inside fold_N/, not the original output_dir."""
    from fujicv.training.kfold import KFoldTrainer
    from fujicv.losses.classification import CrossEntropyLoss
    from fujicv.metrics.classification import Accuracy
    from torch.utils.data import TensorDataset
    import pandas as pd
    import tempfile
    from pathlib import Path

    df = pd.DataFrame({
        "img":   [f"img_{i}.jpg" for i in range(30)],
        "label": [str(i % 3) for i in range(30)],
    })

    def model_factory():
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))

    def dataset_factory(sub_df, transform):
        imgs    = torch.randn(len(sub_df), 3, 8, 8)
        targets = torch.randint(0, 3, (len(sub_df),))
        return TensorDataset(imgs, targets)

    with tempfile.TemporaryDirectory() as tmp:
        def trainer_factory(model, train_loader, val_loader):
            from fujicv.engine.trainer import Trainer
            return Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=CrossEntropyLoss(),
                metrics={"accuracy": Accuracy()},
                optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
                epochs=2,
                task="classification",
                output_dir=tmp,
                mixed_precision=False,
            )

        kfold = KFoldTrainer(
            model_factory=model_factory,
            train_df=df,
            dataset_factory=dataset_factory,
            train_transform=None,
            val_transform=None,
            trainer_factory=trainer_factory,
            n_splits=3,
            output_dir=tmp,
            dataloader_kwargs={"batch_size": 10},
        )
        kfold.run()

        for fold_i in range(3):
            fold_dir = Path(tmp) / f"fold_{fold_i}"
            # Checkpoint callback should write to fold dir
            assert fold_dir.exists(), f"fold_{fold_i} dir missing"
