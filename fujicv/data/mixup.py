"""Mixup and CutMix batch augmentation collators."""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _rand_bbox(size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
    """Return a random bounding box for CutMix."""
    H, W = size[-2], size[-1]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


class MixupCollator:
    """Collate function that applies Mixup to a batch.

    Mixup (Zhang et al., 2018) linearly interpolates two images and their
    one-hot labels:  ``x' = λx_i + (1-λ)x_j``,  ``y' = λy_i + (1-λ)y_j``.

    Use as the ``collate_fn`` argument to a DataLoader.  The loss must accept
    **soft** targets (e.g. ``CrossEntropyLoss`` supports soft labels in PyTorch
    ≥ 1.10 via the ``label_smoothing`` path, or use ``BCEWithLogitsLoss``).

    Args:
        alpha: Beta distribution concentration parameter (default 0.4).
            Larger → stronger mix; set to 0 to disable.
        num_classes: Number of output classes for one-hot encoding.
        prob: Probability of applying Mixup per batch (default 1.0).

    Returns (from ``__call__``):
        ``(images, soft_targets)`` where ``soft_targets`` has shape
        ``(B, num_classes)`` and dtype ``float32``.

    Example::

        from fujicv.data.mixup import MixupCollator
        collator = MixupCollator(alpha=0.4, num_classes=10)
        loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
    """

    def __init__(
        self,
        alpha: float = 0.4,
        num_classes: int = 1000,
        prob: float = 1.0,
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        self.alpha = alpha
        self.num_classes = num_classes
        self.prob = prob

    def __call__(self, batch):
        images, targets = zip(*batch)
        images  = torch.stack(images)
        targets = torch.tensor(targets, dtype=torch.long)

        soft = F.one_hot(targets, self.num_classes).float()

        if self.alpha == 0 or random.random() > self.prob:
            return images, soft

        lam = np.random.beta(self.alpha, self.alpha)
        B = images.size(0)
        idx = torch.randperm(B)

        images = lam * images + (1.0 - lam) * images[idx]
        soft   = lam * soft   + (1.0 - lam) * soft[idx]
        return images, soft


class CutMixCollator:
    """Collate function that applies CutMix to a batch.

    CutMix (Yun et al., 2019) cuts a random rectangular patch from one image
    and pastes it onto another, mixing labels proportionally to patch area.

    Use as the ``collate_fn`` argument to a DataLoader.

    Args:
        alpha: Beta distribution parameter (default 1.0).
        num_classes: Number of output classes.
        prob: Probability of applying CutMix per batch (default 1.0).

    Example::

        from fujicv.data.mixup import CutMixCollator
        collator = CutMixCollator(alpha=1.0, num_classes=10)
        loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        num_classes: int = 1000,
        prob: float = 1.0,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        self.alpha = alpha
        self.num_classes = num_classes
        self.prob = prob

    def __call__(self, batch):
        images, targets = zip(*batch)
        images  = torch.stack(images)
        targets = torch.tensor(targets, dtype=torch.long)

        soft = F.one_hot(targets, self.num_classes).float()

        if random.random() > self.prob:
            return images, soft

        lam_orig = np.random.beta(self.alpha, self.alpha)
        B = images.size(0)
        idx = torch.randperm(B)

        x1, y1, x2, y2 = _rand_bbox(images.shape, lam_orig)
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

        # Recompute lambda from actual patch size
        H, W = images.shape[-2], images.shape[-1]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)

        soft = lam * soft + (1.0 - lam) * soft[idx]
        return images, soft


class MixupCutMixCollator:
    """Randomly applies either Mixup or CutMix each batch.

    Args:
        mixup_alpha: Mixup beta parameter (default 0.4).
        cutmix_alpha: CutMix beta parameter (default 1.0).
        num_classes: Number of output classes.
        mixup_prob: Probability of Mixup per batch (default 0.5).
        cutmix_prob: Probability of CutMix per batch (default 0.5).
            The two probabilities need not sum to 1; each is checked
            independently — if neither fires, the batch is returned unchanged.

    Example::

        from fujicv.data.mixup import MixupCutMixCollator
        collator = MixupCutMixCollator(num_classes=10)
        loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
    """

    def __init__(
        self,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        num_classes: int = 1000,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
    ) -> None:
        self._mixup  = MixupCollator(alpha=mixup_alpha,  num_classes=num_classes, prob=1.0)
        self._cutmix = CutMixCollator(alpha=cutmix_alpha, num_classes=num_classes, prob=1.0)
        self.mixup_prob  = mixup_prob
        self.cutmix_prob = cutmix_prob

    def __call__(self, batch):
        r = random.random()
        if r < self.mixup_prob:
            return self._mixup(batch)
        elif r < self.mixup_prob + self.cutmix_prob:
            return self._cutmix(batch)
        else:
            images, targets = zip(*batch)
            images  = torch.stack(images)
            targets = torch.tensor(targets, dtype=torch.long)
            soft    = F.one_hot(targets, self._mixup.num_classes).float()
            return images, soft
