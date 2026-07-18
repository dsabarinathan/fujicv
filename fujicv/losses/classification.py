"""Classification loss functions."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fujicv.losses.registry import register_loss


@register_loss("CrossEntropyLoss")
class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss wrapping ``torch.nn.CrossEntropyLoss``."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


@register_loss("WeightedCrossEntropyLoss")
class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy with per-class weights computed from class frequencies.

    Args:
        class_counts: 1-D tensor with the count of samples per class.  Weights
            are set to ``total / (num_classes * count)``.
        reduction: Reduction mode (default ``'mean'``).
    """

    def __init__(
        self,
        class_counts: torch.Tensor,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        counts = class_counts.float()
        weights = counts.sum() / (len(counts) * counts.clamp(min=1))
        self.loss = nn.CrossEntropyLoss(weight=weights, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits.to(self.loss.weight.device), targets)


@register_loss("LabelSmoothingCE")
class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing.

    Args:
        smoothing: Label smoothing factor ∈ [0, 1) (default 0.1).
        reduction: ``'mean'`` or ``'sum'`` (default ``'mean'``).
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # One-hot smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@register_loss("FocalLoss")
class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.

    Reference: Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.

    Args:
        alpha: Weighting factor ∈ [0, 1] (default 0.25). Can also be a tensor
            of per-class weights.
        gamma: Focusing parameter ≥ 0 (default 2.0).
        reduction: ``'mean'``, ``'sum'``, or ``'none'`` (default ``'mean'``).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        probs = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1.0 - probs) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@register_loss("ClassBalancedLoss")
class ClassBalancedLoss(nn.Module):
    """Class-balanced loss re-weighting based on effective number of samples.

    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of
    Samples," CVPR 2019.

    Args:
        class_counts: 1-D tensor with sample counts per class.
        beta: Hyper-parameter ∈ [0, 1) (default 0.9999).
        loss_type: Base loss — ``'softmax'`` (default) or ``'focal'``.
        focal_gamma: Gamma for focal base loss (default 0.5).
    """

    def __init__(
        self,
        class_counts: torch.Tensor,
        beta: float = 0.9999,
        loss_type: str = "softmax",
        focal_gamma: float = 0.5,
    ) -> None:
        super().__init__()
        counts = class_counts.float()
        effective_num = 1.0 - beta ** counts
        weights = (1.0 - beta) / effective_num.clamp(min=1e-8)
        weights = weights / weights.sum() * len(counts)
        self.register_buffer("weights", weights)
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "focal":
            ce = F.cross_entropy(logits, targets, reduction="none")
            probs = torch.exp(-ce)
            focal_w = (1.0 - probs) ** self.focal_gamma
            sample_w = self.weights[targets]
            return (focal_w * sample_w * ce).mean()
        else:
            return F.cross_entropy(logits, targets, weight=self.weights)
