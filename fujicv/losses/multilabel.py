"""Multi-label classification loss functions."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fujicv.losses.registry import register_loss


@register_loss("BCEWithLogitsLoss")
class BCEWithLogitsLoss(nn.Module):
    """Standard binary cross-entropy with logits for multi-label tasks."""

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets.float())


@register_loss("WeightedBCELoss")
class WeightedBCELoss(nn.Module):
    """BCE with per-label positive weighting based on label frequency.

    Args:
        label_counts: 1-D tensor with positive sample count per label.
        total_samples: Total number of training samples.
        reduction: Reduction mode (default ``'mean'``).
    """

    def __init__(
        self,
        label_counts: torch.Tensor,
        total_samples: int,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        counts = label_counts.float().clamp(min=1)
        neg_counts = total_samples - counts
        pos_weight = neg_counts / counts
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets.float())


@register_loss("FocalBCELoss")
class FocalBCELoss(nn.Module):
    """Per-label focal loss for multi-label classification.

    Args:
        alpha: Balance factor (default 0.25).
        gamma: Focusing parameter (default 2.0).
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
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * (1.0 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@register_loss("AsymmetricLoss")
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label recognition.

    Different focusing parameters for positive and negative samples allow the
    model to focus more on hard positives while down-weighting easy negatives.

    Reference: Ridnik et al., "Asymmetric Loss For Multi-Label Classification
    and Object Detection," ICCV 2021.

    Args:
        gamma_pos: Focusing parameter for positives (default 0.0).
        gamma_neg: Focusing parameter for negatives (default 4.0).
        clip: Probability margin for clipping negative predictions (default 0.05).
        eps: Numerical stability constant (default 1e-8).
        reduction: Reduction mode (default ``'mean'``).
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        # Clip negatives
        probs_neg = probs + self.clip
        probs_neg = probs_neg.clamp(max=1.0)

        loss_pos = targets * torch.log(probs.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log((1.0 - probs_neg).clamp(min=self.eps))

        loss = loss_pos + loss_neg

        # Apply asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt_pos = probs
            pt_neg = 1.0 - probs_neg
            gamma_t = torch.where(targets == 1,
                                  torch.tensor(self.gamma_pos, device=logits.device),
                                  torch.tensor(self.gamma_neg, device=logits.device))
            pt = torch.where(targets == 1, pt_pos, pt_neg)
            loss = loss * (1.0 - pt) ** gamma_t

        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
