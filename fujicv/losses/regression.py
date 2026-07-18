"""Regression loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fujicv.losses.registry import register_loss


@register_loss("MSELoss")
class MSELoss(nn.Module):
    """Mean squared error loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(preds.float(), targets.float())


@register_loss("MAELoss")
class MAELoss(nn.Module):
    """Mean absolute error (L1) loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(preds.float(), targets.float())


@register_loss("HuberLoss")
class HuberLoss(nn.Module):
    """Huber (smooth L1) loss — quadratic for small errors, linear for large.

    Args:
        delta: Threshold between quadratic and linear regions (default 1.0).
        reduction: Reduction mode (default ``'mean'``).
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta, reduction=reduction)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(preds.float(), targets.float())


@register_loss("LogCoshLoss")
class LogCoshLoss(nn.Module):
    """Log-cosh loss: ``log(cosh(pred - target))``.

    Smoother than MAE and less sensitive to outliers than MSE.

    Args:
        reduction: ``'mean'`` or ``'sum'`` (default ``'mean'``).
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff = preds.float() - targets.float()
        # Numerically stable log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
        loss = diff.abs() + F.softplus(-2.0 * diff.abs()) - torch.log(torch.tensor(2.0))
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


@register_loss("QuantileLoss")
class QuantileLoss(nn.Module):
    """Pinball / quantile loss for quantile regression.

    Args:
        quantile: Target quantile ∈ (0, 1) (default 0.5, i.e. median).
        reduction: ``'mean'`` or ``'sum'`` (default ``'mean'``).
    """

    def __init__(self, quantile: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff = targets.float() - preds.float()
        loss = torch.where(diff >= 0, self.quantile * diff, (self.quantile - 1.0) * diff)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
