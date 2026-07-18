"""Task-specific prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Multi-class classification head.

    Args:
        in_features: Backbone output feature dimension.
        num_classes: Number of target classes.
        dropout: Dropout probability before the linear layer (default 0.0).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        layers: list[nn.Module] = []
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(in_features, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits of shape ``(B, num_classes)``."""
        return self.head(x)


class RegressionHead(nn.Module):
    """Regression head for single or multi-output regression.

    Args:
        in_features: Backbone output feature dimension.
        num_outputs: Number of regression targets (default 1).
        dropout: Dropout probability before the linear layer (default 0.0).
    """

    def __init__(
        self,
        in_features: int,
        num_outputs: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_outputs = num_outputs
        layers: list[nn.Module] = []
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(in_features, num_outputs))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning predictions of shape ``(B, num_outputs)``."""
        out = self.head(x)
        if self.num_outputs == 1:
            out = out.squeeze(-1)  # (B,)
        return out


class MultiLabelHead(nn.Module):
    """Multi-label classification head (outputs raw logits, use BCEWithLogits loss).

    Args:
        in_features: Backbone output feature dimension.
        num_labels: Total number of binary labels.
        dropout: Dropout probability before the linear layer (default 0.0).
    """

    def __init__(
        self,
        in_features: int,
        num_labels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_labels = num_labels
        layers: list[nn.Module] = []
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(in_features, num_labels))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits of shape ``(B, num_labels)``."""
        return self.head(x)
