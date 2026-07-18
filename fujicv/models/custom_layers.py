"""Custom reusable nn.Module building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearBNDropout(nn.Module):
    """Linear → BatchNorm1d → ReLU → Dropout block.

    Useful as an intermediate fully-connected layer before a classification or
    regression head.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        dropout: Dropout probability (default 0.3). Set to 0.0 to disable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GeM(nn.Module):
    """Generalised Mean Pooling (GeM).

    A learnable alternative to global average pooling commonly used in
    image-retrieval tasks. With ``p=1`` it reduces to average pooling;
    with ``p→∞`` it approaches max pooling.

    Reference: Filip Radenović et al., "Fine-tuning CNN Image Retrieval with
    No Human Annotation," TPAMI 2019.

    Args:
        p: Initial power for the mean (default 3.0).
        eps: Small value added before exponentiation for numerical stability.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return (
            x.clamp(min=self.eps)
            .pow(self.p)
            .mean(dim=[-2, -1])
            .pow(1.0 / self.p)
        )


class AttentionPool(nn.Module):
    """Lightweight attention-based pooling over spatial positions.

    Computes a weighted average of spatial feature vectors where the weights
    are predicted by a small MLP.

    Args:
        in_features: Number of input channels.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.Tanh(),
            nn.Linear(in_features // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) or (B, N, C) [ViT patch tokens]
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        # x: (B, N, C)
        w = self.attn(x)  # (B, N, 1)
        w = torch.softmax(w, dim=1)
        out = (w * x).sum(dim=1)  # (B, C)
        return out


class SqueezeExcite(nn.Module):
    """Channel-wise Squeeze-and-Excitation block.

    Args:
        channels: Number of input/output channels.
        reduction: Reduction ratio for the bottleneck (default 16).
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale
