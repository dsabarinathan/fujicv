"""Stochastic Depth (DropPath) regularisation."""

from __future__ import annotations

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic Depth — drop entire residual branches with probability *drop_prob*.

    During training each sample in the batch independently survives (probability
    ``1 - drop_prob``) or is zeroed out.  At inference the layer is a no-op.

    This is the per-sample variant introduced in "Deep Networks with Stochastic
    Depth" (Huang et al., 2016) and widely used in EfficientNet, ConvNeXt, and
    ViT variants.

    Args:
        drop_prob: Probability of dropping a residual branch (default 0.1).

    Example::

        from fujicv.models.stochastic_depth import DropPath

        class ResidualBlock(nn.Module):
            def __init__(self, drop_path_rate=0.1):
                super().__init__()
                self.conv = nn.Conv2d(64, 64, 3, padding=1)
                self.drop_path = DropPath(drop_path_rate)

            def forward(self, x):
                return x + self.drop_path(self.conv(x))
    """

    def __init__(self, drop_prob: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival_prob = 1.0 - self.drop_prob
        # Shape: (batch_size, 1, 1, ...) — broadcast over all spatial/channel dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device)
        noise = noise.bernoulli_(survival_prob)
        if survival_prob > 0.0:
            noise.div_(survival_prob)  # re-scale so expected value is preserved
        return x * noise

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


def build_stochastic_depth_schedule(
    num_stages: int,
    max_drop_rate: float = 0.2,
) -> list[float]:
    """Return a linearly increasing DropPath schedule across *num_stages* stages.

    Earlier layers (closer to input) get a lower drop probability than later
    layers, following the paper's recommendation.

    Args:
        num_stages: Total number of DropPath layers (stages / blocks).
        max_drop_rate: Maximum drop probability assigned to the deepest layer.

    Returns:
        List of *num_stages* drop probabilities increasing from 0 to
        *max_drop_rate*.

    Example::

        rates = build_stochastic_depth_schedule(12, max_drop_rate=0.3)
        drop_paths = [DropPath(r) for r in rates]
    """
    if num_stages <= 0:
        return []
    return [max_drop_rate * i / (num_stages - 1) for i in range(num_stages)] if num_stages > 1 else [max_drop_rate]
