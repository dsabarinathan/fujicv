"""Metric-learning margin heads (ArcFace, CosFace, SubCenter-ArcFace).

These heads replace a plain linear classifier with an angular/cosine-margin
formulation that produces highly discriminative, well-separated embeddings.
They are the workhorse of fine-grained classification and image-retrieval
competitions (Google Landmark, Humpback Whale, Happywhale, …).

Unlike a standard head, a margin head's ``forward`` takes **both** the feature
embedding and the integer labels. During training the additive margin is
applied to the ground-truth class; at inference (``labels=None``) the head
returns the plain scaled cosine similarities, which can be used directly as
logits for ``argmax`` or as a retrieval score.

Example (custom training step)::

    import torch.nn.functional as F
    from fujicv.models.metric_heads import ArcMarginProduct

    head = ArcMarginProduct(in_features=512, num_classes=5000, s=30.0, m=0.5)

    # training
    embeddings = backbone(images)          # (B, 512)
    logits = head(embeddings, labels)      # margin applied to target class
    loss = F.cross_entropy(logits, labels)

    # inference
    logits = head(embeddings)              # labels=None → plain cosine * s
    preds = logits.argmax(dim=1)

References:
    Deng et al., "ArcFace: Additive Angular Margin Loss" (CVPR 2019).
    Wang et al., "CosFace: Large Margin Cosine Loss" (CVPR 2018).
    Deng et al., "Sub-center ArcFace" (ECCV 2020).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """ArcFace head — additive angular margin on the target class.

    Args:
        in_features: Dimension of the input embedding.
        num_classes: Number of classes.
        s: Feature-scale (radius of the hypersphere). Typical: 30–64.
        m: Angular margin in radians. Typical: 0.3–0.5.
        easy_margin: If ``True``, use the softer easy-margin variant that only
            guards ``cosine > 0``; otherwise use the numerically safer hard
            margin from the ArcFace paper (default ``False``).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precomputed margin constants.
        self._cos_m = math.cos(m)
        self._sin_m = math.sin(m)
        self._th = math.cos(math.pi - m)          # threshold for hard margin
        self._mm = math.sin(math.pi - m) * m      # penalty beyond threshold

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            return cosine * self.s

        # clamp_min guards fp16/autocast where (1 - cosine²) can round negative → NaN.
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-7))
        phi = cosine * self._cos_m - sine * self._sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self._th, phi, cosine - self._mm)

        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(cosine.dtype)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, num_classes={self.num_classes}, "
            f"s={self.s}, m={self.m}, easy_margin={self.easy_margin}"
        )


class AddMarginProduct(nn.Module):
    """CosFace head — additive cosine margin on the target class.

    Args:
        in_features: Dimension of the input embedding.
        num_classes: Number of classes.
        s: Feature-scale. Typical: 30–64.
        m: Cosine margin. Typical: 0.35.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.35,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        if labels is None:
            return cosine * self.s

        phi = cosine - self.m
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(cosine.dtype)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, num_classes={self.num_classes}, "
            f"s={self.s}, m={self.m}"
        )


# CosFace is commonly referred to by both names.
CosMarginProduct = AddMarginProduct


class SubCenterArcMarginProduct(nn.Module):
    """Sub-center ArcFace — ``K`` sub-centers per class for noisy-label robustness.

    Each class owns ``K`` weight vectors; the per-class similarity is the
    maximum cosine over its sub-centers before the angular margin is applied.
    This tolerates intra-class variation and label noise better than vanilla
    ArcFace.

    Args:
        in_features: Dimension of the input embedding.
        num_classes: Number of classes.
        k: Number of sub-centers per class (default 3).
        s: Feature-scale.
        m: Angular margin in radians.
        easy_margin: Use the easy-margin variant (default ``False``).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        k: int = 3,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.in_features = in_features
        self.num_classes = num_classes
        self.k = k
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(num_classes * k, in_features))
        nn.init.xavier_uniform_(self.weight)

        self._cos_m = math.cos(m)
        self._sin_m = math.sin(m)
        self._th = math.cos(math.pi - m)
        self._mm = math.sin(math.pi - m) * m

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        # (B, num_classes * k) → (B, num_classes, k) → max over sub-centers.
        cosine = cosine_all.view(-1, self.num_classes, self.k).max(dim=2).values
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            return cosine * self.s

        # clamp_min guards fp16/autocast where (1 - cosine²) can round negative → NaN.
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-7))
        phi = cosine * self._cos_m - sine * self._sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self._th, phi, cosine - self._mm)

        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(cosine.dtype)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, num_classes={self.num_classes}, "
            f"k={self.k}, s={self.s}, m={self.m}, easy_margin={self.easy_margin}"
        )


__all__ = [
    "ArcMarginProduct",
    "AddMarginProduct",
    "CosMarginProduct",
    "SubCenterArcMarginProduct",
]
