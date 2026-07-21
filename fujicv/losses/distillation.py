"""Knowledge Distillation loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fujicv.losses.registry import register_loss


@register_loss("DistillationLoss")
class DistillationLoss(nn.Module):
    """Hinton-style knowledge distillation loss.

    Combines a soft target loss (KL divergence between student and teacher
    logits at temperature *T*) with a hard target loss (cross-entropy against
    ground-truth labels).

    Reference: Hinton et al., "Distilling the Knowledge in a Neural Network"
    (2015). https://arxiv.org/abs/1503.02531

    Args:
        alpha: Weight of the soft distillation loss (default 0.7).
            The hard label loss weight is ``1 - alpha``.
        temperature: Softmax temperature for smoothing logits (default 4.0).
            Higher values produce softer probability distributions.

    Shape:
        - student_logits: ``(N, C)``
        - teacher_logits: ``(N, C)``
        - targets:        ``(N,)`` integer class labels

    Example::

        from fujicv.losses.distillation import DistillationLoss

        loss_fn = DistillationLoss(alpha=0.7, temperature=4.0)

        # Inside training loop:
        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)
        loss = loss_fn(student_logits, teacher_logits, targets)
    """

    def __init__(self, alpha: float = 0.7, temperature: float = 4.0) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        T = self.temperature

        # Soft loss: KL(student || teacher) scaled by T²
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

        # Hard loss: standard cross-entropy with true labels
        hard_loss = F.cross_entropy(student_logits, targets)

        return self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, temperature={self.temperature}"


@register_loss("FeatureDistillationLoss")
class FeatureDistillationLoss(nn.Module):
    """Intermediate feature-map distillation using MSE.

    Minimises the mean squared error between corresponding intermediate
    feature tensors of the teacher and student.  Useful when teacher and
    student share the same architecture (e.g. self-distillation) or when
    a projection layer is used to align channel dimensions.

    Args:
        projector: Optional ``nn.Module`` that maps student features to the
            teacher's feature space.  Required when channel dims differ.

    Shape:
        - student_feat: ``(N, C_s, ...)``
        - teacher_feat: ``(N, C_t, ...)``  (must match after projector)

    Example::

        proj = nn.Linear(256, 512)  # student 256-d → teacher 512-d
        loss_fn = FeatureDistillationLoss(projector=proj)
        loss = loss_fn(student_feat, teacher_feat)
    """

    def __init__(self, projector: nn.Module | None = None) -> None:
        super().__init__()
        self.projector = projector

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        s = self.projector(student_feat) if self.projector is not None else student_feat
        return F.mse_loss(s, teacher_feat.detach())
