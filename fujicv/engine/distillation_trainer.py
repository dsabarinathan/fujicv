"""Distillation-aware Trainer that feeds teacher logits to DistillationLoss."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fujicv.engine.trainer import History, Trainer
from fujicv.losses.distillation import DistillationLoss

logger = logging.getLogger(__name__)


class DistillationTrainer(Trainer):
    """Extends :class:`Trainer` with knowledge distillation support.

    The teacher model is kept frozen throughout training.  Each batch passes
    through both teacher (no grad) and student (with grad), and the
    :class:`~fujicv.losses.distillation.DistillationLoss` combines the soft
    KL-divergence loss with the hard cross-entropy loss.

    Args:
        teacher: Frozen teacher ``nn.Module``.  Will be set to ``eval()``
            and moved to the same device as the student.
        All other args are forwarded to :class:`Trainer`.

    Example::

        from fujicv.engine.distillation_trainer import DistillationTrainer
        from fujicv.losses.distillation import DistillationLoss

        trainer = DistillationTrainer(
            teacher=resnet50_pretrained,
            model=resnet18_student,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=DistillationLoss(alpha=0.7, temperature=4.0),
            metrics={"accuracy": Accuracy()},
            optimizer=optimizer,
            epochs=20,
            task="classification",
            output_dir="runs/distill",
        )
        history = trainer.train()
    """

    def __init__(self, teacher: nn.Module, **kwargs: Any) -> None:
        # Validate loss type before calling super().__init__
        loss_fn = kwargs.get("loss_fn")
        if loss_fn is not None and not isinstance(loss_fn, DistillationLoss):
            raise TypeError(
                f"DistillationTrainer requires a DistillationLoss, got {type(loss_fn).__name__}. "
                "Use: loss_fn=DistillationLoss(alpha=0.7, temperature=4.0)"
            )

        super().__init__(**kwargs)

        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.to(self.device)
        logger.info(
            "DistillationTrainer: teacher=%s → student=%s  device=%s",
            type(teacher).__name__,
            type(self.model).__name__,
            self.device,
        )

    # ------------------------------------------------------------------
    # Override _run_epoch to inject teacher logits
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                images, targets = batch
                images  = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Teacher forward (always no-grad)
                with torch.no_grad():
                    teacher_logits = self.teacher(images)

                # Student forward
                if self._use_amp:
                    from torch.cuda.amp import autocast
                    with autocast():
                        student_logits = self.model(images)
                        loss = self.loss_fn(student_logits, teacher_logits, targets)
                else:
                    student_logits = self.model(images)
                    loss = self.loss_fn(student_logits, teacher_logits, targets)

                if training:
                    self.optimizer.zero_grad()
                    if self._use_amp:
                        self._scaler.scale(loss).backward()
                        if self.grad_clip:
                            self._scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                all_preds.append(student_logits.detach().cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        n = sum(len(t) for t in all_targets)
        avg_loss = total_loss / max(n, 1)
        preds_np   = np.concatenate(all_preds)
        targets_np = np.concatenate(all_targets)

        result = {"loss": avg_loss}
        for name, metric_fn in self.metrics.items():
            try:
                result[name] = float(metric_fn(targets_np, preds_np))
            except Exception:
                result[name] = float("nan")
        return result
