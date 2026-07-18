"""Core training loop."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from fujicv.engine.callbacks import CheckpointCallback, EarlyStopping, LRSchedulerCallback
from fujicv.engine.logger import WandbLogger
from fujicv.utils.seed import get_device

logger = logging.getLogger(__name__)


@dataclass
class History:
    """Container for per-epoch training history.

    Attributes:
        metrics: Dict mapping metric name → list of per-epoch values.
    """

    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def update(self, epoch_metrics: Dict[str, float]) -> None:
        for k, v in epoch_metrics.items():
            self.metrics.setdefault(k, []).append(v)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        if not self.metrics:
            return
        keys = list(self.metrics.keys())
        rows = zip(*[self.metrics[k] for k in keys])
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["epoch"] + keys)
            writer.writeheader()
            for i, row in enumerate(rows):
                writer.writerow({"epoch": i, **dict(zip(keys, row))})


class Trainer:
    """Full training loop with mixed precision, checkpointing, and callbacks.

    Args:
        model: The ``nn.Module`` to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        loss_fn: Loss ``nn.Module``.
        metrics: Dict of metric name → callable ``(y_true, y_pred) → float``.
            ``y_pred`` will be **logits** (numpy arrays).
        optimizer: PyTorch optimiser.
        scheduler: (optional) LR scheduler or ``LRSchedulerCallback``.
        device: Target device (default ``'cuda'`` if available else ``'cpu'``).
        epochs: Number of training epochs.
        task: Task type — ``'classification'``, ``'regression'``, ``'multilabel'``,
            or ``'multiclass'``.
        output_dir: Directory for checkpoints and history CSV.
        wandb_logger: (optional) ``WandbLogger`` instance.
        mixed_precision: Enable automatic mixed precision (default ``True``).
        grad_clip: Gradient clipping max norm (default 1.0, ``None`` to disable).
        monitor_metric: Metric to monitor for checkpointing / early stopping
            (default ``'val_loss'``).
        resume_from: Path to a checkpoint to resume from.
        early_stopping_patience: Epochs without improvement before stopping
            (default ``None`` = disabled).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        metrics: Dict[str, Callable],
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        device: Optional[str] = None,
        epochs: int = 10,
        task: str = "classification",
        output_dir: str | Path = "outputs",
        wandb_logger: Optional[WandbLogger] = None,
        mixed_precision: bool = True,
        grad_clip: Optional[float] = 1.0,
        monitor_metric: str = "val_loss",
        resume_from: Optional[str | Path] = None,
        early_stopping_patience: Optional[int] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer = optimizer
        self.epochs = epochs
        self.task = task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_logger = wandb_logger
        self.mixed_precision = mixed_precision
        self.grad_clip = grad_clip
        self.monitor_metric = monitor_metric
        self.class_to_idx = class_to_idx or {}

        self.device = get_device(device)
        self.model.to(self.device)

        # AMP scaler (only for CUDA)
        self._use_amp = mixed_precision and self.device.type == "cuda"
        self._scaler = GradScaler(enabled=self._use_amp)

        # Checkpoint callback
        monitor_mode = "min" if "loss" in monitor_metric else "max"
        self._ckpt = CheckpointCallback(
            self.output_dir, monitor=monitor_metric, mode=monitor_mode, filename="best.pt"
        )

        # LR scheduler callback
        if scheduler is not None:
            if isinstance(scheduler, LRSchedulerCallback):
                self._lr_callback: Optional[LRSchedulerCallback] = scheduler
            else:
                self._lr_callback = LRSchedulerCallback(scheduler, monitor=monitor_metric)
        else:
            self._lr_callback = None

        # Early stopping
        self._early_stop: Optional[EarlyStopping] = None
        if early_stopping_patience is not None:
            self._early_stop = EarlyStopping(
                patience=early_stopping_patience,
                monitor=monitor_metric,
                mode=monitor_mode,
            )

        self._start_epoch = 0
        self.history = History()

        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._start_epoch = ckpt.get("epoch", 0) + 1
        if "history" in ckpt:
            self.history = ckpt["history"]
        logger.info("Resumed from checkpoint %s (epoch %d)", path, self._start_epoch)

    def _save_last_checkpoint(self, epoch: int) -> None:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "history": self.history,
            "class_to_idx": self.class_to_idx,
        }
        torch.save(payload, self.output_dir / "last.pt")

    # ------------------------------------------------------------------
    # Single epoch helpers
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
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast(enabled=self._use_amp):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, targets)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self._scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        self._scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self._scaler.step(self.optimizer)
                    self._scaler.update()

                total_loss += loss.item() * images.size(0)
                all_preds.append(logits.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())

        n = len(loader.dataset)
        avg_loss = total_loss / max(n, 1)

        preds_arr = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)

        prefix = "train" if training else "val"
        result = {f"{prefix}_loss": avg_loss}
        for name, fn in self.metrics.items():
            try:
                result[f"{prefix}_{name}"] = float(fn(targets_arr, preds_arr))
            except Exception as exc:
                logger.warning("Metric '%s' failed: %s", name, exc)
                result[f"{prefix}_{name}"] = float("nan")

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> History:
        """Run the full training loop.

        Returns:
            A :class:`History` object containing per-epoch metric values.
        """
        logger.info(
            "Starting training: epochs=%d  device=%s  amp=%s",
            self.epochs,
            self.device,
            self._use_amp,
        )

        for epoch in range(self._start_epoch, self.epochs):
            t0 = time.time()
            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_epoch(self.val_loader, training=False)
            epoch_metrics = {**train_metrics, **val_metrics}
            self.history.update(epoch_metrics)

            elapsed = time.time() - t0
            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items())
            logger.info("Epoch %d/%d  [%.1fs]  %s", epoch + 1, self.epochs, elapsed, metric_str)

            # Checkpoint
            self._ckpt.step(
                epoch_metrics,
                self.model,
                extra={
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "history": self.history,
                    "class_to_idx": self.class_to_idx,
                },
            )
            self._save_last_checkpoint(epoch)

            # LR scheduler
            if self._lr_callback is not None:
                self._lr_callback.step(epoch_metrics)

            # W&B logging
            if self.wandb_logger is not None:
                self.wandb_logger.log_epoch(epoch, epoch_metrics)

            # Early stopping
            if self._early_stop is not None and self._early_stop.step(epoch_metrics):
                logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

        # Save history CSV when not using W&B
        if self.wandb_logger is None or not self.wandb_logger.active:
            self.history.to_csv(self.output_dir / "history.csv")

        if self.wandb_logger is not None:
            self.wandb_logger.finish()

        logger.info("Training complete. Outputs saved to %s", self.output_dir)
        return self.history
