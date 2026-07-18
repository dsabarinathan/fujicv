"""Training callbacks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        monitor: Metric name to monitor (e.g. ``'val_loss'``).
        mode: ``'min'`` (lower is better) or ``'max'`` (higher is better).
        min_delta: Minimum change to qualify as improvement (default 0.0).
    """

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self._counter = 0
        self._best: Optional[float] = None

    def step(self, metrics: Dict[str, float]) -> bool:
        """Check whether to stop training.

        Args:
            metrics: Dict of current epoch metrics.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if self.monitor not in metrics:
            logger.warning("EarlyStopping: monitor key '%s' not in metrics.", self.monitor)
            return False

        current = metrics[self.monitor]

        if self._best is None:
            self._best = current
            return False

        if self.mode == "min":
            improved = current < self._best - self.min_delta
        else:
            improved = current > self._best + self.min_delta

        if improved:
            self._best = current
            self._counter = 0
        else:
            self._counter += 1
            logger.info(
                "EarlyStopping counter: %d / %d (best %s=%.6f)",
                self._counter,
                self.patience,
                self.monitor,
                self._best,
            )

        return self._counter >= self.patience

    def reset(self) -> None:
        self._counter = 0
        self._best = None


class CheckpointCallback:
    """Save model checkpoints when a monitored metric improves.

    Args:
        output_dir: Directory to write checkpoint files.
        monitor: Metric name to monitor.
        mode: ``'min'`` or ``'max'``.
        filename: Checkpoint filename template (default ``'best.pt'``).
    """

    def __init__(
        self,
        output_dir: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        filename: str = "best.pt",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self._best: Optional[float] = None

    def step(
        self,
        metrics: Dict[str, float],
        model: nn.Module,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Conditionally save a checkpoint.

        Args:
            metrics: Current epoch metrics.
            model: Model to checkpoint.
            extra: Additional data to include in the checkpoint.

        Returns:
            ``True`` if a checkpoint was saved.
        """
        if self.monitor not in metrics:
            return False

        current = metrics[self.monitor]

        if self._best is None or (
            self.mode == "min" and current < self._best
        ) or (
            self.mode == "max" and current > self._best
        ):
            self._best = current
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                self.monitor: current,
            }
            if extra:
                payload.update(extra)
            path = self.output_dir / self.filename
            torch.save(payload, path)
            logger.info("Checkpoint saved to %s (%s=%.6f)", path, self.monitor, current)
            return True
        return False


class LRSchedulerCallback:
    """Advance a learning-rate scheduler after each epoch.

    Args:
        scheduler: A ``torch.optim.lr_scheduler`` instance.
        monitor: Metric to pass to ``ReduceLROnPlateau`` schedulers (optional).
    """

    def __init__(
        self,
        scheduler: Any,
        monitor: str = "val_loss",
    ) -> None:
        self.scheduler = scheduler
        self.monitor = monitor

    def step(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """Step the scheduler.

        Args:
            metrics: Current epoch metrics (required for ``ReduceLROnPlateau``).
        """
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metrics and self.monitor in metrics:
                self.scheduler.step(metrics[self.monitor])
            else:
                logger.warning(
                    "LRSchedulerCallback: ReduceLROnPlateau requires monitor '%s' in metrics.",
                    self.monitor,
                )
        else:
            self.scheduler.step()
