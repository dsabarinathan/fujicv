"""TensorBoard experiment logging (offline-friendly W&B alternative)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """Thin wrapper around ``torch.utils.tensorboard.SummaryWriter``.

    Unlike :class:`~fujicv.engine.logger.WandbLogger`, TensorBoard runs fully
    offline — no account, no API key, no network. Logs are written to a local
    directory and viewed with ``tensorboard --logdir <log_dir>``.

    All methods are no-ops if TensorBoard is unavailable, so training never
    fails just because logging could not be initialised.

    Args:
        log_dir: Directory where event files are written (default
            ``'runs/tensorboard'``).
        config: Optional flat dict of hyper-parameters. Logged once as text
            and, where all values are numeric, as hparams.
        comment: Optional suffix appended to the auto-generated run directory
            when ``log_dir`` is left at its default.

    Example::

        from fujicv.engine.tensorboard_logger import TensorBoardLogger

        tb = TensorBoardLogger(log_dir="runs/exp1", config={"lr": 1e-3})
        trainer = Trainer(..., tb_logger=tb)
        history = trainer.train()
        # then: tensorboard --logdir runs/exp1
    """

    def __init__(
        self,
        log_dir: str | Path = "runs/tensorboard",
        config: Optional[Dict[str, Any]] = None,
        comment: str = "",
    ) -> None:
        self._writer = None
        self._active = False
        self._config = config or {}

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning(
                "TensorBoardLogger: TensorBoard is not available. Logging will "
                "be skipped. Install with: pip install tensorboard"
            )
            return

        try:
            self._writer = SummaryWriter(log_dir=str(log_dir), comment=comment)
            self._active = True
            if self._config:
                text = "\n".join(f"| {k} | {v} |" for k, v in self._config.items())
                self._writer.add_text(
                    "config", "| key | value |\n|---|---|\n" + text, global_step=0
                )
            logger.info("TensorBoardLogger: writing events to %s", log_dir)
        except Exception as exc:
            logger.warning("TensorBoardLogger: failed to initialise writer: %s", exc)

    @property
    def active(self) -> bool:
        """Return ``True`` if TensorBoard logging is active."""
        return self._active

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log per-epoch scalar metrics.

        Metric keys are grouped by their ``train_``/``val_`` prefix so paired
        curves (e.g. ``loss/train`` and ``loss/val``) share a chart in the UI.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Dict of metric name → value.
        """
        if not self._active or self._writer is None:
            return
        try:
            for key, value in metrics.items():
                tag = self._grouped_tag(key)
                self._writer.add_scalar(tag, value, global_step=epoch)
            self._writer.flush()
        except Exception as exc:
            logger.warning("TensorBoardLogger.log_epoch failed: %s", exc)

    @staticmethod
    def _grouped_tag(key: str) -> str:
        """Turn ``train_loss`` → ``loss/train`` for grouped charts."""
        for split in ("train_", "val_", "test_"):
            if key.startswith(split):
                return f"{key[len(split):]}/{split[:-1]}"
        return key

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log an arbitrary scalar under *tag* at *step*."""
        if not self._active or self._writer is None:
            return
        try:
            self._writer.add_scalar(tag, value, global_step=step)
        except Exception as exc:
            logger.warning("TensorBoardLogger.log_scalar failed: %s", exc)

    def finish(self) -> None:
        """Flush and close the writer."""
        if not self._active or self._writer is None:
            return
        try:
            self._writer.close()
        except Exception as exc:
            logger.warning("TensorBoardLogger.finish failed: %s", exc)
        finally:
            self._active = False
