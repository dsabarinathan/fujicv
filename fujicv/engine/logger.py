"""Experiment logging (W&B wrapper + console fallback)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class WandbLogger:
    """Thin wrapper around Weights & Biases for experiment tracking.

    Authentication is read **exclusively** from the ``WANDB_API_KEY``
    environment variable.  The key is never accepted as a constructor argument,
    stored in attributes, or written to any file.

    Args:
        project: W&B project name.
        entity: W&B entity (team or username). ``None`` uses the default.
        config: Flat or nested dict of hyper-parameters to log.
        use_wandb: Set to ``False`` to disable W&B entirely (all methods
            become no-ops). Also automatically disabled if ``wandb`` is not
            installed or ``WANDB_API_KEY`` is absent.
        run_name: Optional display name for the run.
        tags: Optional list of string tags.
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = True,
        run_name: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> None:
        self._run = None
        self._active = False

        if not use_wandb:
            logger.info("WandbLogger: W&B disabled by use_wandb=False.")
            return

        try:
            import wandb  # type: ignore
        except ImportError:
            logger.warning(
                "WandbLogger: 'wandb' is not installed. Logging will be skipped. "
                "Install with: pip install wandb"
            )
            return

        if not os.environ.get("WANDB_API_KEY"):
            logger.warning(
                "WandbLogger: WANDB_API_KEY environment variable is not set. "
                "W&B logging will be skipped."
            )
            return

        try:
            self._run = wandb.init(
                project=project,
                entity=entity,
                config=config or {},
                name=run_name,
                tags=tags,
            )
            self._active = True
            logger.info("WandbLogger: run started — %s", self._run.url)
        except Exception as exc:
            logger.warning("WandbLogger: failed to initialise W&B run: %s", exc)

    @property
    def active(self) -> bool:
        """Return ``True`` if W&B logging is active."""
        return self._active

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log per-epoch metrics.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Dict of metric name → value.
        """
        if not self._active or self._run is None:
            return
        try:
            self._run.log({"epoch": epoch, **metrics})
        except Exception as exc:
            logger.warning("WandbLogger.log_epoch failed: %s", exc)

    def log_artifact(self, path: str | Path, name: str, artifact_type: str) -> None:
        """Upload a file or directory as a W&B artifact.

        Args:
            path: Local path to upload.
            name: Artifact name.
            artifact_type: Artifact type (e.g. ``'model'``, ``'dataset'``).
        """
        if not self._active or self._run is None:
            return
        try:
            import wandb  # type: ignore

            artifact = wandb.Artifact(name=name, type=artifact_type)
            path = Path(path)
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self._run.log_artifact(artifact)
        except Exception as exc:
            logger.warning("WandbLogger.log_artifact failed: %s", exc)

    def finish(self) -> None:
        """Finalise and close the W&B run."""
        if not self._active or self._run is None:
            return
        try:
            self._run.finish()
        except Exception as exc:
            logger.warning("WandbLogger.finish failed: %s", exc)
        finally:
            self._active = False
