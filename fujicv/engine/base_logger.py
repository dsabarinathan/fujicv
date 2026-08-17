"""Abstract experiment-logger interface.

Any object implementing :class:`BaseLogger` can be passed to
``Trainer(..., loggers=[...])``. This lets FujiCV support MLflow, Neptune,
ClearML, or a bespoke tracker without the trainer knowing the specifics.

The built-in :class:`~fujicv.engine.logger.WandbLogger` and
:class:`~fujicv.engine.tensorboard_logger.TensorBoardLogger` already follow the
same ``log_epoch`` / ``finish`` / ``active`` contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict


class BaseLogger(ABC):
    """Minimal experiment-tracking interface.

    Implementations must be resilient: a failure to log should never crash
    training. Wrap backend calls in ``try/except`` and degrade to a no-op.
    """

    @property
    def active(self) -> bool:
        """Whether the backend is live. Defaults to ``True``."""
        return True

    @abstractmethod
    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log a dict of scalar metrics for *epoch* (0-indexed)."""

    def log_scalar(self, tag: str, value: float, step: int) -> None:  # noqa: B027
        """Log a single scalar. Optional hook — default is a no-op."""

    def log_artifact(self, path: str | Path, name: str, artifact_type: str) -> None:  # noqa: B027
        """Persist a file/dir artifact. Optional hook — default is a no-op."""

    @abstractmethod
    def finish(self) -> None:
        """Flush and close the backend."""
