"""MLflow experiment logger (implements :class:`BaseLogger`)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fujicv.engine.base_logger import BaseLogger

logger = logging.getLogger(__name__)


class MLflowLogger(BaseLogger):
    """Log metrics and artifacts to MLflow tracking.

    All methods degrade to no-ops if ``mlflow`` is not installed, so training
    never fails because of logging.

    Args:
        experiment_name: MLflow experiment to log under (created if absent).
        run_name: Optional display name for the run.
        tracking_uri: Optional tracking server URI (e.g.
            ``'http://localhost:5000'`` or a local ``'file:./mlruns'`` path).
            ``None`` uses MLflow's default (``./mlruns``).
        params: Optional flat dict of hyper-parameters logged once at start.
        tags: Optional dict of run tags.

    Example::

        from fujicv.engine.mlflow_logger import MLflowLogger

        mlf = MLflowLogger("fujicv-experiments", run_name="resnet50-run1",
                           params={"lr": 1e-3, "backbone": "resnet50"})
        trainer = Trainer(..., loggers=[mlf])
        trainer.train()
    """

    def __init__(
        self,
        experiment_name: str = "fujicv",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self._mlflow = None
        self._active = False

        try:
            import mlflow
        except ImportError:
            logger.warning(
                "MLflowLogger: 'mlflow' is not installed. Logging will be "
                'skipped. Install with: pip install "fujicv[mlflow]"'
            )
            return

        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name)
            if params:
                mlflow.log_params(params)
            if tags:
                mlflow.set_tags(tags)
            self._mlflow = mlflow
            self._active = True
            logger.info("MLflowLogger: run started under experiment '%s'.", experiment_name)
        except Exception as exc:
            logger.warning("MLflowLogger: failed to start run: %s", exc)

    @property
    def active(self) -> bool:
        return self._active

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        if not self._active or self._mlflow is None:
            return
        try:
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=epoch)
        except Exception as exc:
            logger.warning("MLflowLogger.log_epoch failed: %s", exc)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if not self._active or self._mlflow is None:
            return
        try:
            self._mlflow.log_metric(tag, float(value), step=step)
        except Exception as exc:
            logger.warning("MLflowLogger.log_scalar failed: %s", exc)

    def log_artifact(self, path: str | Path, name: str, artifact_type: str) -> None:
        if not self._active or self._mlflow is None:
            return
        try:
            p = Path(path)
            if p.is_dir():
                self._mlflow.log_artifacts(str(p), artifact_path=name)
            else:
                self._mlflow.log_artifact(str(p), artifact_path=name)
        except Exception as exc:
            logger.warning("MLflowLogger.log_artifact failed: %s", exc)

    def finish(self) -> None:
        if not self._active or self._mlflow is None:
            return
        try:
            self._mlflow.end_run()
        except Exception as exc:
            logger.warning("MLflowLogger.finish failed: %s", exc)
        finally:
            self._active = False
