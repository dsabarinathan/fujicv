"""Optuna-based hyperparameter search for FujiCV Trainer."""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def run_hpo(
    objective_fn: Callable,
    n_trials: int = 20,
    direction: str = "maximize",
    study_name: str = "fujicv_hpo",
    storage: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimisation.

    Args:
        objective_fn: A callable ``fn(trial) -> float`` that builds and trains
            a model using ``trial.suggest_*`` calls and returns the target metric.
        n_trials: Number of trials (default 20).
        direction: ``'maximize'`` or ``'minimize'``.
        study_name: Optuna study name.
        storage: Optional Optuna storage URL (e.g. ``'sqlite:///hpo.db'``).
        timeout: Optional timeout in seconds.

    Returns:
        Dict with ``best_params``, ``best_value``, and ``study``.

    Example::

        import optuna
        from fujicv.hpo.tuner import run_hpo
        from fujicv.models.builder import ModelBuilder
        from fujicv.engine.trainer import Trainer

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            model = ModelBuilder(
                backbone_name="resnet18", task="classification",
                num_outputs=10, head_kwargs={"dropout": dropout}
            ).build()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            trainer = Trainer(model=model, ..., optimizer=optimizer, epochs=5)
            history = trainer.train()
            return max(history.metrics.get("val_accuracy", [0]))

        result = run_hpo(objective, n_trials=20)
        print(result["best_params"])
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        raise ImportError(
            "Optuna is required for HPO. Install with: pip install optuna"
        ) from e

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective_fn, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    logger.info("Best trial: value=%.4f params=%s", study.best_value, study.best_params)
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }
