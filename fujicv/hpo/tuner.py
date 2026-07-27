"""Optuna-based hyperparameter search for FujiCV Trainer."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def run_hpo(
    objective_fn: Callable,
    n_trials: int = 20,
    direction: str = "maximize",
    study_name: str = "fujicv_hpo",
    storage: Optional[str] = None,
    timeout: Optional[int] = None,
    pruner: Optional[str] = None,
    pruner_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimisation.

    Args:
        objective_fn: Callable ``fn(trial) -> float`` that builds and trains
            a model using ``trial.suggest_*`` calls and returns the metric.
        n_trials: Number of trials (default 20).
        direction: ``'maximize'`` or ``'minimize'``.
        study_name: Optuna study name.
        storage: Optional Optuna storage URL (e.g. ``'sqlite:///hpo.db'``).
        timeout: Optional wall-clock timeout in seconds.
        pruner: Pruning strategy name.  Supported: ``'median'``,
            ``'hyperband'``, ``'percentile'``, ``'successive_halving'``,
            ``None`` (no pruning, default).
        pruner_kwargs: Extra keyword arguments forwarded to the pruner
            constructor (e.g. ``{"n_startup_trials": 5}`` for median pruner).
        n_jobs: Number of parallel jobs (requires a shared *storage*).

    Returns:
        Dict with keys ``best_params``, ``best_value``, and ``study``.

    Example::

        from fujicv.hpo.tuner import run_hpo

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            # ... build model, train, return val_accuracy
            return val_accuracy

        result = run_hpo(objective, n_trials=30, pruner="median")
        print(result["best_params"])
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for HPO. Install with: pip install optuna"
        ) from exc

    _pruner = _build_pruner(pruner, pruner_kwargs or {})

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=_pruner,
    )
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    logger.info("Best trial: value=%.4f params=%s", study.best_value, study.best_params)
    return {
        "best_params": study.best_params,
        "best_value":  study.best_value,
        "study":       study,
    }


# ---------------------------------------------------------------------------
# Pruner factory
# ---------------------------------------------------------------------------

def _build_pruner(name: Optional[str], kwargs: Dict[str, Any]):
    """Instantiate an Optuna pruner by name."""
    if name is None:
        return None
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required. Install with: pip install optuna") from exc

    pruners = {
        "median":            optuna.pruners.MedianPruner,
        "hyperband":         optuna.pruners.HyperbandPruner,
        "percentile":        optuna.pruners.PercentilePruner,
        "successive_halving": optuna.pruners.SuccessiveHalvingPruner,
    }
    if name not in pruners:
        raise ValueError(
            f"Unknown pruner '{name}'. Choose from: {list(pruners)}"
        )
    return pruners[name](**kwargs)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_optimization_history(study, title: str = "Optimization History"):
    """Plot best-value progression across trials.

    Args:
        study: Completed ``optuna.Study`` object.
        title: Plot title.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    import matplotlib.pyplot as plt

    values     = [t.value for t in study.trials if t.value is not None]
    trial_nums = [t.number for t in study.trials if t.value is not None]

    best_so_far: List[float] = []
    current_best = float("-inf") if study.direction.name == "MAXIMIZE" else float("inf")
    for v in values:
        if study.direction.name == "MAXIMIZE":
            current_best = max(current_best, v)
        else:
            current_best = min(current_best, v)
        best_so_far.append(current_best)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(trial_nums, values, alpha=0.5, label="Trial value", s=20)
    ax.plot(trial_nums, best_so_far, color="red", label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig


def plot_param_importances(study, title: str = "Hyperparameter Importances"):
    """Plot hyperparameter importance (requires ``optuna`` ≥ 3.0).

    Args:
        study: Completed ``optuna.Study`` object with ≥ 2 finished trials.
        title: Plot title.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    import matplotlib.pyplot as plt

    try:
        import optuna
        importances = optuna.importance.get_param_importances(study)
    except Exception as exc:
        raise RuntimeError(
            "Could not compute param importances. "
            "Ensure optuna>=3.0 and at least 2 finished trials."
        ) from exc

    params  = list(importances.keys())
    weights = list(importances.values())

    # Sort descending
    pairs   = sorted(zip(weights, params), reverse=True)
    weights = [p[0] for p in pairs]
    params  = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, max(3, len(params) * 0.5)))
    ax.barh(params[::-1], weights[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    return fig
