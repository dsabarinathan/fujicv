"""Tests for Optuna HPO tuner enhancements (pruning + visualization)."""
from __future__ import annotations

import pytest


def _is_optuna_available() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


skip_no_optuna = pytest.mark.skipif(
    not _is_optuna_available(), reason="optuna not installed"
)


@skip_no_optuna
def test_run_hpo_basic():
    from fujicv.hpo.tuner import run_hpo

    def objective(trial):
        x = trial.suggest_float("x", -5, 5)
        return -(x ** 2)

    result = run_hpo(objective, n_trials=5, direction="maximize")
    assert "best_params" in result
    assert "best_value"  in result
    assert "study"       in result
    assert result["best_value"] <= 0.0


@skip_no_optuna
def test_run_hpo_with_median_pruner():
    from fujicv.hpo.tuner import run_hpo

    def objective(trial):
        return trial.suggest_float("x", 0, 1)

    result = run_hpo(objective, n_trials=5, direction="maximize", pruner="median")
    assert result["best_value"] is not None


@skip_no_optuna
def test_run_hpo_invalid_pruner_raises():
    from fujicv.hpo.tuner import run_hpo

    with pytest.raises(ValueError, match="Unknown pruner"):
        run_hpo(lambda t: 0.0, n_trials=1, pruner="nonexistent_pruner")


@skip_no_optuna
def test_run_hpo_minimize():
    from fujicv.hpo.tuner import run_hpo

    def objective(trial):
        x = trial.suggest_float("x", -5, 5)
        return x ** 2

    result = run_hpo(objective, n_trials=5, direction="minimize")
    assert result["best_value"] >= 0.0


@skip_no_optuna
def test_plot_optimization_history():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fujicv.hpo.tuner import plot_optimization_history, run_hpo

    result = run_hpo(
        lambda t: t.suggest_float("x", 0, 1),
        n_trials=5,
        direction="maximize",
    )
    fig = plot_optimization_history(result["study"])
    assert isinstance(fig, plt.Figure)
    plt.close("all")


@skip_no_optuna
def test_plot_param_importances():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fujicv.hpo.tuner import plot_param_importances, run_hpo

    result = run_hpo(
        lambda t: t.suggest_float("lr", 1e-4, 1e-1, log=True)
                + t.suggest_float("wd", 0, 0.1),
        n_trials=10,
        direction="maximize",
    )
    fig = plot_param_importances(result["study"])
    assert isinstance(fig, plt.Figure)
    plt.close("all")
