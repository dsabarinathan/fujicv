"""Unit tests for the HPO module (fujicv.hpo)."""

from __future__ import annotations

import pytest


def test_run_hpo_missing_optuna(monkeypatch):
    """run_hpo raises ImportError with helpful message when optuna is absent."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("No module named 'optuna'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from fujicv.hpo import tuner as _tuner
    import importlib
    importlib.reload(_tuner)

    with pytest.raises(ImportError, match="optuna"):
        _tuner.run_hpo(lambda trial: 0.0, n_trials=1)


def test_run_hpo_with_optuna():
    """run_hpo returns best_params and best_value when optuna is available."""
    pytest.importorskip("optuna")

    from fujicv.hpo.tuner import run_hpo

    def objective(trial):
        x = trial.suggest_float("x", -5.0, 5.0)
        return -(x ** 2)  # maximise → best at x≈0

    result = run_hpo(objective, n_trials=5, direction="maximize", study_name="test_study")
    assert "best_params" in result
    assert "best_value" in result
    assert "study" in result
    assert "x" in result["best_params"]
    assert result["best_value"] <= 0  # -(x^2) is always ≤ 0


def test_run_hpo_minimize():
    """run_hpo works with direction='minimize'."""
    pytest.importorskip("optuna")

    from fujicv.hpo.tuner import run_hpo

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 10.0)
        return x ** 2

    result = run_hpo(objective, n_trials=5, direction="minimize", study_name="test_min")
    assert result["best_value"] >= 0
    assert result["best_value"] < 100  # should be better than worst case
