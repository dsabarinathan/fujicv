"""Unit tests for metric functions."""

from __future__ import annotations

import numpy as np
import pytest

# ---- Classification metrics -----------------------------------------------

def test_accuracy_perfect():
    from fujicv.metrics.classification import Accuracy
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])  # logits
    assert Accuracy()(y_true, y_pred) == 1.0


def test_accuracy_half():
    from fujicv.metrics.classification import Accuracy
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
    acc = Accuracy()(y_true, y_pred)
    assert abs(acc - 0.75) < 1e-6


def test_balanced_accuracy():
    from fujicv.metrics.classification import BalancedAccuracy
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([[1, 0], [1, 0], [0, 1], [1, 0]])
    ba = BalancedAccuracy()(y_true, y_pred)
    assert 0.0 <= ba <= 1.0


def test_f1_perfect():
    from fujicv.metrics.classification import F1
    y_true = np.array([0, 1, 2])
    y_pred = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert abs(F1()(y_true, y_pred) - 1.0) < 1e-6


def test_topk_accuracy():
    from fujicv.metrics.classification import TopKAccuracy
    y_true = np.array([2, 0])
    y_pred = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
    assert TopKAccuracy(k=1)(y_true, y_pred) == 1.0


def test_mcc_perfect():
    from fujicv.metrics.classification import MCC
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    assert abs(MCC()(y_true, y_pred) - 1.0) < 1e-6


def test_cohen_kappa():
    from fujicv.metrics.classification import CohenKappa
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ])
    kappa = CohenKappa()(y_true, y_pred)
    assert abs(kappa - 1.0) < 1e-6


# ---- Regression metrics ---------------------------------------------------

def test_mae_zero():
    from fujicv.metrics.regression import MAE
    y = np.array([1.0, 2.0, 3.0])
    assert MAE()(y, y) == 0.0


def test_mse_zero():
    from fujicv.metrics.regression import MSE
    y = np.array([1.0, 2.0, 3.0])
    assert MSE()(y, y) == 0.0


def test_rmse():
    from fujicv.metrics.regression import RMSE
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 3.0])
    expected = np.sqrt(1.0 / 3.0)
    assert abs(RMSE()(y_true, y_pred) - expected) < 1e-6


def test_r2_perfect():
    from fujicv.metrics.regression import R2Score
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(R2Score()(y, y) - 1.0) < 1e-6


def test_mape():
    from fujicv.metrics.regression import MAPE
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 200.0])
    result = MAPE()(y_true, y_pred)
    # (10/100 + 0/200) / 2 * 100 = 5%
    assert abs(result - 5.0) < 1e-4


def test_pearson_corr_perfect():
    from fujicv.metrics.regression import PearsonCorr
    y = np.arange(10.0)
    assert abs(PearsonCorr()(y, y) - 1.0) < 1e-6


def test_spearman_corr():
    from fujicv.metrics.regression import SpearmanCorr
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert abs(SpearmanCorr()(y_true, y_pred) - 1.0) < 1e-6


# ---- Multi-label metrics --------------------------------------------------

def test_hamming_loss_zero():
    from fujicv.metrics.multilabel import HammingLoss
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.1]])  # perfect
    assert HammingLoss(threshold=0.5)(y_true, y_pred) == 0.0


def test_hamming_loss_nonzero():
    from fujicv.metrics.multilabel import HammingLoss
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[0.1, 0.9], [0.1, 0.9]])  # half wrong
    hl = HammingLoss()(y_true, y_pred)
    assert 0.0 < hl <= 1.0


def test_subset_accuracy_perfect():
    from fujicv.metrics.multilabel import SubsetAccuracy
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
    assert SubsetAccuracy()(y_true, y_pred) == 1.0


def test_map():
    from fujicv.metrics.multilabel import mAP
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, (20, 4)).astype(float)
    y_pred = rng.random((20, 4))
    result = mAP()(y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_per_label_auroc():
    from fujicv.metrics.multilabel import PerLabelAUROC
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, (30, 3)).astype(float)
    # Ensure each column has both classes
    y_true[0, :] = 1
    y_true[1, :] = 0
    y_pred = rng.random((30, 3))
    result = PerLabelAUROC()(y_true, y_pred)
    assert 0.0 <= result <= 1.0


# ---- Registry -----------------------------------------------------------

def test_get_metric_registry():
    from fujicv.metrics import get_metric
    metric = get_metric("Accuracy")
    y_true = np.array([0, 1])
    y_pred = np.array([[1, 0], [0, 1]])
    assert metric(y_true, y_pred) == 1.0


def test_get_metric_unknown():
    from fujicv.metrics import get_metric
    with pytest.raises(KeyError):
        get_metric("NonExistentMetric")
