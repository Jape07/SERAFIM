"""The metrics that judge the dipole CNN (training/metrics.py): numpy only, so they run in CI."""

import numpy as np
import pytest

from training import metrics

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
PREDICTED = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0], bool)  # TP 3, FN 1, FP 1, TN 5
SCORE = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.4, 0.05, 0.15])


def test_the_confusion_matrix_and_its_metrics():
    m = metrics.binary_metrics(Y, PREDICTED, SCORE)
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (3, 1, 1, 5)
    assert m["prevalence"] == pytest.approx(0.4)
    assert m["sensitivity"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(5 / 6)
    assert m["PPV"] == pytest.approx(0.75) and m["NPV"] == pytest.approx(5 / 6)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["balanced_accuracy"] == pytest.approx((0.75 + 5 / 6) / 2)
    assert m["informedness"] == pytest.approx(0.75 + 5 / 6 - 1)
    assert m["markedness"] == pytest.approx(0.75 + 5 / 6 - 1)
    assert m["F1"] == pytest.approx(6 / 8) and m["jaccard"] == pytest.approx(3 / 5)
    assert m["fowlkes_mallows"] == pytest.approx(3 / np.sqrt(16))
    assert m["MCC"] == pytest.approx((15 - 1) / np.sqrt(4 * 4 * 6 * 6))


def test_auc_and_average_precision_match_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    assert metrics.roc_auc(Y, SCORE) == pytest.approx(sklearn.roc_auc_score(Y, SCORE))
    assert metrics.average_precision(Y, SCORE) == pytest.approx(
        sklearn.average_precision_score(Y, SCORE)
    )


def test_auc_without_sklearn():
    assert metrics.roc_auc(Y, SCORE) == pytest.approx(22 / 24)  # 0.6 and 0.4 above 0.3
    assert metrics.roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert np.isnan(metrics.roc_auc([1, 1], [0.2, 0.3]))


def test_precision_falls_when_dipoles_are_rare():
    """Bayes: the same sensitivity and specificity give a poor precision at a low prevalence."""
    assert metrics.ppv_at(0.62, 0.98, 0.45) == pytest.approx(0.962, abs=0.001)
    assert metrics.ppv_at(0.62, 0.98, 0.01) == pytest.approx(0.238, abs=0.005)
    assert metrics.ppv_at(0.62, 0.98, 0.001) < 0.05


def test_the_youden_threshold_maximizes_informedness():
    threshold = metrics.youden_threshold(Y, SCORE)
    assert threshold == pytest.approx(0.7)
    best = metrics.binary_metrics(Y, SCORE >= threshold)["informedness"]
    for other in (0.3, 0.6, 0.8):
        assert metrics.binary_metrics(Y, SCORE >= other)["informedness"] <= best + 1e-12


def test_wilson_interval_and_calibration():
    assert metrics.interval(9, 38) == "13%-39%"  # Wilson by hand: 0.1299 to 0.3921
    assert metrics.interval(0, 0) == "-"
    p = np.repeat([0.1, 0.9], 100)
    y = np.r_[np.zeros(90), np.ones(10), np.zeros(10), np.ones(90)]
    table, ece = metrics.reliability(y, p)
    assert ece == pytest.approx(0.0) and list(table["n"]) == [100, 100]
    _, bad = metrics.reliability(y, np.repeat([0.5, 0.5], 100))
    assert bad == pytest.approx(0.0)  # half are dipoles: 0.5 is calibrated on average
