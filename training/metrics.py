"""Binary-classification metrics for judging the dipole CNN (task A9, stage 2). numpy only.

From a confusion matrix (TP, FN, FP, TN) at one threshold:

- what describes the model whatever the prevalence: sensitivity (recall, TPR), specificity
  (TNR), balanced accuracy = (sensitivity + specificity) / 2, informedness (Youden's J) =
  sensitivity + specificity - 1, and the threshold-free ROC AUC;
- what also depends on how common dipoles are in the test (the prevalence): precision (PPV),
  NPV, accuracy, MCC, markedness = PPV + NPV - 1;
- what ignores the true negatives, for rare events: F1, Jaccard, Fowlkes-Mallows and the
  average precision (area under the precision-recall curve);
- the prevalence threshold PT, below which the precision collapses, and the precision the same
  sensitivity and specificity would give at any other prevalence (Bayes): dipoles are rare in a
  real image, so a 45 % test prevalence flatters PPV.

Plus Wilson intervals for small counts and the calibration table (does p = 0.8 mean 8 in 10?).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    y = np.asarray(y, bool)
    score = np.asarray(score, float)
    positives, negatives = score[y], score[~y]
    if not len(positives) or not len(negatives):
        return float("nan")
    ranks = pd.Series(np.r_[positives, negatives]).rank().to_numpy()
    rank_sum = ranks[: len(positives)].sum()
    return float((rank_sum - len(positives) * (len(positives) + 1) / 2)
                 / (len(positives) * len(negatives)))  # fmt: skip


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (sklearn's average_precision_score)."""
    y = np.asarray(y, bool)
    if not y.any():
        return float("nan")
    order = np.argsort(-np.asarray(score, float), kind="mergesort")
    hits = y[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / hits.sum())


def binary_metrics(y: np.ndarray, predicted: np.ndarray, score: np.ndarray | None = None) -> dict:
    """Every metric of the confusion matrix of `predicted` against the truth `y`."""
    y = np.asarray(y, bool)
    predicted = np.asarray(predicted, bool)
    tp, fn = int((y & predicted).sum()), int((y & ~predicted).sum())
    fp, tn = int((~y & predicted).sum()), int((~y & ~predicted).sum())
    sensitivity, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    ppv, npv = _ratio(tp, tp + fp), _ratio(tn, tn + fn)
    product = float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt(product) if product else float("nan")
    false_positive_rate = 1 - specificity
    out = {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "prevalence": _ratio(tp + fn, len(y)),
        "sensitivity": sensitivity, "specificity": specificity,
        "PPV": ppv, "NPV": npv,
        "accuracy": _ratio(tp + tn, len(y)),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "informedness": sensitivity + specificity - 1,
        "MCC": float(mcc), "markedness": ppv + npv - 1,
        "F1": _ratio(2 * tp, 2 * tp + fp + fn),
        "jaccard": _ratio(tp, tp + fp + fn),
        "fowlkes_mallows": _ratio(tp, np.sqrt(float(tp + fp) * (tp + fn))),
        "prevalence_threshold": _ratio(np.sqrt(false_positive_rate),
                                       np.sqrt(sensitivity) + np.sqrt(false_positive_rate)),
    }  # fmt: skip
    if score is not None:
        out["ROC_AUC"] = roc_auc(y, score)
        out["average_precision"] = average_precision(y, score)
    return out


def ppv_at(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Precision at another prevalence, by Bayes."""
    found = sensitivity * prevalence
    return _ratio(found, found + (1 - specificity) * (1 - prevalence))


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """The threshold that maximizes informedness (sensitivity + specificity - 1)."""
    y = np.asarray(y, bool)
    score = np.asarray(score, float)
    best, best_value = 0.5, -np.inf
    for threshold in np.unique(score):
        predicted = score >= threshold
        value = _ratio((y & predicted).sum(), y.sum()) + _ratio((~y & ~predicted).sum(), (~y).sum())
        if value > best_value:
            best, best_value = float(threshold), value
    return best


def interval(k: int, n: int, z: float = 1.96) -> str:
    """Wilson 95 % interval of k out of n, as text."""
    if n == 0:
        return "-"
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{max(center - half, 0):.0%}-{min(center + half, 1):.0%}"


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Calibration table (low, high, mean_p, observed, n) and the expected calibration error."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        inside = index == b
        if inside.any():
            ece += inside.mean() * abs(p[inside].mean() - y[inside].mean())
            rows.append({"low": edges[b], "high": edges[b + 1], "mean_p": p[inside].mean(),
                         "observed": y[inside].mean(), "n": int(inside.sum())})  # fmt: skip
    return pd.DataFrame(rows), float(ece)
