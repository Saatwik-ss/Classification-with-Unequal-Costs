"""Evaluation metrics for decisions with a reject option (C1, frozen interface).

``y`` holds true labels in {0, 1}; ``a`` holds actions in {0, 1, 2}
(0 deliver, 1 spam folder, 2 inspect) as produced by ``decision.bayes_action``.
"""
from __future__ import annotations

import numpy as np

from .decision import INSPECT, LossMatrix


def _check(y, a):
    y = np.asarray(y).astype(int).ravel()
    a = np.asarray(a).astype(int).ravel()
    if y.shape != a.shape:
        raise ValueError(f"y and a differ in length: {y.shape} vs {a.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    if not np.isin(a, (0, 1, 2)).all():
        raise ValueError("actions must be 0, 1 or 2")
    return y, a


def confusion_with_reject(y, a) -> np.ndarray:
    """2 x 3 count matrix C[true label, action], action columns deliver / spam / inspect."""
    y, a = _check(y, a)
    C = np.zeros((2, 3), dtype=int)
    np.add.at(C, (y, a), 1)
    return C


def realized_loss(y, a, L: LossMatrix) -> float:
    """Average loss per email, (1/N) sum_n L[a_n, y_n]."""
    y, a = _check(y, a)
    return float(L.matrix[a, y].mean())


def accuracy(y, a, on_reject: str = "exclude") -> float:
    """Accuracy of the automatic decisions.

    on_reject:
      "exclude" (default) - accuracy among emails that were not inspected
                            (NaN if every email was inspected);
      "correct"           - inspected emails count as correct (the inspector
                            always finds the true label): system accuracy;
      "error"             - inspected emails count as errors.
    For two-action rules (no 2s in ``a``) all three agree.
    """
    y, a = _check(y, a)
    decided = a != INSPECT
    hits = (a == y) & decided
    if on_reject == "exclude":
        n = decided.sum()
        return float(hits.sum() / n) if n else float("nan")
    if on_reject == "correct":
        return float((hits | ~decided).mean())
    if on_reject == "error":
        return float(hits.mean())
    raise ValueError("on_reject must be 'exclude', 'correct' or 'error'")


def reject_rate(a) -> float:
    """Fraction of emails held for inspection."""
    a = np.asarray(a).astype(int).ravel()
    return float((a == INSPECT).mean())


def error_rates(y, a) -> dict:
    """Class-conditional rates: FPR = P(spam | y=0), FNR = P(deliver | y=1),
    and the inspection rate within each class."""
    C = confusion_with_reject(y, a).astype(float)
    n0, n1 = C[0].sum(), C[1].sum()
    return {
        "fpr": C[0, 1] / n0 if n0 else np.nan,
        "fnr": C[1, 0] / n1 if n1 else np.nan,
        "insp_rate_0": C[0, 2] / n0 if n0 else np.nan,
        "insp_rate_1": C[1, 2] / n1 if n1 else np.nan,
    }


def summarize(y, a, L: LossMatrix) -> dict:
    """All headline numbers for one set of decisions."""
    return {
        "realized_loss": realized_loss(y, a, L),
        "accuracy": accuracy(y, a),
        "system_accuracy": accuracy(y, a, on_reject="correct"),
        "reject_rate": reject_rate(a),
        **error_rates(y, a),
    }


def mean_and_se(values, axis=0):
    """Mean and standard error over repeated splits (for error bars)."""
    v = np.asarray(values, dtype=float)
    n = v.shape[axis]
    se = v.std(axis=axis, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(v.mean(axis=axis))
    return v.mean(axis=axis), se
