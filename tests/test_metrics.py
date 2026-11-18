import numpy as np
import pytest

from project4.decision import LossMatrix
from project4.metrics import (
    accuracy, confusion_with_reject, error_rates, mean_and_se, realized_loss, reject_rate,
)

#            y: 0  0  0  0  1  1  1  1
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
A = np.array([0, 0, 1, 2, 1, 1, 0, 2])
L = LossMatrix(10, 1, 0.5)


def test_confusion():
    assert confusion_with_reject(Y, A).tolist() == [[2, 1, 1], [1, 2, 1]]


def test_realized_loss():
    # one FP (10) + one FN (1) + two inspections (0.5 each) over 8 emails
    assert realized_loss(Y, A, L) == pytest.approx((10 + 1 + 1) / 8)


def test_accuracy_variants():
    assert accuracy(Y, A) == pytest.approx(4 / 6)
    assert accuracy(Y, A, on_reject="correct") == pytest.approx(6 / 8)
    assert accuracy(Y, A, on_reject="error") == pytest.approx(4 / 8)
    assert np.isnan(accuracy(Y, np.full(8, 2)))


def test_reject_and_error_rates():
    assert reject_rate(A) == pytest.approx(0.25)
    r = error_rates(Y, A)
    assert r["fpr"] == pytest.approx(0.25) and r["fnr"] == pytest.approx(0.25)


def test_input_validation():
    with pytest.raises(ValueError):
        realized_loss(Y, A[:-1], L)
    with pytest.raises(ValueError):
        accuracy(Y, A + 1)


def test_mean_and_se():
    m, se = mean_and_se([1.0, 2.0, 3.0])
    assert m == pytest.approx(2.0) and se == pytest.approx(1 / np.sqrt(3))
