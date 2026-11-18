"""Numerical check of T1: the threshold rule equals the brute-force argmin."""
import numpy as np
import pytest

from project4.decision import (
    DELIVER, INSPECT, SPAM, LossMatrix, bayes_action, expected_losses, thresholds,
)


def random_losses(rng, k):
    """Loss matrices spanning several orders of magnitude, with c_I both
    below and above the critical value."""
    c_fp = 10 ** rng.uniform(-2, 2, k)
    c_fn = 10 ** rng.uniform(-2, 2, k)
    crit = c_fp * c_fn / (c_fp + c_fn)
    c_i = crit * rng.uniform(0, 2, k)  # half the cases have no inspect band
    return [LossMatrix(*v) for v in zip(c_fp, c_fn, c_i)]


def test_threshold_rule_matches_bruteforce_argmin():
    rng = np.random.default_rng(0)
    n = 10_000
    p = rng.random(n)
    Ls = random_losses(rng, n)
    a_rule = np.array([bayes_action(p[i : i + 1], L)[0] for i, L in enumerate(Ls)])
    R = np.vstack([expected_losses(p[i], L) for i, L in enumerate(Ls)])
    a_brute = R.argmin(1)
    # Where two actions tie (measure zero) either is optimal; compare losses.
    chosen = R[np.arange(n), a_rule]
    assert np.allclose(chosen, R.min(1), rtol=1e-12, atol=1e-15)
    assert (a_rule == a_brute).mean() == 1.0


def test_vectorised_rule_same_loss_matrix():
    rng = np.random.default_rng(1)
    p = rng.random(10_000)
    for L in random_losses(rng, 50):
        R = expected_losses(p, L)
        assert np.array_equal(bayes_action(p, L), R.argmin(1))


def test_threshold_formulas_and_band_condition():
    L = LossMatrix(10, 1, 0.3)
    th0, th1 = thresholds(L)
    assert th0 == pytest.approx(0.3) and th1 == pytest.approx(0.97)
    assert L.critical_c_insp == pytest.approx(10 / 11) and L.has_reject_band
    L2 = LossMatrix(10, 1, 0.95)
    assert not L2.has_reject_band
    assert thresholds(L2) == pytest.approx((10 / 11, 10 / 11))
    p = np.array([0.0, 0.5, 10 / 11, 0.99])
    assert list(bayes_action(p, L2)) == [DELIVER, DELIVER, SPAM, SPAM]


def test_equal_costs_give_classic_reject_rule():
    """c_fp = c_fn = c: inspect iff max(p, 1-p) < 1 - c_I / c (Bishop 1.5.3)."""
    rng = np.random.default_rng(2)
    p = rng.random(10_000)
    for c, ci in [(1.0, 0.1), (2.0, 0.5), (5.0, 2.0)]:
        a = bayes_action(p, LossMatrix(c, c, ci))
        classic = np.maximum(p, 1 - p) < 1 - ci / c
        assert np.array_equal(a == INSPECT, classic)


def test_general_matrix_expected_losses():
    M = np.array([[0.0, 3.0], [2.0, 0.0], [1.0, 1.0]])
    R = expected_losses([0.0, 0.5, 1.0], M)
    assert np.allclose(R, [[0, 2, 1], [1.5, 1, 1], [3, 0, 1]])


def test_invalid_loss_matrix():
    with pytest.raises(ValueError):
        LossMatrix(0, 1, 0.1)
    with pytest.raises(ValueError):
        LossMatrix(1, 1, -0.1)
