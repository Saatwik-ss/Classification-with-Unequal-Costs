"""Numerical checks of the T6 closed forms."""
import numpy as np
import pytest

from project4.analytic import GaussianModel1D
from project4.data import gaussian_true_posterior, make_gaussian
from project4.decision import LossMatrix, bayes_action
from project4.metrics import accuracy, realized_loss, reject_rate

M = GaussianModel1D(mu=1.0, pi1=0.4)


def test_posterior_and_threshold_mapping():
    x = np.linspace(-3, 3, 7)
    assert M.posterior(x) == pytest.approx(gaussian_true_posterior(x, 0.4, -1, 1, 1, 1))
    for t in (0.1, 0.5, 0.97):
        assert M.posterior(M.x_threshold(t)) == pytest.approx(t)
    assert M.x_threshold(0.0) == -np.inf and M.x_threshold(1.0) == np.inf


def test_closed_form_matches_monte_carlo():
    X, y = make_gaussian(400_000, M.pi1, -M.mu, M.mu, 1.0, 1.0, seed=0)
    p = M.posterior(X[:, 0])
    for ci in (0.05, 0.3, 0.7, 1.0):
        L = LossMatrix(10, 1, ci)
        a = bayes_action(p, L)
        t0, t1 = M.bayes_x_thresholds(L)
        assert realized_loss(y, a, L) == pytest.approx(M.cost(t0, t1, L), abs=0.005)
        assert reject_rate(a) == pytest.approx(M.reject_rate(t0, t1), abs=0.003)
        assert accuracy(y, a) == pytest.approx(M.accuracy(t0, t1), abs=0.003)


def test_bayes_thresholds_minimise_cost_over_all_band_rules():
    L = LossMatrix(10, 1, 0.3)
    best = M.bayes_risk(L)
    grid = np.linspace(-3, 4, 141)
    costs = [M.cost(a, b, L) for i, a in enumerate(grid) for b in grid[i:]]
    assert min(costs) >= best - 1e-12


def test_higher_accuracy_higher_cost_pair():
    """The accuracy-optimal rule is more accurate but costlier than the cost-optimal rule."""
    L = LossMatrix(10, 1, 0.0)
    t_acc = M.accuracy_threshold
    t_cost = float(M.x_threshold(L.two_action_threshold))
    assert M.accuracy(t_acc) > M.accuracy(t_cost)
    assert M.two_action_cost(t_acc, L) > M.two_action_cost(t_cost, L)
    taus = np.linspace(-4, 4, 2001)
    assert max(M.accuracy(t) for t in taus) <= M.accuracy(t_acc) + 1e-12
    assert min(M.two_action_cost(t, L) for t in taus) >= M.two_action_cost(t_cost, L) - 1e-12


def test_bayes_risk_concave_increasing_slope_is_reject_rate():
    c = np.linspace(0, 1.2, 601)
    cur = M.curves_vs_c_insp(c, 10, 1)
    R, rho, crit = cur["bayes_risk"], cur["reject_rate"], cur["critical_c_insp"]
    assert np.all(np.diff(R) >= -1e-12)                      # non-decreasing
    assert np.all(np.diff(R, 2) <= 1e-9)                     # concave
    slope = np.gradient(R, c)
    inner = (c > 0.01) & (np.abs(c - crit) > 0.01)
    assert slope[inner] == pytest.approx(rho[inner], abs=2e-3)  # dR*/dc_I = rho
    assert np.all(rho[c >= crit] == 0)                       # no rejection past c*
    assert np.all(rho[c < crit] > 0)
    assert R[c >= crit] == pytest.approx(cur["two_action_risk"][c >= crit])
    assert np.all(R[c < crit] < cur["two_action_risk"][c < crit])  # rejection helps iff c_I < c*
    assert R[0] == 0 and rho[0] == 1
