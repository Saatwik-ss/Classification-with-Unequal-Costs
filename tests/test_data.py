import numpy as np
import pytest

from project4.data import (
    DATA_DIR, SPAMBASE_FILE, gaussian_true_posterior, load_spambase, make_gaussian,
    preprocess, resample_to_prior, split,
)

MU0, MU1 = [0.0, 0.0], [1.5, 1.0]
S0, S1 = [[1.0, 0.3], [0.3, 1.0]], [[2.0, 0.0], [0.0, 0.5]]


def test_make_gaussian_shapes_prior_and_moments():
    X, y = make_gaussian(200_000, 0.3, MU0, MU1, S0, S1, seed=0)
    assert X.shape == (200_000, 2) and set(np.unique(y)) == {0, 1}
    assert y.mean() == pytest.approx(0.3, abs=0.005)
    assert X[y == 1].mean(0) == pytest.approx(MU1, abs=0.02)
    assert np.cov(X[y == 0].T) == pytest.approx(np.array(S0), abs=0.02)
    X2, y2 = make_gaussian(10, 0.3, MU0, MU1, S0, S1, seed=0)
    X3, y3 = make_gaussian(10, 0.3, MU0, MU1, S0, S1, seed=0)
    assert np.array_equal(X2, X3) and np.array_equal(y2, y3)


def test_make_gaussian_1d():
    X, y = make_gaussian(1000, 0.5, -1.0, 1.0, 1.0, 1.0, seed=3)
    assert X.shape == (1000, 1)


def test_true_posterior_1d_closed_form():
    x = np.linspace(-3, 3, 13)
    p = gaussian_true_posterior(x, 0.4, -1.0, 1.0, 1.0, 1.0)
    assert p == pytest.approx(1 / (1 + np.exp(-(2 * x + np.log(0.4 / 0.6)))))


def test_split_is_stratified_disjoint_and_complete():
    rng = np.random.default_rng(0)
    y = (rng.random(1001) < 0.39).astype(int)
    X = np.arange(y.size)[:, None].astype(float)  # row id as the only feature
    s = split(X, y, seed=5)
    ids = [s.X_tr[:, 0], s.X_va[:, 0], s.X_te[:, 0]]
    allids = np.concatenate(ids)
    assert np.array_equal(np.sort(allids), np.arange(y.size))
    for (Xp, yp), frac in zip(((s.X_tr, s.y_tr), (s.X_va, s.y_va), (s.X_te, s.y_te)), (0.6, 0.2, 0.2)):
        assert len(yp) == pytest.approx(frac * y.size, abs=2)
        assert yp.mean() == pytest.approx(y.mean(), abs=0.01)
        assert np.array_equal(yp, y[Xp[:, 0].astype(int)])
    assert not np.array_equal(split(X, y, seed=6).X_te, s.X_te)  # seeds differ
    assert np.array_equal(split(X, y, seed=5).X_te, s.X_te)


def test_resample_to_prior_keeps_class_conditionals():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.4).astype(int)
    X = np.c_[np.arange(1000), y].astype(float)
    for pi in (0.05, 0.2, 0.6, 0.8):
        Xr, yr = resample_to_prior(X, y, pi, seed=1)
        assert yr.mean() == pytest.approx(pi, abs=0.002)
        assert np.array_equal(Xr[:, 1], yr)  # every row keeps its own label
        assert len(np.unique(Xr[:, 0])) == len(yr)  # no duplicates without replacement
    Xr, yr = resample_to_prior(X, y, 0.9, seed=1, n=2000, replace=True)
    assert len(yr) == 2000 and yr.mean() == pytest.approx(0.9)
    with pytest.raises(ValueError):
        resample_to_prior(X, y, 0.9, seed=1, n=2000)


def test_preprocess_uses_training_statistics_only():
    X, y = make_gaussian(3000, 0.4, MU0, MU1, S0, S1, seed=1)
    s = preprocess(split(X, y, seed=0))
    assert s.X_tr.mean(0) == pytest.approx(0, abs=1e-12)
    assert s.X_tr.std(0) == pytest.approx(1, abs=1e-12)
    assert not np.allclose(s.X_te.mean(0), 0, atol=1e-12)


@pytest.mark.skipif(not (DATA_DIR / SPAMBASE_FILE).exists(), reason="Spambase not cached")
def test_spambase():
    X, y = load_spambase()
    assert X.shape == (4601, 57) and y.sum() == 1813 and (X >= 0).all()
    s = preprocess(split(X, y, seed=0), log1p=True)
    assert np.isfinite(s.X_te).all()
