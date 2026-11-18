"""Data loading, synthetic data, prior resampling and splits (C1, frozen interface).

Public functions
----------------
load_spambase()                                   -> X (4601, 57), y (4601,)
make_gaussian(n, pi1, mu0, mu1, S0, S1, seed)     -> X (n, d), y (n,)
resample_to_prior(X, y, pi1_new, seed)            -> X', y' with class-1 share pi1_new
split(X, y, seed)                                 -> Split(X_tr, y_tr, X_va, y_va, X_te, y_te)
Standardizer / preprocess(split)                  -> preprocessing fitted on training data only
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPAMBASE_FILE = "spambase.data"
SPAMBASE_URLS = (
    "https://archive.ics.uci.edu/static/public/94/spambase.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
)
# SHA-256 of spambase.data as distributed by UCI (checked after download).
SPAMBASE_SHA256 = "b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12"
SPAMBASE_N, SPAMBASE_D, SPAMBASE_N_SPAM = 4601, 57, 1813

SEEDS = tuple(range(20))  # the 20 repeated splits used for error bars


# ----------------------------------------------------------------- Spambase
def _download_spambase(target: Path) -> None:
    errors = []
    for url in SPAMBASE_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                payload = r.read()
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(payload)) as z:
                    payload = z.read(SPAMBASE_FILE)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            return
        except Exception as e:  # try the next mirror
            errors.append(f"{url}: {e}")
    raise RuntimeError(
        "Could not download Spambase. Put spambase.data in "
        f"{target.parent} by hand. Errors:\n  " + "\n  ".join(errors)
    )


def load_spambase(cache_dir: str | Path | None = None):
    """UCI Spambase: 4,601 emails, 57 features, y = 1 for spam (1,813 spam).

    Downloads once and caches ``spambase.data`` in ``data/`` (git-ignored).
    Returns raw features; use ``preprocess`` on a split to standardise using
    training statistics only.
    """
    path = Path(cache_dir or DATA_DIR) / SPAMBASE_FILE
    if not path.exists():
        _download_spambase(path)
    if SPAMBASE_SHA256 is not None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SPAMBASE_SHA256:
            raise RuntimeError(f"{path} has unexpected SHA-256 {digest}; delete it and retry")
    raw = np.loadtxt(path, delimiter=",")
    X, y = raw[:, :-1], raw[:, -1].astype(int)
    assert X.shape == (SPAMBASE_N, SPAMBASE_D) and y.sum() == SPAMBASE_N_SPAM, "corrupt Spambase file"
    return X, y


# ---------------------------------------------------------------- synthetic
def make_gaussian(n, pi1, mu0, mu1, S0, S1, seed):
    """Two-class Gaussian data: y ~ Bernoulli(pi1), x | y=k ~ N(mu_k, S_k).

    Means may be scalars (1-D data) or length-d vectors; covariances scalars
    (variances) or d x d matrices.  The number of class-1 points is drawn
    from Binomial(n, pi1), so the empirical prior fluctuates as in real
    sampling.  Returns X of shape (n, d) and y of shape (n,).
    """
    rng = np.random.default_rng(seed)
    mu0, mu1 = np.atleast_1d(np.asarray(mu0, float)), np.atleast_1d(np.asarray(mu1, float))
    S0, S1 = np.atleast_2d(np.asarray(S0, float)), np.atleast_2d(np.asarray(S1, float))
    d = mu0.size
    if not (mu1.size == d and S0.shape == (d, d) and S1.shape == (d, d)):
        raise ValueError("inconsistent dimensions of means/covariances")
    y = (rng.random(n) < pi1).astype(int)
    X = np.empty((n, d))
    for k, (mu, S) in enumerate(((mu0, S0), (mu1, S1))):
        idx = np.flatnonzero(y == k)
        Lc = np.linalg.cholesky(S)
        X[idx] = mu + rng.standard_normal((idx.size, d)) @ Lc.T
    return X, y


def gaussian_true_posterior(X, pi1, mu0, mu1, S0, S1):
    """Exact P(y=1 | x) for ``make_gaussian`` data (for Bayes-risk references)."""
    X = np.asarray(X, float)
    X = X.reshape(-1, np.atleast_1d(mu0).size)  # (n, d); 1-D input becomes (n, 1)

    def logpdf(mu, S):
        mu, S = np.atleast_1d(np.asarray(mu, float)), np.atleast_2d(np.asarray(S, float))
        Lc = np.linalg.cholesky(S)
        z = np.linalg.solve(Lc, (X - mu).T)
        return -0.5 * (z**2).sum(0) - np.log(np.diag(Lc)).sum() - 0.5 * mu.size * np.log(2 * np.pi)

    a = logpdf(mu1, S1) - logpdf(mu0, S0) + np.log(pi1) - np.log1p(-pi1)
    return 1.0 / (1.0 + np.exp(-a))


# ------------------------------------------------------------- prior shift
def resample_to_prior(X, y, pi1_new, seed, n=None, replace=False):
    """Resample within each class so the class-1 share becomes ``pi1_new``.

    Points are drawn only from their own class, so p(x | y) is unchanged
    (label shift).  By default sampling is without replacement and the
    output is the largest set the available data allow; pass ``n`` for a
    fixed size (``replace=True`` is needed if a class runs out).
    Returns shuffled X', y'.
    """
    if not 0.0 <= pi1_new <= 1.0:
        raise ValueError("pi1_new must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    X, y = np.asarray(X), np.asarray(y).astype(int)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if n is None:
        if replace:
            n = y.size
        else:
            caps = []
            if pi1_new > 0:
                caps.append(idx1.size / pi1_new)
            if pi1_new < 1:
                caps.append(idx0.size / (1 - pi1_new))
            n = int(np.floor(min(caps) + 1e-9))
    n1 = int(round(n * pi1_new))
    n0 = n - n1
    if not replace and (n1 > idx1.size or n0 > idx0.size):
        raise ValueError("not enough points of one class; use replace=True or a smaller n")
    take = np.concatenate([rng.choice(idx0, n0, replace=replace), rng.choice(idx1, n1, replace=replace)])
    rng.shuffle(take)
    return X[take], y[take]


# ------------------------------------------------------------------ splits
class Split(NamedTuple):
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray


def split(X, y, seed, fractions=(0.6, 0.2, 0.2)) -> Split:
    """Stratified train/validation/test split (default 60/20/20).

    Each class is shuffled with ``seed`` and cut in the given proportions,
    so all three parts keep the overall class balance.  Use seeds 0-19
    (``SEEDS``) for the repeated splits.  The test part is never used to
    choose anything.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    X, y = np.asarray(X), np.asarray(y).astype(int)
    parts = ([], [], [])
    for k in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == k))
        n_tr = int(round(fractions[0] * idx.size))
        n_va = int(round(fractions[1] * idx.size))
        for part, chunk in zip(parts, (idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:])):
            part.append(chunk)
    out = []
    for part in parts:
        ids = rng.permutation(np.concatenate(part))
        out += [X[ids], y[ids]]
    return Split(*out)


# ----------------------------------------------------------- preprocessing
class Standardizer:
    """Optional log(1+x), then zero mean / unit variance, fitted on training data."""

    def __init__(self, log1p: bool = False, eps: float = 1e-12):
        self.log1p, self.eps = log1p, eps

    def _pre(self, X):
        X = np.asarray(X, float)
        return np.log1p(X) if self.log1p else X

    def fit(self, X):
        Z = self._pre(X)
        self.mean_ = Z.mean(0)
        self.scale_ = np.maximum(Z.std(0), self.eps)
        return self

    def transform(self, X):
        return (self._pre(X) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def preprocess(s: Split, log1p: bool = False) -> Split:
    """Standardise a split with statistics from its training part only."""
    st = Standardizer(log1p=log1p).fit(s.X_tr)
    return Split(st.transform(s.X_tr), s.y_tr, st.transform(s.X_va), s.y_va, st.transform(s.X_te), s.y_te)
