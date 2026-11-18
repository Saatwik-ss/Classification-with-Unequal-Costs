"""Decision module (C1, frozen interface).

Spam filter with a reject option.  Labels: y = 1 spam, y = 0 real email.
Actions (integer codes used everywhere in the project):

    DELIVER = 0   deliver to inbox
    SPAM    = 1   send to spam folder
    INSPECT = 2   hold for human inspection (the "reject" option)

Loss matrix, rows = actions, columns = true label (y = 0, y = 1):

                 y=0     y=1
    deliver       0      c_fn
    spam         c_fp     0
    inspect      c_I     c_I

With p = P(y=1 | x) the posterior expected losses are
    R(0|x) = c_fn * p,  R(1|x) = c_fp * (1 - p),  R(2|x) = c_I,
and the Bayes action is their argmin (T1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DELIVER = 0
SPAM = 1
INSPECT = 2
ACTION_NAMES = ("deliver", "spam", "inspect")


@dataclass(frozen=True)
class LossMatrix:
    """Loss matrix with zero loss on correct decisions.

    Parameters
    ----------
    c_fp : cost of sending a real email to the spam folder (> 0).
    c_fn : cost of delivering spam to the inbox (> 0).
    c_insp : cost of human inspection, paid whatever the true label (>= 0).
        Assumption: the inspector always finds the true label, so no
        further loss follows an inspection.
    """

    c_fp: float
    c_fn: float
    c_insp: float

    def __post_init__(self):
        if not (self.c_fp > 0 and self.c_fn > 0):
            raise ValueError("c_fp and c_fn must be positive")
        if not self.c_insp >= 0:
            raise ValueError("c_insp must be non-negative")

    @property
    def matrix(self) -> np.ndarray:
        """3 x 2 array L[action, y]."""
        return np.array(
            [[0.0, self.c_fn], [self.c_fp, 0.0], [self.c_insp, self.c_insp]]
        )

    @property
    def critical_c_insp(self) -> float:
        """c_fp * c_fn / (c_fp + c_fn): inspection is used iff c_insp is below this."""
        return self.c_fp * self.c_fn / (self.c_fp + self.c_fn)

    @property
    def has_reject_band(self) -> bool:
        return self.c_insp < self.critical_c_insp

    @property
    def two_action_threshold(self) -> float:
        """Bayes threshold without inspection: spam iff p >= c_fp / (c_fp + c_fn)."""
        return self.c_fp / (self.c_fp + self.c_fn)

    def with_c_insp(self, c_insp: float) -> "LossMatrix":
        return LossMatrix(self.c_fp, self.c_fn, c_insp)


def thresholds(L: LossMatrix) -> tuple[float, float]:
    """Probability thresholds (theta0, theta1) of the Bayes rule.

    Deliver if p <= theta0, spam if p >= theta1, inspect in between.
    If the inspect band is empty (c_insp >= critical value) both thresholds
    equal the two-action threshold c_fp / (c_fp + c_fn).
    """
    if L.has_reject_band:
        return L.c_insp / L.c_fn, 1.0 - L.c_insp / L.c_fp
    t = L.two_action_threshold
    return t, t


def expected_losses(p, L) -> np.ndarray:
    """Posterior expected loss of every action, shape (n, n_actions).

    ``L`` may be a LossMatrix or any (n_actions x 2) array L[action, y], so
    this also serves as the brute-force reference for general loss matrices.
    """
    M = L.matrix if isinstance(L, LossMatrix) else np.asarray(L, dtype=float)
    p = np.asarray(p, dtype=float).reshape(-1, 1)
    return (1.0 - p) * M[:, 0] + p * M[:, 1]


def bayes_action(p, L: LossMatrix) -> np.ndarray:
    """Bayes action for each posterior p = P(y=1|x): array of 0 / 1 / 2 (2 = inspect).

    Ties: p == theta0 -> deliver, p == theta1 -> spam (both are optimal there).
    """
    p = np.asarray(p, dtype=float)
    theta0, theta1 = thresholds(L)
    a = np.full(p.shape, INSPECT, dtype=int)
    a[p <= theta0] = DELIVER
    a[p >= theta1] = SPAM
    return a


def threshold_action(p, t: float) -> np.ndarray:
    """Two-action rule: spam iff p >= t (t = 0.5 is the accuracy-based rule)."""
    return (np.asarray(p, dtype=float) >= t).astype(int)


def band_action(p, theta0: float, theta1: float) -> np.ndarray:
    """Arbitrary reject rule: deliver if p <= theta0, spam if p >= theta1, else inspect."""
    if theta0 > theta1:
        raise ValueError("need theta0 <= theta1")
    p = np.asarray(p, dtype=float)
    a = np.full(p.shape, INSPECT, dtype=int)
    a[p <= theta0] = DELIVER
    a[p >= theta1] = SPAM
    return a
