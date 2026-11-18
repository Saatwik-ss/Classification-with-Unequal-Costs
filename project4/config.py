"""Shared constants. Change values here, never inside experiment scripts."""
from pathlib import Path

from .decision import LossMatrix

ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

SEEDS = tuple(range(20))          # repeated stratified 60/20/20 splits

# Starting loss values from the team plan (swept in the experiments).
C_FP = 10.0                       # real email sent to the spam folder
C_FN = 1.0                        # spam delivered to the inbox
C_INSP = 0.3                      # human inspection (placeholder: agree in meeting 1)
DEFAULT_LOSS = LossMatrix(C_FP, C_FN, C_INSP)

# Synthetic 1-D model used by T6 / C6: x | y ~ N(+-mu, 1), P(y=1) = pi1.
T6_MU = 1.0
T6_PI1 = 0.4                      # close to Spambase's 39% spam
