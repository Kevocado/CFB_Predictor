import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from league_residual import (  # noqa: E402
    LEAGUE_RESIDUAL_WINDOW,
    LEAGUE_RESIDUAL_CAP_MARGIN,
    LEAGUE_RESIDUAL_CAP_TOTAL,
    league_residual_correction,
)


def test_correction_empty_returns_zero():
    assert league_residual_correction([]) == 0.0
    assert league_residual_correction(None) == 0.0


def test_correction_nan_inf_ignored():
    assert league_residual_correction([np.nan, np.inf, 5.0]) == 5.0


def test_correction_mean_of_residuals():
    # residuals = actual - predicted; correction = mean, capped
    r = [2.0, 4.0, 6.0]
    assert league_residual_correction(r, window=10, cap=100) == 4.0


def test_correction_uses_last_window():
    r = [100.0] + [2.0] * 5  # first is huge but outside window
    assert league_residual_correction(r, window=5, cap=100) == 2.0


def test_correction_capped():
    r = [100.0] * 5
    assert league_residual_correction(r, window=10, cap=8.0) == 8.0
    r = [-100.0] * 5
    assert league_residual_correction(r, window=10, cap=8.0) == -8.0


def test_constants_defined():
    assert LEAGUE_RESIDUAL_WINDOW == 300
    assert LEAGUE_RESIDUAL_CAP_MARGIN == 8.0
    assert LEAGUE_RESIDUAL_CAP_TOTAL == 4.0
