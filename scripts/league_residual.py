#!/usr/bin/env python3
"""Research-only trailing league-residual correction (NBA-style).

This module is RESEARCH code. It is deliberately NOT wired into serving.
The correction is OFF by default and must not be applied to live predictions
until the week-12 re-run of scripts/cfb_league_residual_ship_test.py passes
(see docs/cfb_league_residual_correction.md).

Correction = mean(actual - predicted) over the last N played games LEAGUE-WIDE
(not per team), capped, added to the raw prediction. Strictly causal: the
correction for game g uses only games played BEFORE g.

Chosen parameters (justified in docs/cfb_league_residual_correction.md):
  N = 300 games (~4 weeks of CFB at ~75 games/week): matches the NBA window
      (300 games); enough samples for a stable mean (SE ~0.5-1 pt) while
      staying responsive to regime changes within a season.
  cap_margin = +-8 points: the observed margin bias is -5.6, so the cap must
      allow at least 6; +-8 captures the full correction while preventing
      extreme over-correction on thin early-season samples.
  cap_total = +-4 points: matches the NBA total cap; the observed total bias
      is -2.26, so +-4 is ample and prevents over-correction on noisy totals.
"""

from __future__ import annotations

import numpy as np

LEAGUE_RESIDUAL_WINDOW = 300
LEAGUE_RESIDUAL_CAP_MARGIN = 8.0
LEAGUE_RESIDUAL_CAP_TOTAL = 4.0


def league_residual_correction(
    residuals,
    window: int = LEAGUE_RESIDUAL_WINDOW,
    cap: float = LEAGUE_RESIDUAL_CAP_MARGIN,
) -> float:
    """Causal trailing league-wide residual correction.

    `residuals` is a sequence of (actual - predicted) values in chronological
    order. Returns the capped mean of the last `window` residuals, or 0.0 when
    there are no finite residuals. The caller adds this to the raw prediction.
    """
    if residuals is None:
        return 0.0
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    tail = r[-window:]
    mean = float(np.mean(tail))
    return max(-cap, min(cap, mean))
