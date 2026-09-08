"""calibration.py -- measurement-only calibration diagnostics. Does not
correct/adjust any model output; see docs/superpowers/specs/
2026-09-08-cross-sport-accuracy-tracking-parity-design.md for the scope
decision (measurement only, no Platt/isotonic correction layer)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_curve(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges[1:-1]), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin_midpoint": float((bin_edges[b] + bin_edges[b + 1]) / 2),
            "predicted_mean": float(probs[mask].mean()),
            "actual_frequency": float(outcomes[mask].mean()),
            "n_predictions": n,
        })
    return pd.DataFrame(rows)
