import numpy as np

from cfb_predictor.evaluate import calibration


def test_reliability_curve_perfectly_calibrated_predictions_lie_on_the_diagonal():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 2000)
    outcomes = (rng.uniform(0, 1, 2000) < probs).astype(int)

    curve = calibration.reliability_curve(probs, outcomes, n_bins=10)

    assert set(curve.columns) == {"bin_midpoint", "predicted_mean", "actual_frequency", "n_predictions"}
    # Perfectly calibrated by construction -- actual should track predicted within noise.
    assert (curve["actual_frequency"] - curve["predicted_mean"]).abs().max() < 0.25


def test_reliability_curve_drops_empty_bins():
    probs = np.array([0.05, 0.05, 0.95, 0.95])
    outcomes = np.array([0, 0, 1, 1])

    curve = calibration.reliability_curve(probs, outcomes, n_bins=10)

    assert curve["n_predictions"].sum() == 4
    assert (curve["n_predictions"] > 0).all()
