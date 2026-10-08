"""Sigma comes from error the model has not been fitted on, and the ridge is invariant to feature units."""
import numpy as np
import pandas as pd
import pytest

from cfb_predictor.evaluate import walk_forward
from cfb_predictor.models import game_outcome, manifest

NOISE = 10.0
COLS = [f"f{i}" for i in range(6)]


def _folds(n_seasons=7, games=80, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_seasons):
        X = pd.DataFrame(rng.normal(size=(games, len(COLS))), columns=COLS)
        X["season"] = 2018 + s
        X["margin"] = 3 * X["f0"] - 2 * X["f1"] + rng.normal(0, NOISE, games)
        X["total_points"] = 45 + 4 * X["f2"] + rng.normal(0, NOISE, games)
        frames.append(X)
    df = pd.concat(frames, ignore_index=True)
    return [
        {"val_season": 2018 + i, "train_df": df[df["season"] < 2018 + i], "val_df": df[df["season"] == 2018 + i], "feature_cols": COLS}
        for i in range(3, n_seasons)
    ], df


def test_in_sample_sigma_understates_and_out_of_fold_sigma_does_not():
    folds, df = _folds()
    train = df[df["season"] < 2024]
    in_sample = game_outcome.residual_sigma(game_outcome.fit_xgb_margin(train[COLS], train["margin"]), train[COLS], train["margin"])
    oof = game_outcome.sigma_from_residuals(walk_forward.oof_residuals(folds, "xgb", "margin"))
    assert in_sample < 0.8 * NOISE, "an overfit model's training error is optimistic: this is the bug"
    assert oof > 1.15 * in_sample
    assert oof == pytest.approx(NOISE, rel=0.25)


def test_the_manifest_sigmas_are_out_of_fold_for_the_margin_and_the_total():
    folds, df = _folds()
    sigma, total_sigma = manifest.fit_sigmas(folds, "ridge")
    assert sigma == pytest.approx(game_outcome.sigma_from_residuals(walk_forward.oof_residuals(folds, "ridge", "margin")))
    assert total_sigma == pytest.approx(game_outcome.sigma_from_residuals(walk_forward.oof_residuals(folds, "xgb", "total_points")))
    train = df[df["season"] < 2024]
    in_sample_total = game_outcome.residual_sigma(game_outcome.fit_xgb_margin(train[COLS], train["total_points"]), train[COLS], train["total_points"])
    assert total_sigma > in_sample_total


def test_a_folds_sigma_is_the_error_on_the_last_training_season_of_a_model_that_never_saw_it():
    folds, _ = _folds()
    train = folds[-1]["train_df"]
    last = sorted(train["season"].unique())[-1]
    earlier, held_out = train[train["season"] != last], train[train["season"] == last]
    manual = game_outcome.sigma_from_residuals(
        held_out["margin"].to_numpy(float) - game_outcome.fit_xgb_margin(earlier[COLS], earlier["margin"]).predict(held_out[COLS].fillna(0))
    )
    assert walk_forward._honest_sigma("xgb", train, COLS) == pytest.approx(manual)


def test_a_single_training_season_falls_back_to_the_in_sample_spread_and_says_so():
    folds, df = _folds()
    one_season = df[df["season"] == 2018]
    assert walk_forward._honest_sigma("ridge", one_season, COLS) > 0


def test_oof_residuals_refuse_an_unknown_candidate_and_the_elo_total():
    folds, _ = _folds()
    with pytest.raises(ValueError):
        walk_forward.oof_residuals(folds, "nope")
    with pytest.raises(ValueError):
        walk_forward._fit_predict("elo", folds[0]["train_df"], folds[0]["val_df"], COLS, "total_points")


def test_ridge_predictions_do_not_depend_on_the_units_of_a_feature():
    _, df = _folds()
    train, val = df[df["season"] < 2024], df[df["season"] == 2024]
    a = game_outcome.fit_margin_regression(train[COLS], train["margin"]).predict(val[COLS])
    scaled_train, scaled_val = train.copy(), val.copy()
    scaled_train["f0"] *= 1000.0
    scaled_val["f0"] *= 1000.0
    b = game_outcome.fit_margin_regression(scaled_train[COLS], scaled_train["margin"]).predict(scaled_val[COLS])
    np.testing.assert_allclose(a, b, atol=1e-6)