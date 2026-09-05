# tests/test_player_props.py
import numpy as np
import pandas as pd
import pytest

from cfb_predictor.models import player_props


def _toy_player_frame(n=80, seed=1):
    rng = np.random.default_rng(seed)
    passing_roll = rng.normal(240, 40, n)
    rushing_roll = rng.normal(90, 25, n)
    receiving_roll = rng.normal(55, 20, n)
    return pd.DataFrame(
        {
            "passing_yards_roll": passing_roll,
            "rushing_yards_roll": rushing_roll,
            "receiving_yards_roll": receiving_roll,
            "targets_roll": rng.normal(5, 2, n),
            "carries_roll": rng.normal(16, 5, n),
            "passing_yards": passing_roll + rng.normal(0, 15, n),
            "rushing_yards": rushing_roll + rng.normal(0, 15, n),
            "receiving_yards": receiving_roll + rng.normal(0, 15, n),
            "anytime_td": (rng.random(n) < (0.3 + rushing_roll / 500)).astype(int),
        }
    )


FEATURE_COLS = ["passing_yards_roll", "rushing_yards_roll", "receiving_yards_roll", "targets_roll", "carries_roll"]


def test_fit_anytime_td_classifier_predicts_probabilities():
    df = _toy_player_frame()
    model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])

    probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()


def test_fit_yardage_regressor_predicts_reasonable_values():
    df = _toy_player_frame()
    model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["rushing_yards"])

    preds = model.predict(df[FEATURE_COLS])
    assert np.corrcoef(preds, df["rushing_yards"])[0, 1] > 0.3


def test_predict_props_only_returns_relevant_yardage_market_for_position():
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    rushing_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["rushing_yards"])
    models = {"anytime_td": td_model, "rushing_yards": rushing_model, "feature_cols": FEATURE_COLS}

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="RB")

    assert "anytime_td_prob" in result
    assert "rushing_yards" in result
    assert "passing_yards" not in result
    assert "receiving_yards" not in result


def test_predict_props_only_returns_relevant_yardage_market_for_qb():
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    passing_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["passing_yards"])
    models = {"anytime_td": td_model, "passing_yards": passing_model, "feature_cols": FEATURE_COLS}

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="QB")

    assert "passing_yards" in result
    assert "rushing_yards" not in result
    assert "receiving_yards" not in result


def test_predict_props_collapses_wr_and_te_to_the_same_receiving_market():
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    receiving_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["receiving_yards"])
    models = {"anytime_td": td_model, "receiving_yards": receiving_model, "feature_cols": FEATURE_COLS}

    wr_result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="WR")
    te_result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="TE")

    assert "receiving_yards" in wr_result
    assert "receiving_yards" in te_result
