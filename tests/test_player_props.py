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


def test_predict_props_returns_both_markets_for_rb_when_both_models_present():
    """RB gets two markets now -- rushing_yards AND carries -- not just one
    (Phase 8's POSITION_MARKETS replaces the old one-market-per-position
    POSITION_YARDAGE_MARKET dict)."""
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    rushing_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["rushing_yards"])
    carries_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["rushing_yards_roll"])
    models = {
        "anytime_td": td_model, "rushing_yards": rushing_model, "carries": carries_model,
        "feature_cols": FEATURE_COLS,
    }

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="RB")

    assert "rushing_yards" in result
    assert "carries" in result
    assert "passing_yards" not in result


def test_predict_props_returns_receptions_for_wr_when_model_present():
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    receiving_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["receiving_yards"])
    receptions_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["receiving_yards_roll"])
    models = {
        "anytime_td": td_model, "receiving_yards": receiving_model, "receptions": receptions_model,
        "feature_cols": FEATURE_COLS,
    }

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="WR")

    assert "receiving_yards" in result
    assert "receptions" in result


def test_predict_props_gracefully_skips_a_market_not_yet_in_models():
    """Right after this code ships but before the next retrain, `models`
    won't have a "carries"/"receptions" entry yet -- predict_props must not
    KeyError, it should just omit that market."""
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    models = {"anytime_td": td_model, "feature_cols": FEATURE_COLS}

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="RB")

    assert "anytime_td_prob" in result
    assert "rushing_yards" not in result
    assert "carries" not in result


def test_cfb_never_gets_a_targets_based_market():
    """CFB's `targets` column is structurally NaN (data source gap) -- no
    position may ever map to a targets-based market, unlike receptions and
    carries which are real CFB data."""
    for markets in player_props.POSITION_MARKETS.values():
        assert "targets" not in markets
    assert "targets" not in player_props.YARDAGE_TARGETS
