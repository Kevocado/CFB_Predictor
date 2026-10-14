"""player_props.py — anytime-TD classifier and per-position yardage/volume
regressors. Verbatim port from nfl_predictor: props are priced as a
continuous over/under line for both sports, and the fitting/prediction
shape has no NFL-specific assumption. `manifest.py`'s train/save/load
pipeline is entirely generic over YARDAGE_TARGETS's keys, so every entry
here gets its own regressor with zero pipeline changes.

`receptions` and `carries` are real data for CFB (unlike `targets`, which
is structurally NaN for CFB -- see data/player_stats.py -- and must never
get a market): `carries` was already a rolling feature
(features/player_usage.py's ROLL_STATS) but not a priced market here until
now; `receptions` is new on both counts. POSITION_MARKETS is a *separate*,
inference-time-only dict used solely by predict_props to pick which
market(s) apply to a given position.
"""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

YARDAGE_TARGETS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "carries": "carries",
    "receptions": "receptions",
}

POSITION_MARKETS: dict[str, list[str]] = {
    "QB": ["passing_yards"],
    "RB": ["rushing_yards", "carries"],
    "WR": ["receiving_yards", "receptions"],
    "TE": ["receiving_yards", "receptions"],
}


def fit_anytime_td_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train.fillna(0), y_train)
    return model


#: Markets fitted to the MEDIAN (`reg:absoluteerror`) rather than the mean. They are judged on MAE and priced against a
#: line that splits outcomes 50/50, and a squared-error fit chases the right tail of a skewed yardage distribution
#: (median-coverage gap 0.203 -> 0.025 rushing, 0.175 -> 0.026 receiving; MAE +1.39 / +1.14 yds, paired bootstrap,
#: walk-forward 2021-25; docs/research/2026-10-13-nfl-cfb-model-gaps.md in predictor-hub). `passing_yards` is
#: deliberately absent: MAE improves but the gap-change CI does not exclude widening, so it does not clear the bar.
#: `carries` / `receptions` were not measured and stay on squared error.
ABSOLUTE_ERROR_MARKETS = frozenset({"rushing_yards", "receiving_yards"})


def fit_yardage_regressor(X_train: pd.DataFrame, y_train: pd.Series, market: str | None = None) -> XGBRegressor:
    objective = "reg:absoluteerror" if market in ABSOLUTE_ERROR_MARKETS else "reg:squarederror"
    model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42, objective=objective)
    model.fit(X_train.fillna(0), y_train)
    return model


def predict_props(models: dict, feature_row: pd.Series, position: str) -> dict:
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0).to_numpy().reshape(1, -1)

    result = {"anytime_td_prob": float(models["anytime_td"].predict_proba(X)[0, 1])}

    for market in POSITION_MARKETS.get(position, []):
        # Gracefully skip a market not yet in the models dict -- this
        # happens right after this code ships but before the next retrain
        # actually produces a model for it (see manifest.py's train_all,
        # which only saves a market's model when it had positive training
        # rows).
        if market in models:
            result[market] = float(models[market].predict(X)[0])

    return result
