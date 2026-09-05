"""player_props.py — anytime-TD classifier and per-position yardage
regressors. Verbatim port from nfl_predictor: yardage props are priced as a
continuous over/under line for both sports, and the fitting/prediction
shape has no NFL-specific assumption. POSITION_YARDAGE_MARKET's four-way
{QB, RB, WR, TE} mapping is kept as-is even though data/player_stats.py's
CFBD-derived position inference can only ever emit QB/RB/WR (see that
module's docstring) -- WR and TE already map to the same receiving_yards
market, so the extra entry costs nothing and future-proofs this file.
"""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

YARDAGE_TARGETS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
}

POSITION_YARDAGE_MARKET = {"QB": "passing_yards", "RB": "rushing_yards", "WR": "receiving_yards", "TE": "receiving_yards"}


def fit_anytime_td_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train.fillna(0), y_train)
    return model


def fit_yardage_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train.fillna(0), y_train)
    return model


def predict_props(models: dict, feature_row: pd.Series, position: str) -> dict:
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0).to_numpy().reshape(1, -1)

    result = {"anytime_td_prob": float(models["anytime_td"].predict_proba(X)[0, 1])}

    market = POSITION_YARDAGE_MARKET.get(position)
    if market and market in models:
        result[market] = float(models[market].predict(X)[0])

    return result
