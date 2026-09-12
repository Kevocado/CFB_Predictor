"""game_outcome.py — margin-of-victory candidates for the game-outcome
race, plus the shared margin -> win/cover/total-probability conversion.

CFB scoring, like NFL, isn't a low-count Poisson process -- this predicts a
continuous point margin (home_score - away_score) and total_points, then
converts each to probabilities via a fitted-Normal residual distribution.
Three candidates are raced in evaluate/walk_forward.py: Elo (implicit in
features.power_ratings' rating_diff, converted directly via
margin_to_probabilities with a fixed points-per-Elo-point scale), ridge
regression, and XGBoost -- whichever wins held-out log-loss is served.
Verbatim port from nfl_predictor: no NFL-specific column or assumption
anywhere in this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

ELO_POINTS_PER_RATING_POINT = 1.0 / 25.0


def fit_elo_candidate(train_df: pd.DataFrame) -> dict:
    return {"points_per_rating_point": ELO_POINTS_PER_RATING_POINT}


def predict_margin_elo(candidate: dict, rating_diff: float, home_rest_days: float, away_rest_days: float) -> float:
    return rating_diff * candidate["points_per_rating_point"] + 0.05 * (home_rest_days - away_rest_days)


def fit_margin_regression(X_train: pd.DataFrame, y_margin: pd.Series) -> Ridge:
    model = Ridge(alpha=1.0)
    model.fit(X_train.fillna(0), y_margin)
    return model


def fit_xgb_margin(X_train: pd.DataFrame, y_margin: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        reg_lambda=1.0, reg_alpha=0.0, random_state=42,
    )
    model.fit(X_train.fillna(0), y_margin)
    return model


def residual_sigma(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    preds = model.predict(X_val.fillna(0))
    residuals = np.asarray(y_val) - preds
    if len(residuals) == 0:
        return 1.0
    return float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.std(residuals) or 1.0)


def margin_to_probabilities(
    predicted_margin: float,
    sigma: float,
    spread_line: float | None = None,
    total_line: float | None = None,
    predicted_total: float | None = None,
    total_sigma: float | None = None,
) -> dict:
    """margin ~ Normal(predicted_margin, sigma). home_win_prob = P(margin > 0).
    spread_line follows the home team's expected margin (positive means home
    favored by that many points) -- the home team covers when
    margin > spread_line. total_points ~ Normal(predicted_total, total_sigma);
    over_prob = P(total > total_line).
    
    If spread_line is not provided, uses the model's predicted_margin as the
    spread line (model's expected margin)."""
    home_win_prob = float(1.0 - norm.cdf(0.0, loc=predicted_margin, scale=sigma))
    result = {"home_win_prob": home_win_prob, "away_win_prob": 1.0 - home_win_prob}

    # Use model's predicted margin as default spread if no odds provided
    effective_spread = spread_line if spread_line is not None else predicted_margin
    home_cover_prob = float(1.0 - norm.cdf(effective_spread, loc=predicted_margin, scale=sigma))
    result["home_cover_prob"] = home_cover_prob
    result["away_cover_prob"] = 1.0 - home_cover_prob

    if total_line is not None and predicted_total is not None and total_sigma is not None:
        over_prob = float(1.0 - norm.cdf(total_line, loc=predicted_total, scale=total_sigma))
        result["over_prob"] = over_prob
        result["under_prob"] = 1.0 - over_prob

    return result
