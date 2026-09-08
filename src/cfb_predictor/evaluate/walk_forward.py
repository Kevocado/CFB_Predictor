"""walk_forward.py — season-by-season walk-forward validation for the three
models/game_outcome.py candidates. Builds the full feature frame ONCE (no
lookahead -- every feature is already shift(1)/expanding computed before any
slicing), then slices by season so evaluate_candidate can be called
repeatedly without redoing feature engineering. Near-verbatim port of
nfl_predictor's own evaluate/walk_forward.py -- the only change is threading
an optional fbs_teams mapping through to build_training_frame so every
fold's rows already exclude FCS-opponent games.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..features.build import build_training_frame
from ..models import game_outcome


def prepare_folds(
    games_df: pd.DataFrame,
    fbs_teams: dict[int, set[str]] | None = None,
    min_train_seasons: int = 2,
) -> list[dict]:
    df, feature_cols = build_training_frame(games_df, fbs_teams=fbs_teams)
    seasons = sorted(df["season"].unique())

    folds = []
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_seasons = seasons[:i]
        train_df = df[df["season"].isin(train_seasons)]
        val_df = df[df["season"] == val_season]
        if train_df.empty or val_df.empty:
            continue
        folds.append({"val_season": val_season, "train_df": train_df, "val_df": val_df, "feature_cols": feature_cols})
    return folds


def _predict_margin_elo_batch(candidate: dict, df: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            game_outcome.predict_margin_elo(candidate, r, hr, ar)
            for r, hr, ar in zip(df["rating_diff"], df["home_rest_days"], df["away_rest_days"])
        ]
    )


class _EloModelAdapter:
    """Adapts the elo candidate's dict + free function to the model.predict(X)
    interface residual_sigma expects, without ignoring the X it's given."""

    def __init__(self, candidate: dict):
        self.candidate = candidate

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return _predict_margin_elo_batch(self.candidate, X)


def _predict_margins(candidate: str, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]):
    X_train, y_train = train_df[feature_cols], train_df["margin"]
    X_val = val_df[feature_cols]

    if candidate == "elo":
        model = game_outcome.fit_elo_candidate(train_df)
        preds = _predict_margin_elo_batch(model, val_df)
        sigma = game_outcome.residual_sigma(_EloModelAdapter(model), X_train, y_train)
        return preds, sigma

    if candidate == "ridge":
        fit_fn = game_outcome.fit_margin_regression
    elif candidate == "xgb":
        fit_fn = game_outcome.fit_xgb_margin
    else:
        raise ValueError(f"Unknown candidate: {candidate!r}")

    model = fit_fn(X_train, y_train)
    preds = model.predict(X_val.fillna(0))
    sigma = game_outcome.residual_sigma(model, X_train, y_train)
    return preds, sigma


def pooled_predictions(folds: list[dict], candidate: str) -> tuple[np.ndarray, np.ndarray]:
    """All folds' held-out probs/outcomes concatenated -- enough volume for
    a meaningful reliability curve, unlike any single fold alone."""
    all_probs, all_outcomes = [], []
    for fold in folds:
        train_df, val_df, feature_cols = fold["train_df"], fold["val_df"], fold["feature_cols"]
        preds, sigma = _predict_margins(candidate, train_df, val_df, feature_cols)
        probs = np.array([game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds])
        all_probs.append(np.clip(probs, 1e-6, 1 - 1e-6))
        all_outcomes.append((val_df["margin"] > 0).astype(int).to_numpy())
    return np.concatenate(all_probs), np.concatenate(all_outcomes)


def evaluate_candidate(folds: list[dict], candidate: str) -> pd.DataFrame:
    rows = []
    for fold in folds:
        train_df, val_df, feature_cols = fold["train_df"], fold["val_df"], fold["feature_cols"]
        preds, sigma = _predict_margins(candidate, train_df, val_df, feature_cols)

        probs = np.array(
            [game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds]
        )
        actual = (val_df["margin"] > 0).astype(int).to_numpy()
        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        rows.append(
            {
                "val_season": fold["val_season"],
                "n_games": len(val_df),
                "log_loss": log_loss(actual, probs, labels=[0, 1]),
                "brier": brier_score_loss(actual, probs),
            }
        )
    return pd.DataFrame(rows)
