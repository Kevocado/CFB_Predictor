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
    blocks: tuple[str, ...] = (),
    aux=None,
) -> list[dict]:
    # `blocks`/`aux` are forwarded to build_training_frame so a fold can be built with the
    # epa block on. Without this the blocks argument was dead weight: _assemble() would wire
    # epa from aux.efficiency, but nothing could ever reach it from here.
    df, feature_cols = build_training_frame(games_df, fbs_teams=fbs_teams, blocks=blocks, aux=aux)
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


def _fit_predict(candidate: str, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], target: str = "margin") -> np.ndarray:
    """Fit `candidate` on `train_df` and predict `target` for `val_df`. Nothing from `val_df` reaches the fit."""
    if candidate == "elo":
        if target != "margin":
            raise ValueError("the elo candidate predicts the margin only")
        return _predict_margin_elo_batch(game_outcome.fit_elo_candidate(train_df), val_df)
    fit_fn = {"ridge": game_outcome.fit_margin_regression, "xgb": game_outcome.fit_xgb_margin}.get(candidate)
    if fit_fn is None:
        raise ValueError(f"Unknown candidate: {candidate!r}")
    return fit_fn(train_df[feature_cols], train_df[target]).predict(val_df[feature_cols].fillna(0))


def _honest_sigma(candidate: str, train_df: pd.DataFrame, feature_cols: list[str], target: str = "margin") -> float:
    """Sigma for a fold, from error the model has NOT been fitted on: the last TRAINING season is held out of an
    inner fit on the seasons before it. A fold with a single training season has nothing to hold out and falls
    back to the in-sample spread (optimistic; only the smallest folds ever take this branch)."""
    seasons = sorted(train_df["season"].unique())
    if len(seasons) < 2:
        preds = _fit_predict(candidate, train_df, train_df, feature_cols, target)
        return game_outcome.sigma_from_residuals(train_df[target].to_numpy(float) - preds)
    inner_train = train_df[train_df["season"] != seasons[-1]]
    inner_val = train_df[train_df["season"] == seasons[-1]]
    preds = _fit_predict(candidate, inner_train, inner_val, feature_cols, target)
    return game_outcome.sigma_from_residuals(inner_val[target].to_numpy(float) - preds)


def oof_residuals(folds: list[dict], candidate: str, target: str = "margin") -> np.ndarray:
    """Every validation season's residuals, each predicted by a model trained only on EARLIER seasons."""
    pieces = []
    for fold in folds:
        preds = _fit_predict(candidate, fold["train_df"], fold["val_df"], fold["feature_cols"], target)
        pieces.append(fold["val_df"][target].to_numpy(float) - preds)
    return np.concatenate(pieces) if pieces else np.array([])


def _predict_margins(candidate: str, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]):
    preds = _fit_predict(candidate, train_df, val_df, feature_cols, "margin")
    return preds, _honest_sigma(candidate, train_df, feature_cols, "margin")


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
