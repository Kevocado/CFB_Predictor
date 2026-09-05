"""power_ratings.py — Elo-style team power ratings, margin-of-victory
weighted.

Standard football Elo shape (à la 538's NFL model, equally applicable to
CFB): expected score from a logistic function of the rating gap plus a
home-field bonus, update scaled by both the surprise (actual - expected) and
a margin-of-victory multiplier so a 40-point win moves ratings more than a
3-point win. Chronological — every game's *pregame* rating only reflects
games played strictly before it, so this is safe to use as a training
feature with no lookahead. Verbatim port from nfl_predictor: nothing in this
file references anything NFL-specific.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_START_RATING = 1500.0
DEFAULT_K = 20.0
DEFAULT_HOME_FIELD = 65.0
ELO_SCALE = 400.0


def _expected_home_win_prob(home_rating: float, away_rating: float, home_field: float) -> float:
    diff = (home_rating + home_field) - away_rating
    return 1.0 / (1.0 + 10 ** (-diff / ELO_SCALE))


def _margin_multiplier(margin: float, rating_diff: float) -> float:
    """538's Elo margin-of-victory multiplier: log of the margin, damped
    when the favorite already led the ratings by a lot (an autocorrelation
    correction — a huge win over a much weaker team shouldn't move ratings
    as much as the same margin over an evenly matched one).

    `rating_diff` must be winner-relative (winner's pregame rating minus
    loser's), not home-minus-away — otherwise the damping inverts for away
    wins, damping upsets and amplifying expected outcomes backwards."""
    return np.log(max(abs(margin), 1) + 1) * (2.2 / ((rating_diff * 0.001) + 2.2))


def compute_pregame_ratings(
    games_df: pd.DataFrame,
    k: float = DEFAULT_K,
    home_field: float = DEFAULT_HOME_FIELD,
    start_rating: float = DEFAULT_START_RATING,
) -> pd.DataFrame:
    games_df = games_df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    home_pregame = []
    away_pregame = []

    for _, game in games_df.iterrows():
        home, away = game["home_team"], game["away_team"]
        home_rating = ratings.get(home, start_rating)
        away_rating = ratings.get(away, start_rating)
        home_pregame.append(home_rating)
        away_pregame.append(away_rating)

        if pd.isna(game["home_score"]) or pd.isna(game["away_score"]):
            continue  # unplayed game: record pregame rating, no update

        margin = game["home_score"] - game["away_score"]
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        expected = _expected_home_win_prob(home_rating, away_rating, home_field)
        winner_relative_diff = (home_rating - away_rating) if margin >= 0 else (away_rating - home_rating)
        multiplier = _margin_multiplier(margin, winner_relative_diff)
        delta = k * multiplier * (actual - expected)

        ratings[home] = home_rating + delta
        ratings[away] = away_rating - delta

    result = games_df.copy()
    result["home_pregame_rating"] = home_pregame
    result["away_pregame_rating"] = away_pregame
    return result


def final_ratings(
    games_df: pd.DataFrame,
    k: float = DEFAULT_K,
    home_field: float = DEFAULT_HOME_FIELD,
    start_rating: float = DEFAULT_START_RATING,
) -> dict[str, float]:
    """Every team's rating after the last played game in games_df — used to
    seed live predictions for upcoming games."""
    rated = compute_pregame_ratings(games_df, k=k, home_field=home_field, start_rating=start_rating)
    played = rated[rated["home_score"].notna() & rated["away_score"].notna()]
    if played.empty:
        return {}

    ratings: dict[str, float] = {}
    for _, game in played.iterrows():
        home, away = game["home_team"], game["away_team"]
        home_rating = ratings.get(home, game["home_pregame_rating"])
        away_rating = ratings.get(away, game["away_pregame_rating"])
        margin = game["home_score"] - game["away_score"]
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        expected = _expected_home_win_prob(home_rating, away_rating, home_field)
        winner_relative_diff = (home_rating - away_rating) if margin >= 0 else (away_rating - home_rating)
        multiplier = _margin_multiplier(margin, winner_relative_diff)
        delta = k * multiplier * (actual - expected)
        ratings[home] = home_rating + delta
        ratings[away] = away_rating - delta
    return ratings
