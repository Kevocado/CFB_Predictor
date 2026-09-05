"""rolling_form.py — each team's own rolling scoring form, no lookahead.

Reshapes games into one row per team-appearance (home and away rows each
carry "points_scored"/"points_allowed" from that team's own perspective),
computes a shift(1) rolling mean per team so the current game is always
excluded, then reshapes back. Verbatim port from nfl_predictor — operates
only on game_id/gameday/home_team/away_team/home_score/away_score.
"""

from __future__ import annotations

import pandas as pd


def _team_appearances(games_df: pd.DataFrame) -> pd.DataFrame:
    home = games_df[["game_id", "gameday", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "points_scored", "away_score": "points_allowed"}
    )
    away = games_df[["game_id", "gameday", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "points_scored", "home_score": "points_allowed"}
    )
    appearances = pd.concat([home, away], ignore_index=True)
    return appearances.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def add_rolling_form(games_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    appearances = _team_appearances(games_df)
    grouped = appearances.groupby("team")
    appearances["points_scored_roll"] = grouped["points_scored"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    appearances["points_allowed_roll"] = grouped["points_allowed"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    home_form = appearances.rename(
        columns={
            "team": "home_team",
            "points_scored_roll": "home_points_scored_roll",
            "points_allowed_roll": "home_points_allowed_roll",
        }
    )[["game_id", "home_team", "home_points_scored_roll", "home_points_allowed_roll"]]
    away_form = appearances.rename(
        columns={
            "team": "away_team",
            "points_scored_roll": "away_points_scored_roll",
            "points_allowed_roll": "away_points_allowed_roll",
        }
    )[["game_id", "away_team", "away_points_scored_roll", "away_points_allowed_roll"]]

    result = games_df.merge(home_form, on=["game_id", "home_team"], how="left")
    result = result.merge(away_form, on=["game_id", "away_team"], how="left")
    return result
