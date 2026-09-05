"""player_usage.py — player-level rolling usage/production features for
anytime-TD and yardage prop models. Same shift(1)-then-rolling discipline as
features/rolling_form.py, applied per player instead of per team. Verbatim
port from nfl_predictor: "targets" being always NaN for CFB (see
data/player_stats.py) degrades this file's rolling mean to NaN for that one
column, which fit_anytime_td_classifier/fit_yardage_regressor's .fillna(0)
already tolerates -- no change needed here."""

from __future__ import annotations

import pandas as pd

ROLL_STATS = ["passing_yards", "rushing_yards", "receiving_yards", "targets", "carries"]
PLAYER_FEATURE_COLUMNS = [f"{stat}_roll" for stat in ROLL_STATS]


def _add_rolling(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grouped = df.groupby("player_id")
    for stat in ROLL_STATS:
        df[f"{stat}_roll"] = grouped[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return df


def build_player_training_frame(player_stats_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = _add_rolling(player_stats_df)
    df["anytime_td"] = (
        (df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0) + df["passing_tds"].fillna(0)) > 0
    ).astype(int)
    return df, PLAYER_FEATURE_COLUMNS


def build_features_for_player(player_id: str, player_stats_df: pd.DataFrame) -> pd.Series | None:
    history = player_stats_df[player_stats_df["player_id"] == player_id].sort_values(["season", "week"])
    if history.empty:
        return None
    recent = history.tail(5)
    return pd.Series({f"{stat}_roll": float(recent[stat].mean()) for stat in ROLL_STATS})
