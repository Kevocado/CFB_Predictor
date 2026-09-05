# tests/test_player_usage.py
import pandas as pd
import pytest

from cfb_predictor.features import player_usage


def _player_stats():
    rows = []
    for week in range(1, 4):
        rows.append(
            {
                "player_id": "p1", "player_name": "Runner", "position": "RB", "recent_team": "Texas",
                "season": 2025, "week": week,
                "passing_yards": 0, "passing_tds": 0, "rushing_yards": 80 + week, "rushing_tds": 1,
                "receiving_yards": 10, "receiving_tds": 0, "receptions": 2, "targets": float("nan"), "carries": 18,
            }
        )
    return pd.DataFrame(rows)


def test_build_player_training_frame_adds_rolling_features_and_target():
    df, feature_cols = player_usage.build_player_training_frame(_player_stats())

    assert "anytime_td" in df.columns
    assert set(feature_cols).issubset(df.columns)
    week1 = df[df["week"] == 1].iloc[0]
    assert pd.isna(week1["rushing_yards_roll"])
    week3 = df[df["week"] == 3].iloc[0]
    assert week3["rushing_yards_roll"] == (81 + 82) / 2


def test_build_features_for_player_returns_none_with_no_history():
    row = player_usage.build_features_for_player("unknown", _player_stats())
    assert row is None


def test_build_features_for_player_returns_series_with_history():
    row = player_usage.build_features_for_player("p1", _player_stats())
    assert row is not None
    assert row["rushing_yards_roll"] == pytest.approx((81 + 82 + 83) / 3)
