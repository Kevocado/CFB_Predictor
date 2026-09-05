# tests/test_player_stats.py
import pandas as pd
import pytest

from cfb_predictor.data import player_stats


def _games_df():
    return pd.DataFrame(
        [
            {"game_id": "401520145", "season": 2025, "week": 1, "gameday": pd.Timestamp("2025-08-30")},
        ]
    )


def _raw_player_game_stats():
    return [
        {
            "id": 401520145,
            "teams": [
                {
                    "school": "Ohio State",
                    "categories": [
                        {
                            "name": "passing",
                            "types": [
                                {"name": "YDS", "athletes": [{"id": 4567, "name": "J. Doe", "stat": "291"}]},
                                {"name": "TD", "athletes": [{"id": 4567, "name": "J. Doe", "stat": "2"}]},
                            ],
                        },
                        {
                            "name": "rushing",
                            "types": [
                                {"name": "YDS", "athletes": [{"id": 4568, "name": "R. Back", "stat": "112"}]},
                                {"name": "CAR", "athletes": [{"id": 4568, "name": "R. Back", "stat": "18"}]},
                            ],
                        },
                        {
                            "name": "receiving",
                            "types": [
                                {"name": "YDS", "athletes": [{"id": 4569, "name": "W. Out", "stat": "85"}]},
                                {"name": "REC", "athletes": [{"id": 4569, "name": "W. Out", "stat": "6"}]},
                            ],
                        },
                    ],
                }
            ],
        }
    ]


def test_fetch_weekly_player_stats_caches_per_season(monkeypatch, tmp_path):
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    calls = []

    def fake_import(season):
        calls.append(season)
        return _raw_player_game_stats()

    monkeypatch.setattr(player_stats, "_import_player_game_stats", fake_import)

    first = player_stats.fetch_weekly_player_stats([2025], _games_df())
    second = player_stats.fetch_weekly_player_stats([2025], _games_df())

    assert calls == [2025]
    assert len(first) == 3  # one row each for the QB, RB, and WR
    assert second.equals(first)


def test_flattened_stats_attach_week_from_games_df(monkeypatch, tmp_path):
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(player_stats, "_import_player_game_stats", lambda season: _raw_player_game_stats())

    df = player_stats.fetch_weekly_player_stats([2025], _games_df())

    assert (df["week"] == 1).all()
    assert (df["season"] == 2025).all()


def test_flattened_stats_map_category_type_pairs_to_flat_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(player_stats, "_import_player_game_stats", lambda season: _raw_player_game_stats())

    df = player_stats.fetch_weekly_player_stats([2025], _games_df())

    qb = df[df["player_id"] == "4567"].iloc[0]
    assert qb["passing_yards"] == 291.0
    assert qb["passing_tds"] == 2.0
    assert qb["position"] == "QB"

    rb = df[df["player_id"] == "4568"].iloc[0]
    assert rb["rushing_yards"] == 112.0
    assert rb["carries"] == 18.0
    assert rb["position"] == "RB"

    wr = df[df["player_id"] == "4569"].iloc[0]
    assert wr["receiving_yards"] == 85.0
    assert wr["receptions"] == 6.0
    assert wr["position"] == "WR"


def test_targets_column_is_always_present_but_nan(monkeypatch, tmp_path):
    # CFBD's box score does not track targets -- the column must still exist
    # (features/player_usage.py's ROLL_STATS references it) but stays NaN.
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(player_stats, "_import_player_game_stats", lambda season: _raw_player_game_stats())

    df = player_stats.fetch_weekly_player_stats([2025], _games_df())

    assert "targets" in df.columns
    assert df["targets"].isna().all()
