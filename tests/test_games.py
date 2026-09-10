import pandas as pd
import pytest

from cfb_predictor.data import games


def _raw_games_frame():
    return pd.DataFrame(
        [
            {
                "id": 401520145, "season": 2025, "week": 1, "start_date": "2025-08-30T16:00:00.000Z",
                "home_team": "Texas", "away_team": "Ohio State",
                "home_points": 7, "away_points": 14,
                "home_conference": "SEC", "away_conference": "Big Ten",
                "home_division": "fbs", "away_division": "fbs",
                "conference_game": False, "neutral_site": True,
            },
            {
                "id": 401520200, "season": 2025, "week": 1, "start_date": "2025-08-30T19:30:00.000Z",
                "home_team": "Alabama", "away_team": "Western Carolina",
                "home_points": None, "away_points": None,
                "home_conference": "SEC", "away_conference": None,
                "home_division": "fbs", "away_division": "fcs",
                "conference_game": False, "neutral_site": False,
            },
        ]
    )


def _raw_teams_frame():
    return pd.DataFrame(
        [
            {"school": "Texas", "conference": "SEC", "division": "fbs", "classification": "fbs"},
            {"school": "Ohio State", "conference": "Big Ten", "division": "fbs", "classification": "fbs"},
            {"school": "Alabama", "conference": "SEC", "division": "fbs", "classification": "fbs"},
        ]
    )


def test_fetch_schedules_caches_per_season(monkeypatch, tmp_path):
    monkeypatch.setattr(games, "GAMES_CACHE_DIR", tmp_path)
    calls = []

    def fake_import(season):
        calls.append(season)
        return _raw_games_frame()

    monkeypatch.setattr(games, "_import_games", fake_import)

    first = games.fetch_schedules([2025])
    second = games.fetch_schedules([2025])

    assert calls == [2025]  # second call hit the cache, not the network
    assert len(first) == 2
    assert "conference_game" in first.columns
    assert second.equals(first)


def test_fetch_schedules_calls_once_per_missing_season(monkeypatch, tmp_path):
    monkeypatch.setattr(games, "GAMES_CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(games, "_import_games", lambda season: calls.append(season) or _raw_games_frame())

    games.fetch_schedules([2024, 2025])

    assert calls == [2024, 2025]  # exactly one call per season, never per game/team


def test_load_training_data_drops_unplayed_games(monkeypatch, tmp_path):
    monkeypatch.setattr(games, "GAMES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(games, "_import_games", lambda season: _raw_games_frame())

    df = games.load_training_data([2025])

    assert len(df) == 1
    assert df.iloc[0]["game_id"] == "401520145"


def test_default_completed_seasons_excludes_current_season(monkeypatch):
    monkeypatch.setattr(games, "CURRENT_SEASON", 2026)
    seasons = games.default_completed_seasons(n=3)
    assert seasons == [2023, 2024, 2025]


def test_fetch_upcoming_games_filters_season_week(monkeypatch, tmp_path):
    monkeypatch.setattr(games, "GAMES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(games, "_import_games", lambda season: _raw_games_frame())

    upcoming = games.fetch_upcoming_games(2025, 1)

    assert list(upcoming["game_id"]) == ["401520200"]


def test_fetch_week_games_includes_finished_and_upcoming(monkeypatch, tmp_path):
    monkeypatch.setattr(games, "GAMES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(games, "_import_games", lambda season: _raw_games_frame())

    week_games = games.fetch_week_games(2025, 1)

    assert set(week_games["game_id"]) == {"401520145", "401520200"}


def test_fetch_fbs_teams_caches_and_returns_conference_and_division(monkeypatch, tmp_path):
    monkeypatch.setattr(games, "TEAMS_CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(games, "_import_fbs_teams", lambda season: calls.append(season) or _raw_teams_frame())

    first = games.fetch_fbs_teams(2025)
    second = games.fetch_fbs_teams(2025)

    assert calls == [2025]
    assert set(first["team"]) == {"Texas", "Ohio State", "Alabama"}
    assert set(first["conference"]) == {"SEC", "Big Ten"}
    assert second.equals(first)
