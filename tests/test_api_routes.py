import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cfb_predictor.api.main import app
from cfb_predictor.api import routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        routes.games_data, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "401520145", "season": season, "week": week,
              "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State",
              "home_score": None, "away_score": None}]
        ),
    )
    monkeypatch.setattr(
        routes.games_data, "load_training_data",
        lambda seasons: pd.DataFrame(
            [{"game_id": "g0", "season": seasons[0], "week": 1, "gameday": "2025-08-01",
              "home_team": "Texas", "away_team": "Ohio State", "home_score": 24, "away_score": 17}]
        ),
    )
    monkeypatch.setattr(routes.odds_api, "fetch_game_odds", lambda: pd.DataFrame())
    monkeypatch.setattr(
        routes, "_load_models_cached",
        lambda: {
            "game_outcome_model": None, "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
            "total_model": None, "player_models": {"feature_cols": [], "anytime_td": None},
            "feature_cols": [], "player_feature_cols": [],
        },
    )
    monkeypatch.setattr(
        routes, "_predict_game_from_models",
        lambda models, home, away, games_df, spread_line=None, total_line=None: {
            "home_win_prob": 0.4, "away_win_prob": 0.6, "home_cover_prob": 0.45, "away_cover_prob": 0.55,
            "over_prob": 0.52, "under_prob": 0.48,
        },
    )
    monkeypatch.setattr(routes.store, "get_track_record", lambda: {"n_resolved_games": 0, "pct_moneyline_correct": None})
    return TestClient(app)


def test_get_games_returns_week_slate(client):
    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["game_id"] == "401520145"


def test_get_game_prediction(client):
    response = client.get("/api/games/2025/1/401520145/prediction")

    assert response.status_code == 200
    body = response.json()
    assert body["home_win_prob"] == 0.4


def test_get_game_prediction_404s_for_unknown_game(client):
    response = client.get("/api/games/2025/1/nonexistent/prediction")

    assert response.status_code == 404


def test_get_track_record(client):
    response = client.get("/api/track-record")

    assert response.status_code == 200
    assert response.json()["n_resolved_games"] == 0


def test_get_games_handles_nan_scores_for_unplayed_games(client, monkeypatch):
    monkeypatch.setattr(
        routes.games_data, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "401520145", "season": season, "week": week,
              "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State",
              "home_score": float("nan"), "away_score": float("nan")}]
        ),
    )

    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["home_score"] is None
    assert body[0]["away_score"] is None


def test_lines_for_game_reads_spreads_and_totals_from_odds_api():
    odds_df = pd.DataFrame(
        [
            {"home_team": "Texas", "away_team": "Ohio State", "market": "spreads", "outcome_name": "Texas", "point": -3.5},
            {"home_team": "Texas", "away_team": "Ohio State", "market": "totals", "outcome_name": "Over", "point": 51.5},
        ]
    )

    spread_line, total_line = routes._lines_for_game(odds_df, "Texas", "Ohio State")

    assert spread_line == -3.5
    assert total_line == 51.5


def test_lines_for_game_returns_none_when_odds_missing():
    spread_line, total_line = routes._lines_for_game(pd.DataFrame(), "Texas", "Ohio State")

    assert spread_line is None
    assert total_line is None
