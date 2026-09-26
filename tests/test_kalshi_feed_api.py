import pytest
from fastapi.testclient import TestClient

from cfb_predictor.api.main import app
from cfb_predictor.tracking import store


@pytest.fixture
def client(monkeypatch, tmp_path):
    from cfb_predictor import config

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    return TestClient(app)


def test_kalshi_feed_serves_frozen_pregame_rows_and_calibration(client):
    store.record_game_predictions([{
        "game_id": "401871049", "home_team": "New Mexico State", "away_team": "Western Kentucky",
        "commence_time": "2099-10-02 00:15:00", "season": 2026, "week": 4,
        "home_win_prob": 0.40, "away_win_prob": 0.60, "predicted_margin": -3.4, "sigma": 13.2,
        "predicted_total": 39.8, "total_sigma": 12.5, "model_version": "ridge@t",
    }])

    response = client.get("/api/kalshi-feed")

    assert response.status_code == 200
    body = response.json()
    assert body["sport"] == "cfb"
    assert body["lead_hours"] == 48
    assert [g["game_id"] for g in body["games"]] == ["401871049"]
    game = body["games"][0]
    assert game["start_utc"] == "2099-10-02T00:15:00+00:00"
    assert game["p_home"] == 0.40 and game["margin_mu"] == -3.4 and game["backfilled"] is False
    assert set(body["calibration"]) == {"n_buckets", "winner", "spread", "total"}
    assert body["generated_at"].endswith("+00:00")


def test_kalshi_feed_is_empty_but_valid_with_no_snapshots(client):
    body = client.get("/api/kalshi-feed").json()

    assert body["games"] == []
    assert body["calibration"]["winner"][0] == {"lo": 0.0, "hi": 0.1, "n": 0, "mean_prob": None, "hit_rate": None}


def test_kalshi_feed_touches_neither_games_data_nor_models_nor_the_network(client, monkeypatch):
    """The feed must never recompute a prediction. A live forecast is a DIFFERENT number from the
    frozen snapshot the hub graded, so recomputing would swap the series out from under the
    calibration check without any error."""
    import cfb_predictor.api.routes as routes

    def boom(*_a, **_k):
        raise AssertionError("the feed recomputed something instead of reading the store")

    monkeypatch.setattr(routes.games_data, "fetch_upcoming_games", boom)
    monkeypatch.setattr(routes, "_load_models_cached", boom)
    monkeypatch.setattr(routes, "requests", boom)

    assert TestClient(app).get("/api/kalshi-feed").status_code == 200
