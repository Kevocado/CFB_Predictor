import pandas as pd
from fastapi.testclient import TestClient
from cfb_predictor.api import routes
from cfb_predictor.api.main import app

def test_public_mode_serves_hub_from_snapshot(monkeypatch):
    snap = {"season": 2026, "hub_teams": {"season": 2026, "teams": [], "advanced_available": False},
            "hub_players": {"season": 2026, "players": [], "leaderboards": {}}}
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot", lambda: snap)
    monkeypatch.setattr(routes, "_get_hub_teams_live", lambda season: (_ for _ in ()).throw(AssertionError("live")))
    c = TestClient(app)
    assert c.get("/api/hub/teams?season=2026").json()["advanced_available"] is False
    assert c.get("/api/hub/players?season=2026").json()["players"] == []

def test_live_teams_flag_advanced_unavailable_without_key(monkeypatch):
    monkeypatch.setattr(routes, "PUBLIC_MODE", False)
    monkeypatch.setattr(routes, "_load_game_history", lambda s: pd.DataFrame([dict(game_id=1, season=2026, week=1,
        gameday="2026-08-30", home_team="Georgia", away_team="Clemson", home_score=34, away_score=3)]))
    monkeypatch.setattr(routes.advanced_stats, "fetch_advanced", lambda s: [])
    body = TestClient(app).get("/api/hub/teams?season=2026").json()
    assert body["advanced_available"] is False and {t["team"] for t in body["teams"]} == {"Georgia", "Clemson"}


def test_empty_snapshot_hub_falls_back_to_live(monkeypatch):
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot", lambda: {"season": 2026, "hub_teams": {}, "hub_players": {}})
    monkeypatch.setattr(routes, "_get_hub_teams_live", lambda season: {"season": season, "teams": [{"team": "LIVE"}]})
    monkeypatch.setattr(routes, "_get_hub_players_live", lambda season: {"season": season, "players": [], "leaderboards": {"QB": []}})
    c = TestClient(app)
    assert c.get("/api/hub/teams?season=2026").json()["teams"] == [{"team": "LIVE"}]
    assert c.get("/api/hub/players?season=2026").json()["leaderboards"] == {"QB": []}
