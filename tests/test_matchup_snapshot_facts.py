"""Snapshot-carried FBS matchup duels reach /facts context.matchups (offline, real CFBD 2025 data shapes)."""
import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cfb_predictor import public_snapshot
from cfb_predictor.api import facts as facts_mod
from cfb_predictor.api.main import app
from cfb_predictor.data import cfbd_advanced

DATA = Path(__file__).resolve().parents[1] / "data" / "cfbd" / "advanced_2025.json"
SEASON = 2025
WEEK = 12
FBS_SUBSET = (
    "Alabama,Georgia,Ohio State,Texas,Oklahoma,Michigan,Penn State,LSU,Florida,Tennessee,Auburn,Texas A&M,"
    "Notre Dame,Oregon,Washington,USC,UCLA,Utah,Colorado,Arizona,Arizona State,Iowa,Wisconsin,Minnesota,Nebraska,"
    "Illinois,Purdue,Indiana,Clemson,Florida State,Miami,North Carolina,NC State,Virginia Tech,Virginia,Duke,"
    "Louisville,Pittsburgh,Boston College,Syracuse,Wake Forest,Georgia Tech,Missouri,Kentucky,South Carolina,"
    "Arkansas,Ole Miss,Mississippi State,Vanderbilt,Baylor,TCU,Kansas State,Kansas,Iowa State,Oklahoma State,"
    "Texas Tech,West Virginia,Cincinnati,BYU,Houston,UCF,Stanford,California,Oregon State,Washington State,"
    "Maryland,Rutgers,Northwestern,Michigan State,Boise State,Memphis,Tulane,SMU"
).split(",")
START = pd.Timestamp("2025-08-30", tz="UTC")


@pytest.fixture(scope="module")
def world():
    rows = json.loads(DATA.read_text())
    eff = cfbd_advanced.to_team_game_frame(rows)
    # The real FBS list needs CFBD (fetch_fbs_teams); this is a hand-listed subset of genuine 2025 FBS programmes.
    # The efficiency file holds ~310 teams (FCS included), so ranking "everyone" would give n_teams in the hundreds.
    fbs = set(FBS_SUBSET)
    assert fbs <= set(eff["team"]) and len(set(eff["team"])) > 250
    assert {"Western Carolina", "North Dakota State", "Montana"} <= set(eff["team"]) - fbs
    wk = eff.drop_duplicates("game_id").set_index("game_id")["week"]
    pair = eff.groupby("game_id")["team"].apply(list)
    hist = pd.DataFrame({
        "game_id": pair.index, "season": SEASON,
        "gameday": [(START + pd.Timedelta(days=7 * (int(wk[g]) - 1))).isoformat() for g in pair.index],
        "home_team": [p[0] for p in pair], "away_team": [p[1] for p in pair],
    })
    return eff, hist, fbs


def _upcoming_games(world):
    _, hist, fbs = world
    wk = hist[hist["gameday"] == (START + pd.Timedelta(days=7 * (WEEK - 1))).isoformat()]
    return [dict(r, home_score=None, away_score=None) for r in wk.to_dict("records")
            if r["home_team"] in fbs and r["away_team"] in fbs], fbs


@pytest.fixture
def week_matchups(world, monkeypatch):
    eff, hist, fbs = world
    monkeypatch.setattr(public_snapshot, "_matchup_inputs", lambda season: (eff, hist, fbs))
    games, _ = _upcoming_games(world)
    return games, public_snapshot._build_week_matchups(SEASON, games)


def test_snapshot_rows_are_fbs_only_ranked(week_matchups, world):
    games, stored = week_matchups
    fbs = world[2]
    assert stored, "expected at least one game with a >=15 rank-gap duel in real week-12 data"
    by_id = {str(g["game_id"]): g for g in games}
    for gid, rows in stored.items():
        assert by_id[gid]["home_team"] in fbs and by_id[gid]["away_team"] in fbs
        for r in rows:
            assert r["n_teams"] <= len(fbs) < 100 and 1 <= r["attacker_rank"] <= r["n_teams"]
            assert 1 <= r["defender_rank"] <= r["n_teams"]
            assert r["attacker"] in fbs and r["defender"] in fbs
    json.dumps(stored)  # snapshot must be JSON-serialisable


def test_fcs_opponent_gets_no_duel(world, monkeypatch):
    eff, hist, fbs = world
    fcs = "North Dakota State"
    fbs_team = sorted(fbs)[0]
    game = {"game_id": "x", "home_team": fbs_team, "away_team": fcs, "gameday": "2025-11-15T17:00:00+00:00",
            "home_score": None, "away_score": None}
    monkeypatch.setattr(public_snapshot, "_matchup_inputs", lambda season: (eff, hist, fbs))
    assert public_snapshot._build_week_matchups(SEASON, [game]) == {}


def test_started_games_get_no_stored_duels(world, monkeypatch):
    eff, hist, fbs = world
    monkeypatch.setattr(public_snapshot, "_matchup_inputs", lambda season: (eff, hist, fbs))
    games, _ = _upcoming_games(world)
    assert public_snapshot._build_week_matchups(SEASON, [dict(g, home_score=10, away_score=3) for g in games]) == {}


def test_failed_load_keeps_previous_week(monkeypatch):
    def boom(season):
        raise RuntimeError("cfbd down")
    monkeypatch.setattr(public_snapshot, "_matchup_inputs", boom)
    g = {"game_id": "1", "home_team": "A", "away_team": "B", "home_score": None, "away_score": None}
    assert public_snapshot._build_week_matchups(SEASON, [g], {"matchups": {"1": [{"id": "keep"}]}}) == {"1": [{"id": "keep"}]}


def test_facts_route_serves_stored_rows_neutral_with_no_network(week_matchups, monkeypatch):
    games, stored = week_matchups
    gid, rows = next(iter(stored.items()))
    game = next(g for g in games if str(g["game_id"]) == gid)
    snap_game = {**game, "season": 2026, "week": 1, "gameday": "2026-09-12T16:00:00+00:00"}
    snapshot = json.loads(json.dumps({
        "season": 2026, "current_week": 1,
        "weeks": {"1": {"games": [snap_game], "predictions": {gid: {"home_win_prob": 0.6, "away_win_prob": 0.4}},
                        "player_props": [], "matchups": {gid: rows}}},
    }))
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod.routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod, "_now", lambda: pd.Timestamp("2026-09-05T12:00:00Z").to_pydatetime())
    monkeypatch.setattr(facts_mod, "_public_snapshot", lambda: snapshot)
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda *a: [])
    monkeypatch.setattr(facts_mod.store, "get_track_record", lambda: {})

    def no_network(*a, **k):
        raise AssertionError("request path touched the network/loaders")
    monkeypatch.setattr(cfbd_advanced, "current_season_efficiency", no_network)
    monkeypatch.setattr(public_snapshot, "_matchup_inputs", no_network)

    body = TestClient(app).get(f"/facts/{gid}").json()
    got = body["context"]["matchups"]
    assert got and len(got) <= 4
    for r in got:
        assert set(r) == {"id", "attacker", "defender", "stat", "foil", "attacker_rank", "defender_rank",
                          "n_teams", "toward_pick"}
        assert r["toward_pick"] is None
    assert {r["id"] for r in got} <= {r["id"] for r in rows}
    print(json.dumps(got[0]))


def test_game_without_stored_duels_has_no_matchups_key(monkeypatch):
    g = {"game_id": "7", "season": 2026, "week": 1, "gameday": "2026-09-12T16:00:00+00:00", "home_team": "A",
         "away_team": "B", "home_score": None, "away_score": None}
    snapshot = {"season": 2026, "current_week": 1,
                "weeks": {"1": {"games": [g], "predictions": {"7": {"home_win_prob": 0.6, "away_win_prob": 0.4}},
                                "player_props": []}}}
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod.routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod, "_now", lambda: pd.Timestamp("2026-09-05T12:00:00Z").to_pydatetime())
    monkeypatch.setattr(facts_mod, "_public_snapshot", lambda: snapshot)
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda *a: [])
    monkeypatch.setattr(facts_mod.store, "get_track_record", lambda: {})
    assert "matchups" not in TestClient(app).get("/facts/7").json()["context"]


def test_current_season_pull_is_at_most_one_call_per_week(tmp_path):
    rows = json.loads(DATA.read_text())[:200]
    calls = []
    fetch = lambda season: calls.append(season) or rows  # noqa: E731
    cfbd_advanced.current_season_efficiency(2026, tmp_path, fetch=fetch)
    cfbd_advanced.current_season_efficiency(2026, tmp_path, fetch=fetch)  # fresh cache: no call
    assert calls == [2026]
    old = time.time() - 7 * 86400
    os.utime(tmp_path / "advanced_2026.json", (old, old))
    cfbd_advanced.current_season_efficiency(2026, tmp_path, fetch=fetch)  # stale: one more
    assert calls == [2026, 2026]


def test_failed_pull_falls_back_to_stale_cache(tmp_path):
    rows = json.loads(DATA.read_text())
    (tmp_path / "advanced_2026.json").write_text(json.dumps(rows))
    old = time.time() - 30 * 86400
    os.utime(tmp_path / "advanced_2026.json", (old, old))

    def boom(season):
        raise RuntimeError("quota")
    assert len(cfbd_advanced.current_season_efficiency(2026, tmp_path, fetch=boom)) > 0
