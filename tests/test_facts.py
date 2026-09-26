"""Tests for the read-only CFB /facts bundle the match explainer consumes.

Offline throughout: the snapshot, the schedule and the tracking store are
all injected, so nothing reaches CFBD, the odds feed or the model files.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, root_validator

from cfb_predictor.api import facts as facts_mod
from cfb_predictor.api.main import app

# The contract, copied from predictor-hub/services/explainer/explainer/facts.py.
# This repo runs pydantic v1, so the same fields and the same rebuilt/pick_won
# rule are expressed with v1's root_validator rather than v2's model_validator.
class Market(BaseModel):
    market: str

    class Config:
        extra = "allow"


class Facts(BaseModel):
    sport: Literal["pl", "f1", "nfl", "cfb", "nba"]
    id: str
    title: str
    starts_at: str
    status: Literal["upcoming", "live", "final"]
    pick_timing: Literal["pre_kickoff", "rebuilt", "none"]
    pick: dict | None = None
    markets: list[Market] = []
    drivers: list[dict] = []
    context: dict = {}
    players: list[dict] = []
    record: dict | None = None
    result: dict | None = None

    @root_validator
    def _rebuilt_never_won(cls, values):  # noqa: N805
        if values.get("pick_timing") == "rebuilt" and values.get("result") and "pick_won" in values["result"]:
            raise ValueError("a rebuilt pick cannot carry result.pick_won")
        return values


NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
GAME_ID = "401856766"


def _game(**over):
    game = {
        "game_id": GAME_ID,
        "season": 2026,
        "week": 1,
        "gameday": "2026-09-12T16:00:00+00:00",  # CFBD carries an explicit offset
        "home_team": "TCU",
        "away_team": "North Carolina",
        "home_score": None,
        "away_score": None,
        "home_conference": "Big 12",
        "away_conference": "ACC",
        "home_division": "fbs",
        "away_division": "fbs",
        "conference_game": False,
        "neutral_site": False,
    }
    game.update(over)
    return game


def _prediction(**over):
    pred = {
        "home_win_prob": 0.67,
        "away_win_prob": 0.33,
        "predicted_margin": 14.2,
        "predicted_total": 55.5,
    }
    pred.update(over)
    return pred


def _prop(suffix, team, position, **yards):
    prop = {
        "player_id": f"-{suffix}",
        "player_name": f"Player {suffix}",
        "recent_team": team,
        "position": position,
        "anytime_td_prob": 0.5,
    }
    prop.update(yards)
    return prop


def _snapshot(game=None, prediction=None, props=None, games=None):
    return {
        "generated_at": "2026-09-05T06:00:00Z",
        "season": 2026,
        "current_week": 1,
        "weeks": {
            "1": {
                "games": games if games is not None else [game if game is not None else _game()],
                "predictions": {GAME_ID: prediction if prediction is not None else _prediction()},
                "player_props": props if props is not None else [
                    _prop("1", "TCU", "QB", passing_yards=312.0),
                    _prop("2", "TCU", "WR", receiving_yards=88.0),
                    _prop("3", "North Carolina", "RB", rushing_yards=120.0),
                    _prop("4", "North Carolina", "WR", receiving_yards=55.0),
                ],
            }
        },
    }


def _week_rows(**over):
    row = {
        "game_id": GAME_ID,
        "status": "pending",
        "rebuilt": False,
        "home_win_prob": 0.67,
        "away_win_prob": 0.33,
        "verdict": None,
    }
    row.update(over)
    return [row]


@pytest.fixture
def public(monkeypatch):
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod.routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod, "_now", lambda: NOW)
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda season, week, games_df: _week_rows())
    monkeypatch.setattr(
        facts_mod.store, "get_track_record",
        lambda: {"games": {"n_resolved": 30, "pct_moneyline_correct": 0.6, "n_rebuilt": 2}},
    )
    return TestClient(app)


def _install_snapshot(monkeypatch, snapshot):
    monkeypatch.setattr(facts_mod, "_public_snapshot", lambda: snapshot)


# --- the contract -------------------------------------------------------

def test_public_mode_bundle_validates_against_the_contract(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    Facts(**body)
    assert body["sport"] == "cfb"
    assert body["id"] == GAME_ID
    assert body["title"] == "North Carolina at TCU"
    assert body["starts_at"] == "2026-09-12T16:00:00Z"
    assert body["status"] == "upcoming"
    assert body["pick"] == {"label": "TCU", "prob": 0.67}


def test_context_names_the_conferences_and_has_no_injuries(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["context"]["conferences"] == "ACC at Big 12"
    # CFB has no injuries source at all; nothing may be invented here.
    assert "injuries" not in body["context"]


def test_only_the_pick_market_when_no_lines_exist(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    assert [m["market"] for m in body["markets"]] == ["moneyline"]
    assert body["markets"][0]["model"] == {"TCU": pytest.approx(0.67), "North Carolina": pytest.approx(0.33)}


def test_spread_and_total_appear_only_where_the_lines_exist(public, monkeypatch):
    lined = _game(spread_line=10.5, total_line=58.5)
    _install_snapshot(monkeypatch, _snapshot(game=lined))

    body = public.get(f"/facts/{GAME_ID}").json()
    by_market = {m["market"]: m for m in body["markets"]}

    assert set(by_market) == {"moneyline", "spread", "total"}
    # nflverse-style convention (positive = home favoured), worded for the site.
    assert by_market["spread"]["line"] == "TCU -10.5"
    assert by_market["spread"]["model_margin"] == pytest.approx(14.2)
    assert by_market["total"]["line"] == pytest.approx(58.5)
    assert by_market["total"]["model_total"] == pytest.approx(55.5)


def test_a_total_line_without_a_spread_line_keeps_the_total_market(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot(game=_game(total_line=58.5)))

    body = public.get(f"/facts/{GAME_ID}").json()

    assert [m["market"] for m in body["markets"]] == ["moneyline", "total"]


def test_players_are_the_top_three_by_projected_yards(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    # Ranked by projected yards: 312 (Player 1), 120 (Player 3), 88 (Player 2).
    assert [p["name"] for p in body["players"]] == ["Player 1", "Player 3", "Player 2"]
    assert {p["team"] for p in body["players"]} == {"TCU", "North Carolina"}


def test_record_reports_pre_kickoff_hits_over_settled(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["record"] == {"label": "Picks made before kickoff", "hits": 18, "settled": 30}


# --- pick_timing: the three cases ---------------------------------------

def test_pick_timing_is_none_when_no_stored_row(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda season, week, games_df: [])

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["pick_timing"] == "none"
    assert body["pick"] is None


def test_pick_timing_is_rebuilt_when_snapshotted_after_kickoff(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())
    monkeypatch.setattr(
        facts_mod.store, "get_predictions_for_week",
        lambda season, week, games_df: _week_rows(rebuilt=True),
    )

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["pick_timing"] == "rebuilt"
    assert body["pick"] is not None


def test_pick_timing_is_pre_kickoff_for_a_snapshotted_row(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["pick_timing"] == "pre_kickoff"
    assert body["pick"]["label"] == "TCU"


# --- started games: only the stored pre-start pick -----------------------

def test_started_game_uses_the_stored_pre_start_pick(public, monkeypatch):
    started = _game(gameday="2026-08-29T16:00:00+00:00")
    # The public snapshot rebuilds recent rounds after kickoff, so for a
    # started game its prediction is today's model, recomputed after kickoff.
    # Only the tracking row holds the pick made before kickoff.
    _install_snapshot(monkeypatch, _snapshot(game=started, prediction=_prediction(home_win_prob=0.81, away_win_prob=0.19)))

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "live"
    assert body["pick"] == {"label": "TCU", "prob": 0.67}  # the stored row, not the rebuilt snapshot
    # The row carries no margin/total, so no post-kickoff spread or total is quoted.
    assert {m["market"] for m in body["markets"]} <= {"moneyline"}
    assert "0.81" not in str(body["markets"])


def test_final_judges_the_stored_pick_even_when_the_rebuilt_snapshot_flipped(public, monkeypatch):
    final = _game(gameday="2026-08-29T16:00:00+00:00", home_score=10, away_score=34)
    # After kickoff the snapshot rebuilt North Carolina as favourite; before
    # kickoff the stored pick was TCU.
    _install_snapshot(monkeypatch, _snapshot(game=final, prediction=_prediction(home_win_prob=0.40, away_win_prob=0.60)))
    monkeypatch.setattr(
        facts_mod.store, "get_predictions_for_week",
        lambda season, week, games_df: _week_rows(status="resolved"),
    )

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["pick"] == {"label": "TCU", "prob": 0.67}
    assert body["result"]["pick_won"] is False  # TCU lost 10-34
    Facts(**body)

    # Nothing was stored before kickoff: no tracking row means no pick, and
    # the snapshot's post-kickoff rebuild must not stand in for one.
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda season, week, games_df: [])

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["pick"] is None
    assert body["pick_timing"] == "none"
    assert "pick_won" not in body["result"]


def test_started_game_without_a_stored_pick_has_no_pick(public, monkeypatch):
    started = _game(gameday="2026-08-29T16:00:00+00:00")
    snap = _snapshot(game=started)
    snap["weeks"]["1"]["predictions"] = {}
    _install_snapshot(monkeypatch, snap)
    # Nothing was stored before kickoff: no tracking row, and the snapshot's
    # own prediction is a post-kickoff rebuild that must not stand in for one.
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda season, week, games_df: [])

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "live"
    assert body["pick"] is None
    assert body["pick_timing"] == "none"
    assert body["markets"] == []


def test_public_mode_never_computes_a_live_model(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    def explode(*args, **kwargs):
        raise AssertionError("public mode must not compute a live model")

    monkeypatch.setattr(facts_mod.routes, "_get_game_prediction_live", explode)
    monkeypatch.setattr(facts_mod.routes, "_load_models_cached", explode)
    monkeypatch.setattr(facts_mod.routes.feature_build, "build_features_for_game", explode)

    assert public.get(f"/facts/{GAME_ID}").status_code == 200


# --- live (non-public) mode: the rule that actually bites ---------------

@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", False)
    monkeypatch.setattr(facts_mod, "_now", lambda: NOW)
    monkeypatch.setattr(
        facts_mod.routes.games_data, "fetch_schedules",
        lambda seasons, force_refresh=False: pd.DataFrame([_game(gameday="2026-08-29T16:00:00+00:00")]),
    )
    monkeypatch.setattr(facts_mod.routes, "get_player_props", lambda season, week: [])
    monkeypatch.setattr(
        facts_mod.store, "get_track_record",
        lambda: {"games": {"n_resolved": 12, "pct_moneyline_correct": 0.58, "n_rebuilt": 1}},
    )
    return TestClient(app)


def test_live_started_game_never_computes_a_model_and_uses_the_stored_row(live, monkeypatch):
    monkeypatch.setattr(facts_mod.routes, "get_games", lambda season, week: [
        _game(gameday="2026-08-29T16:00:00+00:00"),
    ])
    monkeypatch.setattr(
        facts_mod.store, "get_predictions_for_week",
        lambda season, week, games_df: _week_rows(home_win_prob=0.77, away_win_prob=0.23),
    )

    def explode(*args, **kwargs):
        raise AssertionError("a started game must never be described by today's model")

    monkeypatch.setattr(facts_mod.routes, "get_game_prediction", explode)
    monkeypatch.setattr(facts_mod.routes, "_get_game_prediction_live", explode)
    monkeypatch.setattr(facts_mod.routes, "_load_models_cached", explode)

    body = live.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "live"
    assert body["pick"] == {"label": "TCU", "prob": 0.77}
    assert [m["market"] for m in body["markets"]] == ["moneyline"]


def test_live_upcoming_game_does_use_the_current_model(live, monkeypatch):
    monkeypatch.setattr(facts_mod.routes.games_data, "fetch_schedules", lambda seasons, force_refresh=False: pd.DataFrame([_game()]))
    monkeypatch.setattr(facts_mod.routes, "get_games", lambda season, week: [_game()])
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda season, week, games_df: _week_rows())
    monkeypatch.setattr(
        facts_mod.routes, "get_game_prediction",
        lambda season, week, game_id: _prediction(home_win_prob=0.69, away_win_prob=0.31),
    )

    body = live.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "upcoming"
    assert body["pick"] == {"label": "TCU", "prob": 0.69}


# --- finals -------------------------------------------------------------

def test_final_has_a_score_and_pick_won_when_pre_kickoff(public, monkeypatch):
    final = _game(gameday="2026-08-29T16:00:00+00:00", home_score=34, away_score=10)
    _install_snapshot(monkeypatch, _snapshot(game=final))
    monkeypatch.setattr(
        facts_mod.store, "get_predictions_for_week",
        lambda season, week, games_df: _week_rows(status="resolved"),
    )

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "final"
    assert body["result"]["score"] == "TCU 34-10"
    assert body["result"]["pick_won"] is True
    Facts(**body)


def test_final_omits_pick_won_for_a_rebuilt_pick(public, monkeypatch):
    final = _game(gameday="2026-08-29T16:00:00+00:00", home_score=10, away_score=34)
    _install_snapshot(monkeypatch, _snapshot(game=final))
    monkeypatch.setattr(
        facts_mod.store, "get_predictions_for_week",
        lambda season, week, games_df: _week_rows(status="resolved", rebuilt=True),
    )

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["pick_timing"] == "rebuilt"
    assert "score" in body["result"]
    assert "pick_won" not in body["result"]


def test_upcoming_game_has_no_result(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["result"] is None


# --- /facts/upcoming ----------------------------------------------------

def test_upcoming_lists_only_games_inside_the_window(public, monkeypatch):
    soon = _game(game_id="401856766", gameday=(NOW + timedelta(hours=10)).isoformat())
    later = _game(game_id="401856767", gameday=(NOW + timedelta(hours=100)).isoformat())
    past = _game(game_id="401856768", gameday=(NOW - timedelta(hours=10)).isoformat())
    _install_snapshot(monkeypatch, _snapshot(games=[soon, later, past]))

    body = public.get("/facts/upcoming?hours=72").json()

    assert body["ids"] == [GAME_ID]


def test_unknown_game_id_is_404(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    assert public.get("/facts/999999999").status_code == 404
