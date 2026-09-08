import contextlib
import sqlite3

import pandas as pd
import pytest

from cfb_predictor import config
from cfb_predictor.tracking import store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")


def _future_game(**overrides):
    game = {
        "game_id": "g1", "home_team": "Texas", "away_team": "Oklahoma",
        "commence_time": "2099-01-01T00:00:00Z",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": 0.55, "away_cover_prob": 0.45,
        "over_prob": 0.5, "under_prob": 0.5,
        "home_spread_line": -3.5, "total_line": 51.5,
        "season": 2025,
    }
    game.update(overrides)
    return game


def test_record_game_predictions_persists_spread_and_total_lines():
    store.record_game_predictions([_future_game()])

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = 'g1'", conn).iloc[0]

    assert row["home_spread_line"] == -3.5
    assert row["total_line"] == 51.5


def test_reconcile_grades_moneyline_ats_and_totals():
    store.record_game_predictions([_future_game()])
    # home favored by -3.5 and predicted to cover (home_cover_prob=0.55 > away);
    # over predicted (over_prob=0.5 == under_prob=0.5, home_win predicted).
    results = pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}])

    resolved = store.reconcile_game_predictions(results)

    assert resolved == 1
    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = 'g1'", conn).iloc[0]

    assert row["moneyline_hit"] == 1  # home won, home was favored
    # home won by 10, spread was -3.5 => home covered; predicted_home_cover=True
    assert row["ats_hit"] == 1
    # total = 50, line = 51.5 => actual under; predicted was a coin flip (over_prob==under_prob)
    # tie-break must not crash -- assert it resolved to 0 or 1, not None
    assert row["total_hit"] in (0, 1)


def test_reconcile_leaves_ats_and_total_hit_null_when_lines_were_never_recorded():
    store.record_game_predictions([_future_game(home_spread_line=None, total_line=None)])
    results = pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}])

    store.reconcile_game_predictions(results)

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = 'g1'", conn).iloc[0]

    assert row["moneyline_hit"] == 1
    assert pd.isna(row["ats_hit"])
    assert pd.isna(row["total_hit"])


def test_reconcile_catches_a_game_missed_by_a_prior_tick():
    """Simulates a deploy/restart: the game was snapshotted, its results
    became available, but no tick ran to reconcile it until now."""
    store.record_game_predictions([_future_game(game_id="g1", commence_time="2099-01-01T00:00:00Z")])
    results = pd.DataFrame([{"game_id": "g1", "home_score": 21, "away_score": 14}])

    resolved = store.reconcile_game_predictions(results)

    assert resolved == 1


def test_backfill_catches_a_prior_season_row_current_season_partial_would_miss(monkeypatch):
    # commence_time must still be pre-kickoff for the snapshot to be recorded
    # at all (record_game_predictions silently drops already-kicked-off games);
    # `season` is what marks this as a prior-season row current_season_partial
    # would never look at again.
    store.record_game_predictions([_future_game(game_id="g_old", season=2024, commence_time="2099-01-01T00:00:00Z")])
    from cfb_predictor.data import games as games_data
    monkeypatch.setattr(
        games_data, "load_training_data",
        lambda seasons: pd.DataFrame([{"game_id": "g_old", "home_score": 10, "away_score": 24}]),
    )

    resolved = store.backfill_unresolved_games(games_data)

    assert resolved == 1


def test_get_game_verdict_returns_none_for_unresolved_game():
    store.record_game_predictions([_future_game()])

    assert store.get_game_verdict("g1") is None


def test_get_game_verdict_summarizes_all_three_markets():
    store.record_game_predictions([_future_game()])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}]))

    verdict = store.get_game_verdict("g1")

    assert verdict["resolved"] is True
    assert verdict["moneyline"]["predicted"] == "home_win"
    assert verdict["moneyline"]["actual"] == "home_win"
    assert verdict["moneyline"]["hit"] is True
    assert verdict["ats"]["predicted"] == "home_cover"
    assert verdict["totals"] is not None


def test_get_predictions_for_week_returns_pending_for_unresolved_and_verdict_for_resolved():
    store.record_game_predictions([_future_game(game_id="g1"), _future_game(game_id="g2", home_team="Alabama", away_team="Auburn")])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}]))
    games_df = pd.DataFrame([
        {"game_id": "g1", "home_team": "Texas", "away_team": "Oklahoma"},
        {"game_id": "g2", "home_team": "Alabama", "away_team": "Auburn"},
    ])

    week = store.get_predictions_for_week(2026, 1, games_df)

    by_id = {row["game_id"]: row for row in week}
    assert by_id["g1"]["status"] == "resolved"
    assert by_id["g1"]["verdict"]["moneyline"]["hit"] is True
    assert by_id["g2"]["status"] == "pending"
    assert by_id["g2"]["verdict"] is None
