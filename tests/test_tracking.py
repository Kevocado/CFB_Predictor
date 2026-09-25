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


def test_reconcile_grades_ats_correctly_when_margin_is_smaller_than_the_spread():
    """Discriminating case for the ATS sign convention (this plan's final
    review, finding B1): home_spread_line means "home expected margin"
    (positive = home favored). Home favored by 6, final margin only 3 --
    home did NOT cover. The old buggy formula (`margin + spread > 0`, i.e.
    3 + 6 > 0) claimed home covered; the fixture in
    test_reconcile_grades_moneyline_ats_and_totals above can't catch this
    because it happens to agree under both conventions."""
    store.record_game_predictions([_future_game(
        home_spread_line=6.0, home_cover_prob=0.4087, away_cover_prob=0.5913,
    )])
    # home wins 24-21 -> home_margin = 3 < spread_line = 6 -> home did not cover.
    results = pd.DataFrame([{"game_id": "g1", "home_score": 24, "away_score": 21}])

    store.reconcile_game_predictions(results)

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = 'g1'", conn).iloc[0]

    # model predicted away to cover (away_cover_prob > home_cover_prob) and
    # home indeed did not cover -> the model's call was correct.
    assert row["ats_hit"] == 1


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
    assert verdict["actual_home_score"] == 30
    assert verdict["actual_away_score"] == 20
    assert verdict["home_spread_line"] == -3.5
    assert verdict["total_line"] == 51.5


def test_get_game_verdict_reports_null_lines_when_never_recorded():
    store.record_game_predictions([_future_game(home_spread_line=None, total_line=None)])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}]))

    verdict = store.get_game_verdict("g1")

    assert verdict["home_spread_line"] is None
    assert verdict["total_line"] is None
    assert verdict["actual_home_score"] == 30
    assert verdict["actual_away_score"] == 20


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


def _prop(game_id="g1", player_id="p1", player_name="Test Player", market="passing_yards",
          predicted_value=250.0, position="QB"):
    return {"game_id": game_id, "player_id": player_id, "player_name": player_name,
            "market": market, "predicted_value": predicted_value, "position": position}


def test_track_record_buckets_anytime_td_predictions_by_confidence():
    store.record_player_prop_predictions([
        _prop(market="anytime_td", predicted_value=0.55, player_id="p1"),
        _prop(market="anytime_td", predicted_value=0.65, player_id="p2"),
        _prop(market="anytime_td", predicted_value=0.75, player_id="p3"),
    ])
    stats = pd.DataFrame([
        {"game_id": "g1", "player_id": "p1", "rushing_tds": 1, "receiving_tds": 0, "passing_tds": 0},
        {"game_id": "g1", "player_id": "p2", "rushing_tds": 0, "receiving_tds": 0, "passing_tds": 0},
        {"game_id": "g1", "player_id": "p3", "rushing_tds": 0, "receiving_tds": 1, "passing_tds": 0},
    ])
    assert store.reconcile_player_prop_predictions(stats) == 3

    buckets = {
        b["label"]: b
        for b in store.get_track_record()["player_props"]["anytime_td"]["confidence_buckets"]
    }
    assert buckets["50-60%"]["n"] == 1
    assert buckets["50-60%"]["hit_rate"] == 1.0
    assert buckets["60-70%"]["n"] == 1
    assert buckets["60-70%"]["hit_rate"] == 0.0
    assert buckets["70%+"]["n"] == 1
    assert buckets["70%+"]["hit_rate"] == 1.0


def test_track_record_confidence_buckets_are_empty_when_no_td_predictions_resolved():
    record = store.get_track_record()

    buckets = record["player_props"]["anytime_td"]["confidence_buckets"]
    assert [b["label"] for b in buckets] == ["50-60%", "60-70%", "70%+"]
    assert all(b["n"] == 0 and b["hit_rate"] is None for b in buckets)


def test_track_record_signed_bias_is_positive_for_consistent_overprediction():
    store.record_player_prop_predictions([
        _prop(market="carries", predicted_value=25.0, player_id="p1", position="RB"),
        _prop(market="carries", predicted_value=30.0, player_id="p2", position="RB"),
    ])
    stats = pd.DataFrame([
        {"game_id": "g1", "player_id": "p1", "carries": 20},
        {"game_id": "g1", "player_id": "p2", "carries": 20},
    ])
    assert store.reconcile_player_prop_predictions(stats) == 2

    summary = store.get_track_record()["player_props"]["carries"]
    assert summary["n_resolved"] == 2
    # signed error = mean(predicted - actual): over-prediction is positive.
    assert summary["mean_signed_error"] == pytest.approx(7.5)
    assert summary["mean_absolute_error"] == pytest.approx(7.5)


def test_track_record_signed_bias_cancels_where_mae_does_not():
    store.record_player_prop_predictions([
        _prop(market="receiving_yards", predicted_value=110.0, player_id="p1", position="WR"),
        _prop(market="receiving_yards", predicted_value=90.0, player_id="p2", position="WR"),
    ])
    stats = pd.DataFrame([
        {"game_id": "g1", "player_id": "p1", "receiving_yards": 100},
        {"game_id": "g1", "player_id": "p2", "receiving_yards": 100},
    ])
    assert store.reconcile_player_prop_predictions(stats) == 2

    summary = store.get_track_record()["player_props"]["receiving_yards"]
    # +10 and -10 cancel in the signed mean but both count in the unsigned MAE.
    assert summary["mean_signed_error"] == pytest.approx(0.0)
    assert summary["mean_absolute_error"] == pytest.approx(10.0)


def test_track_record_reports_mae_by_position():
    store.record_player_prop_predictions([
        _prop(market="passing_yards", predicted_value=300.0, player_id="qb1", position="QB"),
        _prop(market="rushing_yards", predicted_value=100.0, player_id="rb1", position="RB"),
        _prop(market="receptions", predicted_value=6.0, player_id="wr1", position="WR"),
    ])
    stats = pd.DataFrame([
        {"game_id": "g1", "player_id": "qb1", "passing_yards": 280},
        {"game_id": "g1", "player_id": "rb1", "rushing_yards": 120},
        {"game_id": "g1", "player_id": "wr1", "receptions": 4},
    ])
    assert store.reconcile_player_prop_predictions(stats) == 3

    props = store.get_track_record()["player_props"]
    assert props["passing_yards"]["mae_by_position"] == {"QB": pytest.approx(20.0)}
    assert props["rushing_yards"]["mae_by_position"] == {"RB": pytest.approx(20.0)}
    assert props["receptions"]["mae_by_position"] == {"WR": pytest.approx(2.0)}


def test_track_record_handles_prop_predictions_recorded_without_position():
    """Rows recorded before the position column existed carry NULL position;
    the summary must not crash and groups them under "unknown"."""
    store.record_player_prop_predictions([
        {"game_id": "g1", "player_id": "p1", "player_name": "Old Row",
         "market": "rushing_yards", "predicted_value": 95.0},
    ])
    stats = pd.DataFrame([{"game_id": "g1", "player_id": "p1", "rushing_yards": 100}])
    assert store.reconcile_player_prop_predictions(stats) == 1

    summary = store.get_track_record()["player_props"]["rushing_yards"]
    assert summary["n_resolved"] == 1
    assert summary["mean_absolute_error"] == pytest.approx(5.0)
    assert summary["mae_by_position"] == {"unknown": pytest.approx(5.0)}


def test_get_predictions_for_week_marks_picks_rebuilt_after_kickoff():
    """A pick backfilled after the game is shown but is not a pre-kickoff call."""
    store.record_game_predictions([_future_game(game_id="g1")])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}]))
    store.record_resolved_game_predictions([{
        "game_id": "g3", "home_team": "Army", "away_team": "Navy", "commence_time": "2025-09-06T17:00:00+00:00",
        "home_win_prob": 0.4, "away_win_prob": 0.6, "actual_home_score": 10, "actual_away_score": 24,
    }])
    games_df = pd.DataFrame([
        {"game_id": "g1", "home_team": "Alabama", "away_team": "Georgia"},
        {"game_id": "g3", "home_team": "Army", "away_team": "Navy"},
    ])

    by_id = {row["game_id"]: row for row in store.get_predictions_for_week(2025, 1, games_df)}

    assert by_id["g1"]["rebuilt"] is False
    assert by_id["g3"]["rebuilt"] is True
