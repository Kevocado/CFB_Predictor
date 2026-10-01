import contextlib
import os
import sqlite3
import time

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
    store.record_game_predictions([_future_game()])  # props need their game's kickoff time
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
    store.record_game_predictions([_future_game()])  # props need their game's kickoff time
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
    store.record_game_predictions([_future_game()])  # props need their game's kickoff time
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
    store.record_game_predictions([_future_game()])  # props need their game's kickoff time
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
    store.record_game_predictions([_future_game()])  # props need their game's kickoff time
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


def test_track_record_counts_a_pick_backfilled_after_kickoff_and_says_so():
    """Picks backfilled after kickoff are counted AND disclosed (Kevin, 2026-10-01).

    This used to read the other way ("shown, never counted", PRODUCT.md). The rule
    changed: the model is rerun constantly, and a pick recorded after kickoff is a
    recorded pick, so it counts in the headline. What must not change is that it is
    never presented as a pre-game pick -- hence the label, not the exclusion.
    """
    store.record_game_predictions([_future_game(game_id="g1")])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 10, "away_score": 20}]))
    store.record_resolved_game_predictions([{
        "game_id": "g3", "home_team": "H", "away_team": "A", "commence_time": "2025-09-07T17:00:00+00:00",
        "home_win_prob": 0.4, "away_win_prob": 0.6, "actual_home_score": 10, "actual_away_score": 24,
    }])

    games = store.get_track_record()["games"]

    assert games["n_resolved"] == 2
    assert games["n_pre_kickoff"] == 1
    assert sorted(r["made_before_kickoff"] for r in games["per_pick"] if r["market"] == "moneyline") == [False, True]


def test_an_unreadable_snapshot_time_is_never_labelled_pre_kickoff():
    """When the timing can't be proven, the honest default is 'not before kickoff'."""
    assert store._snapshotted_after_kickoff("not a time", "2025-09-07T17:00:00+00:00") is True
    assert store._made_before_kickoff("not a time", "2025-09-07T17:00:00+00:00") is False
    assert store._made_before_kickoff("2025-09-07T17:00:00+00:00", None) is False


# --- a missing probability must not become a recorded verdict ------------------
#
# The recurring defect in this repo, and it appears here in its worst form because
# the output is the *track record* -- the page a reader uses to judge the model.
#
# `_compute_hits` guards on the LINE being present and then does
#
#     predicted_home_cover = (home_cover_prob or 0) >= (away_cover_prob or 0)
#
# The probabilities are never checked. So a game snapshotted with a spread but no
# cover probabilities -- which is what the odds feed produces whenever it has a
# spread but not a matching market -- is graded as though the model had called
# `0.0 >= 0.0`, i.e. home cover. Demonstrated both ways:
#
#     home 31-24, spread -3.5, cover probs None  -> ats_hit = 1  (fabricated HIT)
#     home 20-24, spread -3.5, cover probs None  -> ats_hit = 0  (fabricated MISS)
#
# Both are a claim about a prediction the model never made, and both are persisted.
# The existing `..._null_when_lines_were_never_recorded` test covers the opposite
# gap -- line missing, probabilities present -- which is why the suite was green.

def test_reconcile_leaves_ats_null_when_the_line_is_present_but_the_cover_probabilities_are_not():
    store.record_game_predictions([_future_game(
        home_spread_line=-3.5, home_cover_prob=None, away_cover_prob=None,
    )])
    results = pd.DataFrame([{"game_id": "g1", "home_score": 31, "away_score": 24}])

    store.reconcile_game_predictions(results)

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = 'g1'", conn).iloc[0]

    # home did cover, so a fabricated call would read as ats_hit == 1. There was no
    # call to grade, so the honest answer is no grade.
    assert pd.isna(row["ats_hit"]), (
        f"a missing cover probability must not be recorded as a verdict, got {row['ats_hit']}"
    )
    assert row["moneyline_hit"] == 1, "the moneyline call was real and is still gradable"


def test_reconcile_leaves_total_null_when_the_line_is_present_but_the_total_probabilities_are_not():
    store.record_game_predictions([_future_game(
        total_line=51.5, over_prob=None, under_prob=None,
    )])
    results = pd.DataFrame([{"game_id": "g1", "home_score": 31, "away_score": 24}])

    store.reconcile_game_predictions(results)

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = 'g1'", conn).iloc[0]

    assert pd.isna(row["total_hit"]), (
        f"a missing over/under probability must not be recorded as a verdict, got {row['total_hit']}"
    )
    assert row["ats_hit"] == 1, "the ATS call was real (probs present) and is unaffected"


def test_reconcile_does_not_grade_a_one_sided_market():
    """Only one of the pair present is not a usable market either.

    `home_cover_prob=0.6, away_cover_prob=None` is a half-populated market. It is
    not obviously home-cover, but it is not a graded call either, and the old
    expression turns it into a confident `0.6 >= 0` -> home cover.
    """
    store.record_game_predictions([_future_game(
        home_spread_line=-3.5, home_cover_prob=0.6, away_cover_prob=None,
    )])
    results = pd.DataFrame([{"game_id": "g1", "home_score": 31, "away_score": 24}])

    store.reconcile_game_predictions(results)

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = 'g1'", conn).iloc[0]

    assert pd.isna(row["ats_hit"])


def test_get_game_verdict_does_not_re_derive_a_prediction_from_missing_probabilities():
    """The read path, for a row an older build already wrote.

    `_compute_hits` no longer grades a market whose probabilities are missing, so
    new rows never reach this state. But the deployed database *already contains*
    them: every game snapshot by the old code with a spread line and no cover
    probabilities was written with a fabricated `ats_hit`, and those rows are
    permanent unless something backfills them.

    So `get_game_verdict` must refuse to re-derive a `predicted` side from a null
    probability. It used to compute `(None or 0) >= (None or 0)` -> `True` ->
    `"home_cover"`, which put a fabricated prediction next to a stored hit flag
    that the same function then reported as fact. `hit` is stored, `predicted` is
    recomputed, and a disagreement between them is a silent contradiction inside
    one verdict object.
    """
    store.record_game_predictions([_future_game(
        home_spread_line=-3.5, home_cover_prob=None, away_cover_prob=None,
    )])
    # Simulate the row the old build wrote: graded anyway, probabilities still null.
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            "UPDATE game_predictions SET resolved = 1, ats_hit = 1, moneyline_hit = 1, "
            "actual_home_score = 30, actual_away_score = 20 WHERE game_id = 'g1'"
        )

    verdict = store.get_game_verdict("g1")

    assert verdict is not None
    assert verdict["ats"] is None, (
        f"a null-probability row must not report a predicted side, got {verdict['ats']}"
    )
    # The market summary is dropped, not invented -- but the real facts survive.
    assert verdict["moneyline"]["hit"] is True
    assert verdict["actual_home_score"] == 30
    assert verdict["home_spread_line"] == -3.5


def test_get_game_verdict_keeps_reconciling_markets_when_probabilities_are_present():
    """The guard must not swallow a genuinely graded market."""
    store.record_game_predictions([_future_game()])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}]))

    verdict = store.get_game_verdict("g1")

    assert verdict["ats"] is not None
    assert verdict["ats"]["predicted"] == "home_cover"
    assert verdict["ats"]["hit"] is True
    assert verdict["totals"] is not None


def test_track_record_does_not_count_fabricated_ats_hits_from_the_old_build(cfb_store=None):
    """The aggregate reader, which the first fix left unguarded.

    `_compute_hits` no longer writes a hit flag for a market whose probabilities are
    missing, and `get_game_verdict` no longer reports one. But `_summarize_games`
    filtered on `notna()` alone, so every row the old build fabricated still counted
    towards `pct_ats_correct` — and that aggregate is the number the track-record
    page leads with.

    The result was the same game described two ways: the per-game verdict said "no
    ATS market", the track record said "100% ATS accuracy". Demonstrated before the
    fix:

        get_track_record()['games'] = {'n_resolved': 1, 'pct_ats_correct': 1.0, ...}
        get_game_verdict('g1')['ats'] = None
    """
    import contextlib

    import pandas as pd

    from cfb_predictor.tracking import store

    store.record_game_predictions([_future_game(
        home_spread_line=-3.5, home_cover_prob=None, away_cover_prob=None,
        total_line=51.5, over_prob=None, under_prob=None,
    )])
    # Simulate the rows the old build wrote: graded anyway, probabilities still null.
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            "UPDATE game_predictions SET resolved = 1, ats_hit = 1, total_hit = 1, "
            "moneyline_hit = 1, actual_home_score = 30, actual_away_score = 20 "
            "WHERE game_id = 'g1'"
        )

    summary = store._summarize_games(pd.read_sql("SELECT * FROM game_predictions", store._connect()))

    # The moneyline was a real call and stays counted.
    assert summary["n_resolved"] == 1
    assert summary["pct_moneyline_correct"] == 1.0
    # ATS and totals were never called, so there is no percentage to report.
    assert summary["pct_ats_correct"] is None, (
        f"a fabricated ATS hit is still in the headline: {summary['pct_ats_correct']}"
    )
    assert summary["pct_totals_correct"] is None, (
        f"a fabricated totals hit is still in the headline: {summary['pct_totals_correct']}"
    )


def test_a_half_present_market_is_not_counted_in_the_aggregate():
    """One probability present and one missing is not a call either.

    The existing fabricated-row test nulls BOTH probabilities, so a `_pair_present`
    narrowed to a single column still excludes that row and the test still passes.
    Verified: applying that narrowing to `store.py` leaves all 26 tests in this file
    green. This case is what kills it -- `home_cover_prob` is set, so a filter
    checking only that column lets the row through.

    `_present`'s own docstring names the case: "0.6 against None is not obviously
    the home side, but it is not a call either, and the same expression reports it
    as one."
    """
    import pandas as pd

    from cfb_predictor.tracking import store

    # A real graded game, and a HIT: home_cover 0.55 > 0.45, home won by 10 > 3.5.
    store.record_game_predictions([_future_game()])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}]))

    # A second game the old build graded against a half-present market.
    store.record_game_predictions([_future_game(
        game_id="g2", home_cover_prob=0.6, away_cover_prob=None,
    )])
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            "UPDATE game_predictions SET resolved = 1, ats_hit = 0, moneyline_hit = 1, "
            "actual_home_score = 20, actual_away_score = 24 WHERE game_id = 'g2'"
        )

    summary = store._summarize_games(pd.read_sql("SELECT * FROM game_predictions", store._connect()))

    # Only the genuine HIT counts. Including the fabricated MISS gives 0.5, so the
    # two cannot be confused.
    assert summary["pct_ats_correct"] == 1.0


def test_a_genuinely_graded_market_is_still_counted_in_the_aggregate():
    """The guard must not swallow real ATS and totals rows."""
    import pandas as pd

    from cfb_predictor.tracking import store

    store.record_game_predictions([_future_game()])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}]))

    summary = store._summarize_games(pd.read_sql("SELECT * FROM game_predictions", store._connect()))

    # ATS: `_future_game` has home_cover 0.55 > away 0.45 and home won by 10 > 3.5,
    # so the call was right.
    assert summary["pct_ats_correct"] == 1.0
    # Totals: over_prob == under_prob == 0.5, so `>=` calls over; the game totalled
    # 50 against a 51.5 line, so the call was wrong. `0.0` is the point -- the row is
    # *counted*, because it is a real graded market, unlike the fabricated ones.
    assert summary["pct_totals_correct"] == 0.0


# --- every recorded pick counts; the pre-kickoff figure sits beside it ----------
#
# Kevin, 2026-10-01, in `predictor-hub`
# docs/superpowers/specs/2026-10-01-track-record-counts-every-pick.md (merged as
# predictor-hub #66), verbatim:
#
#   "i dont really care about picks made after kickoff because im always re
#    running the models ... with every model change it will stop tracking ...
#    make it that whats recorded remains recorded and then just use every
#    prediction we make for the track record stuff."
#
# That reverses the exclusion rule CFB #26 added eight hours earlier, which
# dropped post-kickoff rows from `n_resolved` and reported them as `n_rebuilt`.
# What replaced it:
#
#   1. Recorded stays recorded. Append-only; a rerun never overwrites a pick.
#   2. One counted pick per (game, market): the EARLIEST recorded one. A later
#      rerun is kept as history but neither replaces nor double-counts -- otherwise
#      re-running until the model is right would be free.
#   3. The headline counts every counted pick, whenever it was made. The figure
#      beside it is the pre-kickoff SUBSET, with its own n.
#   4. Honesty is disclosure, not exclusion: every pick row carries
#      `made_before_kickoff`, derived from its own `snapshotted_at` against the
#      game's start as UTC INSTANTS, plus its timestamp. Never a stored flag, and
#      never a `true` the timestamps do not prove.
#
# So a post-kickoff pick is no longer hidden -- it is counted AND labelled. The
# load-bearing half of #26 that survives is `test_grading_never_restamps_the_
# snapshot_timestamp_the_guard_reads` below: if a reconcile rewrote `snapshotted_at`
# then both the earliest-pick rule and the disclosure label would be corruptible,
# because both are read off that column.


def _insert_game_row(game_id, commence_time):
    """Insert a game row directly.

    `record_game_predictions` refuses a game at/after its own kickoff, which is the
    correct rule for a live tick but makes the post-kickoff case unrepresentable
    through the public writer. The row still has to exist, because the guard reads
    kickoff times from this table.
    """
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO game_predictions
                (game_id, home_team, away_team, commence_time, snapshotted_at,
                 home_win_prob, away_win_prob)
            VALUES (?, 'H', 'A', ?, '2020-01-01T00:00:00+00:00', 0.5, 0.5)
            """,
            (game_id, commence_time),
        )


def _insert_prop_row(game_id, player_id, snapshotted_at, market="rushing_yards",
                     predicted=85.0, resolved=True, actual=None, position=None):
    """Insert a prop row directly, so `snapshotted_at` is the test's to choose.

    `record_player_prop_predictions` stamps `snapshotted_at` with `now()`, which can
    only ever produce a post-kickoff row against a past game. Writing the column is
    the only way to state the boundary the guard is about.
    """
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO player_prop_predictions
                (game_id, player_id, player_name, market, predicted_value, snapshotted_at,
                 resolved, actual_value, position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (game_id, player_id, f"Player {player_id}", market, predicted,
             snapshotted_at, 1 if resolved else 0, actual, position),
        )


def test_prop_headline_counts_every_recorded_pick_with_the_pre_kickoff_subset_beside_it():
    """Rule 3, on the yardage family: the headline is every counted pick.

    Two props resolve against the same yardage market, by different players in
    different games: one snapshotted before its game's kickoff (off by 10) and one
    snapshotted a year after its game's kickoff (off by 70). Under #26 the second
    was dropped and reported as `n_rebuilt`. Under the current rule it COUNTS --
    `n_resolved` 2 and the error is the mean of 10 and 70 -- because it is a
    recorded pick and the model was rerun on that game.

    The figure beside it is the pre-kickoff subset with its own n, which is the
    honest read of live performance: `pre_kickoff.rushing_yards` is the same
    pre-kickoff pick alone, error 10, and `n_pre_kickoff` is its count. Neither
    number replaces the other; the page shows both because they differ.
    """
    _insert_game_row("g_pre", "2099-09-04T20:20:00+00:00")
    _insert_game_row("g_post", "2000-09-04T20:20:00+00:00")
    _insert_prop_row("g_pre", "p_pre", "2099-09-01T00:00:00+00:00",
                     predicted=90.0, actual=80.0, position="RB")
    _insert_prop_row("g_post", "p_post_yards", "2001-01-01T00:00:00+00:00",
                     predicted=10.0, actual=80.0, position="RB")

    props = store.get_track_record()["player_props"]

    # The headline counts both recorded picks: (|90-80| + |10-80|) / 2 = 40.
    assert props["rushing_yards"]["n_resolved"] == 2, (
        "a recorded pick was dropped from the headline; the record is append-only"
    )
    assert props["rushing_yards"]["mean_absolute_error"] == pytest.approx(40.0)
    assert props["rushing_yards"]["mean_signed_error"] == pytest.approx(-30.0)
    # The secondary figure is the pre-kickoff subset alone, with its own n.
    assert props["n_pre_kickoff"] == 1
    assert props["pre_kickoff"]["rushing_yards"]["n_resolved"] == 1
    assert props["pre_kickoff"]["rushing_yards"]["mean_absolute_error"] == pytest.approx(10.0)
    # #26's exclusion key is gone; the count it used to carry is now the pre-kickoff n.
    assert "n_rebuilt" not in props


def test_prop_headline_counts_every_recorded_pick_in_the_brier_and_hit_rate():
    """The same swap on `anytime_td`, where the two numbers a reader quotes live.

    `hit_rate_when_called` and `brier_score` move together with the headline, and
    the pre-kickoff subset keeps #26's figures beside them: the pre-kickoff call
    at 0.8 scored (hit rate 1.0, Brier 0.04), the post-kickoff one at 0.9 missed,
    so the headline is 0.5 and 0.425 -- and the confidence buckets, built from the
    same frame, are counted over the headline.
    """
    _insert_game_row("g_pre", "2099-09-04T20:20:00+00:00")
    _insert_game_row("g_post", "2000-09-04T20:20:00+00:00")
    _insert_prop_row("g_pre", "td_pre", "2099-09-01T00:00:00+00:00",
                     market="anytime_td", predicted=0.8, actual=1.0)
    _insert_prop_row("g_post", "td_post", "2001-01-01T00:00:00+00:00",
                     market="anytime_td", predicted=0.9, actual=0.0)

    props = store.get_track_record()["player_props"]

    assert props["anytime_td"]["n_resolved"] == 2
    assert props["anytime_td"]["n_called"] == 2
    assert props["anytime_td"]["hit_rate_when_called"] == pytest.approx(0.5)
    assert props["anytime_td"]["brier_score"] == pytest.approx(0.425)
    assert sum(b["n"] for b in props["anytime_td"]["confidence_buckets"]) == 2
    assert props["n_pre_kickoff"] == 1
    assert props["pre_kickoff"]["anytime_td"]["n_resolved"] == 1
    assert props["pre_kickoff"]["anytime_td"]["hit_rate_when_called"] == pytest.approx(1.0)
    assert props["pre_kickoff"]["anytime_td"]["brier_score"] == pytest.approx(0.04)
    assert sum(b["n"] for b in props["pre_kickoff"]["anytime_td"]["confidence_buckets"]) == 1


def test_made_before_kickoff_is_the_boundary_and_the_label_never_flips_it():
    """The `>=` boundary, pinned on the label rather than on exclusion.

    A row stamped at the instant of kickoff is not made before kickoff; one stamped
    a second earlier is. Both COUNT in the headline now, so the boundary is only
    observable where the rule says it is: the `made_before_kickoff` label per pick,
    and membership of the pre-kickoff subset.
    """
    _insert_game_row("g_boundary", "2099-09-04T20:20:00+00:00")
    _insert_prop_row("g_boundary", "p_at", "2099-09-04T20:20:00+00:00", actual=80.0)
    _insert_prop_row("g_boundary", "p_before", "2099-09-04T20:19:59+00:00", actual=80.0)

    props = store.get_track_record()["player_props"]

    assert props["rushing_yards"]["n_resolved"] == 2, "both picks are recorded picks"
    labels = {r["player_id"]: r["made_before_kickoff"] for r in props["per_pick"]}
    assert labels == {"p_at": False, "p_before": True}, labels
    assert props["n_pre_kickoff"] == 1
    assert props["pre_kickoff"]["rushing_yards"]["n_resolved"] == 1


def test_a_pick_whose_timing_cannot_be_proven_counts_but_is_never_labelled_pre_kickoff():
    """Disclosure fails closed; exclusion no longer exists.

    An unparseable `snapshotted_at`, and a prop row whose game never reached
    `game_predictions` (so there is no kickoff time to compare against at all),
    are both recorded picks, so both count in the headline. Neither may be labelled
    `made_before_kickoff`: the timestamps do not prove it, and the rule is never to
    backfill a `true` they do not prove. So both are absent from the pre-kickoff
    subset -- `n_pre_kickoff` is 0 even though the headline is 2.
    """
    _insert_game_row("g_bad", "2099-09-04T20:20:00+00:00")
    _insert_prop_row("g_bad", "p_bad_time", "not a time", actual=80.0)
    _insert_prop_row("g_orphan", "p_orphan", "2020-01-01T00:00:00+00:00", actual=80.0)

    props = store.get_track_record()["player_props"]

    assert props["rushing_yards"]["n_resolved"] == 2
    assert props["rushing_yards"]["mean_absolute_error"] == pytest.approx(5.0)
    assert props["n_pre_kickoff"] == 0, (
        "a prop row whose timing cannot be proven was labelled made before kickoff"
    )
    assert props["pre_kickoff"]["rushing_yards"]["n_resolved"] == 0
    assert all(r["made_before_kickoff"] is False for r in props["per_pick"])


def test_every_counted_pick_carries_its_own_timestamp_and_the_pre_kickoff_label():
    """Rule 4, asserted on the published rows rather than on the aggregate.

    `made_before_kickoff` is derived, so the test asserts the two things that make
    it checkable: the row's own stamp is exposed next to the label, and the label
    agrees with that stamp read against the game's start. A `per_pick` list that
    carried the label without the timestamp could not be audited by a reader.
    """
    _insert_game_row("g_pre", "2099-09-04T20:20:00+00:00")
    _insert_game_row("g_post", "2000-09-04T20:20:00+00:00")
    _insert_prop_row("g_pre", "p_pre", "2099-09-01T00:00:00+00:00", actual=80.0)
    _insert_prop_row("g_post", "p_post", "2001-01-01T00:00:00+00:00", actual=80.0)

    rows = {r["player_id"]: r for r in store.get_track_record()["player_props"]["per_pick"]}

    assert set(rows) == {"p_pre", "p_post"}
    for row in rows.values():
        assert row["snapshotted_at"], "a pick row was published without its own timestamp"
        assert isinstance(row["made_before_kickoff"], bool)
        assert row["counted"] is True
    assert rows["p_pre"]["snapshotted_at"] == "2099-09-01T00:00:00+00:00"
    assert rows["p_pre"]["made_before_kickoff"] is True
    assert rows["p_post"]["made_before_kickoff"] is False


# --- rule 2: one counted pick per (game, market), the earliest one -------------
#
# `player_prop_predictions`' primary key is (game_id, player_id, market) and every
# write is `INSERT OR IGNORE`, so a rerun of the same player prop on the same game
# cannot land at all: the earliest recorded pick IS the row, and a later one is
# kept nowhere in the table. The two tests below prove that end to end, because
# "re-running until the model is right must be free" is a claim about the write
# path as much as the read side.
#
# Note the key is (game, PLAYER, market), not (game, market): a yardage pick is one
# pick per player per game, and collapsing a game's five rushers into one pick
# would delete the record rather than deduplicate it.


def test_the_counted_pick_is_the_earliest_recorded_one_not_a_rerun():
    """A rerun neither replaces the counted pick nor counts a second time.

    The same player prop is recorded twice with different predictions -- the model
    changed between the two runs. The headline must be the FIRST pick's number, off
    by 10, over `n_resolved` 1. Had the rerun been counted, re-running until the
    model was right would have been free, and the second run here is deliberately
    the better one (off by 0), so a double count would also be the flattering error.
    """
    _insert_game_row("g_rerun", "2099-09-04T20:20:00+00:00")
    first = store.record_player_prop_predictions([
        _prop(game_id="g_rerun", player_id="p_rerun", market="rushing_yards",
              predicted_value=90.0, player_name="Rerun", position="RB"),
    ])
    with contextlib.closing(store._connect()) as conn, conn:
        stamp = pd.read_sql(
            "SELECT snapshotted_at FROM player_prop_predictions WHERE player_id = 'p_rerun'", conn
        ).iloc[0]["snapshotted_at"]
    store.reconcile_player_prop_predictions(
        pd.DataFrame([{"game_id": "g_rerun", "player_id": "p_rerun", "rushing_yards": 80}])
    )
    rerun = store.record_player_prop_predictions([
        _prop(game_id="g_rerun", player_id="p_rerun", market="rushing_yards",
              predicted_value=80.0, player_name="Rerun", position="RB"),
    ])

    assert first == 1
    assert rerun == 0, "the rerun was admitted; it would have double-counted the pick"

    props = store.get_track_record()["player_props"]

    assert props["rushing_yards"]["n_resolved"] == 1
    assert props["rushing_yards"]["mean_absolute_error"] == pytest.approx(10.0)
    assert [r["snapshotted_at"] for r in props["per_pick"]] == [stamp]


def test_earliest_recorded_keeps_one_row_per_key_and_sorts_by_utc_instant():
    """Rule 2 as a unit, on a frame that CAN hold two rows per key.

    The write path cannot produce the duplicate today, so this drives the helper
    directly: three rows share one key, and the one whose stamp is the EARLIEST UTC
    instant wins -- including a case where the winner is not the first row in the
    frame, and a row whose stamp cannot be parsed, which loses to anything provable.
    A later rerun is dropped from the counted set, not merely deprioritised.
    """
    frame = pd.DataFrame([
        {"game_id": "g1", "player_id": "p1", "market": "rushing_yards",
         "snapshotted_at": "2026-09-12T15:00:00+00:00", "predicted_value": 2.0},
        # Same instant as the row above, expressed as +09:00: 2026-09-13T00:00+09:00
        # is 2026-09-12T15:00Z. Ties go to the first in the frame, stably.
        {"game_id": "g1", "player_id": "p1", "market": "rushing_yards",
         "snapshotted_at": "2026-09-13T00:00:00+09:00", "predicted_value": 3.0},
        # Earlier instant (2026-09-12T14:30Z) despite being listed last: this is
        # the counted pick, and a wall-clock sort would have ranked it 2nd.
        {"game_id": "g1", "player_id": "p1", "market": "rushing_yards",
         "snapshotted_at": "2026-09-12T23:30:00+09:00", "predicted_value": 1.0},
        # A different player on the same game and market is a DIFFERENT pick.
        {"game_id": "g1", "player_id": "p2", "market": "rushing_yards",
         "snapshotted_at": "2026-09-12T16:00:00+00:00", "predicted_value": 4.0},
        # Unparseable stamp: kept (the pick exists) but only if nothing proves earlier.
        {"game_id": "g2", "player_id": "p3", "market": "anytime_td",
         "snapshotted_at": "not a time", "predicted_value": 5.0},
        {"game_id": "g2", "player_id": "p3", "market": "anytime_td",
         "snapshotted_at": "2026-09-12T16:00:00+00:00", "predicted_value": 6.0},
    ])

    counted = store._earliest_recorded(frame, ("game_id", "player_id", "market"))

    # Three rows share (g1, p1, rushing_yards) and only the 14:30Z one counts, so the
    # list has one row per KEY, not one per row in the frame.
    assert list(counted["predicted_value"]) == [1.0, 4.0, 6.0], (
        "the counted pick is not the earliest recorded one per key"
    )


def test_the_games_half_also_counts_every_recorded_game_with_the_pre_kickoff_subset():
    """The swap on `games`, which is the headline the CFB page and facts.py read.

    Two resolved games: one snapshotted before kickoff and missed its moneyline
    call, one recorded after its kickoff (a backfill) and hit. The headline counts
    both -- 2 at 0.5 -- and `pre_kickoff` is the one made before kickoff, 1 at 0.0.
    """
    store.record_game_predictions([_future_game(game_id="g1")])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 10, "away_score": 20}]))
    store.record_resolved_game_predictions([{
        "game_id": "g3", "home_team": "H", "away_team": "A", "commence_time": "2025-09-07T17:00:00+00:00",
        "home_win_prob": 0.4, "away_win_prob": 0.6, "actual_home_score": 10, "actual_away_score": 24,
    }])

    games = store.get_track_record()["games"]

    assert games["n_resolved"] == 2
    assert games["pct_moneyline_correct"] == pytest.approx(0.5)
    assert games["n_pre_kickoff"] == 1
    assert games["pre_kickoff"]["n_resolved"] == 1
    assert games["pre_kickoff"]["pct_moneyline_correct"] == pytest.approx(0.0)
    assert "n_rebuilt" not in games
    moneyline = {r["game_id"]: r for r in games["per_pick"] if r["market"] == "moneyline"}
    assert moneyline["g1"]["made_before_kickoff"] is True
    assert moneyline["g3"]["made_before_kickoff"] is False
    assert moneyline["g1"]["snapshotted_at"] and moneyline["g3"]["snapshotted_at"]


# --- `made_before_kickoff` is derived from UTC instants, not wall clocks --------
#
# Both timestamps are stored as ISO strings and either may carry an offset (a
# commence_time taken from an upstream feed) or none (both are written in UTC).
# Comparing the two STRINGS -- or their naive wall-clock parts -- is the bug this
# derivation is most likely to ship, because a +09:00 feed reads a different
# instant on the same day than a +00:00 one. Each case below is chosen so that the
# wall-clock answer is the OPPOSITE of the instant answer.


def test_made_before_kickoff_is_derived_from_utc_instants_not_wall_clock_strings():
    """Two straddles where wall-clock comparison gives the opposite answer.

    Case A: stamped `2026-09-12T23:30:00+09:00` (14:30Z) against a
    `2026-09-12T15:00:00+00:00` kickoff. The instants say before kickoff, so the
    label is True and the pick is in the pre-kickoff subset. The wall clocks read
    "23:30" against "15:00", which a string or naive comparison calls AFTER.

    Case B: stamped `2026-09-12T23:30:00-11:00` (10:30Z) against a
    `2026-09-12T23:45:00+14:00` kickoff (09:45Z). The instants say after kickoff, so the
    label is False; the wall clocks read "23:30" against "23:45", which a naive
    comparison calls BEFORE. Both pairs are on the SAME calendar date in both notations,
    so nothing but the offset can explain the difference -- and +14:00 against -11:00 is
    the widest gap the zones allow, which is how this pair is forced rather than picked.
    """
    _insert_game_row("g_a", "2026-09-12T15:00:00+00:00")
    _insert_game_row("g_b", "2026-09-12T23:45:00+14:00")
    _insert_prop_row("g_a", "p_a", "2026-09-12T23:30:00+09:00", actual=80.0)
    _insert_prop_row("g_b", "p_b", "2026-09-12T23:30:00-11:00", actual=80.0)

    # The wall-clock reading, stated so the test cannot pass by accident on either side.
    assert "2026-09-12T23:30:00+09:00" > "2026-09-12T15:00:00+00:00"  # would say "after"
    assert "2026-09-12T23:30:00-11:00" < "2026-09-12T23:45:00+14:00"  # would say "before"

    props = store.get_track_record()["player_props"]
    labels = {r["player_id"]: r["made_before_kickoff"] for r in props["per_pick"]}

    assert labels == {"p_a": True, "p_b": False}, labels
    assert props["n_pre_kickoff"] == 1
    assert props["pre_kickoff"]["rushing_yards"]["n_resolved"] == 1
    assert props["rushing_yards"]["n_resolved"] == 2, "both are recorded picks either way"


def test_made_before_kickoff_ignores_the_machine_timezone():
    """The derivation reads the offsets in the data, never the host's clock.

    Same two instants, run under four host timezones (UTC, US Pacific, Japan, and
    Kiritimati at UTC+14). A derivation that dropped the offset and read the
    machine's zone would give different answers in different places; all four runs
    must agree.
    """
    stamp, kickoff = "2026-09-12T23:30:00+09:00", "2026-09-12T15:00:00+00:00"
    original = os.environ.get("TZ")
    try:
        answers = []
        for zone in ("UTC", "America/Los_Angeles", "Asia/Tokyo", "Pacific/Kiritimati"):
            os.environ["TZ"] = zone
            time.tzset()
            answers.append(store._made_before_kickoff(stamp, kickoff))
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()

    assert answers == [True, True, True, True], answers
    # And the two instants that are equal are not "before", whichever way they are written.
    assert store._made_before_kickoff(
        "2026-09-12T15:00:00+00:00", "2026-09-13T00:00:00+09:00"
    ) is False, "an instant equal to kickoff is not made before kickoff"


def test_a_naive_stamp_is_read_as_utc_which_is_how_it_is_written():
    """Both columns are written in UTC, so a stamp with no offset is a UTC instant.

    `record_game_predictions`/`record_player_prop_predictions` write
    `datetime.now(timezone.utc).isoformat()`, and a feed may hand back a kickoff with
    no offset. These are naive UTC, not naive local, so this pair straddles the
    boundary on the day and is labelled from the clock reading.
    """
    assert store._made_before_kickoff("2026-09-12T14:30:00", "2026-09-12T15:00:00") is True
    assert store._made_before_kickoff("2026-09-12T15:00:00", "2026-09-12T15:00:00") is False
    assert store._made_before_kickoff("2026-09-12T15:00:01", "2026-09-12T15:00:00") is False


# --- does the read side see each row's own provenance? -------------------------
#
# #26 asked whether the READ side alone is enough, given that
# `reconcile_player_prop_predictions` also grades post-kickoff rows. Under the
# exclusion rule the answer was "the record must be empty"; under this rule it is
# "the row must be counted AND labelled". The two tests below check the second
# half, and the load-bearing one -- the stamp is never restamped -- is unchanged.


def test_rows_the_write_path_admits_after_kickoff_are_labelled_not_hidden():
    """Proves (1) under the new rule: the label keys on PROVENANCE.

    This drives the real public writers -- `record_player_prop_predictions` and
    `reconcile_player_prop_predictions` -- so the rows are admitted and graded the
    same way production admits them, not hand-planted to suit the summariser. Both
    writers are demonstrably unguarded: the insert stamps `now()` and never looks at
    the kickoff, and the reconcile grades anything that joins to stats. The game
    kicked off in 2000, so `now()` is unambiguously after it.

    They count, and every one of them is labelled `made_before_kickoff: False` and
    kept out of the pre-kickoff subset. A record that counted them silently would
    be presenting a post-kickoff pick as a pre-game one, which is the one thing
    this rule still forbids.
    """
    _insert_game_row("g_done", "2000-09-04T20:20:00+00:00")

    admitted = store.record_player_prop_predictions([
        _prop(game_id="g_done", player_id="late_td", market="anytime_td", predicted_value=0.99),
        _prop(game_id="g_done", player_id="late_yds", market="rushing_yards", predicted_value=999.0),
    ])
    graded = store.reconcile_player_prop_predictions(pd.DataFrame([
        {"game_id": "g_done", "player_id": "late_td",
         "rushing_tds": 0, "receiving_tds": 0, "passing_tds": 0},
        {"game_id": "g_done", "player_id": "late_yds", "rushing_yards": 5},
    ]))
    # Both writers really did admit and grade post-kickoff rows. If a future change
    # makes them refuse, this assert fires and the test below must be re-reasoned --
    # it would then be guarding nothing.
    assert admitted == 2
    assert graded == 2

    props = store.get_track_record()["player_props"]

    assert props["anytime_td"]["n_resolved"] == 1
    assert props["anytime_td"]["hit_rate_when_called"] == pytest.approx(0.0)
    assert props["anytime_td"]["brier_score"] == pytest.approx(0.9801)
    assert props["rushing_yards"]["n_resolved"] == 1
    assert props["rushing_yards"]["mean_absolute_error"] == pytest.approx(994.0)
    # Counted, and accounted for as post-kickoff on every row that carries them.
    assert props["n_pre_kickoff"] == 0
    assert all(r["made_before_kickoff"] is False for r in props["per_pick"])
    assert props["pre_kickoff"]["anytime_td"]["n_resolved"] == 0
    assert props["pre_kickoff"]["rushing_yards"]["mean_absolute_error"] is None


def test_grading_never_restamps_the_snapshot_timestamp_the_guard_reads():
    """Proves (2), and it is the load-bearing half -- unchanged from #26, and now
    load-bearing for MORE.

    Under #26 this pinned the exclusion rule: the read side read pre-kickoff-ness
    off `snapshotted_at`, so a write path able to rewrite that column could put a
    look-forward pick back into the published record. Under the current rule both
    things the record now promises are read off that same column -- which pick
    counts (the earliest recorded one) and whether it is labelled
    `made_before_kickoff`. A restamp breaks both at once: it would let a rerun
    masquerade as the earliest pick, and it would backfill a `true` the timestamps
    do not prove, which is the one thing the spec still forbids outright.

    So the invariant is unchanged and the test is kept verbatim in substance: it
    asserts no particular metric, only that the stamp is byte-identical before and
    after `reconcile_player_prop_predictions` runs. Add `snapshotted_at = ?` to that
    UPDATE and this fails.
    """
    _insert_game_row("g_kept", "2000-09-04T20:20:00+00:00")
    store.record_player_prop_predictions([
        _prop(game_id="g_kept", player_id="p1", market="rushing_yards", predicted_value=90.0),
    ])

    def stamps():
        with contextlib.closing(store._connect()) as conn:
            rows = pd.read_sql(
                "SELECT player_id, snapshotted_at FROM player_prop_predictions ORDER BY player_id", conn
            )
        return dict(zip(rows["player_id"], rows["snapshotted_at"]))

    before = stamps()
    assert store.reconcile_player_prop_predictions(
        pd.DataFrame([{"game_id": "g_kept", "player_id": "p1", "rushing_yards": 80}])
    ) == 1

    assert stamps() == before, (
        "grading rewrote snapshotted_at, so a late row could be relabelled as a "
        "pick and the read-side guard would stop protecting the published record"
    )
    # And the record is still honest for the same reason: the row is counted as the
    # pick it was, and labelled with the truth about when it was made.
    props = store.get_track_record()["player_props"]
    assert props["n_pre_kickoff"] == 0
    assert [r["made_before_kickoff"] for r in props["per_pick"]] == [False]
