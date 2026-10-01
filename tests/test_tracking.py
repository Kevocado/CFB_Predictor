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


def test_track_record_leaves_rebuilt_picks_out_of_every_rate():
    """Picks backfilled after kickoff are shown, never counted (PRODUCT.md)."""
    store.record_game_predictions([_future_game(game_id="g1")])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 10, "away_score": 20}]))
    store.record_resolved_game_predictions([{
        "game_id": "g3", "home_team": "H", "away_team": "A", "commence_time": "2025-09-07T17:00:00+00:00",
        "home_win_prob": 0.4, "away_win_prob": 0.6, "actual_home_score": 10, "actual_away_score": 24,
    }])

    games = store.get_track_record()["games"]

    assert games["n_resolved"] == 1
    assert games["n_rebuilt"] == 1


def test_an_unreadable_snapshot_time_counts_as_rebuilt():
    """When the timing can't be proven, the honest default is 'rebuilt'."""
    assert store._snapshotted_after_kickoff("not a time", "2025-09-07T17:00:00+00:00") is True


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


# --- the pre-kickoff guard on the prop path -----------------------------------
#
# The games half of this guard already exists here: `get_track_record` applies
# `_snapshotted_after_kickoff` to `game_predictions` and reports the excluded rows
# apart under `games.n_rebuilt`, and `get_predictions_for_week` marks them
# `rebuilt` (test_track_record_leaves_rebuilt_picks_out_of_every_rate).
#
# The prop half did not. `_summarize_player_props` counted EVERY resolved row, so
# `player_props.n_resolved`, `hit_rate_when_called` and `brier_score` -- the props
# half of the published track record -- were computed partly from picks the model
# made after the game started. A hit rate built on information the model could not
# have had is exactly the look-forward bias this product exists to measure against,
# and NFL_Predictor's `src/nfl_predictor/tracking/store.py` `_summarize_player_props`
# already refuses those rows and reports them under its own `n_rebuilt`.
#
# `player_prop_predictions` carries no `commence_time`, so the guard joins to
# `game_predictions` on `game_id` (that table's primary key, so the join is
# many-to-one and safe) and fails closed -- an unparseable timestamp, or a prop row
# whose game is absent, is treated as rebuilt rather than counted.


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


def test_prop_track_record_excludes_post_kickoff_picks_from_every_market():
    """The defect, stated as an assertion.

    Two props resolve against the same yardage market: one snapshotted before its
    game's kickoff (a real pick, off by 10) and one snapshotted a year after it (a
    reconstruction, off by 70). The post-kickoff row is exactly the information the
    model could not have had when the line was live.

    Before the guard: `n_resolved` is 2 and the error is the mean of 10 and 70 --
    a published accuracy number computed partly from a look-forward row. After it:
    only the pre-kickoff pick counts, and the excluded row is reported apart under
    `n_rebuilt` rather than silently dropped.
    """
    _insert_game_row("g_pre", "2099-09-04T20:20:00+00:00")
    _insert_game_row("g_post", "2000-09-04T20:20:00+00:00")
    _insert_prop_row("g_pre", "p_pre", "2099-09-01T00:00:00+00:00",
                     predicted=90.0, actual=80.0, position="RB")
    _insert_prop_row("g_post", "p_post_yards", "2001-01-01T00:00:00+00:00",
                     predicted=10.0, actual=80.0, position="RB")

    props = store.get_track_record()["player_props"]

    assert props["rushing_yards"]["n_resolved"] == 1, (
        "a pick snapshotted after kickoff is still in the published record"
    )
    # |90-80| = 10. Leaking the post-kickoff row in gives (10 + 70) / 2 = 40.
    assert props["rushing_yards"]["mean_absolute_error"] == pytest.approx(10.0)
    assert props["rushing_yards"]["mean_signed_error"] == pytest.approx(10.0)
    # Reported apart rather than silently dropped, matching games.n_rebuilt.
    assert props["n_rebuilt"] == 1, (
        f"the post-kickoff prop row was not reported apart: {props['n_rebuilt']}"
    )


def test_prop_track_record_excludes_post_kickoff_picks_from_brier_and_hit_rate():
    """The two metrics the reviewer named, and the two a yardage MAE cannot reach.

    `anytime_td` is scored, not yardage: `hit_rate_when_called` and `brier_score`
    are the numbers a reader is most likely to quote. A post-kickoff anytime-TD row
    is planted alongside a pre-kickoff one that scored, and both must be excluded
    from the scoring pair while still being counted in `n_rebuilt` -- a guard that
    filtered only the yardage family, or only `n_resolved`, would leave these wrong.
    """
    _insert_game_row("g_pre", "2099-09-04T20:20:00+00:00")
    _insert_game_row("g_post", "2000-09-04T20:20:00+00:00")
    _insert_prop_row("g_pre", "td_pre", "2099-09-01T00:00:00+00:00",
                     market="anytime_td", predicted=0.8, actual=1.0)
    _insert_prop_row("g_post", "td_post", "2001-01-01T00:00:00+00:00",
                     market="anytime_td", predicted=0.9, actual=0.0)

    props = store.get_track_record()["player_props"]

    assert props["anytime_td"]["n_resolved"] == 1
    assert props["anytime_td"]["n_called"] == 1
    # Only the pre-kickoff row, which was called at 0.8 and scored: 1.0.
    # Counting the post-kickoff miss as well gives 0.5.
    assert props["anytime_td"]["hit_rate_when_called"] == pytest.approx(1.0)
    # (0.8 - 1.0)^2 = 0.04. With the post-kickoff row the mean is
    # (0.04 + 0.81) / 2 = 0.425 -- a materially different published calibration.
    assert props["anytime_td"]["brier_score"] == pytest.approx(0.04)
    # The confidence buckets are built from the same filtered frame, so they move too.
    assert sum(b["n"] for b in props["anytime_td"]["confidence_buckets"]) == 1
    assert props["n_rebuilt"] == 1


def test_prop_track_record_excludes_a_pick_snapshotted_exactly_at_kickoff():
    """The boundary is the games path's `>=`, pinned so it cannot drift.

    A row stamped at the instant of kickoff is a reconstruction, not a pick; one
    stamped a second earlier is a pick. A test that only asserted "some post-kickoff
    row is excluded" would pass against an off-by-one guard in either direction.
    """
    _insert_game_row("g_boundary", "2099-09-04T20:20:00+00:00")
    _insert_prop_row("g_boundary", "p_at", "2099-09-04T20:20:00+00:00", actual=80.0)
    _insert_prop_row("g_boundary", "p_before", "2099-09-04T20:19:59+00:00", actual=80.0)

    props = store.get_track_record()["player_props"]

    assert props["rushing_yards"]["n_resolved"] == 1
    assert props["n_rebuilt"] == 1


def test_prop_track_record_fails_closed_when_the_kickoff_time_cannot_be_read():
    """Same rule as `test_an_unreadable_snapshot_time_counts_as_rebuilt`.

    When the timing cannot be proven, the honest default is "after kickoff": an
    unparseable `snapshotted_at`, and a prop row whose game never reached
    `game_predictions` (so there is no kickoff time to compare against at all).
    Both are excluded rather than counted on the strength of a timestamp alone.
    """
    _insert_game_row("g_bad", "2099-09-04T20:20:00+00:00")
    _insert_prop_row("g_bad", "p_bad_time", "not a time", actual=80.0)
    _insert_prop_row("g_orphan", "p_orphan", "2020-01-01T00:00:00+00:00", actual=80.0)

    props = store.get_track_record()["player_props"]

    assert props["rushing_yards"]["n_resolved"] == 0
    assert props["rushing_yards"]["mean_absolute_error"] is None
    assert props["n_rebuilt"] == 2, (
        "a prop row that cannot be proven pre-kickoff was counted as a pick"
    )


# --- is the READ-side guard sufficient on its own? ------------------------------
#
# NFL's PR #21 guarded two sites: `_summarize_player_props` and
# `reconcile_player_prop_predictions`. Only the first is ported here, and the
# honest reason is not "out of scope" -- it is that the second cannot reach the
# published record. These two tests are the proof, so that reason is checked
# rather than assumed.
#
# The invariant, stated exactly:
#
#   Every row that reaches `n_resolved`, `hit_rate_when_called` or `brier_score`
#   passes through the `_snapshotted_after_kickoff` guard, regardless of how it
#   entered the table.
#
# It holds for two structural reasons, and each has a test below.
#
# 1. The write path admits post-kickoff rows -- it does NOT have to refuse them
#    for the record to stay honest. `record_player_prop_predictions` (store.py:646)
#    stamps `snapshotted_at` with `now()` and never compares it to the game's
#    kickoff, and `reconcile_player_prop_predictions` (store.py:695) grades any
#    row that joins to stats. Both are demonstrated below.
#
# 2. Neither of them can REWRITE that stamp. `reconcile_player_prop_predictions`
#    is an `UPDATE ... SET resolved = 1, actual_value = ?` -- it does not mention
#    `snapshotted_at`, and no other write against the table exists. So the guard
#    reads the row's original provenance however the row arrived, and
#    `_summarize_player_props` (the only summariser; sole caller
#    `store.py:342`) applies it to the entire `resolved = 1` set.
#
# If (1) or (2) ever changed, the published numbers would go wrong again and these
# tests would fail. That is the point of writing them.


def test_rows_the_write_path_admits_after_kickoff_never_reach_the_published_record():
    """Proves (1): the guard keys on PROVENANCE, not on whether the writer was careful.

    This drives the real public writers -- `record_player_prop_predictions` and
    `reconcile_player_prop_predictions` -- so the rows are admitted and graded the
    same way production admits them, not hand-planted to suit the summariser. Both
    writers are demonstrably unguarded: the insert stamps `now()` and never looks at
    the kickoff, and the reconcile grades anything that joins to stats. The game
    kicked off in 2000, so `now()` is unambiguously after it.

    The published record must still be empty. If it is not, the read-side guard is
    not sufficient and the reconcile guard has to be ported after all.
    """
    _insert_game_row("g_done", "2000-09-04T20:20:00+00:00")

    admitted = store.record_player_prop_predictions([
        _prop(game_id="g_done", player_id="late_td", market="anytime_td", predicted_value=0.99),
        _prop(game_id="g_done", player_id="late_yds", market="rushing_yards", predicted_value=999.0),
    ])
    graded = store.reconcile_player_prop_predictions(pd.DataFrame([
        # The anytime_td row is graded as a MISS, so if it leaked in the published
        # hit rate would fall from None to 0.0 rather than merely changing a count.
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

    assert props["anytime_td"]["n_resolved"] == 0
    assert props["anytime_td"]["hit_rate_when_called"] is None, (
        "a graded post-kickoff anytime_td pick reached the published hit rate"
    )
    assert props["anytime_td"]["brier_score"] is None, (
        "a graded post-kickoff anytime_td pick reached the published Brier score"
    )
    assert props["rushing_yards"]["n_resolved"] == 0
    assert props["rushing_yards"]["mean_absolute_error"] is None
    assert props["n_rebuilt"] == 2, "the write path admitted rows the record did not account for"


def test_grading_never_restamps_the_snapshot_timestamp_the_guard_reads():
    """Proves (2), and it is the load-bearing half.

    The read-side guard decides pre-kickoff-ness from `snapshotted_at`, and the
    write path cannot correct a bad stamp because it cannot alter one. This is the
    only reason a late row cannot be relabelled as a pick: the writer stamps
    `now()`, grading leaves that stamp alone, and the summariser re-reads it every
    time.

    So the invariant to pin is that grading is provenance-preserving. It asserts no
    particular metric -- it asserts that the timestamp the guard depends on is
    byte-identical before and after `reconcile_player_prop_predictions` runs. Add
    `snapshotted_at = ?` to that UPDATE and this fails, which is exactly the change
    that would let the write path defeat the read-side guard and silently put
    look-forward picks back into the published record.
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
    # And the record is still honest for the same reason.
    assert store.get_track_record()["player_props"]["n_rebuilt"] == 1
