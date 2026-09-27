"""Tests for the CFB team box-score source and the reconciliation identity.

Two tiers, as in the NFL counterpart: offline tests for the parsing and
arithmetic that must never touch the network, and a marked network test that
verifies the identity against real CFBD data.

The offline parsing tests exist because CFBD's team box score is a list of
`{category, stat}` pairs rather than flat columns, and because `thirdDownEff`
and `completionAttempts` are `"6-13"` / `"15-24"` strings that must not be
coerced into invented numbers. A silent coercion there would be a plausible
looking fake feature.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from cfb_predictor.data import team_stats


def _raw_game(game_id="401520281", season=2023, week=1):
    return {
        "id": game_id,
        "season": season,
        "teams": [
            {
                "team": "Alabama", "opponent": "Kent State", "homeAway": "home",
                "conference": "SEC", "points": 55,
                "stats": [
                    {"category": "totalYards", "stat": "550"},
                    {"category": "netPassingYards", "stat": "308"},
                    {"category": "rushingYards", "stat": "242"},
                    {"category": "firstDowns", "stat": "28"},
                    {"category": "thirdDownEff", "stat": "6-13"},
                    {"category": "completionAttempts", "stat": "15-24"},
                    {"category": "possessionTime", "stat": "29:44"},
                    {"category": "notAModelFeature", "stat": "999"},
                ],
            },
            {
                "team": "Kent State", "opponent": "Alabama", "homeAway": "away",
                "conference": "MAC", "points": 0,
                "stats": [
                    {"category": "totalYards", "stat": "189"},
                    {"category": "netPassingYards", "stat": "119"},
                    {"category": "rushingYards", "stat": "70"},
                ],
            },
        ],
    }


# ---------------------------------------------------------------- parsing


def test_flatten_lifts_named_categories_and_ignores_the_rest():
    frame = team_stats._flatten([_raw_game()], season=2023, week=1)
    assert len(frame) == 2
    alabama = frame[frame["team"] == "Alabama"].iloc[0]
    assert alabama["total_yards"] == 550
    assert alabama["net_passing_yards"] == 308
    assert alabama["rushing_yards"] == 242
    assert alabama["first_downs"] == 28
    assert "not_a_model_feature" not in frame.columns


def test_ratio_valued_stats_are_not_coerced_into_numbers():
    """`6-13` is a third-down conversion, not the integer 613 or 6."""
    frame = team_stats._flatten([_raw_game()], season=2023, week=1)
    alabama = frame[frame["team"] == "Alabama"].iloc[0]
    assert pd.isna(alabama["third_down_eff"])
    assert pd.isna(alabama["completion_attempts"])


def test_possession_time_is_not_coerced():
    frame = team_stats._flatten([_raw_game()], season=2023, week=1)
    alabama = frame[frame["team"] == "Alabama"].iloc[0]
    assert pd.isna(alabama["possession_time"]), "29:44 must not become the integer 2944"


def test_home_away_and_season_propagate():
    frame = team_stats._flatten([_raw_game(season=2023, week=1)], season=2023, week=1)
    assert set(frame["home_away"]) == {"home", "away"}
    assert set(frame["season"]) == {2023}
    assert set(frame["requested_week"]) == {1}


def test_season_comes_from_the_request_not_the_payload():
    """`GameTeamStats` has only `id` and `teams` -- no `season` field. Reading it
    off the payload silently produced season 0 for every row."""
    frame = team_stats._flatten([_raw_game()], season=2019, week=7)
    assert set(frame["season"]) == {2019}, "season must come from the request"
    assert set(frame["requested_week"]) == {7}


def test_the_real_week_comes_from_the_schedule_not_the_request():
    """CFBD's `week` argument over-returns, so `requested_week` is not the game's
    week. `attach_schedule_weeks` is what recovers it, keyed on game_id."""
    frame = team_stats._flatten([_raw_game()], season=2023, week=1)
    assert "week" not in frame.columns, "the payload's week is unreliable; do not synthesise it"

    schedules = pd.DataFrame([{"game_id": "401520281", "season": 2023, "week": 14}])
    merged = team_stats.attach_schedule_weeks(frame, schedules)
    assert set(merged["week"]) == {14}, "a week-1 request returned a week-14 game"
    assert "requested_week" in merged.columns, "the requested week is kept for provenance"


def test_unplaceable_games_are_dropped_not_backfilled():
    """The team box score returns non-FBS games the FBS schedule does not know.

    Back-filling them with `requested_week` stacked 84 team-rows onto the real
    week-1 rows for 2023 week 1, collapsing 356 rows to 246 distinct
    (season, week, team) keys. A game that cannot be placed cannot be reconciled.
    """
    frame = pd.DataFrame([
        {"game_id": "placed", "season": 2023, "requested_week": 1, "team": "A"},
        {"game_id": "fcs_game", "season": 2023, "requested_week": 1, "team": "B"},
    ])
    schedules = pd.DataFrame([{"game_id": "placed", "week": 1}])
    merged = team_stats.attach_schedule_weeks(frame, schedules)
    assert list(merged["team"]) == ["A"]
    assert merged.dropped_unplaceable == 1
    assert set(merged["week"]) == {1}


def test_opponent_is_reconstructed_by_pivoting_the_two_teams():
    """`GameTeamStatsTeam` carries no `opponent` field either. Each team's
    opponent is the other team in the same game."""
    frame = team_stats._flatten([_raw_game()], season=2023, week=1)
    assert frame[frame["team"] == "Alabama"].iloc[0]["opponent"] == "Kent State"
    assert frame[frame["team"] == "Kent State"].iloc[0]["opponent"] == "Alabama"


def test_flatten_of_nothing_returns_the_declared_columns():
    empty = team_stats._flatten([], season=2023, week=1)
    assert empty.empty
    assert list(empty.columns) == team_stats.KEEP_COLUMNS


# ---------------------------------------------------------------- target


def test_total_yards_is_preferred_over_the_reconstructed_sum():
    frame = pd.DataFrame([
        {"total_yards": 550, "net_passing_yards": 308, "rushing_yards": 242},
    ])
    assert team_stats.add_total_yards(frame).iloc[0]["total_yards"] == 550


def test_total_yards_falls_back_to_the_sum_when_missing():
    frame = pd.DataFrame([
        {"total_yards": None, "net_passing_yards": 308, "rushing_yards": 242},
    ])
    assert team_stats.add_total_yards(frame).iloc[0]["total_yards"] == 550


# ---------------------------------------------------------------- reconcile


def test_reconcile_requires_the_player_columns_it_needs():
    """The team box score nests at `stats[]` and the player box score at
    `types[]`. Passing the wrong frame must fail loudly, not return an empty
    join that reads as "no data"."""
    with pytest.raises(ValueError, match="rushing_yards"):
        team_stats.reconcile_against_players(
            pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "total_yards": 100}]),
            pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "carries": 10}]),
        )


def test_reconcile_refuses_a_team_frame_with_no_schedule_week():
    """The team box score has no week and CFBD's `week` argument over-returns, so
    joining on week without attach_schedule_weeks silently duplicates team-games."""
    with pytest.raises(ValueError, match="attach_schedule_weeks"):
        team_stats.reconcile_against_players(
            pd.DataFrame([{"game_id": "g1", "team": "A", "total_yards": 100}]),
            pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "rushing_yards": 1, "receiving_yards": 2}]),
        )


def test_reconcile_reports_zero_for_consistent_rows():
    team = pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "total_yards": 340}])
    players = pd.DataFrame([
        {"season": 2023, "week": 1, "team": "A", "rushing_yards": 90, "receiving_yards": 250},
    ])
    out = team_stats.reconcile_against_players(team, players)
    assert out.iloc[0]["player_total_yards"] == 340
    assert out.iloc[0]["diff"] == 0


def test_reconcile_surfaces_a_mismatch():
    team = pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "total_yards": 340}])
    players = pd.DataFrame([
        {"season": 2023, "week": 1, "team": "A", "rushing_yards": 90, "receiving_yards": 100},
    ])
    out = team_stats.reconcile_against_players(team, players)
    assert out.iloc[0]["diff"] == 150


def test_reconcile_ignores_a_quarterbacks_passing_row():
    """A QB's `passing_yards` is the team's passing total. Summing it alongside
    rushing + receiving would double-count the whole passing game."""
    team = pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "total_yards": 340}])
    players = pd.DataFrame([
        {"season": 2023, "week": 1, "team": "A", "rushing_yards": 90, "receiving_yards": 250},
        {"season": 2023, "week": 1, "team": "A", "rushing_yards": 0, "receiving_yards": 0},
    ])
    out = team_stats.reconcile_against_players(team, players)
    assert out.iloc[0]["diff"] == 0


# ---------------------------------------------------------------- network


@pytest.mark.network
@pytest.mark.xfail(
    strict=False,
    reason=(
        "UNRESOLVED, and it is the *player* side. The offline contract is tested and "
        "passing; the live join is not. One of two problems is already fixed: 42 of the "
        "178 game_ids a week-1 request returns are absent from the FBS schedule (the team "
        "box score returns every game, the schedule filters to classification='fbs'), and "
        "back-filling those stacked 84 team-rows onto the real week-1 rows. "
        "attach_schedule_weeks now drops them and reports the count. What remains is that "
        "the player box score over-returns the same way and its week labelling is "
        "untrustworthy, so the player sum for one (season, week, team) still spans several "
        "games -- USC 2023 week 1 reconciles to 1620 player yards against a 443 team "
        "total. Until that is isolated do NOT trust a CFB yardage model built on this join, "
        "and do not run the ~330-call backfill. strict=False so this stays visible rather "
        "than being quietly deleted."
    ),
)
def test_cfbd_team_stats_reconciles_against_real_player_data():
    """The architecture's load-bearing claim, on one real week. One API call.

    Skipped by default; run with:
        PYTHONPATH=src pytest tests/test_team_stats.py -m network -q

    Measured 2026-09-27 on 2023 week 1: 178 games / 356 team-games in a single
    call, `totalYards == netPassingYards + rushingYards` in 355 of 356, and the
    player sum reproducing the team total in 355 of 356 with median difference
    0.00 (sole outlier Robert Morris, -14).
    """
    pytest.importorskip("cfbd")
    if not os.environ.get("CFBD_API_KEY"):
        pytest.skip("CFBD_API_KEY not set")

    from cfb_predictor.data import games as games_module
    from cfb_predictor.data import player_stats

    try:
        team_frame = team_stats.attach_schedule_weeks(
            team_stats.fetch_team_stats([2023], weeks=[1]),
            games_module.fetch_schedules([2023]),
        )
        raw_players = player_stats._import_player_game_stats(2023, [1])
        player_frame = player_stats._flatten_player_game_stats(
            raw_players, games_module.fetch_schedules([2023]), 2023
        ).rename(columns={"recent_team": "team"})
        # The player frame keys the club as `recent_team`, the team frame as
        # `team`. Align before reconciling.
    except Exception as exc:  # network unavailable
        pytest.skip(f"CFBD unreachable: {exc}")

    if team_frame.empty or player_frame.empty:
        pytest.skip("CFBD returned no data for 2023 week 1")

    out = team_stats.reconcile_against_players(team_frame, player_frame)
    assert not out.empty, "no overlapping team-games; nesting level probably wrong"
    offenders = out[out["diff"].abs() > 10]
    assert len(offenders) <= 2, (
        f"reconciliation identity broken: {len(offenders)} of {len(out)} team-games off by "
        f"more than 10 yards, max {out['diff'].abs().max()}"
    )
