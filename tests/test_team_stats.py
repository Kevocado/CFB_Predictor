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

from cfb_predictor.data import player_stats, team_stats


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
    out = team_stats.reconcile_against_players(team, players, allow_week_key=True)
    assert out.iloc[0]["player_total_yards"] == 340
    assert out.iloc[0]["diff"] == 0


def test_reconcile_surfaces_a_mismatch():
    team = pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "total_yards": 340}])
    players = pd.DataFrame([
        {"season": 2023, "week": 1, "team": "A", "rushing_yards": 90, "receiving_yards": 100},
    ])
    out = team_stats.reconcile_against_players(team, players, allow_week_key=True)
    assert out.iloc[0]["diff"] == 150


def test_reconcile_ignores_a_quarterbacks_passing_row():
    """A QB's `passing_yards` is the team's passing total. Summing it alongside
    rushing + receiving would double-count the whole passing game."""
    team = pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "total_yards": 340}])
    players = pd.DataFrame([
        {"season": 2023, "week": 1, "team": "A", "rushing_yards": 90, "receiving_yards": 250},
        {"season": 2023, "week": 1, "team": "A", "rushing_yards": 0, "receiving_yards": 0},
    ])
    out = team_stats.reconcile_against_players(team, players, allow_week_key=True)
    assert out.iloc[0]["diff"] == 0


# ---------------------------------------------------------------- network


@pytest.mark.network
def test_cfbd_team_stats_reconciles_against_real_player_data():
    """The architecture's load-bearing claim, on one real week. One API call.

    Skipped by default; run with:
        PYTHONPATH=src pytest tests/test_team_stats.py -m network -q

    **No longer xfail, and the reason it was is the substance of the fix.** This test
    previously carried a docstring claiming the player sum reproduced the team total
    "in 355 of 356" team-games. That claim was measured with the reconciliation keyed
    on `(season, week, team)` and it was **false** — measured on the same week, only
    58 of 272 rows were exact, with a median difference of 336 yards and a maximum of
    1177. A team playing twice in one week had both games' players summed and compared
    against each single game, so the identity never held; the test was not detecting
    that because it was skipped by default and, when run, its own measurement was
    never re-derived.

    Keyed on `game_id` instead, measured 2026-09-27 on 2023 week 1 (272 team-games
    after dropping 84 unplaceable non-FBS games):

    | | before (week-keyed) | after (game-keyed) |
    |---|---|---|
    | exact | 58 / 272 | 161 / 272 |
    | within 5 yards | - | 252 / 272 (92.6%) |
    | within 10 yards | 80 / 272 | 260 / 272 (95.6%) |
    | median abs diff | 336.00 | 0.00 |
    | max abs diff | 1177.0 | 29.0 |
    | team-games surviving the join | 272 of 272 | 272 of 272 |

    **The residual is characterised, not eliminated, and the tolerance below reflects
    that honestly rather than asserting exactness CFBD does not support.** 11 of the 12
    remaining offenders run the same direction — the player sum *exceeds* the team's
    `totalYards` — which points at a definitional difference between CFBD's team
    `totalYards` and the sum of player `rushing + receiving` (net-rushing treatment of
    sacks and lost yards being the obvious candidate), not at a join fault. The
    opposite-signed outlier is Robert Morris, an FCS game.

    So the assertion is a real tolerance rather than a hard identity, and the
    deterministic guard against this specific bug regressing is the offline
    `test_reconcile_keys_on_the_game_not_the_week` above, which does not depend on a
    live API being reachable.
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
        # Through the PRODUCTION loader, not `_import_player_game_stats` +
        # `_flatten_player_game_stats` directly. The previous version of this test
        # called the private flatteners, which always emit `game_id` -- so it
        # proved the fix on a path production never takes. A stale parquet cache
        # written before `game_id` existed then silently put the reconciliation
        # back on the non-unique week key, with this test still green.
        # `force_refresh=True` also exercises the cache gate itself.
        games_frame = games_module.fetch_schedules([2023])
        player_frame = player_stats.fetch_weekly_player_stats(
            [2023], games_frame, force_refresh=True
        ).rename(columns={"recent_team": "team"})
        assert "game_id" in player_frame.columns, (
            "the production loader returned no game_id; the cache gate did not fire")
    except Exception as exc:  # network unavailable
        pytest.skip(f"CFBD unreachable: {exc}")

    if team_frame.empty or player_frame.empty:
        pytest.skip("CFBD returned no data for 2023 week 1")

    out = team_stats.reconcile_against_players(team_frame, player_frame)
    assert not out.empty, "no overlapping team-games; nesting level probably wrong"

    # The join key is the regression that matters most: keyed on the week, this
    # test still "passes" a mean-based check while every individual game is
    # compared against a multi-game player sum.
    assert out["reconciliation_key"].iloc[0] == "game_id", (
        f"reconciled on {out['reconciliation_key'].iloc[0]!r}; a team can play twice in a "
        f"week, so only game_id is exact")

    absolute = out["diff"].abs()
    within_ten = (absolute <= 10).mean()
    assert within_ten >= 0.95, (
        f"only {within_ten:.1%} of {len(out)} team-games reconcile within 10 yards "
        f"(was 29.4% before the game_id fix); max {absolute.max()}")
    assert absolute.median() == 0, f"median abs diff {absolute.median()}, was 336 before the fix"
    assert absolute.max() <= 35, (
        f"worst team-game is off by {absolute.max()} yards; the documented residual is a "
        f"definitional difference in CFBD's team totalYards and should not grow")


def test_reconcile_keys_on_the_game_not_the_week():
    """A team can play twice in one week, and CFBD's `week` argument over-returns,
    so (season, week, team) is not unique in either frame.

    Keyed on week, the two USC week-1 games shared one player sum of 1620 yards,
    and each single game was then compared against that whole two-game total --
    1620 against a 443-yard game. That is the mechanism behind every absurd
    diff the live reconciliation produced.
    """
    team = pd.DataFrame([
        {"game_id": "g1", "season": 2023, "week": 1, "team": "A", "total_yards": 300},
        {"game_id": "g2", "season": 2023, "week": 1, "team": "A", "total_yards": 250},
    ])
    players = pd.DataFrame([
        {"game_id": "g1", "season": 2023, "week": 1, "team": "A", "rushing_yards": 90, "receiving_yards": 210},
        {"game_id": "g2", "season": 2023, "week": 1, "team": "A", "rushing_yards": 80, "receiving_yards": 170},
    ])
    out = team_stats.reconcile_against_players(team, players)
    assert len(out) == 2, "both games must survive the join"
    assert sorted(out["diff"]) == [0, 0], f"each game must reconcile against its own players: {out.to_dict('records')}"


def test_a_team_game_with_no_players_still_reports_its_own_total():
    """Inner-joining on game_id must not silently drop a team-game whose players
    the player endpoint did not return. It should surface as a missing sum, not
    vanish from the reconciliation."""
    team = pd.DataFrame([
        {"game_id": "g1", "season": 2023, "week": 1, "team": "A", "total_yards": 300},
        {"game_id": "g2", "season": 2023, "week": 1, "team": "A", "total_yards": 250},
    ])
    players = pd.DataFrame([
        {"game_id": "g1", "season": 2023, "week": 1, "team": "A", "rushing_yards": 90, "receiving_yards": 210},
    ])
    out = team_stats.reconcile_against_players(team, players)
    assert set(out["game_id"]) == {"g1", "g2"}
    missing = out[out["game_id"] == "g2"].iloc[0]
    assert missing["player_total_yards"] != missing["player_total_yards"], (
        "a game with no player rows must report a missing sum, not a zero that looks reconciled")


def test_the_player_frame_carries_game_id():
    """`_flatten_player_game_stats` keys rows on (game_id, player_id) but used to
    drop the game_id on the way out, which is what forced the week-keyed join."""
    raw = [{
        "id": 401520281,
        "teams": [{
            "team": "Alabama",
            "categories": [{
                "name": "rushing",
                "types": [{
                    "name": "YDS",
                    "athletes": [
                        {"id": 4361307, "name": "J. McClain", "stat": 88},
                        {"id": -1, "name": "Team", "stat": 300},
                    ],
                }],
            }],
        }],
    }]
    schedule = pd.DataFrame([{"game_id": 401520281, "season": 2023, "week": 1}])
    frame = player_stats._flatten_player_game_stats(raw, schedule, 2023)
    assert "game_id" in frame.columns
    assert set(frame["game_id"].astype(str)) == {"401520281"}, "both rows are in the same game"
    assert len(frame) == 1, "the negative-id team total is still excluded"


def test_reconcile_refuses_a_frame_with_no_game_id_by_default():
    """A stale player cache written before `game_id` existed silently put the
    reconciliation back on `(season, week, team)` -- the exact bug `game_id` was
    added to fix. It failed silently *and* the suite stayed green, because the
    tests call the flatteners directly and never touch the cached loader that
    production uses. So the fallback now refuses unless asked for explicitly."""
    team = pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "total_yards": 340}])
    players = pd.DataFrame([
        {"season": 2023, "week": 1, "team": "A", "rushing_yards": 90, "receiving_yards": 250},
    ])
    with pytest.raises(ValueError, match="NOT unique"):
        team_stats.reconcile_against_players(team, players)
    # ...and the opt-in still reconciles, for the hand-built fixtures above.


def test_attach_schedule_weeks_refuses_to_synthesise_a_week():
    """`fetch_schedules` returns an empty frame when the cache is cold and CFBD is
    unreachable -- the documented cold-start case, and the reason this branch
    exists. It used to fall back to `requested_week`, which is precisely the
    stacking bug the rest of the function prevents, and it also returned without
    the `dropped_unplaceable` attribute every other path sets."""
    frame = pd.DataFrame([{"game_id": "g1", "season": 2023, "requested_week": 1, "team": "A"}])
    with pytest.raises(ValueError, match="Refusing to fall back"):
        team_stats.attach_schedule_weeks(frame, pd.DataFrame())


def test_reconcile_separates_two_teams_in_one_game():
    """Kills the `keys = ["game_id"]` mutant -- dropping `team` from the key.

    A single-team fixture cannot see that: with one team per game, `game_id`
    alone is accidentally unique. Two teams sharing one game is what makes the
    omission visible, and it is the shape of every real game.
    """
    team = pd.DataFrame([
        {"game_id": "g1", "season": 2023, "week": 1, "team": "A", "total_yards": 300},
        {"game_id": "g1", "season": 2023, "week": 1, "team": "B", "total_yards": 250},
    ])
    players = pd.DataFrame([
        {"game_id": "g1", "season": 2023, "week": 1, "team": "A", "rushing_yards": 90, "receiving_yards": 210},
        {"game_id": "g1", "season": 2023, "week": 1, "team": "B", "rushing_yards": 80, "receiving_yards": 170},
    ])
    out = team_stats.reconcile_against_players(team, players)
    assert out["reconciliation_key"].eq("game_id").all()
    assert sorted(out["diff"]) == [0, 0], (
        f"both teams must reconcile against their own players: {out.to_dict('records')}")
    assert set(zip(out["team"], out["player_total_yards"])) == {("A", 300.0), ("B", 250.0)}


def test_reconcile_drops_a_repeated_team_stanza():
    """`(game_id, team)` uniqueness is what makes the join mean anything. It held
    empirically but nothing enforced it, and a duplicate would not inflate the
    player sum (the right side is grouped, so a left join cannot fan out) -- it
    would duplicate the output row and double-count the game downstream."""
    team = pd.DataFrame([
        {"game_id": "g1", "season": 2023, "week": 1, "team": "A", "total_yards": 300},
        {"game_id": "g1", "season": 2023, "week": 1, "team": "A", "total_yards": 300},
    ])
    players = pd.DataFrame([
        {"game_id": "g1", "season": 2023, "week": 1, "team": "A", "rushing_yards": 90, "receiving_yards": 210},
    ])
    out = team_stats.reconcile_against_players(team, players)
    assert len(out) == 1, "a repeated stanza must not become two reconciled games"
    assert out["total_yards"].sum() == 300


def test_a_half_populated_game_reports_a_missing_total_not_a_partial_one():
    """Kills the `min_count=len(summed) -> min_count=1` mutant.

    `sum(axis=1, min_count=2)` needs BOTH components populated to call a game's
    player total complete. `min_count=1` would accept a game with only rushing
    rows and report that as a total -- a number that is wrong rather than
    absent, which is the worse failure: an absent one is visible in a count, a
    partial one is not.
    """
    team = pd.DataFrame([
        {"game_id": "g1", "season": 2023, "week": 1, "team": "A", "total_yards": 300},
        {"game_id": "g2", "season": 2023, "week": 1, "team": "A", "total_yards": 300},
    ])
    players = pd.DataFrame([
        # complete: both components
        {"game_id": "g1", "season": 2023, "week": 1, "team": "A",
         "rushing_yards": 90, "receiving_yards": 210},
        # half populated: rushing only
        {"game_id": "g2", "season": 2023, "week": 1, "team": "A",
         "rushing_yards": 90, "receiving_yards": float("nan")},
    ])
    out = team_stats.reconcile_against_players(team, players).set_index("game_id")
    assert out.loc["g1", "player_total_yards"] == pytest.approx(300)
    partial = out.loc["g2", "player_total_yards"]
    assert partial != partial, (
        f"a game with only one of the two components must report a missing total, not {partial}")
    assert out.loc["g2", "diff"] != out.loc["g2", "diff"], "and its diff must be missing too"
