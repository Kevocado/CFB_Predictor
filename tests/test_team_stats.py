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


def test_flatten_of_nothing_declares_a_superset_of_what_a_real_week_carries():
    """An empty week must declare every field a populated one can have.

    This used to assert `list(empty.columns) == KEEP_COLUMNS`, which was a weaker
    and wrong thing to pin. `_flatten` stamps `requested_week` on every real row
    and `KEEP_COLUMNS` does not list it, so the old assertion passed precisely by
    being the shape that did not match a real frame -- 27 columns against 27 but a
    different set.

    That asymmetry is not cosmetic. `_read_cache` requires `requested_week`, so an
    empty frame without it is rejected as pre-dating the column and re-requested
    every run against a metered API.

    Direction matters, and "equal" is the wrong assertion to reach for: a populated
    frame only carries the stat categories that game actually had, so exact
    equality is data-dependent. The invariant that holds for every input is
    *superset* -- an empty week can never lack a field a populated one has.
    """
    empty = team_stats._flatten([], season=2023, week=1)
    populated = team_stats._flatten([_raw_game()], season=2023, week=1)

    assert empty.empty
    assert set(populated.columns) <= set(empty.columns), (
        "a populated week carried fields the empty week does not declare: "
        f"{sorted(set(populated.columns) - set(empty.columns))}"
    )
    assert set(team_stats.KEEP_COLUMNS) <= set(empty.columns)
    assert {"game_id", "season", "requested_week", "team"} <= set(empty.columns)


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


def test_reconcile_refuses_a_week_keyed_frame_with_no_week():
    """Without `game_id` on both frames the only key is (season, week, team), which is
    not unique -- so that path still demands a schedule-derived week. Supplying
    `game_id` on only one side leaves the same problem, so the requirement stands for
    the mixed case too."""
    with pytest.raises(ValueError, match="attach_schedule_weeks"):
        team_stats.reconcile_against_players(
            pd.DataFrame([{"season": 2023, "team": "A", "total_yards": 100}]),
            pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "rushing_yards": 1, "receiving_yards": 2}]),
        )
    with pytest.raises(ValueError, match="attach_schedule_weeks"):
        team_stats.reconcile_against_players(
            pd.DataFrame([{"game_id": "g1", "season": 2023, "team": "A", "total_yards": 100}]),
            pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "rushing_yards": 1, "receiving_yards": 2}]),
        )


def test_the_game_keyed_path_needs_no_week_at_all():
    """`attach_schedule_weeks` drops games absent from the schedule and mislabels
    bowls, so requiring its output cost 31% of the frame for a key the join does not
    read. With `game_id` on both sides the week is never consulted."""
    team = pd.DataFrame([
        {"game_id": "g1", "season": 2023, "requested_week": 1, "team": "A", "total_yards": 300},
        {"game_id": "g2", "season": 2023, "requested_week": 1, "team": "A", "total_yards": 250},
    ])
    players = pd.DataFrame([
        {"game_id": "g1", "season": 2023, "week": 1, "team": "A", "rushing_yards": 90, "receiving_yards": 210},
        {"game_id": "g2", "season": 2023, "week": 1, "team": "A", "rushing_yards": 80, "receiving_yards": 170},
    ])
    out = team_stats.reconcile_against_players(team, players)
    assert sorted(out["diff"]) == [0, 0], "a week-less team frame must still reconcile exactly"
    assert out["reconciliation_key"].eq("game_id").all()


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

    **The residual is upstream, and it is not a definitional difference.** An earlier
    version of this test asserted a maximum of 35 yards and blamed a net-rushing
    treatment of sacks for the residual. Both were wrong, and the second was wrong
    in a way that would have kept being wrong. Checked row by row against the
    unflattened CFBD response, the team and player box scores simply disagree: for
    Campbell vs Monmouth (2023 week 3) the team box reports 182 rushing / 184 net
    passing and the player box reports 133 rushing / 172 receiving. Not a
    flattening fault, not a dropped row, and not one-directional.

    At 22-season scale (42,172 comparable team-games) the distribution is
    60.0% exact, 92.3% within 10, 98.3% within 35, median 0.00, p99 54.0 and
    **max 550** — so the old 35-yard bound held for 2023 week 1 and for nothing
    else. It is therefore removed below rather than loosened: a bound the source
    does not honour is not a weaker assertion, it is a false one that would pass
    on a lucky week and mislead on an unlucky one.

    The
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
    # **No bound on the maximum.** Earlier versions of this file asserted
    # `absolute.max() <= 35`, then a p99 <= 60, then claimed in a comment that the max
    # bound had been removed while it was still there. All three states existed here
    # at some point, which is why the comment is this blunt: the assertion that used
    # to be in this spot was removed, not loosened, and the removal is the point.
    #
    # A bound the source does not honour is not a weaker assertion, it is a false one
    # that passes on a lucky week and misleads on an unlucky one. At 22 seasons the
    # max is 337 (2004) and 92.4% are within 10 -- so a 35-yard bound is false by an
    # order of magnitude, and it survived review because this test only loads
    # 2023 week 1, where the worst case is 29.
    #
    # What this test *does* guard is the join fix, and it guards it well: a week-keyed
    # join fails every one of these three (29.4% within 10, median 336, max 1177).
    # The 22-season figures cannot run in CI -- they need the 351-call CFBD backfill --
    # so they live in the module docstring, and the honesty about what is and is not
    # pinned is deliberate.

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


# --- backfill: an empty week must not be re-requested forever -------------------
#
# `fetch_team_stats` writes a parquet per (season, week) and skips a week whose
# frame is empty -- `if frame.empty: continue` -- without writing anything. The
# next run therefore finds no cache file, re-requests, and skips again. Against a
# metered API that is not free: 2023 has no week 16, and most seasons end before
# week 15, so every subsequent run pays again for every dead week in the range.
#
# For a 2004-2025 sweep that is a few dozen calls per re-run, and it means the
# script is not idempotent, so "just run it again" -- the normal response to an
# interrupted backfill -- silently costs more every time.

class _FakeGamesApi:
    """Counts requests and serves a configurable payload per (season, week)."""

    def __init__(self, payload_by_week):
        self.payload_by_week = payload_by_week
        self.calls: list[tuple[int, int]] = []

    def get_game_team_stats(self, year, week):
        self.calls.append((year, week))
        return self.payload_by_week.get(week, [])


@pytest.fixture
def fake_cfbd(monkeypatch):
    """Install a stub `cfbd` module and hand back the fake GamesApi it built.

    `team_stats` imports `cfbd` lazily inside `_client` and `fetch_team_stats`, so
    the module has to be patched into `sys.modules` rather than onto the
    `team_stats` namespace -- and it needs `ApiClient`, `Configuration` and
    `GamesApi`, because the client is built before the games API is.
    """
    import sys
    import types

    state = {"payload_by_week": {}, "calls": []}

    class _GamesApi:
        def get_game_team_stats(self, year, week):
            state["calls"].append((year, week))
            return state["payload_by_week"].get(week, [])

    class _ApiClient:
        def __init__(self, *a, **k):
            self.default_headers: dict[str, str] = {}

    module = types.ModuleType("cfbd")
    module.ApiClient = _ApiClient
    module.Configuration = lambda *a, **k: None
    module.GamesApi = lambda *a, **k: _GamesApi()
    monkeypatch.setitem(sys.modules, "cfbd", module)
    return state


def test_a_week_with_no_games_is_recorded_so_a_rerun_does_not_re_request_it(monkeypatch, tmp_path, fake_cfbd):
    from cfb_predictor.data import team_stats

    monkeypatch.setattr(team_stats, "TEAM_STATS_CACHE_DIR", tmp_path)
    fake_cfbd["payload_by_week"] = {1: [], 2: []}

    team_stats.fetch_team_stats([2023], weeks=[1, 2])
    assert fake_cfbd["calls"] == [(2023, 1), (2023, 2)], "first run should request both weeks"

    fake_cfbd["calls"].clear()
    team_stats.fetch_team_stats([2023], weeks=[1, 2])
    assert fake_cfbd["calls"] == [], (
        "a re-run must not re-request a week known to be empty; "
        f"it re-requested {fake_cfbd['calls']}"
    )


def test_a_recorded_empty_week_still_yields_a_frame_the_caller_can_concat(monkeypatch, tmp_path, fake_cfbd):
    """The tombstone must be a real, schema-correct frame.

    An empty result has to survive `pd.concat` alongside populated weeks, or the
    backfill's own output is the thing that breaks.
    """
    from cfb_predictor.data import team_stats

    monkeypatch.setattr(team_stats, "TEAM_STATS_CACHE_DIR", tmp_path)
    fake_cfbd["payload_by_week"] = {1: [], 2: []}

    out = team_stats.fetch_team_stats([2023], weeks=[1, 2])
    assert list(out.columns) == team_stats.KEEP_COLUMNS, (
        "an all-empty range must still report the documented schema, "
        f"got {list(out.columns)}"
    )
    assert out.empty


def test_reconcile_accepts_cfbds_own_player_team_column_name():
    """`recent_team` is the name CFBD's player box score uses; `team` is ours.

    `player_stats.KEEP_COLUMNS` carries `recent_team`, taken from the *game's* team
    stanza in the payload nesting (`teams[].team`), so the attribution is per-game
    and correct -- the name is just CFBD's, and it reads like it means something it
    does not. The team box score, by contrast, uses `team`.

    So `reconcile_against_players` was asking for a `team` column that only its
    hand-built fixtures had. Every offline test passed, because every offline test
    supplies its own frame -- and the mismatch only surfaced against real data, at
    42,190 team-games, as a bare `KeyError: 'team'` from inside a `groupby`, which
    says nothing about the cause. This is the same failure shape as the stale-cache
    bug already documented in this file: the private flatteners always emit the
    right column, so tests built on them prove nothing about the cached loader.

    Asserted through the production-facing column name, not by renaming in the test,
    so the alias itself is what is covered.
    """
    team = pd.DataFrame([{
        "game_id": "401", "season": 2023, "week": 1, "team": "A",
        "total_yards": 400.0, "net_passing_yards": 250.0, "rushing_yards": 150.0,
    }])
    players = pd.DataFrame([
        {"game_id": "401", "season": 2023, "week": 1, "recent_team": "A",
         "rushing_yards": 100.0, "receiving_yards": 150.0},
        {"game_id": "401", "season": 2023, "week": 1, "recent_team": "A",
         "rushing_yards": 50.0, "receiving_yards": 100.0},
    ])

    out = team_stats.reconcile_against_players(team, players)

    assert len(out) == 1
    assert out["player_total_yards"].iloc[0] == 400.0
    assert out["diff"].iloc[0] == 0.0


def test_reconcile_still_accepts_a_literal_team_column():
    """The alias must not become the only spelling that works."""
    team = pd.DataFrame([{
        "game_id": "401", "season": 2023, "week": 1, "team": "A",
        "total_yards": 400.0, "net_passing_yards": 250.0, "rushing_yards": 150.0,
    }])
    players = pd.DataFrame([
        {"game_id": "401", "season": 2023, "week": 1, "team": "A",
         "rushing_yards": 100.0, "receiving_yards": 150.0},
        {"game_id": "401", "season": 2023, "week": 1, "team": "A",
         "rushing_yards": 50.0, "receiving_yards": 100.0},
    ])

    out = team_stats.reconcile_against_players(team, players)

    assert out["player_total_yards"].iloc[0] == 400.0


def test_reconcile_raises_a_useful_error_when_the_player_team_column_is_absent_entirely():
    """A `KeyError: 'team'` from inside a `groupby` names the symptom, not the cause."""
    team = pd.DataFrame([{
        "game_id": "401", "season": 2023, "week": 1, "team": "A",
        "total_yards": 400.0, "net_passing_yards": 250.0, "rushing_yards": 150.0,
    }])
    players = pd.DataFrame([
        {"game_id": "401", "season": 2023, "week": 1, "club": "A",
         "rushing_yards": 100.0, "receiving_yards": 150.0},
    ])

    with pytest.raises(ValueError, match="team"):
        team_stats.reconcile_against_players(team, players)


def test_a_zero_total_yards_with_non_zero_components_is_treated_as_missing():
    """CFBD ships `totalYards: 0` for real team-games whose components sum to hundreds.

    48 team-games across 2004-2025 are recorded at *zero* total yards while CFBD's own
    `netPassingYards + rushingYards` for the same row totals in the hundreds — Hawai'i
    2007 is recorded at 0 total yards in a game it won 63-? with 577 by CFBD's own
    numbers. A team that played cannot have gained zero total yards, so the zero is a
    missing value wearing a plausible number.

    The guard was `.notna()`, and `0.0` is not NA. It was not cosmetic: those rows were
    the entire basis of the "max 550, the maximum is unbounded" claim in the module
    docstring, because the single largest residual in 22 seasons was a team-game at
    `total_yards = 0` against a player sum of 550. Excluding them the real worst case
    is 337 and the real p99 is 50.
    """
    frame = pd.DataFrame([{
        "game_id": "1", "season": 2007, "requested_week": 1, "team": "Hawaii",
        # CFBD's own components say 577. The total it shipped says 0.
        "total_yards": 0.0, "net_passing_yards": 540.0, "rushing_yards": 37.0,
    }])

    out = team_stats.add_total_yards(frame)

    assert out["total_yards"].iloc[0] == 577.0, (
        f"a zero total with 577 yards of components must be rebuilt from them, got "
        f"{out['total_yards'].iloc[0]}"
    )


def test_a_genuine_zero_total_is_still_zero():
    """The `> 0` guard must not swallow a real zero.

    A team that never touched the ball downfield *and* gained no rushing yards is
    vanishingly rare but not impossible, and if it happens the components will be zero
    too, so the fallback reconstructs 0 anyway. This pins that the two paths agree
    rather than leaving it to arithmetic.
    """
    frame = pd.DataFrame([{
        "game_id": "1", "season": 2023, "requested_week": 1, "team": "X",
        "total_yards": 0.0, "net_passing_yards": 0.0, "rushing_yards": 0.0,
    }])

    assert team_stats.add_total_yards(frame)["total_yards"].iloc[0] == 0.0


def test_a_genuine_non_zero_total_is_still_preferred_over_the_sum():
    """A fixture where the two candidates differ, so the preference is observable.

    The existing test for this uses `total_yards=550, net=308, rush=242` — and
    `308 + 242 == 550`, so both candidates are identical and the test cannot tell
    "prefers the supplied value" from "always reconstructs". That is the recurring
    defect class: asserting that a value was produced rather than that it was right.
    """
    frame = pd.DataFrame([{
        "game_id": "1", "season": 2023, "requested_week": 1, "team": "X",
        "total_yards": 550.0, "net_passing_yards": 300.0, "rushing_yards": 200.0,
    }])

    out = team_stats.add_total_yards(frame)

    assert out["total_yards"].iloc[0] == 550.0, "CFBD's own total must win when it is present"
