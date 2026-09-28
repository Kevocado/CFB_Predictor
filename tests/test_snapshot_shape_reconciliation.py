"""A reused snapshot week must not be able to silently miss a newly added field.

`build_snapshot()` copies every week outside the rebuild window verbatim. That is
the right call for cost and the wrong call for shape: a copied week cannot pick
up a field the code has started emitting, so a new field lands on the few weeks
inside the window and nowhere else.

This is not hypothetical, and it is the CFB shape exactly as it was the NFL bug
fixed in `NFL_Predictor` commit `0c4ea1e`. The committed CFB snapshot is
`current_week` 5, so with `REBUILD_WEEKS_BEHIND = 1` and
`REBUILD_WEEKS_AHEAD = 3` the window is weeks 4-8 -- and the prop-carrying
weeks that matter sit *outside* it:

    week  1  2558 props   week  2  2843 props   week  3  2663 props
    week  4  2544 props   week  5  2112 props   ...  (in the window)
    week  9  1483 props   week 10  1665 props   ...  (copied verbatim)

(those counts are as of the A3 run; the committed artifact has since been
refreshed and now reads 3360 / 3092 / 2687 for weeks 1-3 and 2016 / 2259 for
weeks 9-10, with 14 and 16 prop-less. Same shape, different week 5.)

So `is_starter` and `depth_slot` would reach five weeks of fifteen and be absent
from the rest. The symptom is a frontend looking for a field the API is
documented to serve and finding it missing on most of the season -- and the
obvious conclusion, "the serialization is broken", is wrong. It is the copy,
not the writer.

The fix: work out the prop-row shape the CURRENT code produces and rebuild any
reused week that does not match it. Narrow on purpose -- only weeks whose row
shape actually changed are rebuilt, so adding a field costs one build rather
than all fifteen, and a week carrying fields the code no longer emits is left
alone rather than caught in a rebuild loop.

Two things that first version of the fix got wrong, and what the tests below
exist to hold down:

* **"Shape" is not one row's key set.** `predict_props` keys off
  `POSITION_MARKETS`, so prop rows are position-heterogeneous on purpose: a QB
  row carries `passing_yards`, an RB row `rushing_yards`, a WR/TE row
  `receiving_yards`, and a K/OL/DL/P row no market at all. Every week of the
  committed artifact has three distinct row shapes for exactly that reason.
  So the required keys are the INTERSECTION of a week's rows -- the keys the
  code writes whatever the position -- and *every* row is checked, not row 0.
* **Every test week in this file used to be single-shape**, which is why the
  row-0 predicate passed. `_row()` below derives its market keys from the real
  `POSITION_MARKETS`, and `TestTheShippedArtifactIsAlreadyReconciled` runs the
  committed 14 MB artifact, so both failure directions are reachable: a week
  whose row 0 is current while the rest are stale, and a week that is current on
  every row but whose row 0 is a different position from the sample's.

The `data/public_snapshot.json` tests read the committed artifact; they do not
write it, and `build_snapshot` never does either -- they assert on the week
list it was asked to build.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from cfb_predictor import config
from cfb_predictor import public_snapshot as ps
from cfb_predictor.models import player_props


# --------------------------------------------------------------------------- #
# Nothing in this file may touch the network.
#
# `build_snapshot` calls four live builders unconditionally and wraps every one
# of them in a bare `except Exception`, so a test that stubs only
# `_get_standings_live` gets a green run *and* a live call: `_get_hub_teams_live`
# -> `advanced_stats.fetch_advanced` and `_get_hub_players_live` ->
# `fetch_player_ppa` (routes.py:838-857), which are CFBD calls against a
# documented 1,000-calls/month quota. On a machine with a cold cache that is
# real quota, spent silently.
#
# Two halves, because raising alone is not enough -- `build_snapshot` swallows
# every one of these:
#   * the four builders are stubbed offline, so no test can reach one by
#     accident, and
#   * the upstream leaves they reach are forbidden, and the fixture asserts at
#     teardown that none was hit. A builder added to `build_snapshot` without a
#     stub here fails the run instead of the quota.
# --------------------------------------------------------------------------- #

OFFLINE_BUILDERS = {
    "_get_standings_live": [],
    "_get_power_rankings_live": {},
    "_get_hub_teams_live": {},
    "_get_hub_players_live": {},
}
# The per-week live calls. `_get_player_props_live` is the one live probe the
# signature is allowed to make, and the tests that exercise it override the
# guard in their own body, which is why they are not tripped by the teardown.
FORBIDDEN_ROUTES_CALLS = ("_get_games_live", "_get_game_prediction_live", "_get_player_props_live")
FORBIDDEN_UPSTREAM_CALLS = ("fetch_advanced", "fetch_player_ppa")

FORBIDDEN_CALLS: list[str] = []


def _forbidden(label: str):
    def stub(*args, **kwargs):
        FORBIDDEN_CALLS.append(label)
        raise AssertionError(f"{label} reached a live upstream from this test file")

    return stub


@pytest.fixture(autouse=True)
def _no_live_upstream(monkeypatch):
    FORBIDDEN_CALLS.clear()
    for name, value in OFFLINE_BUILDERS.items():
        monkeypatch.setattr(ps.routes, name, lambda _season, _value=value: _value)
    for name in FORBIDDEN_ROUTES_CALLS:
        monkeypatch.setattr(ps.routes, name, _forbidden(f"routes.{name}"))
    for name in FORBIDDEN_UPSTREAM_CALLS:
        monkeypatch.setattr(ps.routes.advanced_stats, name, _forbidden(f"advanced_stats.{name}"))
    yield
    assert FORBIDDEN_CALLS == [], f"live upstream reached from this test: {FORBIDDEN_CALLS}"


def _week(*prop_rows: dict, games: list | None = None) -> dict:
    return {"games": games or [], "predictions": {}, "player_props": list(prop_rows)}


# The row shape before A3, and after. `is_starter`/`depth_slot` are None for
# every CFB row because CFBD has no starter field -- see
# tests/test_starter_flag_has_no_upstream.py for the upstream evidence.
OLD_ROW = {
    "player_id": "p", "player_name": "A. Back", "position": "RB",
    "recent_team": "Alabama", "rushing_yards": 88.0, "anytime_td_prob": 0.4,
}
NEW_ROW = {**OLD_ROW, "is_starter": None, "depth_slot": None}

# The keys routes.py writes on every prop row (routes.py:584-615) plus the one
# key `predict_props` always sets. Transcribed rather than derived, on purpose:
# the derivation is checked against it.
CURRENT_ROW_KEYS = frozenset({
    "player_id", "player_name", "recent_team", "position", "anytime_td_prob",
    "is_starter", "depth_slot",
})


def _row(position: str, *, current: bool = True) -> dict:
    """One prop row as the current code emits it for `position`.

    The market keys come from the real `POSITION_MARKETS`, so a row is
    position-heterogeneous by construction -- the thing a row-0 signature could
    not see. `current=False` is the pre-A3 row: same position, same markets, no
    `is_starter`/`depth_slot`.

    The `carries` and `receptions` models do not exist yet, so the committed
    artifact has three row shapes per week rather than four. Including them here
    only makes the fixture more heterogeneous, which is the safe direction for a
    fixture whose job is to catch a position-blind check.
    """
    row = {
        "player_id": f"p-{position}", "player_name": "A. Back", "recent_team": "Alabama",
        "position": position, "anytime_td_prob": 0.4,
    }
    if current:
        row["is_starter"] = None
        row["depth_slot"] = None
    for market in player_props.POSITION_MARKETS[position]:
        row[market] = 1.0
    return row


def _signature() -> frozenset[str]:
    """The required keys for a week that carries every modelled position.

    Which is the only kind of sample a real week ever is: `_get_player_props_live`
    returns every player on every active team, so a real week is multi-position.
    """
    return ps._position_invariant_keys([_row(p) for p in player_props.POSITION_MARKETS])


def _pin_window(monkeypatch, *, current_week: int = 3, max_week: int = 3) -> None:
    """Pin the rebuild window to a single week, so a test states which weeks are
    reused rather than depending on the default -1/+3."""
    monkeypatch.setattr(ps, "MAX_WEEK", max_week)
    monkeypatch.setattr(ps, "REBUILD_WEEKS_BEHIND", 0)
    monkeypatch.setattr(ps, "REBUILD_WEEKS_AHEAD", 0)
    monkeypatch.setattr(ps.routes, "current_season_and_week", lambda: (2026, current_week))


class TestMismatchDetection:
    def test_a_week_missing_the_new_field_is_a_mismatch(self):
        assert ps._prop_shape_mismatch(_week(OLD_ROW), frozenset(NEW_ROW.keys())) is True

    def test_a_week_already_carrying_it_is_not(self):
        assert ps._prop_shape_mismatch(_week(NEW_ROW), frozenset(NEW_ROW.keys())) is False

    def test_a_superset_is_not_a_mismatch(self):
        # Fields the current code no longer emits must not force a rebuild
        # loop; the signature is a subset test on purpose.
        wider = {**NEW_ROW, "retired_field": 1}
        assert ps._prop_shape_mismatch(_week(wider), frozenset(NEW_ROW.keys())) is False

    def test_a_propless_week_is_never_a_mismatch(self):
        # No rows to be stale. Rebuilding it would be pure cost.
        assert ps._prop_shape_mismatch(_week(), frozenset(NEW_ROW.keys())) is False
        assert ps._prop_shape_mismatch({"player_props": []}, frozenset(NEW_ROW.keys())) is False

    def test_the_signature_is_the_position_invariant_keys_not_one_rows_shape(self):
        assert _signature() == CURRENT_ROW_KEYS
        # A single row's shape is NOT the shape, which is the whole point: a QB
        # row carries `passing_yards` and an RB row carries `rushing_yards`.
        assert frozenset(_row("QB").keys()) != _signature()
        assert _signature().isdisjoint(
            {market for markets in player_props.POSITION_MARKETS.values() for market in markets}
        )

    def test_a_row_past_row_zero_being_stale_makes_the_week_stale(self):
        # Row 0 current, rows 1-2 stale. The row-0 version of this predicate
        # called that week current and never rebuilt it -- the shipped artifact
        # came out right only because *every* row was stale, so row 0 was too.
        week = _week(_row("QB"), _row("WR", current=False), _row("RB", current=False))
        stale_rows = [i for i, row in enumerate(week["player_props"]) if not _signature() <= row.keys()]
        assert stale_rows == [1, 2]
        assert ps._prop_shape_mismatch(week, _signature()) is True

    @pytest.mark.parametrize("row_zero", ["QB", "RB", "WR", "TE"])
    def test_a_current_week_is_current_whatever_position_row_zero_is(self, row_zero):
        # The mirror case: a week that is current on every row used to be judged
        # by row 0 alone, so a WR row 0 read against a QB-derived signature said
        # "stale" on a week that needed nothing.
        week = _week(_row(row_zero), _row("QB"), _row("RB"), _row("WR"))
        assert ps._prop_shape_mismatch(week, _signature()) is False

    def test_a_row_missing_one_of_the_two_new_keys_past_row_zero_is_caught(self):
        # A field written as None can go missing on one row alone -- a partial
        # write, a hand-merged snapshot. The check is per row, so one row is
        # enough, and it is not row 0.
        damaged = _row("WR")
        del damaged["depth_slot"]
        assert ps._prop_shape_mismatch(_week(_row("QB"), _row("RB"), damaged), _signature()) is True

    def test_a_row_that_lost_its_position_specific_market_is_not_a_mismatch(self):
        # A market key follows the position, so a WR row without
        # `receiving_yards` is a player whose position changed -- five of them
        # changed between two builds of this same artifact -- not a stale shape.
        # Demanding that key anyway is finding 1 in its pure form: a permanent
        # rebuild of a week that is already current.
        damaged = _row("WR")
        damaged.pop("receiving_yards")
        assert ps._prop_shape_mismatch(_week(_row("QB"), _row("RB"), damaged), _signature()) is False


class TestSignature:
    def test_prefers_a_week_that_was_just_rebuilt(self):
        weeks = {"1": _week(OLD_ROW), "3": _week(NEW_ROW)}
        # Week 3 was rebuilt, so the current shape is known for free.
        assert ps._prop_key_signature(2026, 3, weeks, ["1"]) == frozenset(NEW_ROW.keys())

    def test_falls_back_to_one_live_probe_when_every_rebuilt_week_is_propless(self, monkeypatch):
        calls = []

        def fake(season, week):
            calls.append((season, week))
            return [NEW_ROW]

        monkeypatch.setattr(ps.routes, "_get_player_props_live", fake)
        weeks = {"1": _week(OLD_ROW), "2": _week(), "3": _week()}
        got = ps._prop_key_signature(2026, 3, weeks, ["1", "2"])
        assert got == frozenset(NEW_ROW.keys())
        assert calls == [(2026, 3)]  # exactly one probe, not one per week

    def test_returns_none_when_it_cannot_tell_rather_than_guessing(self, monkeypatch):
        def boom(season, week):
            raise RuntimeError("no models")

        monkeypatch.setattr(ps.routes, "_get_player_props_live", boom)
        # None is load-bearing. An empty frozenset would compare equal against
        # every prop-less week and report "nothing to do" -- the exact bug.
        assert ps._prop_key_signature(2026, 3, {"1": _week(), "2": _week()}, ["1", "2"]) is None

    def test_returns_none_when_the_probe_finds_no_props(self, monkeypatch):
        monkeypatch.setattr(ps.routes, "_get_player_props_live", lambda s, w: [])
        assert ps._prop_key_signature(2026, 3, {"1": _week()}, ["1"]) is None

    def test_it_intersects_every_rebuilt_week_not_just_the_first(self):
        # One rebuilt week that happens to be all-QBs and another all-WRs: the
        # pooled intersection drops both markets, so neither is then demanded of
        # every row in the season. This is the mitigation for a single-position
        # sample, and it costs nothing -- those weeks are already built.
        weeks = {"3": _week(_row("QB")), "4": _week(_row("WR"))}
        assert ps._prop_key_signature(2026, 3, weeks, []) == CURRENT_ROW_KEYS

    def test_the_probe_sample_is_intersected_too_not_just_its_first_row(self, monkeypatch):
        # The probe's row 0 is one arbitrary player, exactly like a rebuilt
        # week's row 0 -- and reading it whole is what rebuilt week 15 forever.
        calls = []

        def fake(season, week):
            calls.append((season, week))
            return [_row("WR"), _row("QB")]

        monkeypatch.setattr(ps.routes, "_get_player_props_live", fake)
        got = ps._prop_key_signature(2026, 3, {"1": _week(), "2": _week()}, ["1", "2"])
        assert got == CURRENT_ROW_KEYS
        assert calls == [(2026, 3)]  # still exactly one probe

    def test_a_reused_week_is_never_the_source_even_when_it_is_the_only_one_with_props(self, monkeypatch):
        # The signature has to describe the CURRENT code. A reused week is a copy
        # of the past, so trusting it would ratify the very shape it exists to
        # check -- and the probe has to happen instead.
        calls = []

        def fake(season, week):
            calls.append((season, week))
            return [_row("WR"), _row("QB")]

        monkeypatch.setattr(ps.routes, "_get_player_props_live", fake)
        got = ps._prop_key_signature(2026, 3, {"1": _week(OLD_ROW)}, ["1"])
        assert got == CURRENT_ROW_KEYS
        assert calls == [(2026, 3)]


class TestBuildSnapshotReconciles:
    def test_a_stale_reused_week_is_rebuilt_and_a_current_one_is_not(self, monkeypatch):
        previous = {
            "season": 2026,
            "weeks": {
                "1": _week(OLD_ROW),      # reused, stale -> should rebuild
                "2": _week(NEW_ROW),      # reused, current -> should survive untouched
                "3": _week(NEW_ROW),      # in the rebuild window
            },
        }
        built: list[int] = []

        def fake_build_week(season, week: int) -> dict:
            built.append(week)
            return _week(NEW_ROW)

        monkeypatch.setattr(ps, "_build_week", fake_build_week)
        monkeypatch.setattr(ps, "MAX_WEEK", 3)
        # Pin the window to week 3 alone, so the test states which weeks are
        # reused rather than depending on the default -1/+3.
        monkeypatch.setattr(ps, "REBUILD_WEEKS_BEHIND", 0)
        monkeypatch.setattr(ps, "REBUILD_WEEKS_AHEAD", 0)
        monkeypatch.setattr(ps.routes, "current_season_and_week", lambda: (2026, 3))
        monkeypatch.setattr(ps.routes, "_get_standings_live", lambda season: [])

        result: dict[str, Any] = ps.build_snapshot(previous)

        # 3 is in the window; 1 is stale and gets rebuilt; 2 is already current
        # and must NOT be rebuilt -- that is the cost control.
        assert built.count(1) == 1
        assert 2 not in built
        assert frozenset(result["weeks"]["1"]["player_props"][0].keys()) == frozenset(NEW_ROW.keys())

    def test_an_undeterminable_shape_skips_reconciliation_rather_than_guessing(self, monkeypatch):
        previous = {
            "season": 2026,
            "weeks": {"1": _week(OLD_ROW), "2": _week(OLD_ROW), "3": _week(OLD_ROW)},
        }

        def fake_build_week(season, week: int) -> dict:
            return _week()  # every rebuilt week is prop-less

        monkeypatch.setattr(ps, "_build_week", fake_build_week)
        monkeypatch.setattr(ps, "MAX_WEEK", 3)
        monkeypatch.setattr(ps, "REBUILD_WEEKS_BEHIND", 0)
        monkeypatch.setattr(ps, "REBUILD_WEEKS_AHEAD", 0)
        monkeypatch.setattr(ps.routes, "current_season_and_week", lambda: (2026, 3))
        monkeypatch.setattr(ps.routes, "_get_standings_live", lambda season: [])

        def boom(season, week):
            raise RuntimeError("offline")

        monkeypatch.setattr(ps.routes, "_get_player_props_live", boom)

        result = ps.build_snapshot(previous)
        # Untouched rather than wrong: the old shape is still there, and the
        # build said so out loud instead of pretending it had reconciled.
        assert "is_starter" not in result["weeks"]["1"]["player_props"][0]

    def test_a_reused_week_stale_past_row_zero_is_rebuilt(self, monkeypatch):
        # The same defect as `test_a_row_past_row_zero_being_stale_makes_the_week_stale`,
        # seen where it costs something: 2 of 3 rows are stale, row 0 is not, and
        # the old predicate left the week alone -- so two thirds of the rows
        # stayed stale for the life of the snapshot.
        previous = {
            "season": 2026,
            "weeks": {
                "1": _week(_row("QB"), _row("WR", current=False)),  # reused, row 1 stale
                "2": _week(NEW_ROW),                                  # reused, current
                "3": _week(_row("QB"), _row("WR"), _row("RB")),      # in the window
            },
        }
        built: list[int] = []

        def fake_build_week(season, week: int) -> dict:
            built.append(week)
            return _week(_row("QB"), _row("WR"), _row("RB"))

        monkeypatch.setattr(ps, "_build_week", fake_build_week)
        _pin_window(monkeypatch)

        ps.build_snapshot(previous)

        assert built.count(1) == 1
        assert 2 not in built  # the current control week is the cost control

    @pytest.mark.parametrize("row_zero", ["QB", "RB", "WR"])
    def test_a_current_reused_week_is_left_alone_whatever_row_zero_is(self, monkeypatch, row_zero):
        # And the mirror, end to end. The rebuilt week is multi-position on
        # purpose, exactly as a real one is: that is the only kind of sample a
        # position-invariant expectation can honestly be read from.
        previous = {
            "season": 2026,
            "weeks": {
                "1": _week(_row(row_zero), _row("QB"), _row("RB"), _row("WR")),  # current throughout
                "2": _week(NEW_ROW),
                "3": _week(_row("QB"), _row("WR"), _row("RB")),                 # in the window
            },
        }
        built: list[int] = []

        def fake_build_week(season, week: int) -> dict:
            built.append(week)
            return _week(_row("QB"), _row("WR"), _row("RB"))

        monkeypatch.setattr(ps, "_build_week", fake_build_week)
        _pin_window(monkeypatch)

        ps.build_snapshot(previous)

        assert 1 not in built  # row 0's position must not decide this
        assert 2 not in built


# --------------------------------------------------------------- shipped file


def _shipped() -> dict:
    return json.loads(Path(config.PUBLIC_SNAPSHOT_PATH).read_text())


def _without(week: dict, *keys: str) -> dict:
    """A week with `keys` removed from every prop row."""
    return {
        **week,
        "player_props": [{k: v for k, v in row.items() if k not in keys} for row in week["player_props"]],
    }


def _damage_last_row(week: dict, key: str) -> dict:
    """A week whose LAST prop row lost `key` -- the case a row-0 check cannot
    see, at artifact scale (row 2,498 of 2,498)."""
    rows = list(week["player_props"])
    rows[-1] = {k: v for k, v in rows[-1].items() if k != key}
    return {**week, "player_props": rows}


def _window(current_week: int) -> list[str]:
    return [
        str(week) for week in range(
            max(1, current_week - ps.REBUILD_WEEKS_BEHIND),
            min(ps.MAX_WEEK, current_week + ps.REBUILD_WEEKS_AHEAD) + 1,
        )
    ]


def _refresh(shipped: dict, monkeypatch) -> list[str]:
    """Feed a snapshot back through `build_snapshot` and record the weeks that
    got rebuilt. `_build_week` returns that snapshot's own week, so anything
    built outside the window was rebuilt for nothing -- which is what
    "idempotent" has to mean here, and what finding 1 was not."""
    built: list[str] = []

    def fake_build_week(season, week: int) -> dict:
        built.append(str(week))
        return shipped["weeks"][str(week)]

    monkeypatch.setattr(ps, "_build_week", fake_build_week)
    monkeypatch.setattr(
        ps.routes, "current_season_and_week",
        lambda: (shipped["season"], shipped["current_week"]),
    )
    return built


class TestTheShippedArtifactIsAlreadyReconciled:
    """The strongest statement available: the committed 14 MB artifact, with
    its real position mix (three distinct row shapes per prop week, 31,641
    rows) and the `current_week` 5 that puts the rebuild window on weeks 4-8.

    Every hand-built week above is single-shape or a three-row sketch, which is
    why the row-0 predicate looked right. This is the fixture that reproduces
    both directions of the bug.
    """

    def test_refreshing_it_rebuilds_nothing_outside_the_window(self, monkeypatch, capsys):
        shipped = _shipped()
        built = _refresh(shipped, monkeypatch)

        result = ps.build_snapshot(shipped)
        out = capsys.readouterr().out

        # Before the fix this read [..., "15"]: week 15 was current on all 38 of
        # its rows, and was rebuilt anyway because the signature came from week
        # 4's row 0 (a WR) and week 15's row 0 is an RB -- so the subset test
        # failed on `receiving_yards`, a market key, forever.
        assert built == _window(shipped["current_week"]) == ["4", "5", "6", "7", "8"]
        assert "reconcil" not in out  # neither "prop shape changed" nor the count line
        # The reused weeks came through with every row intact, not just the
        # window's.
        assert all(
            {"is_starter", "depth_slot"} <= row.keys()
            for week in result["weeks"].values()
            for row in week.get("player_props") or []
        )

    def test_the_two_propless_weeks_are_left_alone(self, monkeypatch):
        # Weeks 14 and 16 carry no props: no rows to be stale, so rebuilding
        # them would be pure cost. Both are inside MAX_WEEK, so nothing else
        # keeps them from being rebuilt.
        shipped = _shipped()
        assert not shipped["weeks"]["14"]["player_props"]
        assert not shipped["weeks"]["16"]["player_props"]

        built = _refresh(shipped, monkeypatch)
        ps.build_snapshot(shipped)

        assert "14" not in built and "16" not in built

    def test_a_stale_row_deep_inside_a_week_is_still_caught(self, monkeypatch):
        # The inverse, and the reason the test above can be trusted: the
        # predicate does still say "stale". One row of week 12 -- row 2,498 of
        # 2,498 -- loses `is_starter`, and week 12 must come back. Row 0 of that
        # week carries every key, so a row-0 check would have missed it.
        shipped = _shipped()
        weeks = shipped["weeks"]
        assert len(weeks["12"]["player_props"]) > 1000
        damaged = {**shipped, "weeks": {**weeks, "12": _damage_last_row(weeks["12"], "is_starter")}}

        built = _refresh(damaged, monkeypatch)
        ps.build_snapshot(damaged)

        assert built == ["4", "5", "6", "7", "8", "12"]

    def test_a_week_missing_a_market_key_is_not_stale(self, monkeypatch):
        # A market key follows the position, so losing one is not a shape
        # change: a player whose position moved between builds no longer
        # carries the old position's market. Rebuilding the week over that is
        # the same permanent quota burn as finding 1, only quieter.
        shipped = _shipped()
        weeks = shipped["weeks"]
        assert any("receiving_yards" in row for row in weeks["9"]["player_props"])
        reshaped = {**shipped, "weeks": {**weeks, "9": _without(weeks["9"], "receiving_yards")}}

        built = _refresh(reshaped, monkeypatch)
        ps.build_snapshot(reshaped)

        assert built == ["4", "5", "6", "7", "8"]


class TestNoTestHereReachesTheNetwork:
    def test_the_guard_records_and_raises(self):
        stub = _forbidden("upstream.thing")
        with pytest.raises(AssertionError):
            stub(1, kw=2)
        assert FORBIDDEN_CALLS == ["upstream.thing"]
        FORBIDDEN_CALLS.clear()  # or the autouse teardown fails on this test

    def test_every_live_builder_public_snapshot_calls_is_accounted_for(self):
        # The guard is only as good as its list. A fifth live builder added to
        # `build_snapshot` fails this rather than quietly reaching CFBD.
        source = Path(ps.__file__).read_text()
        called = set(re.findall(r"routes\.(_get_\w+_live)", source))
        assert called, "the scan matched nothing; it would pass vacuously"
        accounted = set(OFFLINE_BUILDERS) | set(FORBIDDEN_ROUTES_CALLS)
        assert called <= accounted, f"unaccounted live builders: {sorted(called - accounted)}"

    def test_no_test_in_this_file_is_marked_network(self):
        # The repo's convention (`pyproject.toml`: `network: hits a live
        # upstream API; deselect with '-m "not network"'`) is that a test which
        # needs the network says so and is skipped by default. None of these may:
        # the file has to be reproducible on a machine with no CFBD_API_KEY and
        # no warm cache, which is the machine this bug shipped from.
        module = sys.modules[__name__]
        tests = [
            (name, obj)
            for name, obj in vars(module).items()
            if name.startswith("test_") and inspect.isfunction(obj)
        ]
        for obj in list(vars(module).values()):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                tests += [
                    (f"{obj.__name__}.{name}", member)
                    for name, member in vars(obj).items()
                    if name.startswith("test_") and inspect.isfunction(member)
                ]
        assert tests, "the scan matched nothing; it would pass vacuously"
        marked = [
            name for name, fn in tests
            if any(mark.name == "network" for mark in getattr(fn, "pytestmark", []))
        ]
        assert marked == []
