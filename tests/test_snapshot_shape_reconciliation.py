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
"""

from __future__ import annotations

from typing import Any

import pytest

from cfb_predictor import public_snapshot as ps


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
