"""`is_starter` reaches the CFB player props -- as `None`, because CFBD has no such field.

NFL threads a real starter flag: it has a depth chart, and `None` there means
"the chart was unavailable". CFB has no equivalent feed, so the honest value is
`None` on *every* row, permanently. That is not an unfinished feature. Verified
three independent ways against the installed `cfbd` 5.31.0 on 2026-09-28:

  1. Live `GamesApi.get_game_player_stats(year=2026, week=4)` -- 123 games,
     64,007 athlete rows, and the set of athlete key sets is exactly
     `{('id', 'name', 'stat')}`. No depth field on any of them.
  2. `grep -rniE 'starter|is_starter|did_play|home_start|away_start'` over the
     installed package: 0 hits. (Control: `athlete` -> 114.)
  3. `cfbd/models/game_player_stat_player.py:32` declares
     `__properties = ["id", "name", "stat"]`.

The distinction this file exists to protect is `None` vs `False`. `None` means
"this sport has no depth-chart data" and the UI renders a visible
"Projected order -- no depth-chart feed" line. `False` means "this player is
known to be on the bench" and renders a bench row. Only the first is a true
statement about what this project knows. Asserting `is_starter is False` would
let a data source that carries no lineup information assert a lineup, on every
row, forever.

The second temptation is a heuristic, and the flattener's comment says so at the
place a future session would go to add one. `tests/test_no_starter_heuristic_was_invented`
pins the decision mechanically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cfb_predictor.api import routes
from cfb_predictor.data import player_stats


class TestTheFieldIsPresentAndHonest:
    def test_a_prop_row_carries_both_keys(self, monkeypatch):
        rows = _props_for(monkeypatch, ["A. Back", "B. Receiver"])
        assert len(rows) == 2
        for row in rows:
            assert "is_starter" in row
            assert "depth_slot" in row

    def test_both_are_None_not_False(self, monkeypatch):
        # The whole point. `is_starter is None` is the truth; `False` would be
        # an assertion that these players are known to be benched, made by a
        # data source that has no lineup data at all.
        rows = _props_for(monkeypatch, ["A. Back", "B. Receiver"])
        for row in rows:
            assert row["is_starter"] is None
            assert row["depth_slot"] is None

    def test_the_field_pair_matches_NFL_so_no_per_sport_branch_is_needed(self):
        # The reason for adding `depth_slot` alongside `is_starter`: a shared
        # frontend reads both without knowing which sport it is looking at.
        assert set(NFL_FIELD_PAIR) == {"is_starter", "depth_slot"}

    def test_the_starter_value_is_never_a_bool_or_a_number(self, monkeypatch):
        rows = _props_for(monkeypatch, ["A. Back"])
        # Belt and braces on the None/False distinction: `False == 0` in Python,
        # so a plain `==` assertion would let False through.
        assert rows[0]["is_starter"] is not False
        assert type(rows[0]["is_starter"]) is type(None)


class TestNoPlaceholderRows:
    def test_an_empty_props_list_stays_empty(self, monkeypatch):
        # Making the key "always present" by appending a placeholder row would
        # be a fabrication of a player, and would put a ghost in the public API
        # and in the committed snapshot. An empty week must stay empty.
        monkeypatch.setattr(routes, "_load_models_cached", lambda: {"player_models": object()})
        monkeypatch.setattr(routes, "_load_player_history", lambda season: pd.DataFrame(
            [{"player_id": "1", "player_name": "A. Back", "position": "RB",
              "recent_team": "Alabama", "season": 2026}]
        ))
        monkeypatch.setattr(
            routes.games_data, "fetch_upcoming_games",
            lambda season, week: pd.DataFrame([{"home_team": "Alabama", "away_team": "Auburn", "week": week}]),
        )
        monkeypatch.setattr(
            routes.games_data, "fetch_current_season_partial",
            lambda: pd.DataFrame(columns=["week"]),
        )
        # No feature history at all -> every player is skipped.
        monkeypatch.setattr(routes.player_usage, "build_features_for_player", lambda *a, **k: None)

        assert routes._get_player_props_live(2026, 3) == []

    def test_a_week_with_no_games_returns_an_empty_list_not_a_placeholder(self, monkeypatch):
        monkeypatch.setattr(routes, "_load_models_cached", lambda: {"player_models": object()})
        monkeypatch.setattr(routes, "_load_player_history", lambda season: pd.DataFrame(
            [{"player_id": "1", "player_name": "A. Back", "position": "RB",
              "recent_team": "Alabama", "season": 2026}]
        ))
        monkeypatch.setattr(
            routes.games_data, "fetch_upcoming_games",
            lambda season, week: pd.DataFrame(columns=["home_team", "away_team", "week"]),
        )
        monkeypatch.setattr(
            routes.games_data, "fetch_current_season_partial",
            lambda: pd.DataFrame(columns=["home_team", "away_team", "week"]),
        )

        assert routes._get_player_props_live(2026, 3) == []


class TestNoStarterHeuristicWasInvented:
    """The flattener is where a future session would go to "fix" the `None`.

    Both available heuristics are wrong, and the code comment at the athlete
    flatten says so. These assertions make that structural rather than a
    matter of reading the comment: if a `is_starter`/`depth_slot` column ever
    appears in the flattened frame, the invention happened.
    """

    def test_the_flattened_frame_has_no_starter_or_depth_column(self):
        flat = _flatten([{"id": 5079691, "name": "A. Back", "stat": "100"}])
        assert "is_starter" not in flat.columns
        assert "depth_slot" not in flat.columns

    def test_the_kept_columns_list_does_not_declare_one(self):
        assert "is_starter" not in player_stats.KEEP_COLUMNS
        assert "depth_slot" not in player_stats.KEEP_COLUMNS

    def test_a_backup_outranking_a_starter_is_indistinguishable_in_the_flattened_frame(self):
        """Why heuristic #1 ("first in the athletes list") is wrong.

        Measured live on 2026-09-28 (week 4, 123 games): within a stat type
        CFBD returns athletes in descending order **of that stat's value** --
        108/108 passing/YDS groups, 246/246 receiving/YDS, 246/246 rushing/YDS.
        So list position is a ranking of this week's production, not a depth
        chart. A backup who happens to lead the team in a category ranks first.

        And the ordering is not even a consistent rule across stat types --
        receiving/REC is descending in only 19.5% of groups, rushing/CAR in
        23.6% -- so it cannot be leaned on as any kind of depth signal.

        What is asserted here is the structural consequence: the two players are
        byte-identical in the flattened frame. Nothing downstream could tell
        them apart, and inventing a distinction would be a fabrication.
        """
        backup_leads = _flatten([
            {"id": 1, "name": "Real Starter", "stat": "10"},
            {"id": 2, "name": "Backup", "stat": "140"},
        ])
        assert set(backup_leads["player_name"]) == {"Real Starter", "Backup"}
        assert backup_leads.shape[1] == len(player_stats.KEEP_COLUMNS)
        assert "is_starter" not in backup_leads.columns

    def test_bench_appearances_survive_the_team_row_guard_and_still_carry_no_flag(self):
        """Why heuristic #2 ("appeared in the box score") is wrong.

        Measured live on 2026-09-28 (week 4, 246 teams): the median team has 35
        distinct positive-id athletes in its box score (min 26, max 56) against
        an 11-man starting lineup, and the same payload carried 741 `" Team"`
        rows with negative ids. The appearance set is starters plus every backup
        who checked in, plus a team total.

        Filtering the team row out -- which the ingest already does, see
        tests/test_team_rows_are_not_players.py -- still leaves every backup in
        the set, so membership does not imply a start. And `is_real_player_id`
        deliberately keys on the id rather than the name, so a `" Team"` row is
        not the thing that makes this heuristic unusable; the backups are.
        """
        flat = _flatten([
            {"id": 1, "name": "Real Starter", "stat": "10"},
            {"id": 2, "name": "Backup", "stat": "3"},
            {"id": -7352, "name": " Team", "stat": "13"},
        ])
        # The bench player survives the team-row guard -- that is the point.
        assert set(flat["player_name"]) == {"Real Starter", "Backup"}
        assert "is_starter" not in flat.columns
        assert "depth_slot" not in flat.columns


class TestTheCommittedSnapshot:
    def test_the_artifact_carries_the_keys(self):
        # A field missing from public_snapshot.json is invisible in production:
        # public mode is what the VPS serves. A green unit test on the live
        # route proves nothing about what is deployed.
        snapshot = _committed_snapshot()
        rows = [row for week in snapshot["weeks"].values() for row in (week.get("player_props") or [])]
        assert rows, "the committed snapshot has no prop rows at all; this test is vacuous"
        for row in rows:
            assert "is_starter" in row, "a prop row in the artifact is missing is_starter"
            assert "depth_slot" in row, "a prop row in the artifact is missing depth_slot"

    def test_the_artifact_values_are_null_not_false(self):
        snapshot = _committed_snapshot()
        for week in snapshot["weeks"].values():
            for row in week.get("player_props") or []:
                assert row["is_starter"] is None
                assert row["depth_slot"] is None

    def test_the_artifact_is_strict_json(self):
        # `json.dumps` defaults to allow_nan=True, so a non-finite float is
        # written as a bare NaN token -- which is a 500 on the endpoint, not a
        # wrong number. Asserted rather than assumed.
        _reject = lambda name: (_ for _ in ()).throw(AssertionError(f"bare {name} in snapshot"))
        text = _snapshot_path().read_text()
        assert "NaN" not in text
        json.loads(text, parse_constant=_reject)


NFL_FIELD_PAIR = ("is_starter", "depth_slot")


def _snapshot_path() -> Path:
    from cfb_predictor import config

    return Path(config.PUBLIC_SNAPSHOT_PATH)


def _committed_snapshot() -> dict:
    path = _snapshot_path()
    if not path.exists():
        pytest.skip("no committed snapshot in this checkout")
    return json.loads(path.read_text())


def _flatten(athletes: list[dict]) -> pd.DataFrame:
    raw = [{
        "id": 401520145,
        "teams": [{
            "team": "Alabama",
            "categories": [{
                "name": "rushing",
                "types": [{"name": "YDS", "athletes": athletes}],
            }],
        }],
    }]
    games = pd.DataFrame([{"game_id": "401520145", "season": 2026, "week": 3}])
    return player_stats._flatten_player_game_stats(raw, games, 2026)


def _props_for(monkeypatch, names: list[str]) -> list[dict]:
    """Drive `_get_player_props_live` with the model and schedule stubbed out.

    Only the parts that are not under test are replaced; the prop-row assembly
    runs for real, so a change to the row shape cannot pass by stubbing it.
    """
    def fake_predict(models, feature_row, position=None):
        return {"rushing_yards": 99.0, "anytime_td_prob": 0.42}

    players = pd.DataFrame([
        {"player_id": f"p{i}", "player_name": n, "position": "RB",
         "recent_team": "Alabama", "season": 2026}
        for i, n in enumerate(names)
    ])
    games = pd.DataFrame([{"home_team": "Alabama", "away_team": "Auburn", "week": 3}])

    monkeypatch.setattr(routes, "_load_models_cached", lambda: {"player_models": object()})
    monkeypatch.setattr(routes, "_load_player_history", lambda season: players)
    monkeypatch.setattr(routes.games_data, "fetch_upcoming_games", lambda season, week: games)
    monkeypatch.setattr(
        routes.player_usage, "build_features_for_player",
        lambda *a, **k: pd.DataFrame({"x": [1]}),
    )
    monkeypatch.setattr(routes.player_props, "predict_props", fake_predict)

    return routes._get_player_props_live(2026, 3)
