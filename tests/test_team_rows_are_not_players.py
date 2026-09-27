"""Team-level rows from CFBD are not players, and must never become one.

CFBD's game-stats payload carries team totals in the same `athletes` list as
players: an entry with a **negative** id and the name `" Team"`. The ingest walks
that list without checking, so the junk becomes a player row, gets a
`player_id` the model happily predicts against, and ships — in the API response
and in the committed public snapshot.

Measured on the committed snapshot, three consecutive weeks, zero exceptions:

    week 3   2663 rows   125 junk
    week 4   2206 rows   100 junk
    week 5   2120 rows    97 junk

and the two signals coincide **perfectly**: every bare-negative-id row is named
`" Team"`, and every `" Team"` row has a bare-negative id. Real CFB athlete ids
are positive integers (`5079691`). NFL, whose data source has no team-level
rows, has none of this — which is why the same code path is clean there and
this is a CFB-only defect.

Why it matters beyond tidiness: the Sports frontend grew an `isRealPlayer` name
check to hide these, which treats the symptom. The junk is still in the public
API and in an 11 MB committed snapshot, and it still costs a prediction each.
The name check is one CFBD rename away from failing open.

The load-bearing invariant is the **id**, not the name: a name is a display
string CFBD can change, and a negative id is not a player by construction.

The payload below is the same shape `tests/test_player_stats.py` already uses —
scalar `stat` per athlete, real CFBD type names — because the recurring bug in
this repo has been a double shaped like the consumer rather than the producer,
and the first version of this file guessed at the payload and could not have
failed against the buggy code.
"""
from __future__ import annotations

import pandas as pd
import pytest

from cfb_predictor.data import player_stats

# The real player, and the team-level pseudo-athlete CFBD puts in the same list.
_REAL = {"id": 5079691, "name": "J. Doe", "stat": "291"}
_TEAM_ROW = {"id": -7352, "name": " Team", "stat": "291"}


def _raw(athletes: list[dict]) -> list[dict]:
    return [
        {
            "id": 401520145,
            "teams": [
                {
                    "team": "South Carolina",
                    "categories": [
                        {
                            "name": "passing",
                            "types": [{"name": "YDS", "athletes": athletes}],
                        },
                    ],
                }
            ],
        }
    ]


def _games_df() -> pd.DataFrame:
    return pd.DataFrame([{"game_id": "401520145", "season": 2026, "week": 3,
                          "gameday": pd.Timestamp("2026-09-12")}])


def _ingest(monkeypatch, tmp_path, athletes: list[dict]) -> pd.DataFrame:
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(player_stats, "_import_player_game_stats", lambda season, weeks: _raw(athletes))
    # (seasons, games_df) -- the real signature. Passing (2026, [3]) made
    # `seasons` an int and raised "int object is not iterable", which is exactly
    # the harness-guessing this repo's own recurring bug is about.
    return player_stats.fetch_weekly_player_stats([2026], _games_df())


def test_a_team_level_row_never_becomes_a_player(monkeypatch, tmp_path):
    """The regression: 125 junk players in week 3 of the committed snapshot."""
    rows = _ingest(monkeypatch, tmp_path, [_REAL, _TEAM_ROW])

    assert "J. Doe" in set(rows["player_name"]), (
        f"the real player was dropped too: {sorted(set(rows['player_name']))}"
    )
    junk = [n for n in rows["player_name"] if str(n).strip().lower() == "team"]
    assert not junk, (
        f"{len(junk)} team-level rows were ingested as players. CFBD puts team totals in the "
        f"same athletes list with a negative id, and the model then predicts anytime-TD and "
        f"yardage for them — which is how 97 of them reached the public API."
    )


def test_the_real_player_survives_with_its_own_id(monkeypatch, tmp_path):
    """The fix must not be "drop anything that looks odd"."""
    rows = _ingest(monkeypatch, tmp_path, [_REAL, _TEAM_ROW])
    real = rows[rows["player_name"] == "J. Doe"]
    assert len(real) == 1
    assert str(real.iloc[0]["player_id"]) == "5079691"


def test_the_invariant_is_the_id_not_the_name(monkeypatch, tmp_path):
    """A negative id is not a player whatever it is called.

    The name is a display string CFBD can change; the id cannot become positive
    by accident. So this renames the junk row to something perfectly plausible
    and asserts it is still rejected — if the guard ever starts pattern-matching
    on `" Team"`, this fails.
    """
    renamed = {**_TEAM_ROW, "name": "Team Total"}
    rows = _ingest(monkeypatch, tmp_path, [_REAL, renamed])
    assert "Team Total" not in set(rows["player_name"]), (
        "the guard matched the NAME rather than the id; a CFBD rename would put team "
        "totals straight back in the player list"
    )


def test_several_positive_id_players_are_all_kept(monkeypatch, tmp_path):
    """The negative case, so the guard cannot be satisfied by dropping everything."""
    athletes = [
        _REAL,
        {"id": 5301576, "name": "R. Back", "stat": "112"},
        {"id": 5296832, "name": "W. Out", "stat": "85"},
    ]
    rows = _ingest(monkeypatch, tmp_path, athletes)
    assert {"J. Doe", "R. Back", "W. Out"} <= set(rows["player_name"])
