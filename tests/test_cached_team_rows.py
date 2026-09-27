"""Junk already in the cache must not reach the hub, not just junk newly fetched.

`player_stats` now drops CFBD's team totals at ingest (CFB#7), but the VPS
mounts `./volumes/cfb-cache` at `/app/data/cache`, and that parquet was written
*before* the fix. Ingestion never revisits it, so the fix alone changed nothing
about what the live service already had:

    GET /api/cfb/hub/players?season=2026  ->  254 negative-id " Team" rows of 5534

So the filter also has to sit where **cached** rows are read, not only where new
ones are fetched. That is defence in depth doing real work rather than ceremony:
ingest protects the next fetch, the read path protects everything already on
disk, and each would still be right if the other were deleted.

The same read serves `/players/{season}/{week}/props`, so one filter covers both
endpoints. `player_hub` builds its leaderboards from the same frame, so the
leaderboards are covered by the same filter rather than needing a second one.
"""
import pandas as pd
import pytest

from cfb_predictor.api import routes
from cfb_predictor.data import games as games_data
from cfb_predictor.data import player_stats

# The real player, and the team-level pseudo-athlete that a cache written before
# CFB#7 still contains. The pair is measured off the live response, not guessed.
_REAL = {
    "player_id": "5079691", "player_name": "J. Doe", "position": "QB",
    "recent_team": "South Carolina", "season": 2025, "week": 1,
    "passing_yards": 250.0, "passing_tds": 2, "rushing_yards": 10.0, "rushing_tds": 0,
    "receiving_yards": 0.0, "receiving_tds": 0, "receptions": 0, "targets": None, "carries": 3,
}
_TEAM_ROW = {**_REAL, "player_id": "-7352", "player_name": " Team"}


def _cached(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _games_df() -> pd.DataFrame:
    return pd.DataFrame([{"game_id": "401520145", "season": 2025, "week": 1,
                          "gameday": pd.Timestamp("2025-08-30")}])


def test_a_negative_id_is_not_a_player_id():
    """The predicate the ingest and the read path share.

    Kept as a name rather than inlined twice, because the whole point of the
    read-path filter is that it agrees with the ingest filter. Two copies of
    `int(raw) > 0` would agree today and drift the first time one is edited.
    """
    assert player_stats.is_real_player_id(5079691) is True
    assert player_stats.is_real_player_id("5079691") is True
    assert player_stats.is_real_player_id(-7352) is False
    assert player_stats.is_real_player_id("-7352") is False
    # Not a number at all, and the two ways a real id can be spelled.
    assert player_stats.is_real_player_id(None) is False
    assert player_stats.is_real_player_id("") is False
    assert player_stats.is_real_player_id("n/a") is False
    assert player_stats.is_real_player_id("00-0023459") is False, (
        "a hyphenated id is not a CFB athlete id; accepting it would let a "
        "different feed's rows through the filter unchecked"
    )


def test_dropping_team_rows_keeps_the_frame_otherwise_untouched(monkeypatch):
    df = _cached(_REAL, _TEAM_ROW)
    out = player_stats.drop_team_rows(df)
    assert list(out["player_name"]) == ["J. Doe"]
    # Nothing else about the surviving rows may move, or this filter would be a
    # silent data change wearing a cleanup's clothes.
    assert out.iloc[0].to_dict() == _REAL


def test_a_frame_with_no_player_id_column_is_returned_unchanged(monkeypatch):
    """A frame without the column is not a frame with bad ids.

    Returning it as-is keeps the filter safe to call from a path that has not
    been inspected, which is the only way to be sure it is safe to call from two.
    """
    df = pd.DataFrame([{"name": "no id here"}])
    assert player_stats.drop_team_rows(df) is df


def test_cached_team_rows_never_reach_the_hub(monkeypatch):
    """The regression: 254 rows of 5534 on the live endpoint.

    Drives the real `_load_player_history` with a cache seeded to look like the
    one on the VPS, rather than replacing the function under test — stubbing
    `_load_player_history` itself, as the existing props tests do, would pass
    with the filter deleted.
    """
    monkeypatch.setattr(games_data, "default_completed_seasons", lambda n=8: [2024])
    monkeypatch.setattr(games_data, "load_training_data", lambda seasons: _games_df())
    monkeypatch.setattr(player_stats, "fetch_weekly_player_stats",
                        lambda seasons, games_df, force_refresh=False: _cached(_REAL, _TEAM_ROW))

    history = routes._load_player_history(2025)

    assert list(history["player_name"]) == ["J. Doe"], (
        f"the cached team row survived the read: {sorted(set(history['player_name']))}"
    )
    assert "5079691" in set(history["player_id"])


def test_the_hub_response_has_no_team_rows_from_a_seeded_cache(monkeypatch):
    """The endpoint, not just the read — leaderboards included.

    `player_hub` derives its leaderboards from the same frame, so this is where a
    second, separate leak would show up: a filtered `players` list beside a
    leaderboard still naming a team.
    """
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {})
    monkeypatch.setattr(routes.advanced_stats, "fetch_player_ppa", lambda season: [])
    monkeypatch.setattr(games_data, "default_completed_seasons", lambda n=8: [2024])
    monkeypatch.setattr(games_data, "load_training_data", lambda seasons: _games_df())
    monkeypatch.setattr(player_stats, "fetch_weekly_player_stats",
                        lambda seasons, games_df, force_refresh=False: _cached(_REAL, _TEAM_ROW))
    monkeypatch.setattr(routes, "PUBLIC_MODE", False)

    out = routes.get_hub_players(season=2025)

    ids = [p["player_id"] for p in out["players"]]
    assert "-7352" not in ids, f"a team row reached the hub response: {ids}"
    for pos, entries in out["leaderboards"].items():
        for entry in entries:
            assert not str(entry["player_id"]).startswith("-"), (
                f"the {pos} leaderboard still names a team: {entry}"
            )
