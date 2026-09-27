"""Props must be stored against the game they describe.

`background_tracking_tick` maps a prop to a game with `team_to_game`, then
writes the pair with `INSERT OR IGNORE` — and those rows are immutable, so the
first write is the snapshot forever. If the map and the props disagree about
which week they are talking about, a wrong pair is written once and can never be
corrected.

**This file's premise changed, and the correction matters.** When it was first
written, the tick's `games` was `fetch_upcoming_games(season, week)` — one week.
The report that prompted it ("`team_to_game` is now built from this week AND
next week's games") was therefore true of neither repo *at that commit*, and
saying so was correct.

The Kalshi feed (PR #3) then made the report's first clause true:
`_games_to_snapshot` fetches `(week, week + 1)` and concatenates, because
`current_season_and_week()` rolls over on week 1's first kickoff rather than on
a fixed week boundary. So today the map really is built from two weeks of games,
and a team in both weeks is genuinely ambiguous.

What stops it is one line — `for _, g in games[games["week"] == week]` — so that
filter is the load-bearing thing, and it is what these tests pin. The first
version of this file asserted the single-week fetch instead, which was true then
and stopped being true without anyone noticing: the test was shaped like the
consumer, not like the producer. The end-to-end test at the bottom is the one
that would have caught it, and it is now the one that matters.
"""
from __future__ import annotations

import pandas as pd
import pytest

from cfb_predictor.api import routes
from cfb_predictor.data import games as games_data


def _game(game_id: str, week: int, home: str, away: str, played: bool = False) -> dict:
    score = 28 if played else None
    kickoff = f"2026-09-{(week * 7) % 28 + 1:02d}T19:00:00Z"
    return {
        "game_id": game_id, "season": 2026, "week": week,
        "home_team": home, "away_team": away,
        "home_score": score, "away_score": score,
        # The Kalshi window filters on `gameday`, so this field is load-bearing
        # now: leaving it out made this whole file raise KeyError instead of
        # testing anything.
        "gameday": kickoff,
        "commence_time": kickoff,
    }


@pytest.fixture
def store(monkeypatch):
    """A store that records instead of writing, so a wrong pairing is visible."""
    recorded: list[dict] = []

    class FakeStore:
        def record_game_predictions(self, rows):
            pass

        def record_player_prop_predictions(self, rows):
            recorded.extend(rows)

        def reconcile_resolved_games(self, games):
            pass

        def get_untracked_game_ids(self, ids):
            return []

    monkeypatch.setattr(routes, "store", FakeStore())
    # Wide enough that both weeks are inside the window regardless of when the
    # suite runs: the window is about kickoff distance, and this test is about
    # which week the pairing comes from.
    monkeypatch.setattr(routes, "SNAPSHOT_LEAD_HOURS", 24 * 30)
    return recorded


def test_the_snapshot_window_really_does_span_two_weeks(monkeypatch):
    """The report's first clause is true today. Pinning it, because the fix is
    the filter below and not the absence of the second week."""
    frame = pd.DataFrame([
        _game("W3_1", 3, "ALA", "AUB"),
        _game("W4_1", 4, "ALA", "LSU"),  # same team, next week
    ])
    monkeypatch.setattr(games_data, "fetch_schedules", lambda seasons, force_refresh=False: frame)
    monkeypatch.setattr(games_data, "CURRENT_SEASON", 2026)
    monkeypatch.setattr(games_data, "_current_season_needs_refresh", lambda: False)

    # The window filters on the upper bound only (`kickoff <= now + lead`), so
    # `now` has to be near the fixtures' kickoffs for either week to be in it.
    snapshot = routes._games_to_snapshot(
        2026, 3, pd.Timestamp("2026-09-01", tz="UTC").to_pydatetime(), lead_hours=24 * 30,
    )
    assert set(snapshot["week"]) == {3, 4}, (
        "the snapshot window no longer spans next week. If that is deliberate, the "
        "week filter below is no longer load-bearing and this file needs rewriting."
    )


def test_the_props_map_is_built_from_this_week_only(monkeypatch):
    """The load-bearing line: `for _, g in games[games["week"] == week]`.

    Without the filter, a team in both weeks maps to whichever row came last, and
    the prop is frozen against the wrong game.
    """
    both = pd.DataFrame([
        _game("W3_1", 3, "ALA", "AUB"),
        _game("W4_1", 4, "ALA", "LSU"),
    ])
    team_to_game = {}
    for _, g in both[both["week"] == 3].iterrows():
        team_to_game[g["home_team"]] = g["game_id"]
        team_to_game[g["away_team"]] = g["game_id"]

    assert team_to_game == {"ALA": "W3_1", "AUB": "W3_1"}, (
        f"the map is ambiguous across weeks: {team_to_game}"
    )


def test_a_team_playing_both_weeks_is_not_stored_against_next_week(monkeypatch, store):
    """End to end through background_tracking_tick.

    ALA plays in week 3 and week 4. The tick runs for week 3 and the snapshot
    window returns both games. **ALA's** prop must be stored against week 3's
    game: those rows are INSERT OR IGNORE, so a wrong pairing is permanent.

    The prop belongs to ALA and not to AUB, and that is the whole test. AUB
    plays only in week 3, so its prop lands on week 3's game whether or not the
    week filter exists — which is why the first version of this test passed
    with the filter deleted, and was proving nothing.
    """
    both = pd.DataFrame([
        _game("W3_1", 3, "ALA", "AUB"),
        _game("W4_1", 4, "ALA", "LSU"),
    ])
    monkeypatch.setattr(games_data, "fetch_upcoming_games", lambda season, week: both[both["week"] == week])
    monkeypatch.setattr(games_data, "fetch_schedules", lambda seasons, force_refresh=False: both)
    monkeypatch.setattr(games_data, "CURRENT_SEASON", 2026)
    monkeypatch.setattr(games_data, "_current_season_needs_refresh", lambda: False)
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {})
    monkeypatch.setattr(routes, "_load_game_history", lambda season: pd.DataFrame())
    monkeypatch.setattr(routes, "_predict_game_from_models", lambda *a, **k: {"home_win_probability": 0.6})
    monkeypatch.setattr(
        routes, "_get_player_props_live",
        lambda season, week: [{
            "player_id": "1", "player_name": "A Player", "position": "WR",
            # The team in BOTH weeks. See the docstring: a prop for a team that
            # appears once cannot distinguish the two cases.
            "recent_team": "ALA", "anytime_td_prob": 0.4,
        }],
    )

    routes.background_tracking_tick(2026, 3)

    assert store, "the tick stored no props at all, so this proves nothing"
    assert [r["recent_team"] if "recent_team" in r else r.get("player_name") for r in store], store
    for row in store:
        assert row["game_id"] == "W3_1", (
            f"a week-3 prop was stored against {row['game_id']!r}; those rows are "
            f"immutable, so the wrong pairing is permanent"
        )


def test_the_props_fallback_still_filters_to_the_same_week(monkeypatch):
    """The one other place the two sides can disagree. `_get_player_props_live`
    falls back to the season's completed games, which spans every week; it
    re-filters to `week`, and removing that re-filter is what would mispair."""
    this_week = pd.DataFrame([_game("W3_1", 3, "ALA", "AUB", played=True)])
    next_week = pd.DataFrame([_game("W4_1", 4, "ALA", "LSU", played=True)])
    monkeypatch.setattr(
        games_data, "fetch_current_season_partial",
        lambda: pd.concat([this_week, next_week], ignore_index=True),
    )

    fallback = games_data.fetch_current_season_partial()
    fallback = fallback[fallback["week"] == 3]
    assert set(fallback["game_id"]) == {"W3_1"}
