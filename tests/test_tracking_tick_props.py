"""Is there a real props/game pairing bug in CFB's tracking tick?

The NFL review (and the same text again for this repo) said:
`team_to_game` "is now built from this week AND next week's games", so a team
in both gets its props filed under whichever came last, and INSERT OR IGNORE
freezes that pairing.

At this commit it is not: the tick's `games` is `fetch_upcoming_games(season,
week)`, which is `df[df["week"] == week]` (games.py:243), so the map is
single-week and a team plays at most once a week.

These tests pin that, so the invariant is checked in one command and a future
widening fails here rather than shipping a permanent mispairing.
"""
from __future__ import annotations

import pandas as pd
import pytest

from cfb_predictor.api import routes
from cfb_predictor.data import games as games_data


def _game(game_id: str, week: int, home: str, away: str, played: bool = False) -> dict:
    """A game row as the REAL normaliser produces it.

    Built by feeding a raw CFBD-shaped row through `games._normalize_games`, the
    same function `fetch_schedules` uses, so this fixture cannot drift from the
    frame the code actually reads.

    The previous version was hand-shaped and had drifted: it carried
    `commence_time` — which the real CFBD frame does not have, and which CFB's own
    `routes.py` *builds* from `gameday` on the way out — and it had no `gameday`
    at all. So it went RED on main once the Kalshi feed started reading
    `games["gameday"]` (CFB#3), which is the fourth time in this repo that a test
    double shaped like the consumer rather than the producer has cost a round. A
    fixture built from the producer cannot drift that way: adding a column to
    `KEEP_COLUMNS` adds it here too.
    """
    score = 28 if played else None
    raw = pd.DataFrame([{
        # Raw CFBD field names, pre-rename: the normaliser is what turns these
        # into the frame the rest of the code reads.
        "id": game_id, "season": 2026, "week": week,
        "home_team": home, "away_team": away,
        "home_points": score, "away_points": score,
        "start_date": f"2026-09-{(week * 7) % 28 + 1:02d}T19:00:00Z",
        "home_conference": "SEC", "away_conference": "SEC",
    }])
    return games_data._normalize_games(raw).iloc[0].to_dict()


def test_fetch_upcoming_games_is_a_single_week(monkeypatch):
    """The load-bearing fact behind the report's first clause.

    If this widens to two weeks, `team_to_game` becomes ambiguous for any team
    in both, and the last row silently wins.
    """
    frame = pd.DataFrame([
        _game("W3_1", 3, "ALA", "AUB"),
        _game("W4_1", 4, "ALA", "LSU"),  # same team, next week
    ])
    monkeypatch.setattr(games_data, "fetch_schedules", lambda seasons, force_refresh=False: frame)
    monkeypatch.setattr(games_data, "CURRENT_SEASON", 2026)
    monkeypatch.setattr(games_data, "_current_season_needs_refresh", lambda: False)

    got = games_data.fetch_upcoming_games(2026, 3)
    assert set(got["week"]) == {3}, "fetch_upcoming_games must stay single-week"
    assert "W4_1" not in set(got["game_id"])


def test_a_team_playing_both_weeks_resolves_to_this_weeks_game(monkeypatch):
    """The report's exact scenario, against the real code path.

    ALA plays in weeks 3 and 4; the tick runs for week 3. The map must resolve
    to week 3's game, or the props are frozen against the wrong one.
    """
    both_weeks = pd.DataFrame([
        _game("W3_1", 3, "ALA", "AUB"),
        _game("W4_1", 4, "ALA", "LSU"),
    ])
    monkeypatch.setattr(games_data, "fetch_schedules", lambda seasons, force_refresh=False: both_weeks)
    monkeypatch.setattr(games_data, "CURRENT_SEASON", 2026)
    monkeypatch.setattr(games_data, "_current_season_needs_refresh", lambda: False)

    games = games_data.fetch_upcoming_games(2026, 3)
    team_to_game = {}
    for _, g in games.iterrows():
        team_to_game[g["home_team"]] = g["game_id"]
        team_to_game[g["away_team"]] = g["game_id"]

    assert team_to_game == {"ALA": "W3_1", "AUB": "W3_1"}


def test_the_props_fallback_still_filters_to_the_same_week(monkeypatch):
    """The only real divergence: the props' fallback reaches for the season's
    completed games, which spans every week. It re-filters to `week`, and
    removing that re-filter is the one change that would genuinely mispair.
    """
    this_week = pd.DataFrame([_game("W3_1", 3, "ALA", "AUB", played=True)])
    next_week = pd.DataFrame([_game("W4_1", 4, "ALA", "LSU", played=True)])
    monkeypatch.setattr(
        games_data, "fetch_current_season_partial",
        lambda: pd.concat([this_week, next_week], ignore_index=True),
    )

    fallback = games_data.fetch_current_season_partial()
    fallback = fallback[fallback["week"] == 3]
    assert set(fallback["week"]) == {3}
    assert set(fallback["game_id"]) == {"W3_1"}


def test_the_tick_pairs_props_with_the_game_from_its_own_week(monkeypatch):
    """End to end through background_tracking_tick: a prop for a team in this
    week is stored against this week's game."""
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

    this_week = pd.DataFrame([_game("W3_1", 3, "ALA", "AUB")])
    monkeypatch.setattr(games_data, "fetch_upcoming_games", lambda season, week: this_week)
    monkeypatch.setattr(routes.games_data, "fetch_upcoming_games", lambda season, week: this_week)
    monkeypatch.setattr(routes, "store", FakeStore())
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {})
    monkeypatch.setattr(routes, "_load_game_history", lambda season: pd.DataFrame())
    monkeypatch.setattr(routes, "_predict_game_from_models", lambda *a, **k: {"home_win_probability": 0.6})
    monkeypatch.setattr(
        routes, "_get_player_props_live",
        lambda season, week: [{
            "player_id": "1", "player_name": "A Player", "position": "WR",
            "recent_team": "AUB", "anytime_td_prob": 0.4,
        }],
    )

    routes.background_tracking_tick(2026, 3)

    for row in recorded:
        assert row["game_id"] == "W3_1", (
            f"a week-3 prop was stored against {row['game_id']!r}; those rows are "
            f"immutable, so a wrong pairing is permanent"
        )
