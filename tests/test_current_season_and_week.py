"""The current gameweek advances on Sunday, not on the kickoff weekday.

`current_season_and_week` is the only thing that decides which week the site
shows, and it is called by the games list, the player-props board and
`/current-week`. Measuring it as `((today - anchor).days // 7) + 1` put the
rollover on whatever weekday week 1 kicked off — a Friday — so the whole site
moved to the next gameweek on Friday night with Saturday's games still to play.

CFB plays Thursday/Friday/Saturday. Sunday is therefore the first day on which
no game of the closing week remains, and advancing only then is the one
boundary that cannot strand a part-played week.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from cfb_predictor.api import routes

#: A Friday. CFB week 1 kicks off on a Friday, which is what made the old
#: 7-day-from-anchor arithmetic roll over mid-week.
KICKOFF = date(2026, 9, 4)


def _season_and_week(monkeypatch, today: date) -> tuple[int, int]:
    """Run `current_season_and_week` as if it were `today`, with a Friday kickoff."""
    schedule = pd.DataFrame({"week": [1], "gameday": [pd.Timestamp(KICKOFF)]})
    monkeypatch.setattr(routes.games_data, "fetch_schedules", lambda _s: schedule)

    class _FrozenDate(date):
        @classmethod
        def today(cls) -> date:  # noqa: D102
            return today

    monkeypatch.setattr(routes, "date", _FrozenDate)
    return routes.current_season_and_week()


def test_the_precondition_holds_this_season_kicks_off_on_a_friday() -> None:
    # If the real kickoff weekday ever changes, every assertion below is about a
    # scenario that no longer exists, so the premise is pinned rather than assumed.
    assert KICKOFF.weekday() == 4, "the fixture kickoff must be a Friday"


@pytest.mark.parametrize(
    ("offset_days", "expected_week", "why"),
    [
        (0, 1, "kickoff Friday — week 1 has barely started"),
        (1, 1, "Saturday — Saturday's games are still unplayed"),
        (2, 2, "Sunday — the first day with no game of week 1 left"),
        (7, 2, "the next Friday — week 2, not week 3"),
        (8, 2, "the next Saturday — week 2, week 2's Saturday is unplayed"),
        (9, 3, "the next Sunday — week 3"),
        (16, 4, "the Sunday after that — week 4"),
    ],
)
def test_the_week_advances_on_sunday_and_not_before(
    monkeypatch: pytest.MonkeyPatch, offset_days: int, expected_week: int, why: str
) -> None:
    _season, week = _season_and_week(monkeypatch, KICKOFF + timedelta(days=offset_days))
    assert week == expected_week, f"{KICKOFF + timedelta(days=offset_days)} ({why})"


def test_friday_and_saturday_after_kickoff_are_both_still_week_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported bug, as one assertion: the weekend does not advance the week.

    Under the old arithmetic the Saturday case returned 2 while a kickoff-day
    game was still unplayed, which is the whole complaint.
    """
    _season, friday = _season_and_week(monkeypatch, KICKOFF)
    _season, saturday = _season_and_week(monkeypatch, KICKOFF + timedelta(days=1))
    assert friday == saturday == 1


def test_a_missing_schedule_still_uses_the_fallback_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback anchor still works, and still advances on Sunday.

    `fetch_schedules` raising must not resurrect the mid-week rollover, because
    the fallback path is the one that runs when the network or cache is cold —
    exactly when a wrong week is least likely to be noticed.
    """

    def _boom(_season: list[int]) -> pd.DataFrame:
        raise RuntimeError("no schedule")

    monkeypatch.setattr(routes.games_data, "fetch_schedules", _boom)

    class _FrozenDate(date):
        @classmethod
        def today(cls) -> date:  # noqa: D102
            return date(2026, 9, 5)

    monkeypatch.setattr(routes, "date", _FrozenDate)
    season, week = routes.current_season_and_week()
    assert season == 2026
    # The fallback anchor is 2026-08-20 (a Thursday), so its closing Sunday is
    # 2026-08-23 and 2026-09-05 is inside week 3.
    assert week == 3


def test_the_week_stays_inside_the_season_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clamping still applies: never below 1, never above 20."""
    season, week = _season_and_week(monkeypatch, date(2026, 8, 1))
    assert 1 <= week <= 20, f"week {week} out of range"
    assert season == 2026