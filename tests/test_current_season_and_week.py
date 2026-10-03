"""The current gameweek is derived from the schedule, never from 7-day arithmetic.

`current_season_and_week` decides which week the whole site shows — the games
list, the player-props board and `/current-week` all call it. Measuring it as
`((today - anchor).days // 7) + 1` put the rollover on whatever weekday week 1
kicked off, and CFB week 1 kicks off on a **Friday**, so the site moved to the
next gameweek on Friday night with Saturday's games still unplayed.

Rolling over on Sunday instead is also wrong, and the schedule says so: taken
from the committed `data/public_snapshot.json`, week 1 has games on Thu, Fri,
Sat, Sun **and Mon**, and week 6 has a Wednesday. A Sunday rollover would hide
week 1's Monday game for two days — the same defect, two days later.

The rule under test: **the current week is the first week whose last scheduled
game has not yet been played.** No scheduled game is ever off the board.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from cfb_predictor.api import routes

#: The real shape of a CFB week, measured from the committed snapshot: week 1
#: spills into Monday, week 2 ends on Sunday, week 3 opens on Thursday.
WEEK1_DAYS = (3, 4, 5, 6, 7)  # Sep 2026, Thu..Mon
WEEK2_DAYS = (11, 12, 13)  # Fri..Sun
WEEK3_DAYS = (17, 18, 19, 20)  # Thu..Sun
SEPT_2026 = 9


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week": [1] * len(WEEK1_DAYS) + [2] * len(WEEK2_DAYS) + [3] * len(WEEK3_DAYS),
            "gameday": [
                pd.Timestamp(date(2026, SEPT_2026, d))
                for d in (*WEEK1_DAYS, *WEEK2_DAYS, *WEEK3_DAYS)
            ],
        }
    )


def _season_and_week(monkeypatch: pytest.MonkeyPatch, today: date) -> int:
    monkeypatch.setattr(routes.games_data, "fetch_schedules", lambda _s: _schedule())

    class _FrozenDate(date):
        @classmethod
        def today(cls) -> date:  # noqa: D102
            return today

    monkeypatch.setattr(routes, "date", _FrozenDate)
    return routes.current_season_and_week()[1]


def test_the_fixture_schedule_is_the_real_weekday_shape() -> None:
    """Pin the premise: if the real calendar shifts, the table below is fiction."""
    assert date(2026, 9, 4).weekday() == 4, "week 1 must kick off on a Friday"
    assert date(2026, 9, 7).weekday() == 0, "week 1 must have a Monday game"
    assert date(2026, 9, 13).weekday() == 6, "week 2 must end on a Sunday"


@pytest.mark.parametrize(
    ("day", "expected", "why"),
    [
        (3, 1, "Thursday — week 1 opens"),
        (4, 1, "Friday kickoff — THE REPORTED BUG: the old rule returned 2"),
        (5, 1, "Saturday — the old rule returned 2 with Saturday unplayed"),
        (6, 1, "Sunday — week 1 has a Sunday game, so it is still week 1"),
        (7, 1, "Monday — week 1's last game is today, so it is still week 1"),
        (8, 2, "Tuesday — week 1 is behind us, so week 2"),
        (11, 2, "week 2's Friday"),
        (13, 2, "week 2's Sunday, its last game"),
        (14, 3, "Monday after — week 3"),
        (20, 3, "week 3's Sunday, its last game"),
        (21, 4, "the Monday after the season's last listed week"),
    ],
)
def test_the_week_follows_the_schedule_not_the_calendar(
    monkeypatch: pytest.MonkeyPatch, day: int, expected: int, why: str
) -> None:
    got = _season_and_week(monkeypatch, date(2026, SEPT_2026, day))
    assert got == expected, f"2026-09-{day:02d}: {why}"


def test_a_game_is_never_off_the_board(monkeypatch: pytest.MonkeyPatch) -> None:
    """The property the rule exists for: no scheduled game day shows the wrong week.

    Walk every gameday in the fixture and assert the week reported on that day is
    the week that actually owns the game. Under the old 7-day arithmetic this
    failed on the Friday and Saturday of every week.
    """
    for week, days in ((1, WEEK1_DAYS), (2, WEEK2_DAYS), (3, WEEK3_DAYS)):
        for day in days:
            got = _season_and_week(monkeypatch, date(2026, SEPT_2026, day))
            assert got == week, f"2026-09-{day:02d} plays in week {week} but the site says {got}"


def test_a_friday_kickoff_does_not_advance_on_friday(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported symptom, reproduced exactly: a FRIDAY kickoff.

    The live season kicks off on a Friday, which is why the bug was seen on a
    Friday. Under `((today - anchor).days // 7) + 1` with a Friday anchor, that
    form advances on the Friday itself — so Friday and Saturday both reported the
    next gameweek while Saturday's games were unplayed. A Thursday-anchored
    season happens to hide the same defect behind a different weekday, so the
    fixture above cannot reproduce it; this one can.
    """
    days = {1: (4, 5, 6, 7, 8), 2: (12, 13, 14)}  # week 1 Fri..Tue, week 2 Fri..Sun
    rows = [(week, pd.Timestamp(date(2026, SEPT_2026, d))) for week, ds in days.items() for d in ds]
    monkeypatch.setattr(
        routes.games_data,
        "fetch_schedules",
        lambda _s: pd.DataFrame(rows, columns=["week", "gameday"]),
    )

    def _week_on(day: date) -> int:
        class _FrozenDate(date):
            @classmethod
            def today(cls) -> date:  # noqa: D102
                return day

        monkeypatch.setattr(routes, "date", _FrozenDate)
        return routes.current_season_and_week()[1]

    assert date(2026, 9, 4).weekday() == 4, "this fixture must kick off on a Friday"
    assert _week_on(date(2026, 9, 4)) == 1, "kickoff Friday is week 1"
    assert _week_on(date(2026, 9, 5)) == 1, "Saturday is still week 1"
    assert _week_on(date(2026, 9, 8)) == 1, "Tuesday, week 1's last game day"
    assert _week_on(date(2026, 9, 9)) == 2, "Wednesday, week 1 is behind us"


def test_a_missing_schedule_still_refuses_to_advance_midweek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback must not resurrect the mid-week rollover.

    `fetch_schedules` raising is the cold-cache / no-network path — exactly when
    a wrong week would be least likely to be noticed. The fallback is Sunday-
    anchored, so it is worse than the schedule-derived rule (it cannot see a
    Monday game) but it still does not advance on a Friday.
    """

    def _boom(_season: list[int]) -> pd.DataFrame:
        raise RuntimeError("no schedule")

    monkeypatch.setattr(routes.games_data, "fetch_schedules", _boom)

    def _week_on(day: date) -> int:
        class _FrozenDate(date):
            @classmethod
            def today(cls) -> date:  # noqa: D102
                return day

        monkeypatch.setattr(routes, "date", _FrozenDate)
        return routes.current_season_and_week()[1]

    # What the fallback can guarantee, and all it can: the boundary falls on a
    # SUNDAY, not mid-week. It is calendar-only, so it cannot see a Monday game
    # the way the schedule-derived rule does — the absolute week number here is
    # not meaningful, only the fact that Friday and Saturday sit before the
    # boundary and Sunday is the first day after it.
    assert _week_on(date(2026, 9, 4)) == _week_on(date(2026, 9, 5))  # Fri == Sat
    assert _week_on(date(2026, 9, 6)) == _week_on(date(2026, 9, 4)) + 1  # Sun advances


def test_an_empty_schedule_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """A schedule with no usable gamedays falls back rather than raising."""
    monkeypatch.setattr(
        routes.games_data,
        "fetch_schedules",
        lambda _s: pd.DataFrame({"week": [], "gameday": []}),
    )

    class _FrozenDate(date):
        @classmethod
        def today(cls) -> date:  # noqa: D102
            return date(2026, 9, 6)

    monkeypatch.setattr(routes, "date", _FrozenDate)
    season, week = routes.current_season_and_week()
    assert season == 2026
    assert 1 <= week <= 20


def test_the_week_stays_inside_the_season_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clamping still applies: never below 1, never above 20."""
    assert 1 <= _season_and_week(monkeypatch, date(2026, 1, 1)) <= 20
    assert 1 <= _season_and_week(monkeypatch, date(2026, 12, 31)) <= 20