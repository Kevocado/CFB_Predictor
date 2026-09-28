"""The backfill spends metered CFBD quota, so its input handling is a money concern.

`fetch_team_stats`'s own docstring calls itself "a full 2004-2025 backfill is ~330 calls
against a metered budget... Call this once, deliberately". The script is the only thing
standing between an operator typo and a billed call, and a call is made *before* the
response is known to be valid -- so an out-of-range week is money spent on nothing.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backfill_team_stats.py"


def _run(*args):
    """Run the script without spending anything, and return the completed process."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("week", ["0", "-1", "99", "17"])
def test_an_out_of_range_week_is_refused_before_it_can_be_billed(week):
    """`--weeks 0 -1 99` was three billed calls and three junk cache files."""
    result = _run("--weeks", week)

    assert result.returncode != 0
    assert "between 1 and 16" in result.stderr


def test_a_duplicated_week_is_refused_because_it_doubles_every_row():
    """One call, but the cached frame is appended once per occurrence.

    So the caller gets double rows, and the `pairs` count overstates the cost, which is
    worse for a quota tool than an outright failure.
    """
    result = _run("--weeks", "1", "1")

    assert result.returncode != 0
    assert "duplicates" in result.stderr


def test_valid_weeks_are_accepted_and_the_run_is_still_a_dry_run():
    result = _run("--weeks", "1", "16")

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "nothing was fetched" in result.stdout
