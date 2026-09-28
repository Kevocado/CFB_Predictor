"""Backfill CFB team box scores from CFBD, one call per (season, week).

**This is not a route and must not become one.** CFBD's API is metered. The
reconciliation in `data/team_stats.py` is a research tool with no serving caller,
so nothing on the request path needs these rows -- but a `GET` endpoint that
quietly spends 330 calls is a foot-gun with an HTTP trigger, and the cost of
triggering it by accident is real. A script that defaults to a dry run and
requires an explicit flag puts the cost behind a deliberate act instead of a URL.

Usage
-----
    python scripts/backfill_team_stats.py                      # dry run, costs nothing
    python scripts/backfill_team_stats.py --execute           # 2004-2025, ~330 calls
    python scripts/backfill_team_stats.py --execute --to-year 2010
    python scripts/backfill_team_stats.py --execute --season 2023 --weeks 1

Resumability
------------
`fetch_team_stats` caches one parquet per (season, week), including weeks that
return no games, so an interrupted run loses at most the week in flight. Re-running
is therefore cheap and is the correct response to an interruption -- but only
because empty weeks are tombstoned. Without that, every re-run re-requests every
dead week in range (most seasons end before week 15; 2023 has no week 16 at all),
so "just run it again" would cost a few dozen extra calls each time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_api_key() -> None:
    """Populate CFBD_API_KEY from `.env` if it is not already in the environment.

    `.env` is the convention the rest of the project uses (`local.py` does the
    same), so the script should not demand a different one. An existing
    environment value wins, so CI can inject a key without a `.env` file.
    """
    if os.environ.get("CFBD_API_KEY", "").strip():
        return
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if line.startswith("CFBD_API_KEY="):
            key = line.split("=", 1)[1].strip().strip("'\"")
            if key and not key.startswith("Bearer "):
                key = f"Bearer {key}"
            os.environ["CFBD_API_KEY"] = key
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-year", type=int, default=2004, help="first season to fetch (default: 2004)")
    parser.add_argument("--to-year", type=int, default=2025, help="last season to fetch (default: 2025)")
    parser.add_argument("--weeks", type=int, nargs="+", default=list(range(1, 17)), help="weeks to fetch")
    parser.add_argument("--execute", action="store_true", help="actually spend API calls (default: dry run)")
    args = parser.parse_args()

    if args.from_year > args.to_year:
        parser.error(f"--from-year {args.from_year} is after --to-year {args.to_year}")

    # Both of these are money. CFBD bills per call, and a call is made *before* the
    # response is known to be valid, so a nonsense week is a billed call that returns
    # nothing. `--weeks 0 -1 99` was three wasted calls and three junk cache files
    # (`2023_wk-1.parquet`), and a duplicated week is one call but appends the cached
    # frame twice, so the caller gets double rows and a `pairs` count that overstates
    # the cost.
    out_of_range = [week for week in args.weeks if not 1 <= week <= 16]
    if out_of_range:
        parser.error(
            f"--weeks must be between 1 and 16 (CFBD's parameter range); got {out_of_range}. "
            f"Each out-of-range week would be a billed call that returns nothing."
        )
    duplicates = sorted({week for week in args.weeks if args.weeks.count(week) > 1})
    if duplicates:
        parser.error(
            f"--weeks contains duplicates {duplicates}. That costs no extra API call but "
            f"appends the cached frame once per occurrence, so every row comes back doubled."
        )

    # A dry run needs no key. It reads the cache directory and prints a plan; it cannot
    # spend anything whatever the environment looks like, and demanding a secret to
    # find that out is a barrier with no upside. Only `--execute` requires one, and it
    # checks immediately before it would spend.
    if args.execute:
        _load_api_key()
        if not os.environ.get("CFBD_API_KEY", "").strip():
            print(
                "CFBD_API_KEY is not set and was not found in .env -- cannot fetch. "
                "(A dry run needs no key; pass --execute to spend quota.)",
                file=sys.stderr,
            )
            return 2

    from cfb_predictor.data import team_stats

    seasons = list(range(args.from_year, args.to_year + 1))
    pairs = [(season, week) for season in seasons for week in args.weeks]
    cached = [pair for pair in pairs if team_stats._read_cache(team_stats._week_cache_path(*pair)) is not None]
    todo = len(pairs) - len(cached)

    print(f"seasons        {args.from_year}-{args.to_year} ({len(seasons)})")
    print(f"season-weeks   {len(pairs)} total, {len(cached)} already cached, {todo} to fetch")
    print(f"estimated cost {todo} CFBD calls (1 per season-week)")

    if not args.execute:
        print("\nDRY RUN -- nothing was fetched. Pass --execute to spend the quota.")
        return 0
    if todo == 0:
        print("\nNothing to do: every requested season-week is already cached.")
        return 0

    print()
    started = time.monotonic()
    frame = team_stats.fetch_team_stats(
        seasons, weeks=args.weeks, force_refresh=False
    )
    elapsed = time.monotonic() - started

    if frame.empty:
        print("Fetched no rows. If this surprises you, check --from-year/--to-year against the schedule.", file=sys.stderr)
        return 1

    team_games = frame[["game_id", "team"]].drop_duplicates().shape[0]
    print(f"rows           {len(frame)} team-games across {frame['game_id'].nunique()} games")
    print(f"elapsed        {elapsed:.1f}s ({todo / elapsed:.1f} calls/s)" if elapsed else "elapsed        0.0s")

    missing_total = int(frame["total_yards"].isna().sum())
    if missing_total:
        print(
            f"warning        {missing_total} of {len(frame)} team-games carry no total_yards. "
            "These stay NaN by design -- see add_total_yards -- rather than being filled "
            "from a one-sided component sum.",
            file=sys.stderr,
        )

    print(f"\ncache          {team_stats.TEAM_STATS_CACHE_DIR}")
    print("The reconciliation now has a team source to compare players against:")
    print("    from cfb_predictor.data import player_stats, team_stats")
    print("    team_stats.reconcile_against_players(")
    print("        team_stats.add_total_yards(team_stats.fetch_team_stats([2023])),")
    print("        player_stats.fetch_weekly_player_stats([2023]),")
    print("    )")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
