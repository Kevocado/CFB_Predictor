"""Null the ATS / totals hit flags an older build fabricated.

This is a data migration, not a bug fix
---------------------------------------
An older build of `tracking/store.py::_compute_hits` graded the spread and totals
markets from whatever it happened to have:

    if home_spread_line is not None and pd.notna(home_spread_line):
        predicted_home_cover = (home_cover_prob or 0) >= (away_cover_prob or 0)

The probabilities were never checked. A game snapshotted with a spread line but no
cover probabilities -- exactly what the odds feed yields when it has a spread without a
matching market -- was recorded as a graded call the model never made: `(None or 0)`
against `(None or 0)` is `0.0 >= 0.0`, and `>=` resolves that to the home side. So it
fabricated a HIT when the home team covered and a MISS when the home team did not, and
persisted both to the track record.

Both ends of that are fixed and merged. `_compute_hits` now requires presence through
`_present` and leaves the flag NULL; `_summarize_games` excludes such rows from the
aggregate; `get_game_verdict` declines to report the market. What none of them can reach
is the data those fixes could not reach -- rows already written to a deployed
`tracking.db` still carry the fabricated flags. This script nulls them and nothing
else. It does not touch the write path, the read path, the probabilities, the lines,
the scores, or `moneyline_hit`, and it never deletes a row.

The defect is identical in NFL_Predictor -- same code, same line numbers, same fix --
and this script is the same in both repos, so the two track records can be repaired the
same way.

The predicate
-------------
A row is fabricated when a hit flag is set but the probabilities the grade claims to
compare are missing:

    (ats_hit   IS NOT NULL AND (home_cover_prob IS NULL OR away_cover_prob IS NULL))
 OR (total_hit IS NOT NULL AND (over_prob     IS NULL OR under_prob     IS NULL))

`OR` within a market, not "both are NULL": the old expression also fabricated a grade
out of a *half* market -- `0.6` against `None` reads as the home side -- and a row
graded from half a market is exactly as unusable as one graded from none.

Why this cannot catch a legitimately graded row
------------------------------------------------
The predicate is not a heuristic keyed on a suspicious value. It is a contradiction
between two fields of the same row, and the write path makes it unsatisfiable:

  * `_compute_hits` writes a non-NULL `ats_hit` if and only if
    `_present(home_spread_line) and _present(home_cover_prob, away_cover_prob)`.
    `_present` is `all(v is not None and not pd.isna(v))`, so a row the current write
    path graded carries *both* cover probabilities.
  * The predicate requires at least one of them to be NULL.

A row cannot satisfy both, so the predicate is empty over the output of the current
write path. The same argument holds for totals. Two things follow that are worth
stating rather than leaving implicit:

  * **0.0 is not NULL.** A confident zero is a legible probability, not a missing one --
    that is the entire reason the original bug was possible -- so a genuine grade built
    on `home_cover_prob = 0.0, away_cover_prob = 1.0` is preserved. This is the
    distinction a `COALESCE(home_cover_prob, 0) = 0` style predicate would erase, and
    `tests/test_null_fabricated_market_hits.py` pins it with a grid that pushes every
    structural case through both `_compute_hits` and this SQL and asserts the two never
    disagree about a row.
  * **The argument is conditional on `_compute_hits` being the only writer of these two
    columns.** Verified by grep over `src/`, `scripts/` and the frontend: it is. It is
    deliberately *not* conditional on the row also having a spread line, even though
    every row the old build fabricated necessarily had one (its guard was on the line).
    Requiring the line would narrow the predicate on the strength of an assumption about
    provenance rather than about the defect, and would silently leave rows behind if
    that assumption were ever wrong. The survey reports how many matched rows have no
    line so an operator can see it if it happens.

What this cannot know
---------------------
  * **Whether a stored grade that agrees with its probabilities is correct.** This
    verifies the shape of the row, not the truth of the call. A well-formed but wrong
    grade from some other cause is untouched, and nothing here would find it.
  * **What the grade would have been.** The probabilities were never computed, so there
    is nothing to recompute from. The row cannot be recovered, only retired.
  * **Which build wrote a matched row.** The signature identifies the defect, not the
    process. It cannot distinguish "written by the build we know about" from "written by
    something else with the same blind spot".
  * **Anything about rows it does not match.** Fabricating a grade from a *present* but
    nonsensical probability pair, for instance, leaves no signature this can see.

Usage
-----
    python scripts/null_fabricated_market_hits.py                    # dry run on the app db
    python scripts/null_fabricated_market_hits.py --db /path/db      # dry run, explicit
    python scripts/null_fabricated_market_hits.py --db /path/db --execute
    python scripts/null_fabricated_market_hits.py --db /path/db --verify-only

`--execute` is the only thing that writes. The default run opens the database with
SQLite's `mode=ro`, so it cannot create a file either -- a survey pointed at the wrong
path is an error, not an empty `tracking.db` left behind. `--verify-only` re-opens
read-only and exits nonzero if any fabricated row remains, which is the check to run
afterwards to prove the migration did what it said.

`--execute` copies the database to `<db>.bak-<utc timestamp>` first, through SQLite's
online backup API rather than a file copy, because the live database runs in WAL mode
and a plain `shutil.copy` can miss committed pages still sitting in the `-wal` sidecar.

Precedent
---------
The safety shape is `scripts/backfill_team_stats.py`'s, for the same reason: a script
that is dangerous by default needs the danger behind a deliberate act. That one guards
metered CFBD quota, this one guards the track record, and the flag that gates it is
`--execute` in both cases. The write path here is far less expensive than an API call,
which is exactly why it is more likely to be run by accident.
"""

from __future__ import annotations

import argparse
import contextlib
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import pathname2url

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

TABLE = "game_predictions"

# One clause per market, and the single source of truth for both the survey and the
# UPDATE. Derived in exactly two places so the two can never drift apart.
_ATS_GUARD = "ats_hit IS NOT NULL AND (home_cover_prob IS NULL OR away_cover_prob IS NULL)"
_TOTALS_GUARD = "total_hit IS NOT NULL AND (over_prob IS NULL OR under_prob IS NULL)"

FABRICATED_PREDICATE = f"({_ATS_GUARD}) OR ({_TOTALS_GUARD})"

REPAIR_SQL = f"""
UPDATE {TABLE}
SET ats_hit   = CASE WHEN {_ATS_GUARD}   THEN NULL ELSE ats_hit   END,
    total_hit = CASE WHEN {_TOTALS_GUARD} THEN NULL ELSE total_hit END
WHERE {FABRICATED_PREDICATE}
"""


class MigrationError(RuntimeError):
    """A refusal, not a crash: the database was left exactly as it was found."""


def _readonly_uri(path: Path) -> str:
    return "file:" + pathname2url(str(path)) + "?mode=ro"


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open `path` for reading, refusing to create it.

    `mode=ro` is the reason a survey has no side effects at all. A bare
    `sqlite3.connect` creates a missing file, so pointing this at a mistyped path would
    leave an empty `tracking.db` behind -- litter that a later run would then take for a
    real, empty database and report "nothing to do" about.
    """
    if not path.is_file():
        raise MigrationError(f"no such database: {path}")
    conn = sqlite3.connect(_readonly_uri(path), uri=True, timeout=15)
    conn.execute("PRAGMA busy_timeout = 15000")
    try:
        # A real query, not a lazy handle: `mode=ro` alone does not find out whether the
        # open worked. "no such table" is left alone -- it is `_require_tracking_table`'s
        # job to turn that into a useful message.
        conn.execute(f"SELECT 1 FROM {TABLE} LIMIT 0").fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc):
            conn.close()
            raise MigrationError(
                f"cannot read {path} read-only: {exc}. A SQLite database in WAL mode still "
                "needs to create its -wal and -shm sidecars to be read, so the *directory* "
                "has to be writable even though the file is not modified. The live database "
                "lives on local container disk for exactly this reason; if you are trying to "
                "read it off the Azure Files mount, copy it down first."
            ) from exc
    return conn


def _require_tracking_table(conn: sqlite3.Connection) -> None:
    """Refuse anything that is not a tracking database, by table rather than by name.

    Checking the schema instead of trusting the filename is the difference between a
    clear error and `no such column: ats_hit` from an unrelated SQLite file someone
    passed by mistake.
    """
    found = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (TABLE,)
    ).fetchone()
    if not found:
        raise MigrationError(f"no {TABLE} table -- this is not a tracking database")
    required = {
        "game_id", "resolved", "ats_hit", "total_hit",
        "home_cover_prob", "away_cover_prob", "over_prob", "under_prob",
        "home_spread_line", "total_line",
    }
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE})")}
    missing = sorted(required - columns)
    if missing:
        raise MigrationError(f"{TABLE} is missing columns {missing}; refusing to guess")


def survey(conn: sqlite3.Connection) -> dict:
    """Count what is there and what matches the predicate. Read-only by construction."""
    def count(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])

    return {
        "rows": count(f"SELECT COUNT(*) FROM {TABLE}"),
        "resolved": count(f"SELECT COUNT(*) FROM {TABLE} WHERE resolved = 1"),
        "fabricated_ats": count(f"SELECT COUNT(*) FROM {TABLE} WHERE {_ATS_GUARD}"),
        "fabricated_totals": count(f"SELECT COUNT(*) FROM {TABLE} WHERE {_TOTALS_GUARD}"),
        "fabricated_any": count(f"SELECT COUNT(*) FROM {TABLE} WHERE {FABRICATED_PREDICATE}"),
        "fabricated_ats_hits": count(
            f"SELECT COUNT(*) FROM {TABLE} WHERE {_ATS_GUARD} AND ats_hit = 1"
        ),
        "fabricated_ats_misses": count(
            f"SELECT COUNT(*) FROM {TABLE} WHERE {_ATS_GUARD} AND ats_hit = 0"
        ),
        "fabricated_total_hits": count(
            f"SELECT COUNT(*) FROM {TABLE} WHERE {_TOTALS_GUARD} AND total_hit = 1"
        ),
        "fabricated_total_misses": count(
            f"SELECT COUNT(*) FROM {TABLE} WHERE {_TOTALS_GUARD} AND total_hit = 0"
        ),
        # Reported, not enforced. See "Why this cannot catch a legitimately graded row".
        "fabricated_ats_without_line": count(
            f"SELECT COUNT(*) FROM {TABLE} WHERE {_ATS_GUARD} AND home_spread_line IS NULL"
        ),
        "fabricated_totals_without_line": count(
            f"SELECT COUNT(*) FROM {TABLE} WHERE {_TOTALS_GUARD} AND total_line IS NULL"
        ),
    }


def sample(conn: sqlite3.Connection, limit: int) -> list[tuple]:
    """The game_ids the predicate matches, so a dry run says *which* rows, not how many."""
    if limit <= 0:
        return []
    return list(
        conn.execute(
            f"SELECT game_id FROM {TABLE} WHERE {FABRICATED_PREDICATE} ORDER BY game_id LIMIT ?",
            (limit,),
        )
    )


def backup(db_path: Path, dest: Path | None = None) -> Path:
    """Copy the database through SQLite's online backup API.

    Not `shutil.copy`: the live database runs in WAL mode, so committed transactions can
    still be sitting in the `-wal` sidecar, and a byte copy of the main file alone can
    miss them. The backup API checkpoints as it goes and produces a self-contained file.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = dest or db_path.with_name(f"{db_path.name}.bak-{stamp}")
    if dest.exists():
        raise MigrationError(f"backup destination already exists, refusing to overwrite: {dest}")
    with contextlib.closing(open_readonly(db_path)) as source:
        with contextlib.closing(sqlite3.connect(dest)) as target:
            source.backup(target)
    return dest


def repair(conn: sqlite3.Connection) -> int:
    """Null the fabricated flags. One statement, in one transaction.

    `rowcount` is the number of rows the UPDATE matched, and the WHERE requires at least
    one non-NULL flag whose probabilities are missing, so every matched row really does
    have a column changed to NULL -- it is not a count of rows that happened to be
    touched and left alone. The post-repair re-survey is what actually proves it.
    """
    conn.isolation_level = None  # explicit BEGIN, not sqlite3's implicit transaction
    conn.execute("BEGIN IMMEDIATE")
    try:
        changed = conn.execute(REPAIR_SQL).rowcount
        conn.execute("COMMIT")
    except Exception:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise
    return int(changed)


def default_db_path() -> Path:
    """The application's own tracking database, resolved from the package config."""
    from cfb_predictor.config import TRACKING_DB_PATH

    return Path(TRACKING_DB_PATH).resolve()


def _print_survey(label: str, counts: dict) -> None:
    print(f"{label}")
    print(f"  rows in {TABLE:<18} {counts['rows']}")
    print(f"  resolved                    {counts['resolved']}")
    print(f"  fabricated ats_hit           {counts['fabricated_ats']}"
          f"  ({counts['fabricated_ats_hits']} hits, {counts['fabricated_ats_misses']} misses)")
    print(f"  fabricated total_hit         {counts['fabricated_totals']}"
          f"  ({counts['fabricated_total_hits']} hits, {counts['fabricated_total_misses']} misses)")
    print(f"  fabricated in either market  {counts['fabricated_any']}")
    print(f"  ...with no spread line       {counts['fabricated_ats_without_line']}")
    print(f"  ...with no total line        {counts['fabricated_totals_without_line']}")


def _print_consequences() -> None:
    """The part an operator has to read before typing --execute."""
    print(
        "\nconsequences\n"
        "  This REWRITES STORED HISTORY. The fabricated flags are the track record's own\n"
        "  numbers, and nulling them is not a cosmetic edit.\n"
        "\n"
        "  No *reported* figure moves. Every reader of these columns -- _summarize_games\n"
        "  and get_game_verdict -- already filters on the probabilities and already declines\n"
        "  to report a market whose probabilities are missing, both merged in the fix that\n"
        "  introduced this script. So pct_ats_correct and pct_totals_correct are identical\n"
        "  before and after, and the per-game verdict is identical before and after. If you\n"
        "  were told this migration raises pct_ats_correct, that was true of the build before\n"
        "  that fix and is not true of this one.\n"
        "\n"
        "  What it does change is the stored data, so the table agrees with the code that\n"
        "  reads it: a raw query, a restored backup, a replica, or a future reader that\n"
        "  drops the read-path guard can no longer pick the fabricated flags back up. That\n"
        "  guard is currently the only thing standing between these rows and a published\n"
        "  accuracy number.\n"
        "\n"
        "  The repair is not reversible from this script: there is no stored copy of the\n"
        "  removed flags. The backup it takes is.\n"
        "\n"
        "  Deleting the rows instead was considered and is not implemented. It would remove\n"
        "  n_resolved and the actual scores along with the flag, so it trades one wrong\n"
        "  number for a smaller sample with no way to audit the difference afterwards."
    )


def _run(args: argparse.Namespace, db_path: Path) -> int:
    if args.verify_only:
        with contextlib.closing(open_readonly(db_path)) as conn:
            _require_tracking_table(conn)
            counts = survey(conn)
        print(f"database        {db_path}")
        print("mode            VERIFY ONLY -- read-only, no writes, no backup")
        print(f"\nfabricated rows remaining   {counts['fabricated_any']}")
        if counts["fabricated_any"]:
            print(
                f"\nFAILED: {counts['fabricated_any']} row(s) still match the predicate.\n"
                "Either the migration never ran, or something wrote the shape back.",
                file=sys.stderr,
            )
            return 1
        print("\nVERIFIED: no fabricated ats_hit or total_hit remains in this database.")
        return 0

    with contextlib.closing(open_readonly(db_path)) as conn:
        _require_tracking_table(conn)
        before = survey(conn)
        example = sample(conn, args.sample)

    print(f"database        {db_path}")
    print("mode            DRY RUN -- opened read-only; no writes, no backup, nothing spent"
          if not args.execute else "mode            EXECUTE -- this will write")
    _print_survey("survey (before)", before)

    print("\nplan")
    print(f"  would set ats_hit   to NULL on {before['fabricated_ats']} row(s)")
    print(f"  would set total_hit to NULL on {before['fabricated_totals']} row(s)")
    print(f"  would touch {before['fabricated_any']} row(s) in total, changing no other column")
    print("  deletes nothing: a row with no grade is still a game, and dropping it would")
    print("  also drop its actual scores and its n_resolved contribution")
    if example:
        print(f"\n  first {len(example)} affected game_id(s):")
        for (game_id,) in example:
            print(f"    {game_id}")

    if not args.execute:
        _print_consequences()
        print("\nDRY RUN -- nothing was changed. Pass --execute to write.")
        return 0

    if before["fabricated_any"] == 0:
        _print_consequences()
        print("\nNothing to do: no row matches the predicate. Already migrated.")
        return 0

    _print_consequences()
    backup_path = backup(db_path, args.backup_path)
    print(f"\nbackup          {backup_path}")

    conn = sqlite3.connect(str(db_path), timeout=15)
    conn.execute("PRAGMA busy_timeout = 15000")
    try:
        _require_tracking_table(conn)
        changed = repair(conn)
        after = survey(conn)
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.close()

    _print_survey("\nafter", after)
    print(f"\nrows changed    {changed}")
    print(f"predicate       {before['fabricated_any']} matched before, {after['fabricated_any']} after")
    if changed != before["fabricated_any"] or after["fabricated_any"] != 0:
        print(
            f"\nFAILED: expected to null {before['fabricated_any']} row(s); "
            f"changed {changed}, {after['fabricated_any']} still match. The backup is at {backup_path}.",
            file=sys.stderr,
        )
        return 1
    print(f"\nDONE. Re-run with --verify-only to prove it independently (it exits nonzero if any remain).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help="tracking.db to operate on (default: the app's own TRACKING_DB_PATH)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="actually write the repair (default: dry run, which changes nothing)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="read-only: report how many fabricated rows remain and exit nonzero if any do",
    )
    parser.add_argument(
        "--backup-path", type=Path, default=None,
        help="where --execute copies the database first (default: <db>.bak-<utc timestamp>)",
    )
    parser.add_argument(
        "--sample", type=int, default=5,
        help="how many affected game_ids to list in the plan (default: 5; 0 for none)",
    )
    args = parser.parse_args(argv)

    if args.execute and args.verify_only:
        parser.error("--execute and --verify-only are opposite modes; pick one")

    db_path = args.db.resolve() if args.db else default_db_path()
    if args.execute and not db_path.is_file():
        # Never let --execute be the thing that creates a database. An empty file here
        # would be reported as "nothing to do", which is a lie about a live database.
        print(f"--execute refuses to create a database: {db_path} does not exist", file=sys.stderr)
        return 2

    try:
        return _run(args, db_path)
    except MigrationError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
