"""The migration has to be safe, verifiable, and provably unable to touch a good row.

Three separate claims are under test and they need separate instruments:

  * **Safety** is a property of the *process*, not of a function call, so the default-run
    and `--execute`-gate tests run the real script in a subprocess against a temp
    database and then check the bytes on disk. A dry run that quietly wrote would pass
    every in-process assertion here while still being the thing that hurts someone.
  * **Narrowness** is checked by comparing whole rows before and after, so a migration
    that also rewrote a probability or a score would fail even though the flag counts
    looked right.
  * **Soundness** -- that the predicate cannot select a legitimately graded row -- cannot
    be shown by example. Two examples prove nothing about a rule. It is shown by
    enumerating every *structural* case (missing / zero / real) through both the current
    write path and the SQL, and asserting the two never disagree about a row.

Nothing here opens a network connection or a database outside `tmp_path`; the autouse
fixture below also repoints the app's own `TRACKING_DB_PATH` at the temp file so that a
test which forgot to pass `--db` still could not reach `data/tracking.db`.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import pandas as pd

from cfb_predictor.tracking import store

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "null_fabricated_market_hits.py"

ATS_GUARD = "(ats_hit IS NOT NULL AND (home_cover_prob IS NULL OR away_cover_prob IS NULL))"
TOTALS_GUARD = "(total_hit IS NOT NULL AND (over_prob IS NULL OR under_prob IS NULL))"
PREDICATE = f"{ATS_GUARD} OR {TOTALS_GUARD}"

# Probabilities that always accompany the totals grid, so the ATS flag is gradeable and
# only the totals column is under test.
_ATS_PROBS = (0.55, 0.45)
_TOTAL_PROBS = (0.7, 0.3)


def _load_script():
    spec = importlib.util.spec_from_file_location("null_fabricated_market_hits", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_script()


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    """Point the app's own tracking path at a temp file for every test in this file."""
    from cfb_predictor import config

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    return db_path


# --------------------------------------------------------------------------------------
# fixture construction
# --------------------------------------------------------------------------------------

def _connect_to(path: Path) -> sqlite3.Connection:
    """`store._connect()` against `path`, so the schema is the one the app really uses
    and not a hand-written imitation that could drift from it."""
    original = store.TRACKING_DB_PATH
    store.TRACKING_DB_PATH = path
    try:
        return store._connect()
    finally:
        store.TRACKING_DB_PATH = original


def _insert(conn, game_id, *, ats_hit, home_cover_prob, away_cover_prob,
            total_hit=None, over_prob=None, under_prob=None,
            home_spread_line=-3.5, total_line=51.5, home_score=31, away_score=24):
    conn.execute(
        """
        INSERT INTO game_predictions (
            game_id, home_team, away_team, commence_time, snapshotted_at,
            home_win_prob, away_win_prob, home_cover_prob, away_cover_prob,
            over_prob, under_prob, home_spread_line, total_line,
            resolved, actual_home_score, actual_away_score, moneyline_hit,
            ats_hit, total_hit
        ) VALUES (?, 'SF', 'DAL', '2025-09-14T20:20:00', '2025-09-14T17:00:00',
                  0.55, 0.45, ?, ?, ?, ?, ?, ?,
                  1, ?, ?, 1, ?, ?)
        """,
        (game_id, home_cover_prob, away_cover_prob, over_prob, under_prob,
         home_spread_line, total_line, home_score, away_score, ats_hit, total_hit),
    )
    conn.commit()


def _mixed_db(path: Path) -> sqlite3.Connection:
    """Two genuinely graded rows and five in the shape the old build fabricated.

    The five cover the shapes the predicate has to get right: no probabilities at all, a
    *half* market, a fabricated HIT, a fabricated MISS, and a row where one market was
    fabricated while the other was graded for real -- which is the only shape that can
    tell the two per-market guards apart.
    """
    conn = _connect_to(path)
    _insert(conn, "2025_01_genuine_hit", ats_hit=1, total_hit=1,
            home_cover_prob=0.6, away_cover_prob=0.4, over_prob=0.7, under_prob=0.3)
    _insert(conn, "2025_02_genuine_miss", ats_hit=0, total_hit=0,
            home_cover_prob=0.6, away_cover_prob=0.4, over_prob=0.7, under_prob=0.3)
    # Both probabilities missing: the old build called `0.0 >= 0.0`, i.e. the home side.
    _insert(conn, "2025_03_fabricated_ats_hit", ats_hit=1,
            home_cover_prob=None, away_cover_prob=None)
    # Half a market: `0.6` against `None` also reads as the home side.
    _insert(conn, "2025_04_fabricated_half_market", ats_hit=1,
            home_cover_prob=0.6, away_cover_prob=None)
    # The other direction: home did not cover, so the same bug wrote a fabricated MISS.
    _insert(conn, "2025_05_fabricated_ats_miss", ats_hit=0,
            home_cover_prob=None, away_cover_prob=None, home_score=20)
    # Totals fabricated, ATS genuinely graded on the same row.
    _insert(conn, "2025_06_fabricated_total", ats_hit=1,
            home_cover_prob=0.55, away_cover_prob=0.45, total_hit=1,
            over_prob=None, under_prob=None)
    # ...and the mirror image, which is the only shape that can catch one market's guard
    # being applied to the other market's column.
    _insert(conn, "2025_07_ats_fabricated_total_genuine", ats_hit=1,
            home_cover_prob=None, away_cover_prob=None, total_hit=1,
            over_prob=0.7, under_prob=0.3)
    return conn


def _rows(path: Path) -> dict:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        return {row["game_id"]: dict(row) for row in conn.execute("SELECT * FROM game_predictions")}
    finally:
        conn.close()


def _matches(path: Path) -> set:
    conn = sqlite3.connect(str(path))
    try:
        return {row[0] for row in conn.execute(f"SELECT game_id FROM game_predictions WHERE {PREDICATE}")}
    finally:
        conn.close()


def _field(out: str, label: str) -> int:
    """Read one number out of the survey block by its label."""
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith(label):
            return int(stripped[len(label):].strip().split()[0])
    raise AssertionError(f"no {label!r} line in output:\n{out}")


def _run(*args):
    """Run the script in a real process, with no API keys in sight."""
    child_env = dict(os.environ)
    child_env.pop("ODDS_API_KEY", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, timeout=180, env=child_env,
        cwd=str(SCRIPT.parent.parent),
    )


# --------------------------------------------------------------------------------------
# 1. the default run cannot write
# --------------------------------------------------------------------------------------

def test_the_default_run_writes_nothing(tmp_path):
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()
    before = db.read_bytes()

    result = _run("--db", str(db))

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "nothing was changed" in result.stdout
    assert db.read_bytes() == before, "a default run modified the database file"
    assert not list(tmp_path.glob("*.bak-*")), "a dry run took a backup"


def test_the_default_run_cannot_create_a_database(tmp_path):
    """`sqlite3.connect` creates a missing file, so a survey pointed at a mistyped path
    would leave an empty tracking.db behind -- and a later run would take that for a real,
    empty database and report "nothing to do" about it."""
    missing = tmp_path / "nope" / "tracking.db"

    result = _run("--db", str(missing))

    assert result.returncode != 0
    assert "no such database" in result.stderr
    assert not missing.exists(), "a read-only survey created the file it was asked to inspect"
    assert not missing.parent.exists()


def test_a_missing_database_is_refused_in_every_mode_including_execute(tmp_path):
    """`--execute` must not be the thing that creates a database: an empty file there
    would be reported as "nothing to do", which is a lie about a live database."""
    missing = tmp_path / "tracking.db"

    for extra in ([], ["--execute"], ["--verify-only"]):
        result = _run("--db", str(missing), *extra)
        assert result.returncode != 0, f"{extra} accepted a missing database"
        assert not missing.exists()


# --------------------------------------------------------------------------------------
# 2. the dry run reports what it would do
# --------------------------------------------------------------------------------------

def test_a_dry_run_reports_the_counts_it_would_change(tmp_path, capsys):
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()

    assert migration.main(["--db", str(db)]) == 0

    out = capsys.readouterr().out
    assert _field(out, "rows in game_predictions") == 7
    assert _field(out, "fabricated ats_hit") == 4
    assert _field(out, "fabricated total_hit") == 1
    assert _field(out, "fabricated in either market") == 5
    assert _field(out, "...with no spread line") == 0
    assert "--execute" in out, "the dry run must say how to actually run it"


def test_a_dry_run_names_the_rows_not_just_the_count(tmp_path, capsys):
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()

    migration.main(["--db", str(db), "--sample", "2"])
    out = capsys.readouterr().out

    assert "affected game_id(s)" in out
    assert "2025_03_fabricated_ats_hit" in out
    assert "2025_04_fabricated_half_market" in out
    assert "2025_05_fabricated_ats_miss" not in out, "--sample 2 listed more than two"
    assert "2025_01_genuine_hit" not in out, "the plan listed a row it would not touch"


def test_a_dry_run_states_that_stored_history_is_being_rewritten(tmp_path, capsys):
    """The consequence has to be in the output, not only in the PR body.

    A reader who runs this against a live database has to be told, before typing
    --execute, that it edits the track record rather than tidying a cache -- and also
    told what does *not* change, so they can weigh it correctly.
    """
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()

    migration.main(["--db", str(db)])
    out = capsys.readouterr().out

    assert "REWRITES STORED HISTORY" in out
    assert "No *reported* figure moves" in out
    assert "deletes nothing" in out


# --------------------------------------------------------------------------------------
# 3. --execute writes only the flags
# --------------------------------------------------------------------------------------

def test_execute_nulls_only_the_fabricated_flags(tmp_path):
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()
    before = _rows(db)

    assert migration.main(["--db", str(db), "--execute"]) == 0

    after = _rows(db)
    assert set(before) == set(after), "the row set changed; this migration must delete nothing"
    changed = set()
    for game_id, old in before.items():
        new = after[game_id]
        differing = {k for k in old if old[k] != new[k]}
        if not differing:
            continue
        changed.add(game_id)
        assert differing <= {"ats_hit", "total_hit"}, (
            f"{game_id}: the migration changed {differing}, which is out of scope"
        )
        for column in differing:
            assert old[column] is not None and new[column] is None, (
                f"{game_id}.{column} went {old[column]!r} -> {new[column]!r}; "
                "the repair only ever nulls a non-NULL flag"
            )
    assert changed == {
        "2025_03_fabricated_ats_hit",
        "2025_04_fabricated_half_market",
        "2025_05_fabricated_ats_miss",
        "2025_06_fabricated_total",
        "2025_07_ats_fabricated_total_genuine",
    }, f"wrong set of rows touched: {changed}"


def test_execute_preserves_a_genuine_grade_that_rests_on_a_zero_probability(tmp_path):
    """`0.0` is a probability, not an absence.

    The original bug is *only* possible because `(None or 0)` turned an absence into a
    legible zero. A predicate written as `COALESCE(home_cover_prob, 0) = 0` would null
    this row too, and a genuine call at 0.0 against 1.0 is exactly the row that most
    needs keeping.
    """
    db = tmp_path / "tracking.db"
    conn = _connect_to(db)
    _insert(conn, "2025_07_zero_probability", ats_hit=1, home_cover_prob=0.0, away_cover_prob=1.0,
            total_hit=1, over_prob=0.0, under_prob=1.0)
    # The same side of the rule, one-sided, for contrast.
    _insert(conn, "2025_08_half_market", ats_hit=1, home_cover_prob=0.0, away_cover_prob=None)
    conn.close()

    assert migration.main(["--db", str(db), "--execute"]) == 0

    rows = _rows(db)
    assert rows["2025_07_zero_probability"]["ats_hit"] == 1, "a 0.0-vs-1.0 call is a real call"
    assert rows["2025_07_zero_probability"]["total_hit"] == 1
    assert rows["2025_08_half_market"]["ats_hit"] is None, (
        "a one-sided market was never a call; 0.0 against None must not survive as one"
    )


def test_execute_reports_before_and_after_counts_and_the_backup(tmp_path, capsys):
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()

    assert migration.main(["--db", str(db), "--execute"]) == 0

    out = capsys.readouterr().out
    assert "survey (before)" in out
    assert "fabricated in either market  5" in out
    assert "rows changed    5" in out
    assert "5 matched before, 0 after" in out
    assert len(list(db.parent.glob("*.bak-*"))) == 1


def test_the_backup_is_a_pre_migration_snapshot_that_still_holds_the_fabricated_rows(tmp_path):
    """A backup taken *after* the UPDATE is worthless, and a byte copy of a WAL-mode
    database can be missing committed pages. Assert on the content, not the filename."""
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()

    assert migration.main(["--db", str(db), "--execute"]) == 0

    (backup_path,) = db.parent.glob("*.bak-*")
    assert _matches(backup_path) == {
        "2025_03_fabricated_ats_hit", "2025_04_fabricated_half_market",
        "2025_05_fabricated_ats_miss", "2025_06_fabricated_total",
        "2025_07_ats_fabricated_total_genuine",
    }


def test_a_backup_destination_that_exists_is_not_overwritten(tmp_path):
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()
    occupied = tmp_path / "keepme.db"
    occupied.write_bytes(b"not a database")

    assert migration.main(["--db", str(db), "--execute", "--backup-path", str(occupied)]) == 2

    assert occupied.read_bytes() == b"not a database"
    assert _matches(db), "the refusal happened before any repair, so the rows are still there"


# --------------------------------------------------------------------------------------
# 4. idempotence, and the proof-of-effect affordance
# --------------------------------------------------------------------------------------

def test_running_twice_is_a_no_op_the_second_time(tmp_path, capsys):
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()

    assert migration.main(["--db", str(db), "--execute"]) == 0
    after_first = db.read_bytes()
    capsys.readouterr()

    assert migration.main(["--db", str(db), "--execute"]) == 0

    out = capsys.readouterr().out
    assert "Nothing to do" in out
    assert _field(out, "fabricated in either market") == 0
    assert db.read_bytes() == after_first, "a second --execute modified the database"


def test_verify_only_exits_nonzero_until_the_rows_are_gone_and_zero_after(tmp_path, capsys):
    """The proof of effect, as a command an operator can run afterwards on a real
    database. It has to be capable of failing, or it is a rubber stamp."""
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()

    assert migration.main(["--db", str(db), "--verify-only"]) == 1
    assert "FAILED" in capsys.readouterr().err

    assert migration.main(["--db", str(db), "--execute"]) == 0
    capsys.readouterr()

    assert migration.main(["--db", str(db), "--verify-only"]) == 0
    out = capsys.readouterr().out
    assert "VERIFIED" in out
    assert _field(out, "fabricated rows remaining") == 0


def test_verify_only_does_not_write(tmp_path, capsys):
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()
    before = db.read_bytes()

    migration.main(["--db", str(db), "--verify-only"])

    capsys.readouterr()
    assert db.read_bytes() == before
    assert not list(tmp_path.glob("*.bak-*"))


def test_execute_and_verify_only_cannot_both_be_asked_for(tmp_path):
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()

    with pytest.raises(SystemExit) as excinfo:
        migration.main(["--db", str(db), "--execute", "--verify-only"])

    assert excinfo.value.code == 2


# --------------------------------------------------------------------------------------
# 5. it refuses anything that is not a tracking database
# --------------------------------------------------------------------------------------

def test_it_refuses_an_sqlite_file_that_is_not_a_tracking_database(tmp_path, capsys):
    other = tmp_path / "other.db"
    conn = sqlite3.connect(str(other))
    conn.execute("CREATE TABLE predictions (game_id TEXT)")
    conn.commit()
    conn.close()

    assert migration.main(["--db", str(other), "--execute"]) == 2
    assert "not a tracking database" in capsys.readouterr().err


def test_it_refuses_a_tracking_table_missing_the_columns_it_needs(tmp_path, capsys):
    other = tmp_path / "partial.db"
    conn = sqlite3.connect(str(other))
    conn.execute("CREATE TABLE game_predictions (game_id TEXT, ats_hit INTEGER)")
    conn.commit()
    conn.close()

    assert migration.main(["--db", str(other)]) == 2
    assert "missing columns" in capsys.readouterr().err


def test_the_default_database_is_the_apps_own_tracking_path(tmp_path, capsys, _isolated_db):
    """No `--db` means the app's `TRACKING_DB_PATH`, and the path is printed either way
    so a misdirected run is visible in the log before anyone acts on it."""
    _mixed_db(_isolated_db).close()

    assert migration.main([]) == 0

    out = capsys.readouterr().out
    assert str(_isolated_db.resolve()) in out
    assert "DRY RUN" in out


# --------------------------------------------------------------------------------------
# 6. soundness: the predicate cannot select a legitimately graded row
# --------------------------------------------------------------------------------------
#
# The grid is over *structural* cases -- missing, zero, a real probability, and a few
# line values -- not over all floats. That is the right shape: every way this predicate
# can be wrong is a way of confusing 0.0 with NULL, and both are in the grid. A bug that
# needed some particular float to appear would be a bug about a value, and this
# predicate keys on no value at all.

_ATS_LINES = [None, 0.0, -3.5, 3.5]
_ATS_HOME = [None, 0.0, 0.5, 0.6, 1.0]
_ATS_AWAY = [None, 0.0, 0.4, 0.5, 1.0]
_TOTAL_LINES = [None, 0.0, 30.0, 51.5]
_TOTAL_OVER = [None, 0.0, 0.5, 0.7, 1.0]
_TOTAL_UNDER = [None, 0.0, 0.3, 0.5, 1.0]
_SCORES = [(31, 24), (20, 24), (30, 30), (24, 31)]


def _old_compute_hits(home_score, away_score, home_spread_line, home_cover_prob, away_cover_prob,
                      total_line, over_prob, under_prob):
    """`_compute_hits` exactly as it was *before* the `_present` fix (CFB commit 18a2708).

    Copied out of git history rather than paraphrased, because the whole completeness
    argument is "the predicate catches every row this function wrote", and that claim is
    only worth anything if the function is the real one. The moneyline half is omitted:
    the migration does not touch `moneyline_hit`.
    """
    ats_hit = None
    if home_spread_line is not None and pd.notna(home_spread_line):
        home_covered = (home_score - away_score) > home_spread_line
        ats_hit = int(((home_cover_prob or 0) >= (away_cover_prob or 0)) == home_covered)
    total_hit = None
    if total_line is not None and pd.notna(total_line):
        went_over = (home_score + away_score) > total_line
        total_hit = int(((over_prob or 0) >= (under_prob or 0)) == went_over)
    return ats_hit, total_hit


def _assert_predicate_agrees_with_the_write_path(tmp_path, grid, *, label):
    """The predicate, against both writers, over the whole grid.

    Two rows per grid point: one exactly as the *current* write path would store it, and
    one identical except that the flag is forced non-NULL -- the shape a fabricated row
    has. Then three things have to hold:

      * **Soundness.** The predicate selects none of the first kind, and none of the
        second kind that the current write path *would* grade. This is the claim that
        matters: it can never null a legitimately graded row. Neither half is an
        assumption; both are asserted directly over the grid.
      * **Completeness against the historical writer.** Every row the *old* build
        fabricated -- wrote a flag for a market whose probabilities are missing -- is
        selected. `_old_compute_hits` is the pre-`_present` implementation copied out of
        git history, so this is checked against the writer that actually produced the
        rows, not a paraphrase of it.
      * **Exactness.** Among forced-flag rows the predicate selects exactly those with a
        missing probability, which is the condition it keys on and nothing else.
      * **The writing SQL.** `repair()` is then run over the same database, so the
        statement that *writes* is pinned to the same predicate as the one that reads.

    The predicate is deliberately a *strict superset* of what the old build fabricated:
    it also selects a forced-flag row that has no spread line, which the old build could
    not have written (its guard was on the line). That is asserted rather than left
    implicit -- such a row is still wrong, since a flag on a market whose probabilities
    are missing is not a call, and the survey's "no spread line" counter is what makes
    one visible to an operator.
    """
    db = tmp_path / f"grid_{label}.db"
    conn = _connect_to(db)
    fabricated: set[str] = set()
    missing_probability: set[str] = set()
    no_line_and_missing_probability: set[str] = set()
    graded_by_current: set[str] = set()
    graded_by_old: set[str] = set()
    for i, point in enumerate(grid):
        line, probs = point["line"], point["probs"]
        home_score, away_score = point["home_score"], point["away_score"]
        if label == "ats":
            current = store._compute_hits(
                home_score, away_score, 0.55, 0.45, line, probs[0], probs[1],
                51.5, *_TOTAL_PROBS,
            )[1]
            old = _old_compute_hits(
                home_score, away_score, line, probs[0], probs[1], 51.5, *_TOTAL_PROBS
            )[0]
            real = dict(ats_hit=current, home_cover_prob=probs[0], away_cover_prob=probs[1],
                        home_spread_line=line)
            fake = dict(real, ats_hit=1)
        else:
            current = store._compute_hits(
                home_score, away_score, 0.55, 0.45, -3.5, *_ATS_PROBS,
                line, probs[0], probs[1],
            )[2]
            old = _old_compute_hits(
                home_score, away_score, -3.5, *_ATS_PROBS, line, probs[0], probs[1]
            )[1]
            real = dict(ats_hit=1, home_cover_prob=_ATS_PROBS[0], away_cover_prob=_ATS_PROBS[1],
                        total_hit=current, over_prob=probs[0], under_prob=probs[1], total_line=line)
            fake = dict(real, total_hit=1)
        _insert(conn, f"{label}_real_{i:04d}", home_score=home_score, away_score=away_score, **real)
        _insert(conn, f"{label}_fake_{i:04d}", home_score=home_score, away_score=away_score, **fake)
        if current is not None:
            graded_by_current.add(f"{label}_real_{i:04d}")
        if old is not None:
            graded_by_old.add(f"{label}_real_{i:04d}")
        if old is not None and current is None:
            # The old build wrote a grade for a market it could not see. This is the
            # definition of "fabricated": a flag the current build refuses to write.
            fabricated.add(f"{label}_fake_{i:04d}")
        if probs[0] is None or probs[1] is None:
            missing_probability.add(f"{label}_fake_{i:04d}")
            if line is None:
                no_line_and_missing_probability.add(f"{label}_fake_{i:04d}")

    selected = _matches(db)
    real_selected = {gid for gid in selected if "_real_" in gid}
    fake_selected = {gid for gid in selected if "_fake_" in gid}

    assert graded_by_current, f"{label}: the grid produced no gradable rows, so it proves nothing"
    assert graded_by_old, f"{label}: the old build graded nothing here, so there is nothing to catch"
    assert fabricated, f"{label}: the old build fabricated nothing here, so it proves nothing"
    assert not real_selected, (
        f"{label}: the predicate selected {len(real_selected)} row(s) the current write path "
        f"would have graded, e.g. {sorted(real_selected)[:3]}"
    )
    assert not (fake_selected & graded_by_current), (
        f"{label}: the predicate selected a row the current write path grades: "
        f"{sorted(fake_selected & graded_by_current)[:3]}"
    )
    assert fabricated <= fake_selected, (
        f"{label}: the predicate missed {len(fabricated - fake_selected)} row(s) the old build "
        f"fabricated: {sorted(fabricated - fake_selected)[:3]}"
    )
    assert fake_selected == missing_probability, (
        f"{label}: selected {len(fake_selected - missing_probability)} row(s) whose "
        f"probabilities are all present, and missed {len(missing_probability - fake_selected)} "
        "row(s) whose probabilities are missing"
    )
    assert fake_selected - fabricated == no_line_and_missing_probability, (
        f"{label}: the predicate is meant to be a strict superset of what the old build "
        "fabricated, differing only by rows with neither a line nor probabilities"
    )

    writer = sqlite3.connect(str(db))
    try:
        before_rows = _rows(db)
        changed = migration.repair(writer)
    finally:
        writer.close()
    flag_column = "ats_hit" if label == "ats" else "total_hit"
    after_rows = _rows(db)
    nulled = {
        gid for gid, row in after_rows.items()
        if row[flag_column] is None and before_rows[gid][flag_column] is not None
    }
    assert nulled == fake_selected, (
        f"{label}: repair() reported {changed} rows but it nulled {len(nulled)} flag(s), "
        f"and {len(fake_selected - nulled)} of them were not ones the predicate selected"
    )
    assert changed == len(fake_selected)


def test_the_ats_predicate_agrees_with_the_write_path_on_every_structural_case(tmp_path):
    grid = [
        {"line": line, "probs": probs, "home_score": hs, "away_score": aws}
        for line, probs, (hs, aws) in itertools.product(
            _ATS_LINES, itertools.product(_ATS_HOME, _ATS_AWAY), _SCORES
        )
    ]
    assert len(grid) == 4 * 25 * 4
    _assert_predicate_agrees_with_the_write_path(tmp_path, grid, label="ats")


def test_the_totals_predicate_agrees_with_the_write_path_on_every_structural_case(tmp_path):
    grid = [
        {"line": line, "probs": probs, "home_score": hs, "away_score": aws}
        for line, probs, (hs, aws) in itertools.product(
            _TOTAL_LINES, itertools.product(_TOTAL_OVER, _TOTAL_UNDER), _SCORES
        )
    ]
    assert len(grid) == 4 * 25 * 4
    _assert_predicate_agrees_with_the_write_path(tmp_path, grid, label="totals")


# --------------------------------------------------------------------------------------
# 7. the honest consequence: no reported number moves
# --------------------------------------------------------------------------------------

def test_the_migration_does_not_change_any_reported_number(tmp_path, capsys):
    """Nulling these flags changes the *stored* track record and nothing a reader sees.

    This is the claim the script makes in its own output, so it needs a test. If a future
    change to the read path makes it fail, that is the interesting event: either a reader
    started trusting the stored flag, in which case the migration matters more than this
    test says, or a reader started reporting something new.
    """
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()
    game_ids = list(_rows(db))

    before_track_record = store.get_track_record()["games"]
    before_verdicts = {gid: store.get_game_verdict(gid) for gid in game_ids}

    assert migration.main(["--db", str(db), "--execute"]) == 0
    capsys.readouterr()

    assert store.get_track_record()["games"] == before_track_record
    assert {gid: store.get_game_verdict(gid) for gid in game_ids} == before_verdicts


def test_the_migration_still_nulls_the_flags_even_though_no_reported_number_moves(tmp_path):
    """The other half of the previous test, so it cannot pass by the repair doing nothing."""
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()

    assert migration.main(["--db", str(db), "--execute"]) == 0

    rows = _rows(db)
    assert rows["2025_03_fabricated_ats_hit"]["ats_hit"] is None
    assert rows["2025_05_fabricated_ats_miss"]["ats_hit"] is None
    assert rows["2025_06_fabricated_total"]["total_hit"] is None
    assert rows["2025_01_genuine_hit"]["ats_hit"] == 1


def test_the_survey_reports_a_fabricated_row_with_no_line_so_a_wrong_provenance_shows(tmp_path, capsys):
    """A fabricated row always had a spread line, because the old build's guard was on
    the line. The predicate deliberately does not require one, so the survey reports the
    count instead: a nonzero number here means a row matched that the build we know about
    could not have written, which is the signal to stop and look rather than proceed."""
    db = tmp_path / "tracking.db"
    conn = _connect_to(db)
    _insert(conn, "2025_09_no_line", ats_hit=1, home_cover_prob=None, away_cover_prob=None,
            home_spread_line=None)
    conn.close()

    assert migration.main(["--db", str(db)]) == 0

    out = capsys.readouterr().out
    assert _field(out, "...with no spread line") == 1
    assert _field(out, "fabricated in either market") == 1


def test_the_survey_separates_fabricated_hits_from_fabricated_misses(tmp_path, capsys):
    """A count is not a direction. The whole point of the defect was that it fabricated
    a HIT and a MISS, and the difference between "your ATS record is 40%" and "we never
    made an ATS call" is the difference between the two numbers, so both are reported."""
    db = tmp_path / "tracking.db"
    conn = _connect_to(db)
    _insert(conn, "2025_10_hit", ats_hit=1, home_cover_prob=None, away_cover_prob=None)
    _insert(conn, "2025_11_miss", ats_hit=0, home_cover_prob=None, away_cover_prob=None)
    _insert(conn, "2025_12_total_hit", ats_hit=1, home_cover_prob=0.5, away_cover_prob=0.5,
            total_hit=1, over_prob=None, under_prob=None)
    conn.close()

    assert migration.main(["--db", str(db)]) == 0

    out = capsys.readouterr().out
    assert "(1 hits, 1 misses)" in out
    assert "(1 hits, 0 misses)" in out


# --------------------------------------------------------------------------------------
# 8. the script must not be able to claim success it did not achieve
# --------------------------------------------------------------------------------------

def test_execute_fails_loudly_when_the_repair_does_not_match_the_survey(tmp_path, monkeypatch, capsys):
    """A migration that reports "DONE" without having done the thing is worse than no
    migration, because it is trusted. The statement is narrowed here to simulate a
    partial repair, and the exit code has to be nonzero."""
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()
    monkeypatch.setattr(migration, "REPAIR_SQL", migration.REPAIR_SQL.replace(
        "(ats_hit IS NOT NULL AND (home_cover_prob IS NULL OR away_cover_prob IS NULL))",
        "(ats_hit = 1 AND (home_cover_prob IS NULL OR away_cover_prob IS NULL))",
    ))

    assert migration.main(["--db", str(db), "--execute"]) == 1

    assert "FAILED" in capsys.readouterr().err


def test_a_genuine_flag_survives_on_a_row_whose_other_market_was_fabricated(tmp_path):
    """One market's guard must not be applied to the other market's column.

    A row whose ATS grade was fabricated and whose totals grade is real is the only
    shape that can tell the two apart: a `total_hit = NULL` or a totals-flavoured CASE
    guard leaks onto it while every count still comes out right.
    """
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()

    assert migration.main(["--db", str(db), "--execute"]) == 0

    rows = _rows(db)
    mirror = rows["2025_07_ats_fabricated_total_genuine"]
    assert mirror["ats_hit"] is None, "the fabricated ATS grade should have been nulled"
    assert mirror["total_hit"] == 1, "the real totals grade was destroyed"
    other = rows["2025_06_fabricated_total"]
    assert other["total_hit"] is None
    assert other["ats_hit"] == 1, "the real ATS grade was destroyed"


def test_a_failed_repair_leaves_the_rows_alone(tmp_path, monkeypatch):
    """One statement in one transaction: a repair that raises must not have half-applied.

    Checked by pointing the statement at a column that does not exist, which is the
    closest a test can get to a failure partway through a real multi-row update.
    """
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()
    before = _rows(db)
    monkeypatch.setattr(migration, "REPAIR_SQL", "UPDATE game_predictions SET nope = 1")

    conn = sqlite3.connect(str(db))
    try:
        with pytest.raises(sqlite3.Error):
            migration.repair(conn)
        assert not conn.in_transaction, (
            "a failed repair left a transaction open, so a later write on this connection "
            "would be committed inside a transaction nobody is watching"
        )
    finally:
        conn.close()

    assert _rows(db) == before
    assert _matches(db), "the fabricated rows must still be there to be retried"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the file mode")
def test_a_dry_run_explains_a_database_it_cannot_open(tmp_path, capsys):
    """An unreadable database must produce an explanation, not a traceback.

    A SQLite database in WAL mode needs to create its `-wal` and `-shm` sidecars even to
    be read, so a read-only *directory* makes `mode=ro` fail -- which is the situation an
    operator lands in when they try to audit the database off the Azure Files mount. The
    message has to say what is wrong and where, because the underlying
    "attempt to write a readonly database" points at the wrong thing entirely.
    """
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()
    db.parent.chmod(0o500)
    try:
        assert migration.main(["--db", str(db)]) == 2
    finally:
        db.parent.chmod(0o700)

    err = capsys.readouterr().err
    assert "-wal" in err and "read-only" in err, f"unhelpful refusal: {err!r}"
    assert "Traceback" not in err


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the file mode")
def test_a_dry_run_can_read_a_database_the_process_cannot_write(tmp_path):
    """A dry run must not need write access, whatever mode the file is in.

    NOTE: this test is *not* able to distinguish `mode=ro` from a plain read-write open,
    and the mutation log says so. SQLite's unix VFS retries a failed `O_RDWR` with
    `O_RDONLY` and silently downgrades, on macOS and Linux alike, so both URIs read a
    0o444 database. `mode=ro` is still what the script should ask for -- it states the
    intent instead of relying on a fallback inside a C library -- but the test pins the
    behaviour, not the spelling. What the two create-guards together pin is the thing
    that matters, and the combined mutant is caught.
    """
    db = tmp_path / "tracking.db"
    _mixed_db(db).close()
    db.chmod(0o444)
    try:
        assert migration.main(["--db", str(db)]) == 0
    finally:
        db.chmod(0o644)


def test_the_backup_includes_rows_that_are_still_in_the_write_ahead_log(tmp_path):
    """A byte copy of a WAL database can be missing committed rows.

    The live database runs in WAL mode and this test holds a writer open, so committed
    transactions are still sitting in the `-wal` sidecar that a `shutil.copy` of the main
    file would not carry. This is why `backup()` uses SQLite's online backup API.
    """
    db = tmp_path / "tracking.db"
    writer = _connect_to(db)
    _insert(writer, "2025_01_in_the_wal", ats_hit=1, home_cover_prob=None, away_cover_prob=None)
    assert list(db.parent.glob("*.db-wal")), "the precondition: rows are still only in the WAL"

    try:
        assert migration.main(["--db", str(db), "--execute", "--backup-path",
                               str(tmp_path / "copy.db")]) == 0
    finally:
        writer.close()

    assert _matches(tmp_path / "copy.db") == {"2025_01_in_the_wal"}, (
        "the backup lost a committed row that was only in the write-ahead log"
    )
