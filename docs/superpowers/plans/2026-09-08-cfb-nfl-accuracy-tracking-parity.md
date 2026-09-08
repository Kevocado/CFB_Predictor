# CFB/NFL Accuracy Tracking & Week-View Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring CFB_Predictor and NFL_Predictor to parity with the user's actual ask: grade ATS/totals (not just moneyline), never silently miss a reconcile window, show a human-readable post-match verdict per game, show a calibration reliability curve, and expose one endpoint per repo that lists predictions by week — past (resolved, with verdict) and future (upcoming, pending).

**Architecture:** Both repos' `tracking/store.py` are verbatim copies of each other (same schema, same functions) and both `evaluate/walk_forward.py` are near-verbatim copies of each other (same Brier/log_loss metrics). The work below is therefore the same change applied twice, once per package (`cfb_predictor` / `nfl_predictor`), each its own task so either repo can ship independently.

**Tech Stack:** Python, SQLite (stdlib `sqlite3`), pandas, scikit-learn (`sklearn.metrics`), FastAPI.

**Spec:** `/Users/sigey/Documents/Projects/PL_Predictor/docs/superpowers/specs/2026-09-08-cross-sport-accuracy-tracking-parity-design.md`

## Global Constraints

- Measurement only — no probability-correction/calibration layer (Platt/isotonic). Applies to every task touching `evaluate/`.
- Backfill is targeted (catch games missed by the normal tick), not a full-history rebuild.
- No RPS metric: for a 2-way market, RPS is algebraically identical to Brier score (already computed by both repos' `walk_forward.py`), so it is not implemented separately.
- Snapshot rows in `game_predictions` remain immutable once resolved — reconciliation only ever fills previously-NULL outcome columns on `resolved = 0` rows, never overwrites a resolved row.

---

## Task 1: CFB — store spread/total lines and grade ATS/totals on reconcile

**Files:**
- Modify: `src/cfb_predictor/tracking/store.py`
- Modify: `src/cfb_predictor/api/routes.py:353-392` (`background_tracking_tick`)
- Test: `tests/test_tracking.py` (new file if it doesn't exist — confirm first)

**Interfaces:**
- Produces: `store.record_game_predictions(games: list[dict])` now also persists `home_spread_line: float | None` and `total_line: float | None` when present in each game dict.
- Produces: `store.reconcile_game_predictions(results_df: pd.DataFrame) -> int` now also fills `ats_hit: int | None` and `total_hit: int | None` alongside the existing `moneyline_hit`.

- [ ] **Step 1: Check for an existing tracking test file**

Run: `ls /Users/sigey/Documents/Projects/CFB_Predictor/tests/test_tracking.py 2>/dev/null || echo "none"`

If it exists, read it first and add to it; if not, create it fresh with the imports below.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_tracking.py
import contextlib
import sqlite3

import pandas as pd
import pytest

from cfb_predictor import config
from cfb_predictor.tracking import store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")


def _future_game(**overrides):
    game = {
        "game_id": "g1", "home_team": "Texas", "away_team": "Oklahoma",
        "commence_time": "2099-01-01T00:00:00Z",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": 0.55, "away_cover_prob": 0.45,
        "over_prob": 0.5, "under_prob": 0.5,
        "home_spread_line": -3.5, "total_line": 51.5,
    }
    game.update(overrides)
    return game


def test_record_game_predictions_persists_spread_and_total_lines():
    store.record_game_predictions([_future_game()])

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = 'g1'", conn).iloc[0]

    assert row["home_spread_line"] == -3.5
    assert row["total_line"] == 51.5


def test_reconcile_grades_moneyline_ats_and_totals():
    store.record_game_predictions([_future_game()])
    # home favored by -3.5 and predicted to cover (home_cover_prob=0.55 > away);
    # over predicted (over_prob=0.5 == under_prob=0.5, home_win predicted).
    results = pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}])

    resolved = store.reconcile_game_predictions(results)

    assert resolved == 1
    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = 'g1'", conn).iloc[0]

    assert row["moneyline_hit"] == 1  # home won, home was favored
    # home won by 10, spread was -3.5 => home covered; predicted_home_cover=True
    assert row["ats_hit"] == 1
    # total = 50, line = 51.5 => actual under; predicted was a coin flip (over_prob==under_prob)
    # tie-break must not crash -- assert it resolved to 0 or 1, not None
    assert row["total_hit"] in (0, 1)


def test_reconcile_leaves_ats_and_total_hit_null_when_lines_were_never_recorded():
    store.record_game_predictions([_future_game(home_spread_line=None, total_line=None)])
    results = pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}])

    store.reconcile_game_predictions(results)

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = 'g1'", conn).iloc[0]

    assert row["moneyline_hit"] == 1
    assert pd.isna(row["ats_hit"])
    assert pd.isna(row["total_hit"])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest tests/test_tracking.py -v`
Expected: FAIL — `home_spread_line` column doesn't exist / `ats_hit`/`total_hit` not produced.

- [ ] **Step 4: Add the schema columns and grading logic**

In `src/cfb_predictor/tracking/store.py`, extend the `CREATE TABLE IF NOT EXISTS game_predictions` block and add a migration for pre-existing DB files (SQLite's `CREATE TABLE IF NOT EXISTS` won't add columns to a table that already exists):

```python
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(TRACKING_DB_PATH), timeout=15)
    conn.execute("PRAGMA busy_timeout = 15000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS game_predictions (
            game_id TEXT PRIMARY KEY,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            commence_time TEXT NOT NULL,
            snapshotted_at TEXT NOT NULL,
            home_win_prob REAL NOT NULL,
            away_win_prob REAL NOT NULL,
            home_cover_prob REAL,
            away_cover_prob REAL,
            over_prob REAL,
            under_prob REAL,
            home_spread_line REAL,
            total_line REAL,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_home_score INTEGER,
            actual_away_score INTEGER,
            moneyline_hit INTEGER,
            ats_hit INTEGER,
            total_hit INTEGER
        )
        """
    )
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(game_predictions)")}
    for column in ("home_spread_line", "total_line", "ats_hit", "total_hit"):
        if column not in existing_cols:
            conn.execute(f"ALTER TABLE game_predictions ADD COLUMN {column} REAL" if column in ("home_spread_line", "total_line")
                         else f"ALTER TABLE game_predictions ADD COLUMN {column} INTEGER")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_prop_predictions (
            game_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            market TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            snapshotted_at TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_value REAL,
            PRIMARY KEY (game_id, player_id, market)
        )
        """
    )
    return conn
```

Update `record_game_predictions` to include the two new columns:

```python
    rows = [
        (
            game["game_id"], game["home_team"], game["away_team"], game["commence_time"], now,
            float(game["home_win_prob"]), float(game["away_win_prob"]),
            game.get("home_cover_prob"), game.get("away_cover_prob"),
            game.get("over_prob"), game.get("under_prob"),
            game.get("home_spread_line"), game.get("total_line"),
        )
        for game in valid_games
    ]
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO game_predictions
                (game_id, home_team, away_team, commence_time, snapshotted_at,
                 home_win_prob, away_win_prob, home_cover_prob, away_cover_prob, over_prob, under_prob,
                 home_spread_line, total_line)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount
```

Update `reconcile_game_predictions` to grade ATS and totals when lines were recorded:

```python
def reconcile_game_predictions(results_df: pd.DataFrame) -> int:
    if results_df.empty:
        return 0
    with contextlib.closing(_connect()) as conn, conn:
        unresolved = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 0", conn)
        if unresolved.empty:
            return 0

        merged = unresolved.merge(results_df, on="game_id", how="inner")
        resolved_count = 0
        for _, row in merged.iterrows():
            home_win = row["home_score"] > row["away_score"]
            predicted_home_win = row["home_win_prob"] >= row["away_win_prob"]
            moneyline_hit = int(predicted_home_win == home_win)

            ats_hit = None
            if pd.notna(row.get("home_spread_line")):
                home_margin = row["home_score"] - row["away_score"]
                home_covered = (home_margin + row["home_spread_line"]) > 0
                predicted_home_cover = (row.get("home_cover_prob") or 0) >= (row.get("away_cover_prob") or 0)
                ats_hit = int(predicted_home_cover == home_covered)

            total_hit = None
            if pd.notna(row.get("total_line")):
                actual_total = row["home_score"] + row["away_score"]
                went_over = actual_total > row["total_line"]
                predicted_over = (row.get("over_prob") or 0) >= (row.get("under_prob") or 0)
                total_hit = int(predicted_over == went_over)

            cursor = conn.execute(
                """
                UPDATE game_predictions
                SET resolved = 1, actual_home_score = ?, actual_away_score = ?,
                    moneyline_hit = ?, ats_hit = ?, total_hit = ?
                WHERE game_id = ? AND resolved = 0
                """,
                (int(row["home_score"]), int(row["away_score"]), moneyline_hit, ats_hit, total_hit, row["game_id"]),
            )
            resolved_count += cursor.rowcount
        return resolved_count
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest tests/test_tracking.py -v`
Expected: PASS

- [ ] **Step 6: Wire the lines through `background_tracking_tick`**

In `src/cfb_predictor/api/routes.py`, add `home_spread_line`/`total_line` to the dict appended in the prediction loop (around line 374):

```python
                predictions.append(
                    {
                        "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                        "commence_time": str(game["gameday"]),
                        "home_spread_line": spread_line, "total_line": total_line,
                        **pred,
                    }
                )
```

- [ ] **Step 7: Run the full test suite**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest -v`
Expected: all pass (note: this repo's local `.venv` has a known unrelated `pydantic`/`cfbd` version conflict breaking collection — if that surfaces, run inside a clean venv or Docker instead of fighting it here).

- [ ] **Step 8: Commit**

```bash
cd /Users/sigey/Documents/Projects/CFB_Predictor
git add src/cfb_predictor/tracking/store.py src/cfb_predictor/api/routes.py tests/test_tracking.py
git commit -m "feat: grade ATS and totals on reconcile, not just moneyline"
```

---

## Task 2: NFL — store spread/total lines and grade ATS/totals on reconcile

**Files:**
- Modify: `src/nfl_predictor/tracking/store.py`
- Modify: `src/nfl_predictor/api/routes.py` (the tracking-tick function, ~line 197-221)
- Test: `tests/test_tracking.py`

**Interfaces:**
- Produces: same as Task 1, applied to `nfl_predictor.tracking.store`.

- [ ] **Step 1: Copy Task 1's schema/record/reconcile changes verbatim into `src/nfl_predictor/tracking/store.py`** (identical file today — same `CREATE TABLE`, `record_game_predictions`, `reconcile_game_predictions` bodies).

- [ ] **Step 2: Write the same three tests as Task 1** into `tests/test_tracking.py`, importing `from nfl_predictor.tracking import store` and `from nfl_predictor import config` instead.

- [ ] **Step 3: Run tests to verify they fail, then pass after the change**

Run: `cd /Users/sigey/Documents/Projects/NFL_Predictor && python -m pytest tests/test_tracking.py -v`

- [ ] **Step 4: Wire spread/total lines into the tracking tick**

NFL's `schedules.fetch_upcoming_games` rows already carry `spread_line`/`total_line` (confirmed: `routes.py` already reads `game.get("spread_line")`/`game.get("total_line")` when building predictions, ~line 207). Add them to the recorded-predictions dict at the same point predictions are appended for storage:

```python
                predictions.append(
                    {
                        "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                        "commence_time": str(game["gameday"]),
                        "home_spread_line": game.get("spread_line"), "total_line": game.get("total_line"),
                        **pred,
                    }
                )
```

(Read the exact surrounding block first — `routes.py:197-221` — before editing, since the dict-construction line numbers may shift slightly from CFB's equivalent.)

- [ ] **Step 5: Run the full test suite**

Run: `cd /Users/sigey/Documents/Projects/NFL_Predictor && python -m pytest -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/sigey/Documents/Projects/NFL_Predictor
git add src/nfl_predictor/tracking/store.py src/nfl_predictor/api/routes.py tests/test_tracking.py
git commit -m "feat: grade ATS and totals on reconcile, not just moneyline"
```

---

## Task 3: CFB — targeted backfill for missed reconcile windows

**Files:**
- Modify: `src/cfb_predictor/tracking/store.py`
- Modify: `src/cfb_predictor/api/routes.py:388-392` (end of `background_tracking_tick`)
- Test: `tests/test_tracking.py`

**Interfaces:**
- Consumes: `store._connect()` (Task 1), `games_data.fetch_current_season_partial()` (already used in `background_tracking_tick`).
- Produces: `store.find_unresolved_games_missing_from_results(results_df: pd.DataFrame) -> pd.DataFrame` — returns rows from `game_predictions` that are `resolved = 0` but whose `game_id` IS present in `results_df` (i.e., a finished game the last tick should have reconciled but didn't). This is what `reconcile_game_predictions` already reconciles when called — the "backfill" is really about *making sure this call happens even after a missed tick*, which is a call-site guarantee, not new grading logic.

- [ ] **Step 1: Write a failing test proving a missed tick still gets caught on the next one**

```python
def test_reconcile_catches_a_game_missed_by_a_prior_tick():
    """Simulates a deploy/restart: the game was snapshotted, its results
    became available, but no tick ran to reconcile it until now."""
    store.record_game_predictions([_future_game(commence_time="2099-01-01T00:00:00Z")])
    # Days later, a tick finally runs with this game now finished.
    results = pd.DataFrame([{"game_id": "g1", "home_score": 21, "away_score": 14}])

    resolved = store.reconcile_game_predictions(results)

    assert resolved == 1
```

- [ ] **Step 2: Run it**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest tests/test_tracking.py -v -k missed_by_a_prior_tick`
Expected: this already PASSES — `reconcile_game_predictions` is idempotent and re-checks all `resolved = 0` rows every call, so the real gap isn't the grading function, it's that `background_tracking_tick` only ever reconciles against `fetch_current_season_partial()`'s **current season**. A game snapshotted in a season that has since rolled over (e.g. bowl season into a new year) would never be looked at again. Confirm this reasoning by reading `games_data.fetch_current_season_partial` (`src/cfb_predictor/data/games.py:212-218`) — it's hardcoded to `CURRENT_SEASON`.

- [ ] **Step 3: Write the actual gap-closing test — a still-unresolved row from a past season**

```python
def test_reconcile_all_seasons_catches_a_prior_season_row_current_season_partial_would_miss(monkeypatch):
    store.record_game_predictions([_future_game(game_id="g_old", commence_time="2024-09-01T00:00:00Z")])
    # games_data.fetch_current_season_partial() only ever returns CURRENT_SEASON rows,
    # so a still-unresolved prior-season game would never be reconciled by the normal tick.
    from cfb_predictor.data import games as games_data
    monkeypatch.setattr(
        games_data, "load_training_data",
        lambda seasons: pd.DataFrame([{"game_id": "g_old", "home_score": 10, "away_score": 24}]),
    )

    resolved = store.backfill_unresolved_games(games_data)

    assert resolved == 1
```

- [ ] **Step 4: Run to verify it fails**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest tests/test_tracking.py -v -k backfill`
Expected: FAIL — `store.backfill_unresolved_games` doesn't exist yet.

- [ ] **Step 5: Implement `backfill_unresolved_games`**

Add to `src/cfb_predictor/tracking/store.py`:

```python
def backfill_unresolved_games(games_data_module) -> int:
    """Targeted backfill: reconciles any still-unresolved snapshot whose
    season isn't CURRENT_SEASON, which the normal tick's
    fetch_current_season_partial() never looks at again. Takes the
    games_data module (not a season list) so it can look up whichever
    season each unresolved row actually belongs to."""
    with contextlib.closing(_connect()) as conn:
        unresolved = pd.read_sql("SELECT game_id FROM game_predictions WHERE resolved = 0", conn)
    if unresolved.empty:
        return 0

    seasons = sorted({int(g[:4]) for g in unresolved["game_id"] if g[:4].isdigit()} | {pd.Timestamp.now().year})
    finished = games_data_module.load_training_data(seasons=seasons)
    if finished.empty:
        return 0
    return reconcile_game_predictions(finished[["game_id", "home_score", "away_score"]])
```

(Note: CFBD `game_id`s are not guaranteed to start with the season year — confirm the real `game_id` format from `src/cfb_predictor/data/games.py`'s `fetch_schedules` before trusting the `g[:4]` season-extraction above; if it doesn't hold, join `unresolved` against a stored `season` column instead — this requires adding `season INTEGER` to the schema in Task 1's migration block, which is a safer, format-independent fix. Prefer that: add `season` to the `game_predictions` table and `record_game_predictions` args now if the `game_id` format check fails.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest tests/test_tracking.py -v`

- [ ] **Step 7: Call it from `background_tracking_tick`**

In `src/cfb_predictor/api/routes.py`, after the existing reconcile call (~line 392):

```python
    try:
        store.backfill_unresolved_games(games_data)
    except Exception:
        logger.exception("backfill_unresolved_games failed")
```

- [ ] **Step 8: Run the full suite and commit**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest -v`

```bash
git add src/cfb_predictor/tracking/store.py src/cfb_predictor/api/routes.py tests/test_tracking.py
git commit -m "feat: backfill unresolved predictions a missed tick would otherwise never catch"
```

---

## Task 4: NFL — targeted backfill for missed reconcile windows

**Files:**
- Modify: `src/nfl_predictor/tracking/store.py`
- Modify: `src/nfl_predictor/api/routes.py`
- Test: `tests/test_tracking.py`

**Interfaces:**
- Produces: `store.backfill_unresolved_games(schedules_module)` — same shape as Task 3, calling whatever NFL's `data/schedules.py` exposes as its all-finished-games loader (confirm the exact function name in `src/nfl_predictor/data/schedules.py` before writing the test — do not assume it matches CFB's `load_training_data` name).

- [ ] **Step 1: Read `src/nfl_predictor/data/schedules.py` to find its finished-games loader function name and signature.**

- [ ] **Step 2: Port Task 3's test and implementation**, substituting the confirmed function name/module.

- [ ] **Step 3: Run tests to verify fail then pass**

Run: `cd /Users/sigey/Documents/Projects/NFL_Predictor && python -m pytest tests/test_tracking.py -v`

- [ ] **Step 4: Wire into the tracking tick, run full suite, commit**

```bash
cd /Users/sigey/Documents/Projects/NFL_Predictor
python -m pytest -v
git add src/nfl_predictor/tracking/store.py src/nfl_predictor/api/routes.py tests/test_tracking.py
git commit -m "feat: backfill unresolved predictions a missed tick would otherwise never catch"
```

---

## Task 5: CFB — per-game post-match verdict

**Files:**
- Modify: `src/cfb_predictor/tracking/store.py`
- Modify: `src/cfb_predictor/api/routes.py`
- Test: `tests/test_tracking.py`

**Interfaces:**
- Consumes: `game_predictions` table columns from Task 1 (`moneyline_hit`, `ats_hit`, `total_hit`, `home_win_prob`, `away_win_prob`, `actual_home_score`, `actual_away_score`).
- Produces: `store.get_game_verdict(game_id: str) -> dict | None`. Returns `None` if the game isn't resolved yet or isn't tracked. Otherwise:
  ```python
  {
      "game_id": str, "resolved": True,
      "moneyline": {"hit": bool, "predicted": "home_win" | "away_win", "actual": "home_win" | "away_win"},
      "ats": {"hit": bool, "predicted": "home_cover" | "away_cover"} | None,
      "totals": {"hit": bool, "predicted": "over" | "under"} | None,
  }
  ```

- [ ] **Step 1: Write failing tests**

```python
def test_get_game_verdict_returns_none_for_unresolved_game():
    store.record_game_predictions([_future_game()])

    assert store.get_game_verdict("g1") is None


def test_get_game_verdict_summarizes_all_three_markets():
    store.record_game_predictions([_future_game()])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}]))

    verdict = store.get_game_verdict("g1")

    assert verdict["resolved"] is True
    assert verdict["moneyline"]["predicted"] == "home_win"
    assert verdict["moneyline"]["actual"] == "home_win"
    assert verdict["moneyline"]["hit"] is True
    assert verdict["ats"]["predicted"] == "home_cover"
    assert verdict["totals"] is not None
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest tests/test_tracking.py -v -k verdict`

- [ ] **Step 3: Implement**

```python
def get_game_verdict(game_id: str) -> dict | None:
    with contextlib.closing(_connect()) as conn:
        rows = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = ?", conn, params=(game_id,))
    if rows.empty or not rows.iloc[0]["resolved"]:
        return None
    row = rows.iloc[0]

    predicted_home_win = row["home_win_prob"] >= row["away_win_prob"]
    actual_home_win = row["actual_home_score"] > row["actual_away_score"]
    verdict = {
        "game_id": game_id,
        "resolved": True,
        "moneyline": {
            "hit": bool(row["moneyline_hit"]),
            "predicted": "home_win" if predicted_home_win else "away_win",
            "actual": "home_win" if actual_home_win else "away_win",
        },
        "ats": None,
        "totals": None,
    }
    if pd.notna(row["ats_hit"]):
        predicted_home_cover = (row["home_cover_prob"] or 0) >= (row["away_cover_prob"] or 0)
        verdict["ats"] = {"hit": bool(row["ats_hit"]), "predicted": "home_cover" if predicted_home_cover else "away_cover"}
    if pd.notna(row["total_hit"]):
        predicted_over = (row["over_prob"] or 0) >= (row["under_prob"] or 0)
        verdict["totals"] = {"hit": bool(row["total_hit"]), "predicted": "over" if predicted_over else "under"}
    return verdict
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest tests/test_tracking.py -v`

- [ ] **Step 5: Expose via API**

In `src/cfb_predictor/api/routes.py`, add near `get_track_record` (~line 340):

```python
@router.get("/games/{game_id}/verdict")
def get_game_verdict(game_id: str):
    verdict = store.get_game_verdict(game_id)
    if verdict is None:
        raise HTTPException(status_code=404, detail="Game not tracked or not yet resolved")
    return verdict
```

Add a route test to `tests/test_api_routes.py`:

```python
def test_get_game_verdict_404s_when_not_resolved(client, monkeypatch):
    monkeypatch.setattr(routes.store, "get_game_verdict", lambda game_id: None)

    response = client.get("/api/games/nope/verdict")

    assert response.status_code == 404
```

- [ ] **Step 6: Run full suite and commit**

```bash
cd /Users/sigey/Documents/Projects/CFB_Predictor
python -m pytest -v
git add src/cfb_predictor/tracking/store.py src/cfb_predictor/api/routes.py tests/test_tracking.py tests/test_api_routes.py
git commit -m "feat: add per-game post-match verdict (moneyline/ATS/totals)"
```

---

## Task 6: NFL — per-game post-match verdict

**Files:**
- Modify: `src/nfl_predictor/tracking/store.py`
- Modify: `src/nfl_predictor/api/routes.py`
- Test: `tests/test_tracking.py`, `tests/test_api_routes.py`

**Interfaces:** identical to Task 5, package `nfl_predictor`.

- [ ] **Step 1-6:** Port Task 5's tests and implementation verbatim into the `nfl_predictor` package paths, run `python -m pytest -v` in `/Users/sigey/Documents/Projects/NFL_Predictor`, then commit:

```bash
cd /Users/sigey/Documents/Projects/NFL_Predictor
git add src/nfl_predictor/tracking/store.py src/nfl_predictor/api/routes.py tests/test_tracking.py tests/test_api_routes.py
git commit -m "feat: add per-game post-match verdict (moneyline/ATS/totals)"
```

---

## Task 7: CFB — calibration reliability curve

**Files:**
- Create: `src/cfb_predictor/evaluate/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: nothing beyond `numpy`/`pandas`/`sklearn.calibration.calibration_curve`.
- Produces: `calibration.reliability_curve(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> pd.DataFrame` with columns `bin_midpoint`, `predicted_mean`, `actual_frequency`, `n_predictions`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_calibration.py
import numpy as np

from cfb_predictor.evaluate import calibration


def test_reliability_curve_perfectly_calibrated_predictions_lie_on_the_diagonal():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 2000)
    outcomes = (rng.uniform(0, 1, 2000) < probs).astype(int)

    curve = calibration.reliability_curve(probs, outcomes, n_bins=10)

    assert set(curve.columns) == {"bin_midpoint", "predicted_mean", "actual_frequency", "n_predictions"}
    # Perfectly calibrated by construction -- actual should track predicted within noise.
    assert (curve["actual_frequency"] - curve["predicted_mean"]).abs().max() < 0.25


def test_reliability_curve_drops_empty_bins():
    probs = np.array([0.05, 0.05, 0.95, 0.95])
    outcomes = np.array([0, 0, 1, 1])

    curve = calibration.reliability_curve(probs, outcomes, n_bins=10)

    assert curve["n_predictions"].sum() == 4
    assert (curve["n_predictions"] > 0).all()
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest tests/test_calibration.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

```python
"""calibration.py -- measurement-only calibration diagnostics. Does not
correct/adjust any model output; see docs/superpowers/specs/
2026-09-08-cross-sport-accuracy-tracking-parity-design.md for the scope
decision (measurement only, no Platt/isotonic correction layer)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_curve(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges[1:-1]), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin_midpoint": float((bin_edges[b] + bin_edges[b + 1]) / 2),
            "predicted_mean": float(probs[mask].mean()),
            "actual_frequency": float(outcomes[mask].mean()),
            "n_predictions": n,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest tests/test_calibration.py -v`

- [ ] **Step 5: Wire into `evaluate/walk_forward.py`'s output** — add a `reliability_curve` field to `evaluate_candidate`'s per-fold output isn't right (folds have too few rows individually); instead expose it as a second function callable on `evaluate_candidate`'s pooled predictions. Add to `walk_forward.py`:

```python
def pooled_predictions(folds: list[dict], candidate: str) -> tuple[np.ndarray, np.ndarray]:
    """All folds' held-out probs/outcomes concatenated -- enough volume for
    a meaningful reliability curve, unlike any single fold alone."""
    all_probs, all_outcomes = [], []
    for fold in folds:
        train_df, val_df, feature_cols = fold["train_df"], fold["val_df"], fold["feature_cols"]
        preds, sigma = _predict_margins(candidate, train_df, val_df, feature_cols)
        probs = np.array([game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds])
        all_probs.append(np.clip(probs, 1e-6, 1 - 1e-6))
        all_outcomes.append((val_df["margin"] > 0).astype(int).to_numpy())
    return np.concatenate(all_probs), np.concatenate(all_outcomes)
```

Add a matching test:

```python
def test_pooled_predictions_concatenates_every_folds_probs_and_outcomes(monkeypatch):
    # Reuses existing walk_forward test fixtures/mocks -- read
    # tests/test_walk_forward.py first to match its existing fold-building
    # setup instead of duplicating it here.
    ...
```

(Read `tests/test_walk_forward.py` before writing this test — reuse its existing fixture for building `folds`, do not hand-roll a second one.)

- [ ] **Step 6: Run full suite and commit**

```bash
cd /Users/sigey/Documents/Projects/CFB_Predictor
python -m pytest -v
git add src/cfb_predictor/evaluate/calibration.py src/cfb_predictor/evaluate/walk_forward.py tests/test_calibration.py tests/test_walk_forward.py
git commit -m "feat: add calibration reliability curve (measurement only)"
```

---

## Task 8: NFL — calibration reliability curve

**Files:**
- Create: `src/nfl_predictor/evaluate/calibration.py`
- Modify: `src/nfl_predictor/evaluate/walk_forward.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1-6:** Port Task 7 verbatim into `nfl_predictor` paths (its `walk_forward.py` has the same `_predict_margins`/`game_outcome.margin_to_probabilities` shape). Run `python -m pytest -v` in `/Users/sigey/Documents/Projects/NFL_Predictor`, then commit:

```bash
cd /Users/sigey/Documents/Projects/NFL_Predictor
git add src/nfl_predictor/evaluate/calibration.py src/nfl_predictor/evaluate/walk_forward.py tests/test_calibration.py
git commit -m "feat: add calibration reliability curve (measurement only)"
```

---

## Task 9: CFB — week-view endpoint (past + future predictions)

**Files:**
- Modify: `src/cfb_predictor/tracking/store.py`
- Modify: `src/cfb_predictor/api/routes.py`
- Test: `tests/test_tracking.py`, `tests/test_api_routes.py`

**Interfaces:**
- Consumes: `get_game_verdict` (Task 5), `game_predictions` table.
- Produces: `store.get_predictions_for_week(season: int, week: int, games_df: pd.DataFrame) -> list[dict]`. `games_df` supplies the week's game_ids (from `games_data.fetch_upcoming_games`/`load_training_data`) since the tracking DB has no `season`/`week` columns of its own; for each `game_id`, look up its tracked prediction row and, if resolved, its verdict.

- [ ] **Step 1: Write failing test**

```python
def test_get_predictions_for_week_returns_pending_for_unresolved_and_verdict_for_resolved():
    store.record_game_predictions([_future_game(game_id="g1"), _future_game(game_id="g2", home_team="Alabama", away_team="Auburn")])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}]))
    games_df = pd.DataFrame([
        {"game_id": "g1", "home_team": "Texas", "away_team": "Oklahoma"},
        {"game_id": "g2", "home_team": "Alabama", "away_team": "Auburn"},
    ])

    week = store.get_predictions_for_week(2026, 1, games_df)

    by_id = {row["game_id"]: row for row in week}
    assert by_id["g1"]["status"] == "resolved"
    assert by_id["g1"]["verdict"]["moneyline"]["hit"] is True
    assert by_id["g2"]["status"] == "pending"
    assert by_id["g2"]["verdict"] is None
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest tests/test_tracking.py -v -k predictions_for_week`

- [ ] **Step 3: Implement**

```python
def get_predictions_for_week(season: int, week: int, games_df: pd.DataFrame) -> list[dict]:
    if games_df.empty:
        return []
    game_ids = tuple(games_df["game_id"])
    placeholders = ",".join("?" * len(game_ids))
    with contextlib.closing(_connect()) as conn:
        tracked = pd.read_sql(
            f"SELECT * FROM game_predictions WHERE game_id IN ({placeholders})", conn, params=game_ids
        )
    tracked_by_id = {row["game_id"]: row for _, row in tracked.iterrows()}

    results = []
    for _, game in games_df.iterrows():
        row = tracked_by_id.get(game["game_id"])
        if row is None:
            results.append({"game_id": game["game_id"], "status": "untracked", "verdict": None})
            continue
        resolved = bool(row["resolved"])
        results.append({
            "game_id": game["game_id"],
            "status": "resolved" if resolved else "pending",
            "home_win_prob": row["home_win_prob"],
            "away_win_prob": row["away_win_prob"],
            "verdict": get_game_verdict(game["game_id"]) if resolved else None,
        })
    return results
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/sigey/Documents/Projects/CFB_Predictor && python -m pytest tests/test_tracking.py -v`

- [ ] **Step 5: Expose via API**

In `src/cfb_predictor/api/routes.py`:

```python
@router.get("/predictions/{season}/{week}")
def get_predictions_for_week(season: int, week: int):
    games = games_data.fetch_upcoming_games(season, week)
    if games.empty:
        games = games_data.load_training_data(seasons=[season])
        games = games[games["week"] == week]
    return store.get_predictions_for_week(season, week, games)
```

Add a route test to `tests/test_api_routes.py` using the existing `client` fixture's monkeypatched `games_data`/`store`.

- [ ] **Step 6: Run full suite and commit**

```bash
cd /Users/sigey/Documents/Projects/CFB_Predictor
python -m pytest -v
git add src/cfb_predictor/tracking/store.py src/cfb_predictor/api/routes.py tests/test_tracking.py tests/test_api_routes.py
git commit -m "feat: add week-view endpoint (past resolved + future pending predictions)"
```

---

## Task 10: NFL — week-view endpoint (past + future predictions)

**Files:**
- Modify: `src/nfl_predictor/tracking/store.py`
- Modify: `src/nfl_predictor/api/routes.py`
- Test: `tests/test_tracking.py`, `tests/test_api_routes.py`

- [ ] **Step 1-6:** Port Task 9 verbatim into `nfl_predictor` paths, substituting `schedules.fetch_upcoming_games`/whatever finished-games loader Task 4 confirmed for the "past week already finished" branch. Run `python -m pytest -v` in `/Users/sigey/Documents/Projects/NFL_Predictor`, then commit:

```bash
cd /Users/sigey/Documents/Projects/NFL_Predictor
git add src/nfl_predictor/tracking/store.py src/nfl_predictor/api/routes.py tests/test_tracking.py tests/test_api_routes.py
git commit -m "feat: add week-view endpoint (past resolved + future pending predictions)"
```

---

## Self-Review Notes

- **Spec coverage:** Measurement layer → Tasks 7-8. Backfill → Tasks 3-4. Post-match verdict → Tasks 5-6. Week view → Tasks 9-10. ATS/totals grading (a prerequisite the spec assumed existed but doesn't — `moneyline_hit` was the only graded market in both repos before this plan) → Tasks 1-2. Code modularity → no separate task; every new function slots into the existing `tracking/store.py` / `evaluate/` modules per repo, matching the spec's call.
- **Deferred/conditional items not included as tasks:** CFB secondary data-provider research (explicitly conditional on CFBD proving insufficient — not triggered by anything in this plan) and the Platt/isotonic correction layer (explicitly out of scope) are correctly absent.
- **Known open risk flagged inline:** Task 3, Step 5 flags that CFBD's `game_id` format needs verifying before trusting the season-extraction shortcut — the safer fallback (add a `season` column to the schema) is spelled out as a fallback in that same step rather than left as a TODO.
