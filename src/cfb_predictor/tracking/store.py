"""SQLite persistence for the live prediction track record.

Snapshots each game's core-market predictions and each tracked player prop
before kickoff, then reconciles them against actual results once games are
played. Snapshot rows are immutable (``INSERT OR IGNORE``); reconciliation
only fills outcome columns on existing, unresolved rows. Verbatim port of
nfl_predictor/tracking/store.py — this schema has zero sport-specific logic.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from ..config import TRACKING_DB_PATH


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
    for column in ("home_spread_line", "total_line", "ats_hit", "total_hit", "season"):
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


def _require_pre_kickoff(commence_time: str) -> None:
    kickoff = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    if kickoff <= datetime.now(timezone.utc):
        raise ValueError("Game predictions must be snapshotted before kickoff")


def record_game_predictions(games: list[dict]) -> int:
    """Snapshots only the games that are still pre-kickoff. A single
    already-kicked-off (or malformed) game in the batch used to raise for
    the whole call, silently dropping every other valid game in the same
    tick along with it -- skip just the invalid ones instead (see this
    plan's final-review fix, Task 23, finding I4)."""
    if not games:
        return 0
    valid_games = []
    for game in games:
        try:
            _require_pre_kickoff(game["commence_time"])
            valid_games.append(game)
        except ValueError:
            continue
    if not valid_games:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            game["game_id"], game["home_team"], game["away_team"], game["commence_time"], now,
            float(game["home_win_prob"]), float(game["away_win_prob"]),
            game.get("home_cover_prob"), game.get("away_cover_prob"),
            game.get("over_prob"), game.get("under_prob"),
            game.get("home_spread_line"), game.get("total_line"), game.get("season"),
        )
        for game in valid_games
    ]
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO game_predictions
                (game_id, home_team, away_team, commence_time, snapshotted_at,
                 home_win_prob, away_win_prob, home_cover_prob, away_cover_prob, over_prob, under_prob,
                 home_spread_line, total_line, season)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


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


def backfill_unresolved_games(games_data_module) -> int:
    """Targeted backfill: reconciles any still-unresolved snapshot whose
    season isn't CURRENT_SEASON, which the normal tick's
    fetch_current_season_partial() never looks at again. Takes the
    games_data module (not a season list) so it can look up whichever
    season each unresolved row actually belongs to."""
    with contextlib.closing(_connect()) as conn:
        unresolved = pd.read_sql("SELECT season FROM game_predictions WHERE resolved = 0", conn)
    if unresolved.empty:
        return 0

    seasons = sorted({int(s) for s in unresolved["season"].dropna()} | {pd.Timestamp.now().year})
    finished = games_data_module.load_training_data(seasons=seasons)
    if finished.empty:
        return 0
    return reconcile_game_predictions(finished[["game_id", "home_score", "away_score"]])


def get_track_record() -> dict:
    with contextlib.closing(_connect()) as conn, conn:
        resolved = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 1", conn)
    if resolved.empty:
        return {"n_resolved_games": 0, "pct_moneyline_correct": None}
    return {
        "n_resolved_games": int(len(resolved)),
        "pct_moneyline_correct": float(resolved["moneyline_hit"].mean()),
    }


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


def record_player_prop_predictions(props: list[dict]) -> int:
    if not props:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (prop["game_id"], prop["player_id"], prop["player_name"], prop["market"], float(prop["predicted_value"]), now)
        for prop in props
    ]
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO player_prop_predictions
                (game_id, player_id, player_name, market, predicted_value, snapshotted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


_MARKET_TO_STAT_COLUMN = {
    "anytime_td": None,
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
}


def reconcile_player_prop_predictions(player_stats_df: pd.DataFrame) -> int:
    if player_stats_df.empty:
        return 0
    with contextlib.closing(_connect()) as conn, conn:
        unresolved = pd.read_sql("SELECT * FROM player_prop_predictions WHERE resolved = 0", conn)
        if unresolved.empty:
            return 0

        merged = unresolved.merge(player_stats_df, on=["game_id", "player_id"], how="inner")
        resolved_count = 0
        for _, row in merged.iterrows():
            if row["market"] == "anytime_td":
                actual = float(
                    (row.get("rushing_tds", 0) or 0)
                    + (row.get("receiving_tds", 0) or 0)
                    + (row.get("passing_tds", 0) or 0)
                    > 0
                )
            else:
                stat_col = _MARKET_TO_STAT_COLUMN[row["market"]]
                if stat_col not in row or pd.isna(row[stat_col]):
                    continue
                actual = float(row[stat_col])
            cursor = conn.execute(
                """
                UPDATE player_prop_predictions
                SET resolved = 1, actual_value = ?
                WHERE game_id = ? AND player_id = ? AND market = ? AND resolved = 0
                """,
                (actual, row["game_id"], row["player_id"], row["market"]),
            )
            resolved_count += cursor.rowcount
        return resolved_count
