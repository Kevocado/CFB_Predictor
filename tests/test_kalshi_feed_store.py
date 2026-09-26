"""Task 1: the store's feed columns, feed reader and calibration reader.

Adapted from the plan, which added a `backfilled` INTEGER column and a one-time legacy
classification. PR #1 already merged a better version of that idea: `_snapshotted_after_kickoff`,
i.e. `snapshotted_at >= commence_time`, computed live from two columns that always existed, and
failing CLOSED (an unparseable row counts as not-pre-game). So this task:

- adds the five distribution columns (that part is still a schema change), and
- derives "pre-game" from the existing predicate instead of storing a flag, which cannot go stale
  and needs no migration pass over legacy rows.

`/api/track-record` already reports `n_rebuilt` from that same predicate, so the plan's
`n_backfilled` is not added -- it would be the same number under a second name.
"""
import contextlib
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import pytest

from cfb_predictor.tracking import store

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from cfb_predictor import config

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield


def _upcoming(**overrides):
    game = {
        "game_id": "401871100", "home_team": "Stanford", "away_team": "Georgia Tech",
        "commence_time": "2099-09-27 20:25:00", "season": 2026, "week": 3,
        "home_win_prob": 0.41, "away_win_prob": 0.59,
        "home_cover_prob": 0.47, "away_cover_prob": 0.53, "over_prob": 0.52, "under_prob": 0.48,
        "home_spread_line": -2.5, "total_line": 47.5,
        "predicted_margin": -2.9, "sigma": 13.2, "predicted_total": 48.1, "total_sigma": 12.5,
        "model_version": "ridge@2026-08-20T10:00:00+00:00",
    }
    game.update(overrides)
    return game


def _finished(**overrides):
    game = _upcoming(game_id="401869941", home_team="Coastal Carolina", away_team="Liberty",
                     commence_time="2026-09-20 17:00:00", home_win_prob=0.7, away_win_prob=0.3,
                     actual_home_score=30, actual_away_score=10)
    game.update(overrides)
    return game


def test_record_game_predictions_persists_the_distribution():
    store.record_game_predictions([_upcoming()])

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions", conn).iloc[0]

    assert row["predicted_margin"] == pytest.approx(-2.9)
    assert row["sigma"] == pytest.approx(13.2)
    assert row["predicted_total"] == pytest.approx(48.1)
    assert row["total_sigma"] == pytest.approx(12.5)
    assert row["model_version"] == "ridge@2026-08-20T10:00:00+00:00"


def test_record_resolved_game_predictions_also_persists_the_distribution():
    """A resolved row still has a distribution behind it; the feed does not serve it, but
    calibration and any future analysis read the same table."""
    store.record_resolved_game_predictions([_finished()])

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions", conn).iloc[0]

    assert row["predicted_margin"] == pytest.approx(-2.9)
    assert row["resolved"] == 1


def test_legacy_rows_need_no_migration():
    """The plan added a `backfilled` column and a one-time UPDATE over legacy rows. There is no
    column to migrate here: pre-game-ness is derived, so a database created before this change
    works unchanged and cannot drift."""
    conn = sqlite3.connect(str(store.TRACKING_DB_PATH))
    conn.execute(
        """CREATE TABLE game_predictions (
            game_id TEXT PRIMARY KEY, home_team TEXT NOT NULL, away_team TEXT NOT NULL,
            commence_time TEXT NOT NULL, snapshotted_at TEXT NOT NULL,
            home_win_prob REAL NOT NULL, away_win_prob REAL NOT NULL,
            home_cover_prob REAL, away_cover_prob REAL, over_prob REAL, under_prob REAL,
            resolved INTEGER NOT NULL DEFAULT 0, actual_home_score INTEGER, actual_away_score INTEGER,
            moneyline_hit INTEGER)"""
    )
    conn.executemany(
        "INSERT INTO game_predictions (game_id, home_team, away_team, commence_time, snapshotted_at,"
        " home_win_prob, away_win_prob, resolved) VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
        [
            ("pregame", "Stanford", "Georgia Tech", "2099-09-13 00:00:00", "2026-09-12T15:00:00+00:00", 0.6, 0.4),
            ("rebuilt", "Coastal Carolina", "Liberty", "2026-09-13 00:00:00", "2026-09-24T02:00:00+00:00", 0.6, 0.4),
        ],
    )
    conn.commit()
    conn.close()

    # _connect() adds the new columns, and the classification comes from the existing predicate.
    with contextlib.closing(store._connect()) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(game_predictions)")}
    assert {"predicted_margin", "sigma", "predicted_total", "total_sigma", "model_version"} <= cols
    assert "backfilled" not in cols, "a stored flag is exactly what the rebuilt rule replaced"
    assert store._snapshotted_after_kickoff("2026-09-12T15:00:00+00:00", "2099-09-13 00:00:00") is False
    assert store._snapshotted_after_kickoff("2026-09-24T02:00:00+00:00", "2026-09-13 00:00:00") is True


def test_feed_predictions_returns_only_upcoming_pregame_rows():
    store.record_game_predictions([_upcoming()])
    store.record_resolved_game_predictions([_finished()])

    rows = store.get_feed_predictions(now=NOW)

    assert [r["game_id"] for r in rows] == ["401871100"]
    row = rows[0]
    assert row["home"] == "Stanford" and row["away"] == "Georgia Tech"
    assert row["start_utc"] == "2099-09-27T20:25:00+00:00"
    assert row["p_home"] == pytest.approx(0.41)
    assert row["margin_mu"] == pytest.approx(-2.9) and row["sigma"] == pytest.approx(13.2)
    assert row["total_mu"] == pytest.approx(48.1) and row["total_sigma"] == pytest.approx(12.5)
    assert row["model_version"] == "ridge@2026-08-20T10:00:00+00:00"
    # The feed contract keeps the key: the hub rejects a truthy value, and False is the honest
    # statement about a row that passed the rebuilt rule.
    assert row["backfilled"] is False
    assert datetime.fromisoformat(row["snapshotted_at"]) < datetime.fromisoformat(row["start_utc"])


def test_feed_predictions_drop_games_that_have_started():
    store.record_game_predictions([_upcoming()])

    assert store.get_feed_predictions(now=datetime(2100, 1, 1, tzinfo=timezone.utc)) == []


def test_feed_predictions_exclude_a_row_rebuilt_after_the_fact():
    """The leakage rule (spec 5a). A row written at or after kickoff is exactly what PR #1 calls
    `rebuilt`, and it must never reach the feed even if the game has not started."""
    with contextlib.closing(store._connect()) as conn:
        conn.execute(
            """INSERT INTO game_predictions (game_id, home_team, away_team, commence_time, snapshotted_at,
                   home_win_prob, away_win_prob, resolved)
               VALUES ('rebuilt', 'DAL', 'BAL', '2099-09-27 20:25:00', '2099-09-27 21:00:00+00:00', 0.6, 0.4, 0)"""
        )
        conn.commit()

    store.record_game_predictions([_upcoming()])

    assert [r["game_id"] for r in store.get_feed_predictions(now=NOW)] == ["401871100"]


def test_feed_predictions_keep_null_distribution_fields_as_none():
    legacy = _upcoming()
    for key in ("predicted_margin", "sigma", "predicted_total", "total_sigma", "model_version"):
        legacy.pop(key)
    store.record_game_predictions([legacy])

    row = store.get_feed_predictions(now=NOW)[0]

    assert row["margin_mu"] is None and row["sigma"] is None
    assert row["total_mu"] is None and row["model_version"] is None


def test_calibration_uses_only_pregame_resolved_rows():
    store.record_game_predictions([
        _upcoming(game_id=f"g{i}", home_win_prob=0.65, away_win_prob=0.35) for i in range(4)
    ])
    store.reconcile_game_predictions(pd.DataFrame([
        {"game_id": "g0", "home_score": 24, "away_score": 17},
        {"game_id": "g1", "home_score": 24, "away_score": 17},
        {"game_id": "g2", "home_score": 24, "away_score": 17},
        {"game_id": "g3", "home_score": 10, "away_score": 17},
    ]))
    # A rebuilt 0.65 miss must not count.
    store.record_resolved_game_predictions([
        _finished(game_id="bf", home_win_prob=0.65, away_win_prob=0.35, actual_home_score=0, actual_away_score=7)
    ])

    calibration = store.get_calibration()

    bucket = next(b for b in calibration["winner"] if b["lo"] == 0.6)
    assert bucket == {"lo": 0.6, "hi": 0.7, "n": 4, "mean_prob": pytest.approx(0.65), "hit_rate": 0.75}
    assert len(calibration["winner"]) == 10
    empty = next(b for b in calibration["winner"] if b["lo"] == 0.1)
    assert empty == {"lo": 0.1, "hi": 0.2, "n": 0, "mean_prob": None, "hit_rate": None}


def test_calibration_grades_spread_and_total_against_recorded_lines():
    store.record_game_predictions([_upcoming(game_id="s1", home_spread_line=3.0, home_cover_prob=0.55,
                                             total_line=40.0, over_prob=0.35)])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "s1", "home_score": 27, "away_score": 20}]))

    calibration = store.get_calibration()

    spread = next(b for b in calibration["spread"] if b["lo"] == 0.5)
    total = next(b for b in calibration["total"] if b["lo"] == 0.3)
    assert (spread["n"], spread["hit_rate"]) == (1, 1.0)   # margin 7 > 3
    assert (total["n"], total["hit_rate"]) == (1, 1.0)     # 47 > 40


def test_the_track_record_keeps_reporting_the_rebuilt_count():
    """PR #1 already reports `n_rebuilt` from the same predicate, and excludes rebuilt rows from
    `n_resolved` before counting. The plan's `n_backfilled` would be the same number under a
    second name, so it is not added."""
    store.record_resolved_game_predictions([_finished()])
    # A genuinely pre-game snapshot, resolved: counted as resolved.
    store.record_game_predictions([_upcoming(game_id="tracked")])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "tracked", "home_score": 21, "away_score": 14}]))

    games = store.get_track_record()["games"]

    assert games["n_rebuilt"] == 1, games
    assert games["n_resolved"] == 1, games          # only the tracked one
    assert "n_backfilled" not in games
