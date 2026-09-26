"""Task 6: freeze each game's snapshot inside a lead window, and record which model made it.

`current_season_and_week()` anchors on the date of week 1's first kickoff, so it rolls over mid
week. Two games were therefore never snapshotted pre-kickoff and never tracked at all:

- Friday-night games, because "this week" had already rolled to the next one by the time the
  window opened;
- CFB week-N+1 games, frozen on the previous Saturday, before that day's results.

A per-game lead window fixes both, and the first snapshot inside it is kept forever
(`INSERT OR IGNORE`), so it is the one the track record grades and the Kalshi feed serves.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd

from cfb_predictor.api import routes
from cfb_predictor.models import manifest

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def _game(game_id, week, hours_from_now):
    # CFBD's Game model hands back tz-aware UTC start dates.
    kickoff = pd.Timestamp(NOW + timedelta(hours=hours_from_now))
    return {"game_id": game_id, "season": 2026, "week": week, "gameday": kickoff,
            "home_team": "Stanford", "away_team": "Georgia Tech", "home_score": None, "away_score": None}


def test_games_to_snapshot_takes_both_weeks_but_only_inside_the_lead_window(monkeypatch):
    weeks = {
        4: pd.DataFrame([_game("in_window", 4, 40), _game("too_early", 4, 100)]),
        5: pd.DataFrame([_game("next_week_soon", 5, 47), _game("next_week_later", 5, 200)]),
    }
    monkeypatch.setattr(routes.games_data, "fetch_upcoming_games", lambda season, week: weeks.get(week, pd.DataFrame()))

    games = routes._games_to_snapshot(2026, 4, NOW, lead_hours=48)

    assert list(games["game_id"]) == ["in_window", "next_week_soon"]


def test_games_to_snapshot_handles_an_empty_next_week(monkeypatch):
    weeks = {4: pd.DataFrame([_game("in_window", 4, 24)])}
    monkeypatch.setattr(routes.games_data, "fetch_upcoming_games", lambda season, week: weeks.get(week, pd.DataFrame()))

    games = routes._games_to_snapshot(2026, 4, NOW, lead_hours=48)

    assert list(games["game_id"]) == ["in_window"]


def test_the_lead_window_keeps_past_games_so_the_reconciler_still_sees_them(monkeypatch):
    """The window filters on the upper bound only. `record_game_predictions` rejects a game that
    has already kicked off, and the tick's existing test depends on past games reaching it."""
    weeks = {4: pd.DataFrame([_game("already_played", 4, -30), _game("in_window", 4, 24)])}
    monkeypatch.setattr(routes.games_data, "fetch_upcoming_games", lambda season, week: weeks.get(week, pd.DataFrame()))

    games = routes._games_to_snapshot(2026, 4, NOW, lead_hours=48)

    assert set(games["game_id"]) == {"already_played", "in_window"}


def test_games_to_snapshot_drops_a_game_with_an_unparseable_kickoff(monkeypatch):
    """A NaT kickoff cannot be placed relative to the window, so it must not be snapshotted at a
    moment that is not 48h before kickoff."""
    bad = _game("no_kickoff", 4, 24)
    bad["gameday"] = pd.NaT
    weeks = {4: pd.DataFrame([bad, _game("in_window", 4, 24)])}
    monkeypatch.setattr(routes.games_data, "fetch_upcoming_games", lambda season, week: weeks.get(week, pd.DataFrame()))

    games = routes._games_to_snapshot(2026, 4, NOW, lead_hours=48)

    assert list(games["game_id"]) == ["in_window"]


def test_the_lead_window_is_configurable_and_defaults_to_48_hours(monkeypatch):
    assert routes.SNAPSHOT_LEAD_HOURS == 48.0


def test_tracking_tick_records_distribution_and_each_games_own_week(monkeypatch):
    recorded = []
    monkeypatch.setattr(routes, "_games_to_snapshot", lambda season, week, now, lead_hours=None: pd.DataFrame(
        [_game("401871049", 5, 30)]))
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {"model_version": "ridge@t"})
    monkeypatch.setattr(routes, "_load_game_history", lambda season: pd.DataFrame())
    monkeypatch.setattr(routes.sportsbook_api, "fetch_game_odds", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(routes, "_predict_game_from_models", lambda *a, **k: {
        "home_win_prob": 0.4, "away_win_prob": 0.6, "predicted_margin": -3.4, "sigma": 16.0,
        "predicted_total": 55.0, "total_sigma": 14.0, "model_version": "ridge@t",
    })
    monkeypatch.setattr(routes.store, "record_game_predictions", lambda games: recorded.extend(games) or len(games))
    monkeypatch.setattr(routes, "_get_player_props_live", lambda season, week: [])
    monkeypatch.setattr(routes.store, "record_player_prop_predictions", lambda rows: 0)
    monkeypatch.setattr(routes.games_data, "fetch_current_season_partial",
                        lambda: pd.DataFrame(columns=["game_id", "home_score", "away_score"]))
    monkeypatch.setattr(routes.store, "reconcile_game_predictions", lambda df: 0)
    monkeypatch.setattr(routes.store, "backfill_unresolved_games", lambda module: 0)

    routes.background_tracking_tick(season=2026, week=4)

    assert len(recorded) == 1
    row = recorded[0]
    # The game is a week-5 game being snapshotted during the week-4 tick. Labelling it week 4
    # would put it in the wrong week everywhere downstream.
    assert row["week"] == 5
    assert row["predicted_margin"] == -3.4 and row["sigma"] == 16.0
    assert row["model_version"] == "ridge@t"


def test_predict_game_from_models_reports_model_version(monkeypatch):
    class _FakeTotalModel:
        def predict(self, _X):
            return [55.0]

    monkeypatch.setattr(
        routes.feature_build, "build_features_for_game",
        lambda home, away, games_df: pd.Series({"rating_diff": 50.0, "home_rest_days": 7.0, "away_rest_days": 7.0}),
    )
    models = {
        "feature_cols": ["rating_diff", "home_rest_days", "away_rest_days"],
        "chosen_candidate": "elo", "sigma": 16.0, "total_sigma": 14.0,
        "total_model": _FakeTotalModel(), "model_version": "elo@2026-08-20T10:00:00+00:00",
    }

    result = routes._predict_game_from_models(models, "Stanford", "Georgia Tech", pd.DataFrame())

    assert result["model_version"] == "elo@2026-08-20T10:00:00+00:00"


def test_model_version_combines_candidate_and_training_time():
    assert manifest.model_version(
        {"chosen_candidate": "ridge", "trained_at": "2026-08-20T10:00:00+00:00"}
    ) == "ridge@2026-08-20T10:00:00+00:00"


def test_load_models_exposes_the_version(monkeypatch):
    monkeypatch.setattr(manifest, "load_manifest", lambda: {
        "chosen_candidate": "ridge", "trained_at": "2026-08-20T10:00:00+00:00", "sigma": 16.0,
        "total_sigma": 14.0, "feature_cols": ["rating_diff"], "player_feature_cols": ["p"],
        "yardage_metrics": [],
    })
    monkeypatch.setattr(manifest, "_load_pickle", lambda path: object())
    monkeypatch.setattr(manifest, "_artifact_path", lambda name: object())

    models = manifest.load_models()

    assert models["model_version"] == "ridge@2026-08-20T10:00:00+00:00"
