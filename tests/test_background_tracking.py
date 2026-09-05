import pandas as pd
import pytest

from cfb_predictor.api import routes


def test_background_tracking_tick_records_and_reconciles(monkeypatch):
    calls = {"recorded": 0, "reconciled": 0}

    monkeypatch.setattr(
        routes.games_data, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "g1", "home_team": "Texas", "away_team": "Ohio State", "gameday": "2025-08-30"}]
        ),
    )
    monkeypatch.setattr(
        routes.games_data, "load_training_data",
        lambda seasons: pd.DataFrame(
            [{"game_id": "g0", "season": seasons[0], "week": 1, "gameday": "2025-08-01",
              "home_team": "Texas", "away_team": "Ohio State", "home_score": 24, "away_score": 17}]
        ),
    )
    monkeypatch.setattr(routes.odds_api, "fetch_game_odds", lambda: pd.DataFrame())
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {
        "game_outcome_model": None, "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
        "total_model": None, "player_models": {"feature_cols": [], "anytime_td": None},
        "feature_cols": [], "player_feature_cols": [],
    })
    monkeypatch.setattr(routes, "_predict_game_from_models", lambda *a, **k: {
        "home_win_prob": 0.4, "away_win_prob": 0.6, "home_cover_prob": 0.45,
        "away_cover_prob": 0.55, "over_prob": 0.52, "under_prob": 0.48,
    })
    monkeypatch.setattr(
        routes.store, "record_game_predictions",
        lambda games: calls.__setitem__("recorded", calls["recorded"] + len(games)) or len(games),
    )
    monkeypatch.setattr(
        routes.store, "reconcile_game_predictions",
        lambda results_df: calls.__setitem__("reconciled", calls["reconciled"] + 1) or 0,
    )
    monkeypatch.setattr(routes.games_data, "fetch_current_season_partial", lambda: pd.DataFrame(columns=["game_id", "home_score", "away_score"]))

    routes.background_tracking_tick(season=2025, week=1)

    assert calls["recorded"] == 1
    assert calls["reconciled"] == 1
