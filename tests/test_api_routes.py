import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cfb_predictor.api.main import app
from cfb_predictor.api import routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        routes.games_data, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "401520145", "season": season, "week": week,
              "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State",
              "home_score": None, "away_score": None}]
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
    monkeypatch.setattr(
        routes, "_load_models_cached",
        lambda: {
            "game_outcome_model": None, "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
            "total_model": None, "player_models": {"feature_cols": [], "anytime_td": None},
            "feature_cols": [], "player_feature_cols": [],
        },
    )
    monkeypatch.setattr(
        routes, "_predict_game_from_models",
        lambda models, home, away, games_df, spread_line=None, total_line=None: {
            "home_win_prob": 0.4, "away_win_prob": 0.6, "home_cover_prob": 0.45, "away_cover_prob": 0.55,
            "over_prob": 0.52, "under_prob": 0.48,
        },
    )
    monkeypatch.setattr(routes.store, "get_track_record", lambda: {"n_resolved_games": 0, "pct_moneyline_correct": None})
    return TestClient(app)


def test_get_games_returns_week_slate(client):
    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["game_id"] == "401520145"


def test_get_game_prediction(client):
    response = client.get("/api/games/2025/1/401520145/prediction")

    assert response.status_code == 200
    body = response.json()
    assert body["home_win_prob"] == 0.4


def test_get_game_prediction_404s_for_unknown_game(client):
    response = client.get("/api/games/2025/1/nonexistent/prediction")

    assert response.status_code == 404


def test_get_track_record(client):
    response = client.get("/api/track-record")

    assert response.status_code == 200
    assert response.json()["n_resolved_games"] == 0


def test_get_game_verdict_404s_when_not_resolved(client, monkeypatch):
    monkeypatch.setattr(routes.store, "get_game_verdict", lambda game_id: None)

    response = client.get("/api/games/nope/verdict")

    assert response.status_code == 404


def test_get_games_handles_nan_scores_for_unplayed_games(client, monkeypatch):
    monkeypatch.setattr(
        routes.games_data, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "401520145", "season": season, "week": week,
              "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State",
              "home_score": float("nan"), "away_score": float("nan")}]
        ),
    )

    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["home_score"] is None
    assert body[0]["away_score"] is None


def test_lines_for_game_reads_spreads_and_totals_from_odds_api():
    odds_df = pd.DataFrame(
        [
            {"home_team": "Texas", "away_team": "Ohio State", "market": "spreads", "outcome_name": "Texas", "point": -3.5},
            {"home_team": "Texas", "away_team": "Ohio State", "market": "totals", "outcome_name": "Over", "point": 51.5},
        ]
    )

    spread_line, total_line = routes._lines_for_game(odds_df, "Texas", "Ohio State")

    assert spread_line == -3.5
    assert total_line == 51.5


def test_lines_for_game_returns_none_when_odds_missing():
    spread_line, total_line = routes._lines_for_game(pd.DataFrame(), "Texas", "Ohio State")

    assert spread_line is None
    assert total_line is None


def test_lines_for_game_matches_cfbd_school_name_to_odds_api_full_name():
    """CFBD's Game model carries short school names ("Texas") while The
    Odds API uses full team names ("Texas Longhorns") -- exact-string
    matching between the two never matches in practice (see this plan's
    final-review fix, Task 23, finding I5)."""
    odds_df = pd.DataFrame(
        [
            {"home_team": "Texas Longhorns", "away_team": "Ohio State Buckeyes", "market": "spreads",
             "outcome_name": "Texas Longhorns", "point": -3.5},
            {"home_team": "Texas Longhorns", "away_team": "Ohio State Buckeyes", "market": "totals",
             "outcome_name": "Over", "point": 51.5},
        ]
    )

    spread_line, total_line = routes._lines_for_game(odds_df, "Texas", "Ohio State")

    assert spread_line == -3.5
    assert total_line == 51.5


def test_lines_for_game_returns_none_for_genuinely_non_matching_teams():
    assert routes._team_name_matches("Texas", "Oklahoma Sooners") is False

    # Odds feed only has a different matchup on file -- no row should match
    # "Texas" vs "Oklahoma", so this should degrade to (None, None) same as
    # a missing key or empty odds_df.
    odds_df = pd.DataFrame(
        [
            {"home_team": "Oklahoma Sooners", "away_team": "Alabama Crimson Tide", "market": "spreads",
             "outcome_name": "Oklahoma Sooners", "point": -3.5},
        ]
    )

    spread_line, total_line = routes._lines_for_game(odds_df, "Texas", "Oklahoma")

    assert spread_line is None
    assert total_line is None


def test_lines_for_game_refuses_to_guess_between_ambiguous_prefix_matches():
    """_team_name_matches is a prefix match, not an exact one -- two
    distinct FBS teams can share a short-name prefix (e.g. CFBD's "Miami"
    matches both "Miami Hurricanes" and "Miami (OH) RedHawks"). If the
    odds feed's matched rows disagree on which real (home, away) pair
    they belong to, _lines_for_game must refuse to guess rather than
    silently attach a different game's line to this one."""
    odds_df = pd.DataFrame(
        [
            {"home_team": "Miami Hurricanes", "away_team": "Clemson Tigers", "market": "spreads",
             "outcome_name": "Miami Hurricanes", "point": -3.5},
            {"home_team": "Miami (OH) RedHawks", "away_team": "Clemson Tigers", "market": "spreads",
             "outcome_name": "Miami (OH) RedHawks", "point": 10.0},
        ]
    )

    spread_line, total_line = routes._lines_for_game(odds_df, "Miami", "Clemson")

    assert spread_line is None
    assert total_line is None

def test_get_player_props_includes_recent_team_and_position(client, monkeypatch):
    monkeypatch.setattr(
        routes, "_load_player_history",
        lambda season: pd.DataFrame(
            [{"player_id": "cfb-001", "player_name": "Quinn Ewers", "position": "QB",
              "recent_team": "Texas", "season": season}]
        ),
    )
    monkeypatch.setattr(
        routes.player_usage, "build_features_for_player",
        lambda player_id, history: pd.Series({"dummy_feature": 1.0}),
    )
    monkeypatch.setattr(
        routes.player_props, "predict_props",
        lambda player_models, feature_row, position: {"anytime_td_prob": 0.37, "passing_yards": 260.0},
    )

    response = client.get("/api/players/2025/1/props")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["recent_team"] == "Texas"
    assert body[0]["position"] == "QB"


def _games_df_with_final_scores(season=2026):
    return pd.DataFrame([
        {"season": season, "home_team": "Florida State", "away_team": "SMU", "home_score": 24, "away_score": 27},
    ])


def test_player_stats_needs_refresh_when_cache_missing_a_finished_teams_stats(monkeypatch, tmp_path):
    """The SMU class of bug: the cache was written before CFBD finished
    posting one side's box score, so it has rows for Florida State but none
    for SMU even though SMU's game already has a final score. This must
    force a refresh regardless of how fresh the cache file's mtime is."""
    monkeypatch.setattr(routes.player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    path = routes.player_stats._season_cache_path(2026)
    pd.DataFrame([{"recent_team": "Florida State", "season": 2026, "week": 1}]).to_parquet(path)

    assert routes._player_stats_needs_refresh(2026, _games_df_with_final_scores()) is True


def test_player_stats_does_not_need_refresh_when_cache_has_every_finished_teams_stats(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    path = routes.player_stats._season_cache_path(2026)
    pd.DataFrame([
        {"recent_team": "Florida State", "season": 2026, "week": 1},
        {"recent_team": "SMU", "season": 2026, "week": 1},
    ]).to_parquet(path)

    assert routes._player_stats_needs_refresh(2026, _games_df_with_final_scores()) is False


def test_player_stats_refresh_check_ignores_games_with_no_final_score_yet(monkeypatch, tmp_path):
    """A team with no rows in the cache is expected and fine as long as
    their game hasn't finished yet -- only a *finished* game missing stats
    is the stale-cache signal."""
    monkeypatch.setattr(routes.player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    path = routes.player_stats._season_cache_path(2026)
    pd.DataFrame([{"recent_team": "Texas", "season": 2026, "week": 1}]).to_parquet(path)
    upcoming_game = pd.DataFrame([
        {"season": 2026, "home_team": "Texas", "away_team": "Oklahoma", "home_score": None, "away_score": None},
    ])

    assert routes._player_stats_needs_refresh(2026, upcoming_game) is False


def test_player_stats_needs_refresh_when_no_cache_exists_yet(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    assert routes._player_stats_needs_refresh(2026, _games_df_with_final_scores()) is True
