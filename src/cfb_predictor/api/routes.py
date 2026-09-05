"""routes.py — game/player prop/track-record endpoints. Thin HTTP layer
over data/features/models/tracking, near-verbatim port of nfl_predictor's
own api/routes.py. Two CFB-specific adaptations: (1) schedules -> games
(data.games); (2) CFBD's Game model carries no spread_line/total_line, so
_lines_for_game pulls them from The Odds API's own spreads/totals markets
instead, matched by team name.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..config import CURRENT_SEASON, PUBLIC_MODE
from ..data import games as games_data
from ..data import odds_api, player_stats
from ..features import build as feature_build
from ..features import player_usage
from ..models import game_outcome, manifest, player_props
from ..odds import value_bets
from ..tracking import store

router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_models_cached() -> dict:
    return manifest.load_models()


def _lines_for_game(odds_df: pd.DataFrame, home_team: str, away_team: str) -> tuple[float | None, float | None]:
    """CFBD's Game model (unlike nflverse's schedule frame) carries no
    market spread/total -- pull them from The Odds API's own spreads/totals
    markets instead, matched by team name. Best-effort: returns (None, None)
    for either line the odds feed doesn't have for this matchup (a common
    case outside Power-conference games, per the design spec)."""
    if odds_df is None or odds_df.empty:
        return None, None
    matches = odds_df[(odds_df["home_team"] == home_team) & (odds_df["away_team"] == away_team)]
    if matches.empty:
        return None, None

    spread_line = None
    spread_rows = matches[(matches["market"] == "spreads") & (matches["outcome_name"] == home_team)]
    if not spread_rows.empty:
        spread_line = float(spread_rows.iloc[0]["point"])

    total_line = None
    total_rows = matches[(matches["market"] == "totals") & (matches["outcome_name"] == "Over")]
    if not total_rows.empty:
        total_line = float(total_rows.iloc[0]["point"])

    return spread_line, total_line


def _predict_game_from_models(
    models: dict, home: str, away: str, games_df: pd.DataFrame,
    spread_line: float | None = None, total_line: float | None = None,
) -> dict:
    feature_row = feature_build.build_features_for_game(home, away, games_df)
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0)

    if models["chosen_candidate"] == "elo":
        predicted_margin = game_outcome.predict_margin_elo(
            {"points_per_rating_point": game_outcome.ELO_POINTS_PER_RATING_POINT},
            feature_row["rating_diff"], feature_row["home_rest_days"], feature_row["away_rest_days"],
        )
    else:
        predicted_margin = float(models["game_outcome_model"].predict(X.to_numpy().reshape(1, -1))[0])

    predicted_total = float(models["total_model"].predict(X.to_numpy().reshape(1, -1))[0])

    return game_outcome.margin_to_probabilities(
        predicted_margin, models["sigma"],
        spread_line=spread_line, total_line=total_line,
        predicted_total=predicted_total, total_sigma=models["total_sigma"],
    )


def _load_game_history(season: int) -> pd.DataFrame:
    """Historical + requested-season game data for feature building. A
    still-in-progress CURRENT_SEASON needs the always-refetch
    fetch_current_season_partial() instead of load_training_data()'s
    per-season cache, or every request that season would return the same
    stale games-so-far snapshot forever."""
    history_seasons = games_data.default_completed_seasons(n=8)
    if season == CURRENT_SEASON:
        historical_df = games_data.load_training_data(history_seasons)
        current_df = games_data.fetch_current_season_partial()
        return pd.concat([historical_df, current_df], ignore_index=True)
    return games_data.load_training_data(history_seasons + [season])


def _load_player_history(season: int) -> pd.DataFrame:
    """Historical + requested-season player stats for feature building.
    player_stats.fetch_weekly_player_stats needs each season's games_df to
    attach week numbers to CFBD's raw per-game box scores (see
    data/player_stats.py), so this loads game history first."""
    history_seasons = games_data.default_completed_seasons(n=8)
    if season == CURRENT_SEASON:
        historical_games = games_data.load_training_data(history_seasons)
        current_games = games_data.fetch_current_season_partial()
        games_df = pd.concat([historical_games, current_games], ignore_index=True)
        historical_df = player_stats.fetch_weekly_player_stats(history_seasons, games_df)
        current_df = player_stats.fetch_weekly_player_stats([season], games_df, force_refresh=True)
        return pd.concat([historical_df, current_df], ignore_index=True)
    all_seasons = history_seasons + [season]
    games_df = games_data.load_training_data(all_seasons)
    return player_stats.fetch_weekly_player_stats(all_seasons, games_df)


@router.get("/games")
def get_games(season: int, week: int):
    games = games_data.fetch_upcoming_games(season, week)
    games = games.astype(object).where(pd.notna(games), None)
    return games.to_dict("records")


@router.get("/games/{season}/{week}/{game_id}/prediction")
def get_game_prediction(season: int, week: int, game_id: str):
    games = games_data.fetch_upcoming_games(season, week)
    matches = games[games["game_id"] == game_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    game = matches.iloc[0]

    models = _load_models_cached()
    history = _load_game_history(season)
    odds_df = odds_api.fetch_game_odds()
    spread_line, total_line = _lines_for_game(odds_df, game["home_team"], game["away_team"])
    prediction = _predict_game_from_models(
        models, game["home_team"], game["away_team"], history,
        spread_line=spread_line, total_line=total_line,
    )
    return prediction


@router.get("/players/{season}/{week}/props")
def get_player_props(season: int, week: int):
    models = _load_models_cached()
    player_history = _load_player_history(season)

    latest_players = (
        player_history[player_history["season"] == season]
        [["player_id", "player_name", "position", "recent_team"]]
        .drop_duplicates("player_id")
    )
    results = []
    for _, player in latest_players.iterrows():
        feature_row = player_usage.build_features_for_player(player["player_id"], player_history)
        if feature_row is None:
            continue
        props = player_props.predict_props(models["player_models"], feature_row, position=player["position"])
        results.append({"player_id": player["player_id"], "player_name": player["player_name"], **props})
    return results


@router.get("/track-record")
def get_track_record():
    return store.get_track_record()


@router.post("/retrain")
def retrain():
    if PUBLIC_MODE:
        raise HTTPException(status_code=403, detail="Retraining is disabled in public mode")
    result = manifest.train_all()
    _load_models_cached.cache_clear()
    return {"trained_at": result["trained_at"], "chosen_candidate": result["chosen_candidate"]}


def background_tracking_tick(season: int, week: int) -> None:
    """Snapshot this week's upcoming-game predictions, then reconcile
    anything now resolved. Called on a timer from api/main.py's lifespan."""
    games = games_data.fetch_upcoming_games(season, week)
    if not games.empty:
        models = _load_models_cached()
        history = _load_game_history(season)
        odds_df = odds_api.fetch_game_odds()
        predictions = []
        for _, game in games.iterrows():
            try:
                spread_line, total_line = _lines_for_game(odds_df, game["home_team"], game["away_team"])
                pred = _predict_game_from_models(
                    models, game["home_team"], game["away_team"], history,
                    spread_line=spread_line, total_line=total_line,
                )
                predictions.append(
                    {
                        "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                        "commence_time": str(game["gameday"]), **pred,
                    }
                )
            except Exception:
                logger.exception("prediction failed for game_id=%s", game.get("game_id"))
                continue
        store.record_game_predictions(predictions)

    completed = games_data.fetch_current_season_partial()
    store.reconcile_game_predictions(completed[["game_id", "home_score", "away_score"]])


def warm_caches() -> None:
    """Pre-fetch schedules/player stats/odds so the first real request
    after startup isn't slow — best-effort, never raises."""
    try:
        seasons = games_data.default_completed_seasons(n=8)
        games_df = games_data.fetch_schedules(seasons)
        player_stats.fetch_weekly_player_stats(seasons, games_df)
        odds_api.fetch_game_odds()
    except Exception:
        pass
