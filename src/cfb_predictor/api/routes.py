"""routes.py — game/player prop/track-record endpoints. Thin HTTP layer
over data/features/models/tracking, near-verbatim port of nfl_predictor's
own api/routes.py. Two CFB-specific adaptations: (1) schedules -> games
(data.games); (2) CFBD's Game model carries no spread_line/total_line, so
_lines_for_game pulls them from The Odds API's own spreads/totals markets
instead, matched by team name.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from functools import lru_cache

import pandas as pd
import requests
from fastapi import APIRouter, HTTPException

from ..config import (
    CURRENT_SEASON,
    PUBLIC_MODE,
    PUBLIC_SNAPSHOT_PATH,
    PUBLIC_SNAPSHOT_REFRESH_URL,
)
from ..data import games as games_data
from ..data import odds_api, player_stats
from ..features import build as feature_build
from ..features import player_usage
from ..models import game_outcome, manifest, player_props
from ..odds import value_bets
from ..tracking import store

router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)

_public_snapshot_cache: dict | None = None
_public_snapshot_etag: str | None = None


def _public_snapshot() -> dict:
    """The precomputed data public_snapshot.py generates. Cold-start value
    is whatever was baked into this image at build time; a running process
    then keeps it current via refresh_public_snapshot_from_remote below,
    polled on a timer (see api/main.py's lifespan). Empty dict if none has
    been generated yet, so every PUBLIC_MODE branch below just falls
    through to a live compute instead of a 500."""
    global _public_snapshot_cache
    if _public_snapshot_cache is None:
        _public_snapshot_cache = (
            json.loads(PUBLIC_SNAPSHOT_PATH.read_text()) if PUBLIC_SNAPSHOT_PATH.exists() else {}
        )
    return _public_snapshot_cache


def refresh_public_snapshot_from_remote() -> bool:
    """Re-fetch the precomputed public snapshot from its GitHub raw URL and
    swap it in -- lets the scheduled snapshot-refresh GitHub Action reach
    this running process without a Docker rebuild+redeploy. ETag-conditional
    so an unchanged snapshot costs one small request. Any fetch/parse
    failure is swallowed and leaves the previous snapshot serving -- a
    transient network hiccup here must never blank an already-working
    public site."""
    global _public_snapshot_cache, _public_snapshot_etag
    try:
        headers = {"If-None-Match": _public_snapshot_etag} if _public_snapshot_etag else {}
        resp = requests.get(PUBLIC_SNAPSHOT_REFRESH_URL, headers=headers, timeout=30)
        if resp.status_code == 304:
            return False
        resp.raise_for_status()
        snapshot = resp.json()
    except Exception as exc:
        logger.info("public snapshot remote refresh skipped: %s", exc)
        return False
    _public_snapshot_cache = snapshot
    _public_snapshot_etag = resp.headers.get("ETag")
    return True


def _snapshot_week(season: int, week: int) -> dict | None:
    snap = _public_snapshot()
    if snap.get("season") != season:
        return None
    return snap.get("weeks", {}).get(str(week))


def current_season_and_week() -> tuple[int, int]:
    """Current CFB season/week, anchored to the real schedule's own week-1
    kickoff date rather than a hardcoded month/day guess. A fixed
    `date(season, 8, 20)` anchor drifts every year the season's actual
    opening week doesn't start exactly then (confirmed live: it was a full
    week ahead of the real current week)."""
    today = date.today()
    season = today.year if today.month >= 2 else today.year - 1
    try:
        schedule = games_data.fetch_schedules([season])
        week1_start = schedule.loc[schedule["week"] == 1, "gameday"].min()
        anchor = week1_start.date() if pd.notna(week1_start) else date(season, 8, 20)
    except Exception:
        anchor = date(season, 8, 20)
    week = max(1, min(20, ((today - anchor).days // 7) + 1))
    return season, week


@router.get("/current-week")
def get_current_week():
    season, week = current_season_and_week()
    return {"season": season, "week": week}

# Re-fetch the current season's weekly player stats at most this often,
# same TTL rationale as data/games.py's _CURRENT_SEASON_TTL_SECONDS --
# unconditionally force-refreshing here cost ~15-20 CFBD calls per single
# Player Props page view once Task 17 changed the fetch to one call per
# week (see this plan's final-review fix, Task 23, findings C1+C2).
_CURRENT_SEASON_PLAYER_STATS_TTL_SECONDS = 6 * 60 * 60


def _player_stats_cache_missing_a_finished_teams_stats(season: int, games_df: pd.DataFrame) -> bool:
    """True when the cached player-stats parquet exists but has zero rows
    for a team whose game already has a final score -- confirmed live as
    the actual SMU-returns-0-stats bug: CFBD hadn't finished posting one
    side's box score at the moment this cache was written (a fetch that
    landed kickoff-adjacent caught one team's stats but not the other's),
    so the incomplete snapshot got cached and nothing re-checks it until
    the blanket TTL lapses. Checked in *addition* to the TTL below, never
    instead of it -- this only adds a faster trigger, it never skips a
    refresh the TTL would still want."""
    path = player_stats._season_cache_path(season)
    if not path.exists():
        return False
    season_games = games_df[games_df["season"] == season]
    finished = season_games[season_games["home_score"].notna() & season_games["away_score"].notna()]
    if finished.empty:
        return False
    finished_teams = set(finished["home_team"]) | set(finished["away_team"])
    cached_teams = set(pd.read_parquet(path)["recent_team"])
    return not finished_teams.issubset(cached_teams)


def _player_stats_needs_refresh(season: int, games_df: pd.DataFrame) -> bool:
    path = player_stats._season_cache_path(season)
    if not path.exists():
        return True
    if (time.time() - path.stat().st_mtime) > _CURRENT_SEASON_PLAYER_STATS_TTL_SECONDS:
        return True
    return _player_stats_cache_missing_a_finished_teams_stats(season, games_df)


@lru_cache(maxsize=1)
def _load_models_cached() -> dict:
    return manifest.load_models()


def _load_models_or_503() -> dict:
    """manifest.load_models() raises FileNotFoundError when no manifest has
    ever been trained -- turn that into a friendly 503 instead of letting it
    propagate as a bare unhandled 500. Does not address the larger "ship
    models in the image" / persistent-disk deploy question -- that's out of
    scope for this fix (see this plan's final-review fix, Task 23, finding
    C3)."""
    try:
        return _load_models_cached()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503, detail="No trained model yet — run POST /api/retrain first"
        ) from exc


def _team_name_matches(cfbd_name: str, odds_name: str) -> bool:
    """Best-effort match between CFBD's short school name (e.g. "Texas")
    and The Odds API's full team name (e.g. "Texas Longhorns") -- these
    are different naming conventions with no shared id space. Matches
    when the CFBD name's words are a prefix of the Odds API name's words
    (case-insensitive), which covers the common case without a full
    manual mapping table. Real gap acknowledged in the plan/ledger: this
    will still miss genuinely divergent names (e.g. abbreviations,
    "St." vs "State") -- degrades gracefully to no line found, same as
    a missing key or empty odds_df already does."""
    cfbd_words = cfbd_name.lower().split()
    odds_words = odds_name.lower().split()
    return odds_words[: len(cfbd_words)] == cfbd_words


def _lines_for_game(odds_df: pd.DataFrame, home_team: str, away_team: str) -> tuple[float | None, float | None]:
    """CFBD's Game model (unlike nflverse's schedule frame) carries no
    market spread/total -- pull them from The Odds API's own spreads/totals
    markets instead, matched by team name (best-effort -- see
    _team_name_matches). Best-effort: returns (None, None) for either line
    the odds feed doesn't have for this matchup (a common case outside
    Power-conference games, per the design spec)."""
    if odds_df is None or odds_df.empty:
        return None, None
    matches = odds_df[
        odds_df["home_team"].map(lambda t: _team_name_matches(home_team, t))
        & odds_df["away_team"].map(lambda t: _team_name_matches(away_team, t))
    ]
    if matches.empty:
        return None, None
    # _team_name_matches is a prefix match, not an exact one -- two distinct
    # FBS teams can share a short-name prefix (e.g. "Miami" matches both
    # "Miami Hurricanes" and "Miami (OH) RedHawks"). If the matched rows
    # disagree on which real (home_team, away_team) pair they belong to,
    # trusting the first row would silently attach a DIFFERENT game's line
    # to this one -- worse than the graceful (None, None) this function is
    # documented to return on a genuine miss. Refuse to guess: treat an
    # ambiguous match the same as no match.
    if matches[["home_team", "away_team"]].drop_duplicates().shape[0] > 1:
        return None, None

    spread_line = None
    spread_rows = matches[
        (matches["market"] == "spreads")
        & matches["outcome_name"].map(lambda t: _team_name_matches(home_team, t))
    ]
    if not spread_rows.empty:
        # The Odds API's `point` is the raw handicap (negative when the home
        # team is favored). Negate it so spread_line means "home expected
        # margin" -- the same convention nflverse uses and that
        # game_outcome.margin_to_probabilities documents and is pinned
        # against (see this plan's final review, finding B2).
        spread_line = -float(spread_rows.iloc[0]["point"])

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
        current_df = player_stats.fetch_weekly_player_stats(
            [season], games_df, force_refresh=_player_stats_needs_refresh(season, games_df)
        )
        return pd.concat([historical_df, current_df], ignore_index=True)
    all_seasons = history_seasons + [season]
    games_df = games_data.load_training_data(all_seasons)
    return player_stats.fetch_weekly_player_stats(all_seasons, games_df)


@router.get("/games")
def get_games(season: int, week: int):
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            return snap["games"]
    return _get_games_live(season, week)


def _get_games_live(season: int, week: int):
    games = games_data.fetch_week_games(season, week)
    games = games.astype(object).where(pd.notna(games), None)
    return games.to_dict("records")


@router.get("/games/{season}/{week}/{game_id}/prediction")
def get_game_prediction(season: int, week: int, game_id: str):
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            pred = snap["predictions"].get(game_id)
            if pred is not None:
                return pred
    return _get_game_prediction_live(season, week, game_id)


def _get_game_prediction_live(season: int, week: int, game_id: str):
    games = games_data.fetch_week_games(season, week)
    matches = games[games["game_id"] == game_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    game = matches.iloc[0]

    models = _load_models_or_503()
    # Exclude the game's own row from its feature history -- fetch_week_games
    # (unlike the old fetch_upcoming_games) makes already-finished games
    # reachable here too, and without this exclusion a finished game's
    # prediction would leak its own result into its own features.
    history = _load_game_history(season)
    history = history[history["game_id"] != game_id]
    odds_df = odds_api.fetch_game_odds()
    spread_line, total_line = _lines_for_game(odds_df, game["home_team"], game["away_team"])
    prediction = _predict_game_from_models(
        models, game["home_team"], game["away_team"], history,
        spread_line=spread_line, total_line=total_line,
    )
    return prediction


@router.get("/players/{season}/{week}/props")
def get_player_props(season: int, week: int):
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            return snap["player_props"]
    return _get_player_props_live(season, week)


def _get_player_props_live(season: int, week: int):
    try:
        models = _load_models_cached()
        player_history = _load_player_history(season)

        # Standardize team name variants to prevent mismatches
        player_history["recent_team"] = player_history["recent_team"].replace({
            "Southern Methodist": "SMU"
        })

        # 1. Fetch upcoming games for this specific week to find active teams (e.g., FSU vs SMU)
        upcoming_games = games_data.fetch_upcoming_games(season, week)
        if upcoming_games.empty:
            upcoming_games = games_data.fetch_current_season_partial()
            upcoming_games = upcoming_games[upcoming_games["week"] == week]

        if upcoming_games.empty:
            return []

        active_teams = set(upcoming_games["home_team"]).union(set(upcoming_games["away_team"]))

        # 2. Extract players present in historical/current stats for active teams
        latest_players = (
            player_history[
                (player_history["season"] == season) & 
                (player_history["recent_team"].isin(active_teams))
            ]
            [["player_id", "player_name", "position", "recent_team"]]
            .drop_duplicates("player_id")
        )

        found_teams = set(latest_players["recent_team"].unique())
        missing_teams = active_teams - found_teams

        # 3. Fallback: a team with no current-season stats yet (week 1, or a
        # bye-to-opener gap) pulls from the cached full-FBS roster instead —
        # fetched at most once a week (data/player_stats.py's
        # fetch_season_roster), not once per request per missing team like
        # the old CFBD TeamsApi.get_roster fallback did.
        if missing_teams:
            try:
                roster = player_stats.fetch_season_roster(season)
                fallback = roster[
                    roster["recent_team"].isin(missing_teams) & roster["position"].isin({"WR", "TE", "RB", "QB"})
                ]
                if not fallback.empty:
                    latest_players = pd.concat([latest_players, fallback], ignore_index=True).drop_duplicates("player_id")
            except Exception as roster_err:
                logger.warning("Failed to load roster fallback for teams=%s: %s", missing_teams, roster_err)

        results = []
        for _, player in latest_players.iterrows():
            try:
                if "roster_" in str(player["player_id"]) or (player["player_id"].isdigit() and not (player_history["player_id"] == player["player_id"]).any()):
                    feature_row = pd.Series({"attempts": 5.0, "completions": 3.0, "passing_yards": 40.0, "carries": 2.0, "rushing_yards": 10.0, "receptions": 2.0, "receiving_yards": 20.0, "targets": 3.0})
                else:
                    feature_row = player_usage.build_features_for_player(player["player_id"], player_history)
                
                if feature_row is None:
                    continue
                
                props = player_props.predict_props(models["player_models"], feature_row, position=player["position"])
                results.append({
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "recent_team": player["recent_team"],
                    "position": player["position"],
                    **props,
                })
            except Exception as player_err:
                logger.warning("Failed to predict props for player_id=%s: %s", player.get("player_id"), player_err)
                continue

        # 4. Strict position filter to ensure only offensive skill players make it into the final API output
        valid_positions = {"WR", "TE", "RB", "QB"}
        results = [p for p in results if p["position"] in valid_positions]

        return results
    except Exception as e:
        logger.exception("Failed to load player props for season=%s week=%s", season, week)
        return []

@router.get("/track-record")
def get_track_record():
    return store.get_track_record()


@router.get("/games/{game_id}/verdict")
def get_game_verdict(game_id: str):
    verdict = store.get_game_verdict(game_id)
    if verdict is None:
        raise HTTPException(status_code=404, detail="Game not tracked or not yet resolved")
    return verdict


@router.get("/predictions/{season}/{week}")
def get_predictions_for_week(season: int, week: int):
    # A union, not an if/else fallback: a week in progress has both
    # already-final games and still-upcoming ones, and the old if/else
    # dropped every finished game (and its verdict) whenever any game in
    # the week was still unplayed (see this plan's final review, finding B3).
    upcoming = games_data.fetch_upcoming_games(season, week)
    finished = games_data.load_training_data(seasons=[season])
    finished = finished[finished["week"] == week]
    games = pd.concat([finished, upcoming], ignore_index=True).drop_duplicates(subset="game_id")
    return store.get_predictions_for_week(season, week, games)


@router.post("/retrain")
def retrain():
    if PUBLIC_MODE:
        raise HTTPException(status_code=403, detail="Retraining is disabled in public mode")
    result = manifest.train_all()
    _load_models_cached.cache_clear()
    return {"trained_at": result["trained_at"], "chosen_candidate": result["chosen_candidate"]}


def _attach_game_id(stats_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """Join weekly player stats to the game_id of the game each row's
    player actually played in, matched by season/week/team -- weekly
    stats carry a team and week but no game_id of their own."""
    if stats_df.empty or games_df.empty:
        return stats_df.iloc[0:0]
    home = games_df[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "recent_team"})
    away = games_df[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "recent_team"})
    team_game = pd.concat([home, away], ignore_index=True)
    return stats_df.merge(team_game, on=["season", "week", "recent_team"], how="inner")


def background_tracking_tick(season: int, week: int) -> None:
    """Snapshot this week's upcoming-game (and player-prop) predictions,
    then reconcile anything now resolved. Called on a timer from
    api/main.py's lifespan."""
    try:
        models = _load_models_cached()
    except Exception as exc:
        logger.warning("background_tracking_tick skipped: %s", exc)
        return

    games = games_data.fetch_upcoming_games(season, week)
    if not games.empty:
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
                        "commence_time": str(game["gameday"]),
                        "home_spread_line": spread_line, "total_line": total_line,
                        "season": season, "week": week,
                        **pred,
                    }
                )
            except Exception:
                logger.exception("prediction failed for game_id=%s", game.get("game_id"))
                continue
        try:
            store.record_game_predictions(predictions)
        except Exception:
            logger.exception("record_game_predictions failed")

        try:
            team_to_game = {}
            for _, g in games.iterrows():
                team_to_game[g["home_team"]] = g["game_id"]
                team_to_game[g["away_team"]] = g["game_id"]
            prop_rows = []
            for prop in _get_player_props_live(season, week):
                game_id = team_to_game.get(prop["recent_team"])
                if game_id is None:
                    continue
                prop_rows.append({
                    "game_id": game_id, "player_id": prop["player_id"], "player_name": prop["player_name"],
                    "market": "anytime_td", "predicted_value": prop["anytime_td_prob"],
                })
                for market in ("passing_yards", "rushing_yards", "receiving_yards"):
                    value = prop.get(market)
                    if value is not None:
                        prop_rows.append({
                            "game_id": game_id, "player_id": prop["player_id"], "player_name": prop["player_name"],
                            "market": market, "predicted_value": value,
                        })
            store.record_player_prop_predictions(prop_rows)
        except Exception:
            logger.exception("player prop snapshot failed for season=%s week=%s", season, week)

    try:
        completed = games_data.fetch_current_season_partial()
        store.reconcile_game_predictions(completed[["game_id", "home_score", "away_score"]])
    except Exception:
        logger.exception("reconcile_game_predictions failed")
        completed = pd.DataFrame()

    # Backfill games that finished before this tracker ever ran once
    # (e.g. background_tracking_tick wasn't wired up / wasn't running yet)
    # -- reconcile only ever updates an EXISTING snapshot, so a game with
    # no snapshot at all would otherwise never get a verdict. Excludes
    # each game from its own history so the prediction still reflects
    # strictly pre-game information, not this game's own result.
    try:
        if not completed.empty:
            untracked_ids = store.get_untracked_game_ids(list(completed["game_id"]))
            if untracked_ids:
                models = _load_models_cached()
                history_all = _load_game_history(season)
                odds_df = odds_api.fetch_game_odds()
                backfill_games = []
                for _, game in completed[completed["game_id"].isin(untracked_ids)].iterrows():
                    try:
                        spread_line, total_line = _lines_for_game(odds_df, game["home_team"], game["away_team"])
                        history_excl = history_all[history_all["game_id"] != game["game_id"]]
                        pred = _predict_game_from_models(
                            models, game["home_team"], game["away_team"], history_excl,
                            spread_line=spread_line, total_line=total_line,
                        )
                        backfill_games.append({
                            "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                            "commence_time": str(game["gameday"]), "season": season,
                            "week": int(game["week"]) if pd.notna(game.get("week")) else None,
                            "home_spread_line": spread_line, "total_line": total_line,
                            "actual_home_score": game["home_score"], "actual_away_score": game["away_score"],
                            **pred,
                        })
                    except Exception:
                        logger.exception("backfill prediction failed for game_id=%s", game.get("game_id"))
                        continue
                store.record_resolved_game_predictions(backfill_games)
    except Exception:
        logger.exception("backfill of untracked finished games failed")

    try:
        if not completed.empty:
            actual_stats = player_stats.fetch_weekly_player_stats([season], completed)
            store.reconcile_player_prop_predictions(_attach_game_id(actual_stats, completed))
    except Exception:
        logger.exception("player prop reconciliation failed")

    try:
        store.backfill_unresolved_games(games_data)
    except Exception:
        logger.exception("backfill_unresolved_games failed")