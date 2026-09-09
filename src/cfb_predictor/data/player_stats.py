"""player_stats.py — weekly player-level stats, cache-or-fetch from CFBD.

CFBD's per-game player stats (cfbd.GamesApi.get_player_game_stats) return a
deeply nested box-score tree (game -> team -> stat category -> stat type ->
athlete), not nfl_data_py's flat per-player-per-week row -- this module's job
is fetching that tree (one call per season) and flattening it into the same
flat shape nfl_predictor's data/player_stats.py already produces downstream
(features/player_usage.py, models/player_props.py both expect one row per
player-week with named stat columns). Needs the season's games frame (from
data/games.py) to attach a week/gameday to each stat row, since the raw CFBD
payload only carries a game id per top-level entry, not a week number.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from ..config import CFBD_API_KEY, PLAYER_STATS_CACHE_DIR, ROSTER_CACHE_DIR

logger = logging.getLogger(__name__)

# Full FBS roster (player identity: name/position/team) changes rarely —
# a handful of transfers/injuries a week at most, nothing that needs
# same-day freshness. Re-fetching it on every player-props request (as the
# old per-team CFBD TeamsApi.get_roster fallback did) was the single
# biggest driver of CFBD's 1,000-calls/month quota getting exhausted.
# One call covers every FBS team for the season; refresh at most weekly.
_ROSTER_TTL_SECONDS = 7 * 24 * 60 * 60

KEEP_COLUMNS = [
    "player_id", "player_name", "position", "recent_team", "season", "week",
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "receptions", "targets", "carries",
]

# CFBD's box-score category/stat-type names -> this project's flat column
# names. NOTE: CFBD's official box score does not track targets (only
# receptions/yards/tds), so the "targets" column is always NaN here -- a
# real data-source gap, not a bug: features/player_usage.py's rolling mean
# over an all-NaN column degrades to NaN, and every downstream consumer
# already .fillna(0)s before feeding a model.
_CATEGORY_TYPE_TO_COLUMN = {
    ("passing", "YDS"): "passing_yards",
    ("passing", "TD"): "passing_tds",
    ("rushing", "YDS"): "rushing_yards",
    ("rushing", "TD"): "rushing_tds",
    ("rushing", "CAR"): "carries",
    ("receiving", "YDS"): "receiving_yards",
    ("receiving", "TD"): "receiving_tds",
    ("receiving", "REC"): "receptions",
}

_STAT_COLUMNS = [
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "receptions", "carries",
]


def _cfbd_configuration():
    import cfbd

    return cfbd.Configuration(access_token=CFBD_API_KEY)


def _import_player_game_stats(season: int, weeks: list[int]) -> list[dict]:
    """One call per (season, week) -- confirmed against real CFBD data in
    Task 17 that cfbd.GamesApi's real method is get_game_player_stats (NOT
    get_player_game_stats, which does not exist on the installed cfbd
    5.26.0), and that it rejects year-only requests: the live API requires
    one of week/team/conference and returns HTTP 400
    ("either week, team, or conference are required") otherwise. So this
    module's original "one call per season" design assumption was wrong;
    the per-season parquet cache below still holds, just built from one
    call per week instead of a single season-wide call."""
    import time

    import cfbd

    with cfbd.ApiClient(_cfbd_configuration()) as api_client:
        games_api = cfbd.GamesApi(api_client)
        player_games = []
        for week in weeks:
            # Observed transient CFBD-origin 5xx (500, 525 SSL handshake)
            # errors during Task 17's real run at this call volume (~130
            # calls/train_all() run) -- a short retry-with-backoff instead
            # of failing the whole multi-minute fetch on one flaky call.
            for attempt in range(5):
                try:
                    player_games.extend(games_api.get_game_player_stats(year=season, week=int(week)))
                    break
                except cfbd.exceptions.ServiceException:
                    if attempt == 4:
                        raise
                    time.sleep(3 * (2**attempt))
    return [g.to_dict() for g in player_games]


def _infer_position(row: dict) -> str:
    """CFBD's box score carries no roster position -- infer one from which
    stat category dominates this player-game (a real limitation, not a
    placeholder: fetching per-team rosters for a real position would cost
    one CFBD call per team per season, which the 1,000-calls/month budget
    can't absorb). WR and TE are indistinguishable from box-score stats
    alone and both collapse to "WR" -- see models/player_props.py's
    POSITION_YARDAGE_MARKET, which already maps WR and TE to the same
    receiving_yards market, so this collapse costs nothing downstream."""
    totals = {
        "QB": row.get("passing_yards") or 0.0,
        "RB": row.get("rushing_yards") or 0.0,
        "WR": row.get("receiving_yards") or 0.0,
    }
    return max(totals, key=totals.get)


def _flatten_player_game_stats(raw_games: list[dict], games_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Walk CFBD's nested game -> team -> category -> type -> athlete tree
    into one row per (game, player), with one column per stat."""
    game_lookup = games_df.set_index("game_id")[["week"]].to_dict("index")

    rows: dict[tuple, dict] = {}
    for game in raw_games:
        game_id = str(game.get("id"))
        game_meta = game_lookup.get(game_id)
        week = game_meta["week"] if game_meta else None
        for team in game.get("teams", []) or []:
            # Real GamePlayerStatsTeam field is "team", not "school" --
            # confirmed against real CFBD data in Task 17.
            school = team.get("team")
            for category in team.get("categories", []) or []:
                cat_name = category.get("name")
                for stat_type in category.get("types", []) or []:
                    type_name = stat_type.get("name")
                    column = _CATEGORY_TYPE_TO_COLUMN.get((cat_name, type_name))
                    if column is None:
                        continue
                    for athlete in stat_type.get("athletes", []) or []:
                        player_id = str(athlete.get("id"))
                        key = (game_id, player_id)
                        if key not in rows:
                            rows[key] = {
                                "player_id": player_id,
                                "player_name": athlete.get("name"),
                                "recent_team": school,
                                "season": season,
                                "week": week,
                                **{col: 0.0 for col in _STAT_COLUMNS},
                                "targets": float("nan"),
                            }
                        try:
                            rows[key][column] = float(athlete.get("stat"))
                        except (TypeError, ValueError):
                            continue

    if not rows:
        return pd.DataFrame(columns=KEEP_COLUMNS)

    flat_rows = list(rows.values())
    for row in flat_rows:
        row["position"] = _infer_position(row)

    df = pd.DataFrame(flat_rows)
    return df[KEEP_COLUMNS]


def _season_cache_path(season: int):
    return PLAYER_STATS_CACHE_DIR / f"{season}.parquet"


def fetch_weekly_player_stats(
    seasons: list[int], games_df: pd.DataFrame, force_refresh: bool = False
) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = _season_cache_path(season)
        if not force_refresh and path.exists():
            frames.append(pd.read_parquet(path))
            continue
        try:
            season_games = games_df[games_df["season"] == season]
            weeks = sorted(int(w) for w in season_games["week"].dropna().unique())
            raw = _import_player_game_stats(season, weeks)
            flattened = _flatten_player_game_stats(raw, season_games, season)
            flattened.to_parquet(path)
            frames.append(pd.read_parquet(path))
        except Exception:
            if path.exists():
                logger.warning("CFBD player-stats fetch failed for season=%s; serving stale cache", season)
                frames.append(pd.read_parquet(path))
            else:
                logger.warning("CFBD player-stats fetch failed for season=%s; no cache available, skipping", season)

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["season", "week"]).reset_index(drop=True)


def _roster_cache_path(season: int):
    return ROSTER_CACHE_DIR / f"{season}.parquet"


def _import_season_roster(season: int) -> pd.DataFrame:
    """One call for every FBS team's full roster this season — replaces the
    old fallback's one-call-per-missing-team-per-request pattern."""
    import os

    import cfbd

    from .games import _cfbd_configuration

    with cfbd.ApiClient(_cfbd_configuration()) as api_client:
        key = os.getenv("CFBD_API_KEY", "").strip()
        if key:
            if not key.startswith("Bearer "):
                key = f"Bearer {key}"
            api_client.default_headers["Authorization"] = key
        teams_api = cfbd.TeamsApi(api_client)
        roster = teams_api.get_roster(year=season, classification="fbs")

    rows = []
    for player in roster:
        p_id = getattr(player, "id", None) or getattr(player, "athlete_id", None)
        if p_id is None:
            continue
        fname = getattr(player, "first_name", "") or ""
        lname = getattr(player, "last_name", "") or ""
        full_name = f"{fname} {lname}".strip() or "Unknown Player"
        rows.append({
            "player_id": str(p_id),
            "player_name": full_name,
            "position": getattr(player, "position", "ATH") or "ATH",
            "recent_team": getattr(player, "team", "") or "",
        })
    return pd.DataFrame(rows, columns=["player_id", "player_name", "position", "recent_team"])


def fetch_season_roster(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Player identity (name/position/team) for the whole FBS, persisted to
    disk and only re-fetched once the cache is missing or older than
    _ROSTER_TTL_SECONDS — this is the data that "only updates when it
    changes"; weekly *predictions* are computed fresh from it every time,
    but the roster fetch itself is not repeated per request."""
    path = _roster_cache_path(season)
    is_stale = not path.exists() or (time.time() - path.stat().st_mtime) > _ROSTER_TTL_SECONDS
    if not force_refresh and not is_stale:
        return pd.read_parquet(path)
    try:
        roster = _import_season_roster(season)
        roster.to_parquet(path)
        return roster
    except Exception:
        if path.exists():
            logger.warning("CFBD roster fetch failed for season=%s; serving stale cache", season)
            return pd.read_parquet(path)
        logger.warning("CFBD roster fetch failed for season=%s; no cache available", season)
        return pd.DataFrame(columns=["player_id", "player_name", "position", "recent_team"])
