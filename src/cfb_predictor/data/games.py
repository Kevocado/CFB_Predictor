"""games.py — CFB schedule/results and FBS team universe, cache-or-fetch
from CollegeFootballData (CFBD) via the cfbd Python package.

One parquet file per season under GAMES_CACHE_DIR/TEAMS_CACHE_DIR, mirroring
nfl_predictor's data/schedules.py per-season cache-or-fetch pattern — a
season's results never change once played, and a season's FBS membership
never changes once the season starts, so per-season caching is safe. This
caching is *load-bearing* here, not just a latency optimization: CFBD's free
tier caps API access at 1,000 calls/month, so every fetch below is exactly
one call per season (never per-team or per-game).

CFBD's Game model already exposes `conference_game` as a real boolean and
`home_division`/`away_division` directly — unlike NFL's div_game (which
nfl_data_py also supplies pre-computed), no team-conference join is needed
to populate `conference_game` here either. `fetch_fbs_teams` is kept as the
authoritative FBS team-universe source (features/build.py uses it to exclude
FCS-opponent games from training rows), with the per-game division fields
available as a fallback/cross-check when a team-universe lookup isn't
supplied.
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd
import cfbd

from ..config import CFBD_API_KEY, CURRENT_SEASON, GAMES_CACHE_DIR, TEAMS_CACHE_DIR

logger = logging.getLogger(__name__)

# Re-fetch the in-progress season at most this often. CFBD's free tier caps
# API access at 1,000 calls/month; unconditionally force-refreshing the
# current season on every request (as this module originally did) burns
# through that budget in about a day once the background tracking loop
# (api/main.py, every _TRACKING_INTERVAL_SECONDS) is added in -- see this
# plan's final-review fix (Task 23, findings C1+C2).
_CURRENT_SEASON_TTL_SECONDS = 6 * 60 * 60

KEEP_COLUMNS = [
    "game_id", "season", "week", "gameday", "home_team", "away_team",
    "home_score", "away_score", "home_conference", "away_conference",
    "home_division", "away_division", "conference_game", "neutral_site",
]

TEAM_KEEP_COLUMNS = ["team", "conference", "division", "classification"]


def _cfbd_configuration():
    """Configures the CFBD client configuration object."""
    config = cfbd.Configuration()
    return config


def _import_games(season: int) -> pd.DataFrame:
    """One call per season — cfbd.GamesApi.get_games(year=season) returns
    every week of that season's games in a single response, so this never
    needs a per-week loop. Thin wrapper so tests can monkeypatch just this
    one function rather than the whole cfbd client.

    classification="fbs" filters server-side to games with at least one FBS
    team (confirmed against real data in Task 17: without it, get_games
    returns every division's games -- FCS/II/III included -- which flooded
    /api/games' weekly slate with non-FBS matchups this project's spec
    scopes out entirely. FBS-vs-FCS "buy games" still come through, which is
    correct for a real schedule display; features/build.py's own FCS
    exclusion (via fbs_teams/home_division/away_division) still separately
    keeps those out of *training* rows).

    Retries on transient CFBD-origin 5xx errors (observed 503s and 502s
    from CFBD's Cloudflare front end during Task 17's real run) with a
    short backoff, matching the same resilience data/player_stats.py's
    _import_player_game_stats already needed for the exact same problem."""
    
    with cfbd.ApiClient(_cfbd_configuration()) as api_client:
        # Directly inject Authorization header to bypass OpenAPI client key-mapping bugs
        key = os.getenv("CFBD_API_KEY", "").strip()
        if key:
            if not key.startswith("Bearer "):
                key = f"Bearer {key}"
            api_client.default_headers['Authorization'] = key

        games_api = cfbd.GamesApi(api_client)
        for attempt in range(5):
            try:
                fetched = games_api.get_games(year=season, classification="fbs")
                break
            except cfbd.exceptions.ServiceException:
                if attempt == 4:
                    raise
                time.sleep(3 * (2**attempt))
    # NOTE: Game.to_dict() serializes with by_alias=True (CFBD's camelCase
    # wire format, e.g. "homeTeam"), which does not match KEEP_COLUMNS'
    # snake_case names below. .dict() uses the model's actual (snake_case)
    # field names instead. Confirmed against real CFBD data in Task 17.
    return pd.DataFrame([g.dict() for g in fetched])


def _import_fbs_teams(season: int) -> pd.DataFrame:
    """One call per season — cfbd.TeamsApi.get_fbs_teams(year=season) is
    this project's team-universe source. FBS teams move conferences (and
    occasionally divisions) year to year, so hardcoding a list would go
    stale; this is the "one additional data-module function" the design
    spec calls for rather than a separate task."""
    
    with cfbd.ApiClient(_cfbd_configuration()) as api_client:
        # Directly inject Authorization header here as well
        key = os.getenv("CFBD_API_KEY", "").strip()
        if key:
            if not key.startswith("Bearer "):
                key = f"Bearer {key}"
            api_client.default_headers['Authorization'] = key

        teams_api = cfbd.TeamsApi(api_client)
        fetched = teams_api.get_fbs_teams(year=season)
    return pd.DataFrame([t.to_dict() for t in fetched])


def _games_cache_path(season: int):
    return GAMES_CACHE_DIR / f"{season}.parquet"


def _teams_cache_path(season: int):
    return TEAMS_CACHE_DIR / f"{season}.parquet"


def _normalize_games(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(
        columns={
            "id": "game_id",
            "start_date": "gameday",
            "home_points": "home_score",
            "away_points": "away_score",
            # Real CFBD Game model field is home_classification/
            # away_classification (an enum), not home_division/
            # away_division — confirmed against real data in Task 17.
            "home_classification": "home_division",
            "away_classification": "away_division",
        }
    )
    for col in KEEP_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[KEEP_COLUMNS].copy()
    for col in ("home_division", "away_division"):
        df[col] = df[col].map(lambda v: getattr(v, "value", v))
    df["gameday"] = pd.to_datetime(df["gameday"])
    df["game_id"] = df["game_id"].astype(str)
    return df


def fetch_schedules(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """One row per game across every requested season. Seasons already
    cached on disk are read from cache; anything missing (or force_refresh)
    is fetched from CFBD one season at a time — CFBD's get_games only takes
    a single `year`, so a per-season loop is inherent to the endpoint, not a
    violation of the one-call-per-season budget."""
    frames = []
    for season in seasons:
        path = _games_cache_path(season)
        if not force_refresh and path.exists():
            frames.append(pd.read_parquet(path))
            continue
        try:
            fetched = _normalize_games(_import_games(season))
            fetched.to_parquet(path)
            frames.append(pd.read_parquet(path))
        except Exception:
            # CFBD errors here (most commonly the free tier's monthly quota,
            # or a cold-started container with no cache on disk yet) used to
            # 500 every endpoint built on this. Fall back to a stale cache
            # if one exists rather than a hard failure; skip the season
            # entirely (not a crash) if there's nothing to fall back to.
            if path.exists():
                logger.warning("CFBD fetch failed for season=%s; serving stale cache", season)
                frames.append(pd.read_parquet(path))
            else:
                logger.warning("CFBD fetch failed for season=%s; no cache available, skipping", season)

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def fetch_fbs_teams(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """This season's FBS team universe (team, conference, division,
    classification) — used to cross-check conference_game and to exclude
    FCS-opponent games from training. Cached per season since conference
    realignment only happens between seasons, not mid-season."""
    path = _teams_cache_path(season)
    if not force_refresh and path.exists():
        return pd.read_parquet(path)

    raw = _import_fbs_teams(season)
    df = raw.rename(columns={"school": "team"})
    for col in TEAM_KEEP_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[TEAM_KEEP_COLUMNS].copy()
    df.to_parquet(path)
    return df


def default_completed_seasons(n: int = 8) -> list[int]:
    return list(range(CURRENT_SEASON - n, CURRENT_SEASON))


def load_training_data(seasons: list[int]) -> pd.DataFrame:
    """Only games with a final score — excludes future/postponed games from
    the same fetch_schedules() call."""
    df = fetch_schedules(seasons)
    return df[df["home_score"].notna() & df["away_score"].notna()].reset_index(drop=True)


def _current_season_needs_refresh() -> bool:
    """True when CURRENT_SEASON's cache file is missing or older than
    _CURRENT_SEASON_TTL_SECONDS. Shared by fetch_current_season_partial and
    fetch_upcoming_games so the in-progress season is re-fetched on a TTL
    instead of on every single call."""
    path = _games_cache_path(CURRENT_SEASON)
    return not path.exists() or (time.time() - path.stat().st_mtime) > _CURRENT_SEASON_TTL_SECONDS


def fetch_current_season_partial() -> pd.DataFrame:
    """Completed games so far in CURRENT_SEASON. Re-fetched at most every
    _CURRENT_SEASON_TTL_SECONDS (not on every call, which would burn the
    1,000-calls/month cap in about a day given the background tick's
    interval -- see this plan's final-review fix, Task 23)."""
    df = fetch_schedules([CURRENT_SEASON], force_refresh=_current_season_needs_refresh())
    return df[df["home_score"].notna() & df["away_score"].notna()].reset_index(drop=True)


def fetch_upcoming_games(season: int, week: int) -> pd.DataFrame:
    """Games in a given season/week that haven't been played yet. Uses the
    same TTL as fetch_current_season_partial for CURRENT_SEASON rather than
    unconditionally force-refreshing (see this plan's final-review fix,
    Task 23, findings C1+C2)."""
    force = season == CURRENT_SEASON and _current_season_needs_refresh()
    df = fetch_schedules([season], force_refresh=force)
    week_df = df[df["week"] == week]
    return week_df[week_df["home_score"].isna()].reset_index(drop=True)