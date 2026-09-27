"""team_stats.py — per-team, per-game yardage from CFBD's team box score.

Why this module exists. The dashboard labels a number "Team Yardage Predictions";
it is the sum of every active player's position-specific yardage projection
across the whole roster (`Sports_Predictor/src/lib/playerRank.ts` `keyYardage`).
Measured 2026-09-27 on the current deployment, CFB week 5: median **841** yards
per team, max 1343, and **0 of 118** teams inside the realistic 300-450 band.
NFL is the same story (median 1110, 0 of 32).

There was no yardage model to aggregate because no yardage data existed. This
project's `data/games.py` only ever called `get_games`, `get_fbs_teams`,
`get_game_player_stats` and `get_roster`; the per-team box score, which is where
yardage lives, was never requested. `nflverse` has no college-football data at
all (its 25 release tags contain no `cfbd_*` asset; all probe 404), so CFBD is
the only route.

`GamesApi.get_game_team_stats(year, week)` returns **every game's team box score
for that week in one call** — 178 games / 356 team-games for 2023 week 1. A
full 2004-2025 backfill is therefore ~330 calls, once. This project documents a
1,000-calls/month budget in config.py, though the live `X-CallLimit-Remaining`
header read 2196 on 2026-09-27, so the effective cap is roughly double the
documented figure. Either way the backfill is a one-time spike and must never
be scheduled.

Target is CFBD's own `totalYards`, which the API documents as net offensive
yards. Verified on real data: `totalYards == netPassingYards + rushingYards` in
355 of 356 team-games (2023 week 1), median difference 0.00.

Reconciliation constraint. Because both sides come from the same game payload,
`sum_players(rushing + receiving)` reproduces the team `totalYards` -- verified
355 of 356, median error 0.00, with the single outlier Robert Morris at -14.
That makes allocate-by-share the correct architecture: project the team total
first, then allocate. Summing independent player projections cannot reproduce
it, which is what the dashboard does today and why it is out by 2.2-3.2x.

Note the quarterback special case when reconciling: a QB's `passing.YDS` is the
*team's* passing total, so a QB must be counted via `passing` or excluded from
the receiving-side sum, never both.
"""

from __future__ import annotations

import os

import pandas as pd

from ..config import CFBD_API_KEY, TEAM_STATS_CACHE_DIR

# CFBD's team box score is a list of {category, stat} pairs per team, not flat
# columns. Only the ones a team-offence model would read are lifted; the full
# per-team category list is 26 entries including kicking, penalties, third-down
# efficiency and possession time.
CATEGORIES = {
    "totalYards": "total_yards",
    "netPassingYards": "net_passing_yards",
    "rushingYards": "rushing_yards",
    "firstDowns": "first_downs",
    "rushingAttempts": "rushing_attempts",
    "passingTDs": "passing_tds",
    "rushingTDs": "rushing_tds",
    "interceptionTDs": "interception_tds",
    "interceptions": "interceptions",
    "passesIntercepted": "passes_intercepted",
    "turnovers": "turnovers",
    "totalFumbles": "total_fumbles",
    "fumblesLost": "fumbles_lost",
    "possessionTime": "possession_time",
    "completionAttempts": "completion_attempts",
    "yardsPerPass": "yards_per_pass",
    "yardsPerRushAttempt": "yards_per_rush_attempt",
    "thirdDownEff": "third_down_eff",
    "fourthDownEff": "fourth_down_eff",
}

KEEP_COLUMNS = [
    "game_id", "season", "week", "team", "opponent", "home_away", "conference", "points",
    *CATEGORIES.values(),
]

# The identity the architecture rests on, matching the CFB box-score shape:
# rushing + receiving for skill players, and passing for the QB (whose passing
# figure is the team's).
TARGET_COLUMNS = ["total_yards"]


def _client():
    import cfbd

    client = cfbd.ApiClient(cfbd.Configuration())
    key = os.getenv("CFBD_API_KEY", "").strip()
    if key:
        if not key.startswith("Bearer "):
            key = f"Bearer {key}"
        client.default_headers["Authorization"] = key
    return client


def _week_cache_path(season: int, week: int):
    return TEAM_STATS_CACHE_DIR / f"{season}_wk{week:02d}.parquet"


def _read_cache(path):
    """Read a cached week, or None if it predates the current schema.

    Without this a change to the flattened shape silently serves stale-shaped
    data. It happened during development: renaming `week` to `requested_week`
    left a cached file whose columns no longer matched, and
    `attach_schedule_weeks` then raised a bare `KeyError` far from the cause. A
    cache that cannot tell it is stale is worse than no cache.
    """
    if not path.exists():
        return None
    try:
        cached = pd.read_parquet(path)
    except Exception:
        return None
    required = {"game_id", "season", "requested_week", "team"}
    return cached if required <= set(cached.columns) else None


def _flatten(raw_games: list, season: int, week: int) -> pd.DataFrame:
    """Walk CFBD's nested game -> team -> stats tree into one row per team-game.

    Three things the payload does *not* carry, all verified against real data:

    - `GameTeamStats` has only `id` and `teams` — no `season`.
    - `GameTeamStatsTeam` has no `opponent`; it is reconstructed by pivoting the
      game's two teams against each other.
    - **There is no `week` field at all, and the endpoint's `week` argument does
      not reliably filter.** `get_game_team_stats(year=2023, week=1)` returns 178
      games, but only 246 of the resulting 356 team-rows carry a distinct
      (season, week, team) key — Air Force appears twice, against Robert Morris
      and against James Madison, the latter a December bowl. The `week` written
      here is therefore the *requested* week and is wrong for any game the
      endpoint over-returns.

    Consequence: **`game_id` is the only trustworthy key in this frame**, and the
    true week has to come from the schedule. `attach_schedule_weeks` does that
    join, and reconciliation keys on `game_id` + `team`, never on week.
    """
    rows = []
    for game in raw_games:
        game_id = str(game.get("id"))
        teams = game.get("teams") or []
        names = [team.get("team") for team in teams]
        for team in teams:
            name = team.get("team")
            record = {
                "game_id": game_id,
                "season": season,
                "requested_week": week,
                "team": name,
                "opponent": next((other for other in names if other != name), None),
                "home_away": team.get("homeAway"),
                "conference": team.get("conference"),
                "points": team.get("points"),
            }
            for entry in team.get("stats") or []:
                column = CATEGORIES.get(entry.get("category"))
                if column is None:
                    continue
                record[column] = _to_number(entry.get("stat"))
            rows.append(record)
    if not rows:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.DataFrame(rows)


def attach_schedule_weeks(frame: pd.DataFrame, schedules_df: pd.DataFrame) -> pd.DataFrame:
    """Replace `requested_week` with each game's real `week` from the schedule.

    Needed because CFBD's team box score has no week field and its `week` request
    argument over-returns. Games the schedule does not know about fall back to
    `requested_week` rather than being dropped.
    """
    out = frame.copy()
    if schedules_df.empty or not {"game_id", "week"} <= set(schedules_df.columns):
        out["week"] = out["requested_week"]
        return out
    lookup = schedules_df[["game_id", "week"]].drop_duplicates(["game_id"]).copy()
    lookup["game_id"] = lookup["game_id"].astype(str)
    out["game_id"] = out["game_id"].astype(str)
    out = out.drop(columns=["week"], errors="ignore").merge(lookup, on="game_id", how="left")
    out["week"] = out["week"].fillna(out["requested_week"])
    return out


def _to_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def fetch_team_stats(
    seasons: list[int], weeks: list[int] | None = None, force_refresh: bool = False
) -> pd.DataFrame:
    """Team box scores for the requested (season, week) pairs, cache-or-fetch.

    One CFBD call per (season, week), each covering every game that week. Weeks
    default to 1-16. Cache is per (season, week) because a season's team
    yardage never changes once played.

    **Cost warning:** a full 2004-2025 backfill is ~330 calls against a metered
    budget. Call this once, deliberately; never put it on a schedule.
    """
    import cfbd

    TEAM_STATS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    api = cfbd.GamesApi(_client())
    frames = []
    for season in seasons:
        for week in weeks or range(1, 17):
            path = _week_cache_path(season, week)
            cached = None if force_refresh else _read_cache(path)
            if cached is not None:
                frames.append(cached)
                continue
            raw = api.get_game_team_stats(year=season, week=week)
            frame = _flatten([g.to_dict() for g in raw], season, week)
            if frame.empty:
                continue
            frame.to_parquet(path, index=False)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["season", "requested_week", "game_id", "team"]
    ).reset_index(drop=True)


def add_total_yards(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the model target, defaulting to `net_passing_yards + rushing_yards`.

    CFBD supplies `totalYards` directly, but it is also reconstructible from the
    two components, and the two are not identical on every row. Prefer the
    supplied value; fill gaps from the sum only where both components are
    present, so a frame carrying neither is left alone rather than raising.
    """
    out = frame.copy()
    if "total_yards" not in out.columns:
        return out
    components = [column for column in ("net_passing_yards", "rushing_yards") if column in out.columns]
    if len(components) == 2:
        fallback = out[components[0]] + out[components[1]]
        out["total_yards"] = out["total_yards"].where(out["total_yards"].notna(), fallback)
    return out


def reconcile_against_players(
    team_frame: pd.DataFrame, player_frame: pd.DataFrame
) -> pd.DataFrame:
    """Per (season, week, team): team `totalYards`, the player sum, and the gap.

    **Both frames must carry a `week` sourced from the schedule**
    (`attach_schedule_weeks` for the team side; the player frame already picks
    its week up from a schedule merge). That is what makes the join key safe:
    CFBD's team box score has no `week` field and its `week` request argument
    over-returns, so the team frame's own week is unreliable until replaced, and
    the player frame has no `game_id` to join on instead.

    `player_frame` is the CFB player box score with `rushing_yards` and
    `receiving_yards` already extracted per player-game (see
    `data/player_stats.py`'s `_CATEGORY_TYPE_TO_COLUMN`). A quarterback's
    `passing_yards` is the team's passing total and must **not** be summed here,
    or the quarterback's own row double-counts against `net_passing_yards`.
    """
    required = ["rushing_yards", "receiving_yards"]
    missing = [column for column in required if column not in player_frame.columns]
    if missing:
        raise ValueError(
            f"player_frame is missing {missing}; the CFB player box score nests at "
            f"`types[]` while the team box score nests at `stats[]`, and mixing them up "
            f"produces a silent empty join rather than an error"
        )
    for name, frame in (("team_frame", team_frame), ("player_frame", player_frame)):
        if "week" not in frame.columns:
            raise ValueError(
                f"{name} has no `week`. The CFBD team box score carries none and its `week` "
                f"request argument over-returns, so call attach_schedule_weeks(team_frame, "
                f"schedules) first -- joining on week without it silently duplicates team-games."
            )

    team_side = team_frame.copy()
    player_side = player_frame.copy()
    for side in (team_side, player_side):
        side["week"] = pd.to_numeric(side["week"], errors="coerce").astype("Int64")
        side["season"] = pd.to_numeric(side["season"], errors="coerce").astype("Int64")

    keys = ["season", "week", "team"]
    player_totals = (
        player_side.groupby(keys, as_index=False)[required]
        .sum()
        .rename(columns={"rushing_yards": "player_rushing_yards", "receiving_yards": "player_receiving_yards"})
    )
    summed = ["player_rushing_yards", "player_receiving_yards"]
    # Presence is checked on the summed components, not with `min_count` on the
    # groupby: there it is per *column* (N non-null observations of that one
    # column), not N populated components, so it NaN'd out any game with a single
    # player row.
    player_totals["player_total_yards"] = player_totals[summed].sum(axis=1, min_count=len(summed))
    team_totals = add_total_yards(team_side)[keys + ["total_yards"]]
    merged = team_totals.merge(player_totals[keys + ["player_total_yards"]], on=keys, how="inner")
    merged["diff"] = merged["total_yards"] - merged["player_total_yards"]
    return merged
