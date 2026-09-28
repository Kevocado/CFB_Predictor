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
**271 of 272** team-games (2023 week 1, after dropping 42 unplaceable non-FBS
games). The single exception is a 1-yard difference at Incarnate Word
(245 + 64 = 309 against a reported 308).

Reconciliation constraint. Because both sides come from the same game payload,
`sum_players(rushing + receiving)` reproduces the team `totalYards` to within a
small residual: **161 of 272 exact, 260 of 272 (95.6%) within 10 yards, median
absolute difference 0.00, maximum 29.0** (2023 week 1, keyed on `game_id`).

That earlier figure of "355 of 356, median error 0.00, sole outlier Robert Morris
at -14" was **wrong on all three counts** -- it was measured on the broken
week-keyed join, on a row count taken before unplaceable games were dropped, and
the outlier is a residual, not a single row. Reproduced below.

That makes allocate-by-share the correct architecture: project the team total
first, then allocate. Summing independent player projections cannot reproduce
it, which is what the dashboard does today and why it is out by 2.2-3.2x.

Note the quarterback special case when reconciling: a QB's `passing.YDS` is the
*team's* passing total, so a QB must be counted via `passing` or excluded from
the receiving-side sum, never both.

**Reconciliation status, measured 2026-09-27.** Both problems that once blocked
it are fixed, and the join is now sound enough to build on:

- 42 of the 178 game_ids a week-1 request returns are absent from the schedule.
  The reason is **division, not week**: `get_games(classification="fbs")` matches
  games with *at least one* FBS team, so FBS-vs-FCS "buy games" are present and
  all 42 absent games are **FCS-vs-FCS** -- Morgan State vs Richmond
  (`401539978`), Jackson State vs South Carolina State, Lafayette vs Sacred Heart,
  Fordham vs Wagner, and so on. An earlier version of this note cited
  "Air Force vs Robert Morris" as one of the 42; that is wrong, Air Force vs
  Robert Morris (`401532570`) **is** in the schedule and reconciles normally.
  Back-filling those with `requested_week` stacked 84 team-rows on top of the
  genuine week-1 rows; `attach_schedule_weeks` now drops them and counts them.
- The player sum was keyed on `(season, week, team)`, which is **not unique** — a
  team can play twice in one week — so both games' players were summed and
  compared against each single game. That is where a 1,620-yard "player total"
  for one 443-yard game came from. `game_id` is now carried through the player
  frame and is the join key.

On 2023 week 1, 272 team-games: 161 exact, 260 within 10 yards (95.6%), median
absolute difference 0.00, maximum 29 yards. Previously 58 exact, median 336,
maximum 1177.

**The residual is characterised, not eliminated.** 11 of the 12 remaining
offenders run the same direction — the player sum *exceeds* the team's
`totalYards` — which points at a definitional difference between CFBD's team
`totalYards` and the sum of player `rushing + receiving`, with net-rushing
treatment of sacks the obvious candidate. NFL's equivalent identity is verified
clean (544/544 exact), so this is a property of the CFB source rather than a
shared bug. A yardage model can be built on this join; it should be validated
against the same 10-yard tolerance rather than assumed exact.

There is still no backfill writing team yardage into the tracking DB, and the
store must be keyed on `game_id` — keying it on the week is what caused the bug
above, and would hide it at the persistence layer.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from ..config import CFBD_API_KEY, TEAM_STATS_CACHE_DIR

logger = logging.getLogger(__name__)

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


def _empty_week_frame(season: int, week: int) -> pd.DataFrame:
    """A schema-correct empty frame for a (season, week) that returned no games.

    A populated `_flatten` result carries `requested_week` and whatever stat
    categories happened to be present, so an empty week has to declare at least
    all of that -- never less, or the two are not interchangeable. Two details that
    are load-bearing rather than cosmetic:

    - `requested_week` is required by `_read_cache` and is not in `KEEP_COLUMNS`.
      Omit it and the staleness gate rejects the tombstone as pre-dating the
      column, so the week is re-requested on every run against a metered API --
      the exact failure the tombstone exists to prevent.
    - `week` *is* declared, even though a freshly flattened frame has none. `week`
      is added later by `attach_schedule_weeks`; declaring it here means a
      tombstoned week and a freshly-fetched one are the same shape to every
      caller, and an absent column is far easier to mistake for a real zero than
      an absent row is.
    """
    return pd.DataFrame(
        {
            "game_id": pd.Series(dtype="object"),
            "season": pd.Series(dtype="int64"),
            "requested_week": pd.Series(dtype="int64"),
            **{column: pd.Series(dtype="object") for column in KEEP_COLUMNS if column not in {"game_id", "season"}},
        }
    )


def _flatten(raw_games: list, season: int, week: int) -> pd.DataFrame:
    """Walk CFBD's nested game -> team -> stats tree into one row per team-game.

    Three things the payload does *not* carry, all verified against real data:

    - `GameTeamStats` has only `id` and `teams` — no `season`.
    - `GameTeamStatsTeam` has no `opponent`; it is reconstructed by pivoting the
      game's two teams against each other.
    - **There is no `week` field at all, and the endpoint's `week` argument does
      not reliably filter.** `get_game_team_stats(year=2023, week=1)` returns 178
      games, but only 246 of the resulting 356 team-rows carry a distinct
      (season, week, team) key: Air Force appears twice, against Robert Morris
      and against James Madison, the latter a December bowl. The `week` written
      here is therefore the *requested* week and is wrong for any game the
      endpoint over-returns. (This is why `attach_schedule_weeks` exists, and why
      the reconciliation is keyed on `game_id` rather than on the week.)

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
        return _empty_week_frame(season, week)
    return pd.DataFrame(rows)


def attach_schedule_weeks(frame: pd.DataFrame, schedules_df: pd.DataFrame) -> pd.DataFrame:
    """Give each team-game its real `week` from the schedule, dropping unplaceable rows.

    Needed because CFBD's team box score has no week field and its `week` request
    argument over-returns.

    **Rows whose `game_id` is absent from the schedule are dropped, not
    back-filled with `requested_week`.** Measured 2026-09-27 on 2023 week 1: 42 of
    178 returned game_ids are not in the schedule. The reason is **division, not
    week**: `get_games(classification="fbs")` matches games with at least one FBS
    team, so FBS-vs-FCS games are present and all 42 absent games are FCS-vs-FCS
    (Morgan State vs Richmond, `401539978`, is one). Back-filling those 42 with
    `requested_week=1` silently stacked 84 team-rows on top of the genuine week-1
    rows, collapsing 356 rows to 246 distinct (season, week, team) keys and
    inflating any player sum that joined across them. A game that cannot be placed
    cannot be reconciled, so it is excluded and counted.

    Returns the frame with a `week` column. `dropped_unplaceable` is attached as an
    attribute on the returned object for callers that want to log it.
    """
    out = frame.copy()
    if schedules_df.empty or not {"game_id", "week"} <= set(schedules_df.columns):
        # Previously this back-filled `requested_week`, which is the exact bug the
        # rest of this function exists to prevent: it stacks several real games
        # onto one week, which then reconciles against each other. `fetch_schedules`
        # returns an empty frame when the cache is cold and CFBD is unreachable, so
        # this branch is reachable, and it was silent.
        raise ValueError(
            "attach_schedule_weeks needs a schedule carrying game_id and week, and got "
            f"neither (rows={len(schedules_df)}, columns={list(schedules_df.columns)[:6]}). "
            "Refusing to fall back to `requested_week`: CFBD's `week` argument over-returns, "
            "so back-filling stacks several games onto one week. Call fetch_schedules() "
            "first and let it raise if the schedule is genuinely unavailable."
        )
    lookup = schedules_df[["game_id", "week"]].drop_duplicates(["game_id"]).copy()
    lookup["game_id"] = lookup["game_id"].astype(str)
    out["game_id"] = out["game_id"].astype(str)
    placed = out.merge(lookup, on="game_id", how="left")
    dropped = int(placed["week"].isna().sum())
    out = placed[placed["week"].notna()].copy()
    out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
    out.dropped_unplaceable = dropped
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
            # Record an empty week rather than skipping it. Skipping looks harmless
            # and is not: no cache file means the next run re-requests, and most
            # seasons end before week 15 (2023 has no week 16 at all), so every
            # re-run of a 2004-2025 sweep pays again for every dead week in range.
            # That is what makes "just run it again" -- the normal response to an
            # interrupted backfill -- quietly more expensive each time.
            #
            # The tombstone is a real schema-correct empty frame, not a sentinel, so
            # it concatenates and compares like any other week.
            if frame.empty:
                frame = _empty_week_frame(season, week)
            frame.to_parquet(path, index=False)
            if not frame.empty:
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
    team_frame: pd.DataFrame,
    player_frame: pd.DataFrame,
    allow_week_key: bool = False,
) -> pd.DataFrame:
    """Per team-game: team `totalYards`, that game's player sum, and the gap.

    **Keyed on `game_id` when both frames carry one**, which is the only key that
    is actually unique. A team can play twice in one week and CFBD's `week`
    request argument over-returns, so `(season, week, team)` is not unique in
    either frame: joining on it summed *both* games' players and compared the lot
    against each single game. Measured 2023 week 1, that is where a 1,620-yard
    "player total" for one 443-yard game came from.

    Falls back to `(season, week, team)` for frames without a `game_id` so
    hand-built fixtures and pre-`game_id` callers still work -- but a week-keyed
    reconciliation cannot be exact, and says so in `reconciliation_key`.

    **Both frames must still carry a `week` sourced from the schedule**
    (`attach_schedule_weeks` for the team side; the player frame picks its week
    up from a schedule merge). CFBD's team box score has no `week` field and its
    `week` request argument over-returns, so the team frame's own week is
    unreliable until replaced.

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
    # `week` is required only where it is actually used. Since the join is keyed on
    # `game_id`, requiring a schedule-derived week bought nothing and cost 31% of the
    # frame: `attach_schedule_weeks` drops any game absent from the schedule, and
    # CFBD labels *every* bowl `week: 1`, so the recovered week is also wrong for a
    # third of what survives. The week-keyed path still needs it, and still demands
    # it -- that path is not unique, which is the whole reason `game_id` exists.
    if "game_id" not in team_frame.columns or "game_id" not in player_frame.columns:
        for name, frame in (("team_frame", team_frame), ("player_frame", player_frame)):
            if "week" not in frame.columns:
                raise ValueError(
                    f"{name} has no `week`, and without `game_id` on both frames the only "
                    f"available key is (season, week, team) -- which is not unique. Either "
                    f"call attach_schedule_weeks() first, or supply `game_id`."
                )

    if "game_id" in team_frame.columns and "game_id" in player_frame.columns:
        keys = ["game_id", "team"]
    elif not allow_week_key:
        # Previously this fell back silently, and that is how a stale player cache
        # (written before `game_id` existed) reintroduced the exact bug this key
        # was introduced to fix -- with a green test run, because the tests call
        # the flatteners directly and never touch the cached loader. A frame
        # without `game_id` can only be joined on a key that is not unique, so
        # refuse rather than return a number that looks reconciled.
        missing = [
            name for name, frame in (("team_frame", team_frame), ("player_frame", player_frame))
            if "game_id" not in frame.columns
        ]
        raise ValueError(
            f"{missing} lack `game_id`, so the only available key is (season, week, team) -- "
            f"which is NOT unique, because a team can play twice in one week and CFBD's "
            f"`week` argument over-returns. Reconciling on it sums several games' players "
            f"and compares the lot against each single game. Fix the cache "
            f"(data.player_stats._read_cache now refetches a cache missing required columns) "
            f"or pass allow_week_key=True if you are deliberately testing that failure."
        )
    else:
        keys = ["season", "week", "team"]

    team_side = team_frame.copy()
    if keys == ["game_id", "team"]:
        # The merge's correctness rests on this being unique. It holds empirically
        # (verified 272/272 on the 2023 week-1 cache, 600/600 across weeks 1+2 with
        # zero game_id overlap between the two responses) but nothing enforced it,
        # and a duplicated stanza would not inflate the player sum -- the right side
        # is grouped, so a left join cannot fan out -- it would silently duplicate
        # the output row and double-count that game downstream.
        before = len(team_side)
        team_side = team_side.drop_duplicates(["game_id", "team"])
        if len(team_side) != before:
            logger.warning(
                "dropped %d duplicate (game_id, team) rows before reconciling; the team box "
                "score returned a repeated stanza", before - len(team_side),
            )
        assert not team_side.duplicated(["game_id", "team"]).any(), (
            "(game_id, team) must be unique for this join to mean anything")
    player_side = player_frame.copy()
    for side in (team_side, player_side):
        if "week" in side.columns:
            side["week"] = pd.to_numeric(side["week"], errors="coerce").astype("Int64")
        if "season" in side.columns:
            side["season"] = pd.to_numeric(side["season"], errors="coerce").astype("Int64")

    player_totals = (
        player_side.groupby(keys, as_index=False)[required]
        .sum(min_count=1)
        .rename(columns={"rushing_yards": "player_rushing_yards", "receiving_yards": "player_receiving_yards"})
    )
    summed = ["player_rushing_yards", "player_receiving_yards"]
    # `min_count=1` on the groupby is load-bearing and was missing. pandas'
    # `groupby.sum()` treats an all-NaN column as 0, so a game where *no* player
    # has a `receiving_yards` value came back with `player_receiving_yards = 0`
    # -- a real zero -- and the `min_count` on the axis sum below then saw two
    # populated values and never fired. Result: a game with only rushing rows
    # reported `player_total_yards = 90.0` instead of missing, i.e. a partial
    # figure that looks like a complete one. Caught by mutating that axis
    # `min_count` to 1, which the offline suite did not notice.
    #
    # Both are needed. The groupby `min_count=1` stops an all-NaN component
    # collapsing to 0; the axis `min_count=len(summed)` then requires both
    # components to be genuinely present before calling a total complete.
    player_totals["player_total_yards"] = player_totals[summed].sum(axis=1, min_count=len(summed))
    team_totals = add_total_yards(team_side)[keys + ["total_yards"]]
    # `how="left"`, not "inner": a team-game whose players the player endpoint
    # did not return must surface as a missing sum. An inner join drops it, and a
    # dropped row reads as "reconciled" in any count of matches.
    merged = team_totals.merge(player_totals[keys + ["player_total_yards"]], on=keys, how="left")
    merged["diff"] = merged["total_yards"] - merged["player_total_yards"]
    merged["reconciliation_key"] = "game_id" if keys[0] == "game_id" else "season_week_team"
    return merged
