"""public_snapshot.py — precomputes games/predictions/player-props for a
window of weeks so the public deployment never has to run this project's
live feature-building pipeline (CFBD fetches, feature build, model
predict, the whole-FBS roster fallback) on an actual request. This is also
what keeps CFBD's 1,000-calls/month quota from being exhausted by request
traffic -- every CFBD call now happens here, on a schedule, not per visit.

Run this locally, or from .github/workflows/refresh-public-snapshot.yml:

    python -m cfb_predictor.public_snapshot

Then commit + push data/public_snapshot.json -- the running deployment
picks it up within PUBLIC_SNAPSHOT_POLL_SECONDS via
api/routes.py::refresh_public_snapshot_from_remote, no redeploy needed.
See config.py's PUBLIC_SNAPSHOT_PATH for the rest of that mechanism.
"""

from __future__ import annotations

import json

from fastapi.encoders import jsonable_encoder

from . import config
from .api import routes

# How many weeks past the current one get freshly rebuilt every run.
# Everything else reuses the previous snapshot verbatim (if that week was
# already in it). A full-season rebuild is both slow (player props alone
# runs several seconds per player-position pass across every FBS team) and
# burns CFBD calls for weeks nobody's currently browsing.
REBUILD_WEEKS_AHEAD = 3
REBUILD_WEEKS_BEHIND = 1
MAX_WEEK = 16


def _build_week(season: int, week: int) -> dict:
    games = routes._get_games_live(season, week)
    predictions: dict[str, dict] = {}
    for game in games:
        game_id = game["game_id"]
        try:
            predictions[game_id] = routes._get_game_prediction_live(season, week, game_id)
        except Exception as exc:
            print(f"    ! skipped prediction for {game_id}: {exc}")
    try:
        player_props = routes._get_player_props_live(season, week)
    except Exception as exc:
        print(f"    ! skipped player props for week {week}: {exc}")
        player_props = []
    return {"games": games, "predictions": predictions, "player_props": player_props}


def _position_invariant_keys(rows: list[dict]) -> frozenset[str] | None:
    """The prop-row keys the current code emits on EVERY row, whatever the position.

    Prop rows are position-heterogeneous, and deliberately so: `predict_props`
    keys off `POSITION_MARKETS` (models/player_props.py:30), so a QB row
    carries `passing_yards`, an RB row `rushing_yards`, a WR/TE row
    `receiving_yards`, and a K/OL/DL/P row no market at all. Every week of the
    committed artifact has three distinct row shapes for exactly that reason.
    The only keys every row shares are the ones routes.py writes literally
    (`player_id`, `player_name`, `recent_team`, `position`, `is_starter`,
    `depth_slot`) plus `anytime_td_prob`, which `predict_props` always sets.

    So the signature is the INTERSECTION of the rows, never one row's key set.
    Returns None for an empty sample, which the caller reports rather than
    guesses from (see `_prop_key_signature`).
    """
    if not rows:
        return None
    invariant = frozenset(rows[0].keys())
    for row in rows[1:]:
        invariant &= row.keys()
    return invariant


def _prop_key_signature(
    season: int,
    current_week: int,
    weeks: dict[str, dict],
    reused: list[str],
) -> frozenset[str] | None:
    """The position-invariant prop-row keys the CURRENT code produces, or None if
    undeterminable.

    Every rebuilt week's rows are pooled and intersected, because prop rows are
    not one shape (see `_position_invariant_keys`). Pooling costs nothing extra
    -- those weeks are already built -- and it is what makes the result
    position-invariant: a single week that happens to be all-QBs would put
    `passing_yards` in the signature and then demand it of every WR row in the
    season.

    Only if every rebuilt week is prop-less -- which is the real situation early
    in a season, when the rebuild window holds no games yet -- does it make a
    single live call for the current week, and it intersects that sample the
    same way. Still one call, never one per week.

    Returns None rather than an empty set when it genuinely cannot tell. An
    empty set would compare equal against every prop-less week and report
    "nothing to do", which is the bug being fixed.
    """
    current_rows: list[dict] = []
    for key, week in weeks.items():
        if key in reused:
            continue
        current_rows.extend(week.get("player_props") or [])
    signature = _position_invariant_keys(current_rows)
    if signature is not None:
        return signature

    try:
        sample = routes._get_player_props_live(season, current_week)
    except Exception as exc:  # noqa: BLE001 - reported by the caller
        print(f"  ! live prop probe failed: {exc}")
        return None
    return _position_invariant_keys(sample or [])


def _prop_shape_mismatch(week: dict, required: frozenset[str]) -> bool:
    """True when ANY prop row of a reused week predates the current row shape.

    Every row is checked, not row 0. `required` is position-invariant (see
    `_position_invariant_keys`), so a week whose row 0 happens to be current
    while its other rows are stale is stale -- the case a row-0 subset test
    could not see at all, because it only ever asked about one row of three.

    A prop-less week is not a mismatch: it has no rows to be stale, and
    rebuilding it would be pure cost.
    """
    props = week.get("player_props") or []
    if not props:
        return False
    return any(not required <= row.keys() for row in props)


def build_snapshot(previous: dict | None = None) -> dict:
    season, current_week = routes.current_season_and_week()
    previous = previous or {}
    previous_weeks = previous.get("weeks", {}) if previous.get("season") == season else {}

    rebuild_from = max(1, current_week - REBUILD_WEEKS_BEHIND)
    rebuild_to = min(MAX_WEEK, current_week + REBUILD_WEEKS_AHEAD)

    print(
        f"Building weeks 1-{MAX_WEEK} (current: {current_week}); "
        f"rebuilding {rebuild_from}-{rebuild_to}, reusing the rest..."
    )
    weeks: dict[str, dict] = {}
    reused: list[str] = []
    for week in range(1, MAX_WEEK + 1):
        key = str(week)
        if rebuild_from <= week <= rebuild_to or key not in previous_weeks:
            print(f"  week {week}")
            weeks[key] = _build_week(season, week)
        else:
            weeks[key] = previous_weeks[key]
            reused.append(key)

    # A reused week is a copy, so it can never pick up a field that the code has
    # since started emitting. That is not a hypothetical: `is_starter` and
    # `depth_slot` were added to the CFB prop rows, and every week that actually
    # carries props sits OUTSIDE the rebuild window -- at current_week 5 the
    # window is weeks 4-8, while weeks 1-3 and 9-13 are copies. So the new
    # fields reached five weeks of fifteen and nowhere else. The symptom is a
    # frontend looking for a field the API is supposed to serve and finding it
    # missing on most of the season -- and the obvious "it's a serialization
    # bug" conclusion is wrong. It is the copy, not the writer.
    #
    # So: work out the shape the CURRENT code produces, and rebuild any reused
    # week whose props do not match it. Narrow on purpose -- only the weeks
    # whose row shape actually changed are rebuilt, so adding a field costs one
    # build rather than all fifteen. A week carrying fields the code no longer
    # emits is left alone rather than caught in a rebuild loop.
    #
    # "Shape" has to mean the position-INVARIANT keys and be checked on every
    # row, or the reconciliation is wrong in two directions at once: judging
    # one row leaves the other ~2,000 of the week unchecked (a position-
    # specific new field would land on row 0 and nowhere else), and judging a
    # whole row's key set against a *different* row's key set rebuilds a week
    # that is already current, forever, on the strength of a market key.
    #
    # This is the same defect and the same fix as NFL_Predictor commit 0c4ea1e,
    # ported because the two modules are copies of each other and only one of
    # them had it.
    signature = _prop_key_signature(season, current_week, weeks, reused)
    if signature is None:
        print("  ! could not determine the current prop shape; reused weeks were NOT reconciled")
    else:
        stale = [key for key in reused if _prop_shape_mismatch(weeks[key], signature)]
        for key in stale:
            print(f"  week {key}: prop shape changed, rebuilding")
            weeks[key] = _build_week(season, int(key))
        if stale:
            print(f"  reconciled {len(stale)} reused week(s) onto the new prop shape")

    print("Building season standings projection (this predicts every remaining FBS game -- slow)...")
    try:
        standings = routes._get_standings_live(season)
    except Exception as exc:
        print(f"  ! skipped standings: {exc}")
        standings = previous.get("standings", []) if previous.get("season") == season else []

    print("Building power rankings...")
    try:
        power_rankings = routes._get_power_rankings_live(season)
    except Exception as exc:
        print(f"  ! skipped power rankings: {exc}")
        power_rankings = previous.get("power_rankings", {}) if previous.get("season") == season else {}

    print("Building Data Hub tables...")
    try:
        hub_teams = routes._get_hub_teams_live(season)
    except Exception as exc:
        print(f"  ! skipped hub teams: {exc}")
        hub_teams = previous.get("hub_teams", {}) if previous.get("season") == season else {}
    try:
        hub_players = routes._get_hub_players_live(season)
    except Exception as exc:
        print(f"  ! skipped hub players: {exc}")
        hub_players = previous.get("hub_players", {}) if previous.get("season") == season else {}

    import pandas as pd

    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "season": season,
        "current_week": current_week,
        "weeks": weeks,
        "standings": standings,
        "power_rankings": power_rankings,
        "hub_teams": hub_teams,
        "hub_players": hub_players,
    }


def main() -> None:
    previous = json.loads(config.PUBLIC_SNAPSHOT_PATH.read_text()) if config.PUBLIC_SNAPSHOT_PATH.exists() else None
    snapshot = jsonable_encoder(build_snapshot(previous))
    config.PUBLIC_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {config.PUBLIC_SNAPSHOT_PATH} ({config.PUBLIC_SNAPSHOT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
