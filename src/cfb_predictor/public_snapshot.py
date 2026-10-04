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
from .models import manifest
from .models import player_props

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


def _market_keys() -> frozenset[str]:
    """Every key `POSITION_MARKETS` can put on a prop row.

    `predict_props` (models/player_props.py:53-68) emits `anytime_td_prob`
    unconditionally and then one key per market, and only `if market in models`
    -- so the reachable set is exactly the union of the per-position lists, and
    which subset a given row gets is decided inside the position. The callers
    here treat that set as position-specific, which is the safe reading: a
    market is present or absent for reasons that have nothing to do with the
    code's age, so requiring one of these keys says nothing about whether a
    week predates a change.

    `anytime_td_prob` is deliberately NOT in here: `predict_props` sets it on
    every row, whatever the position, so it IS position-invariant and is
    required like any other invariant key.
    """
    return frozenset(
        market for markets in player_props.POSITION_MARKETS.values() for market in markets
    )


def _position_invariant_keys(rows: list[dict]) -> frozenset[str] | None:
    """The prop-row keys the current code emits on EVERY row, whatever the position.

    Prop rows are position-heterogeneous, and deliberately so: `predict_props`
    keys off `POSITION_MARKETS` (models/player_props.py:30), so a QB row
    carries `passing_yards`, an RB row `rushing_yards`, and a WR/TE row
    `receiving_yards`/`receptions`. Every week of the committed artifact has
    three distinct row shapes for exactly that reason. A K/OL/DL/P row cannot
    appear in a prop week at all -- routes.py filters prop output to
    {WR, TE, RB, QB} (routes.py:620-622) -- so "no market at all" is not a
    shape any row here can have. What *does* vary is which of the four modelled
    positions a week happens to contain, and even that is not a guarantee worth
    leaning on: within one position the market set is decided per row by
    `if market in models` (models/player_props.py:65), so an RB row can be
    `rushing_yards` with no `carries` on it while those models are missing.

    So the invariant key set is computed two ways, and both are needed:

    * every key that `POSITION_MARKETS` can introduce is SUBTRACTED, by name,
      from the sample. That makes the result independent of which positions
      the sample contains, including a single-position sample -- so an all-QB
      week cannot put `passing_yards` in the signature and then demand it of
      every WR row in the season. This is the property the check actually
      needs, and it is unconditional rather than a property of the sample.
    * what is left is the INTERSECTION of the sample's rows, so any *other*
      position-specific key added later without touching `POSITION_MARKETS`
      still cannot leak in by being row 0's shape. Defence in depth, not the
      mechanism.

    Returns None for an empty sample, which the caller reports rather than
    guesses from (see `_prop_key_signature`).
    """
    if not rows:
        return None
    invariant = frozenset(rows[0].keys())
    for row in rows[1:]:
        invariant &= row.keys()
    return invariant - _market_keys()


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
    -- those weeks are already built. It is defence in depth: the subtraction of
    `POSITION_MARKETS` is what makes the result position-invariant, and it holds
    for a sample of one row, so pooling is not what stands between an all-QB
    week and a season-wide demand for `passing_yards`.

    Only if every rebuilt week is prop-less -- which is the case when the rebuild
    window holds no games, before the season starts -- does it make a single
    live call for the current week, and it intersects that sample the same way.
    Still one call, never one per week.

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
    # row. Both halves are load-bearing and they do different jobs:
    #
    #   * Checking every row is what makes the predicate CONSERVATIVE. A
    #     position-invariant key (`is_starter`, `depth_slot`, `anytime_td_prob`)
    #     is required of each row independently, so a week where row 0 happens to
    #     be current and rows 1-2,499 are stale is stale. Judging one row left
    #     ~2,000 of a week unchecked, and the shipped artifact came out right
    #     only because every row was stale, so row 0 was stale too.
    #   * Being position-invariant is what makes it CHEAP. A key that follows the
    #     position is excluded from `required` by construction, so no mix of
    #     rows can demand it of a row that legitimately lacks it.
    #
    # What this deliberately does NOT catch, because it cannot: a NEW
    # position-specific field. `POSITION_MARKETS` is per-position, so the most
    # likely next change to this row shape is a market added to one position --
    # and the subtraction excludes it from `required` on purpose, before any
    # row is looked at. So a market added to, say, RB alone will reach the
    # rebuilt window weeks and no others, exactly the partial-season symptom
    # this whole block exists to prevent, and nothing here will say so. A
    # position-specific field needs its own decision: either a one-off rebuild
    # of the season, or a deliberately widened `required` for that run. Do not
    # "fix" it by making `required` position-dependent again -- that is the
    # defect this signature replaced, and it costs a full `_build_week` per
    # affected week on every scheduled run, against a 1,000-calls/month quota.
    #
    # Provenance, because the two repos are copies of each other and this one
    # was AHEAD of the other. NFL_Predictor commit 0c4ea1e is the ORIGINAL row-0
    # fix: it took props[0]'s key set as the signature and tested it against
    # props[0]'s keys. That is insufficient in both directions and this module
    # is the correction of it, not a copy.
    #
    # As of NFL_Predictor 7f50b82 that is no longer one-sided: NFL carried the
    # same defect, has now been corrected with the same design (its
    # `_position_invariant_keys` subtracts ITS `POSITION_MARKETS`, and its
    # `_get_player_props_live` does not filter by position, so a K/OL/DL/P row
    # with no market is a shape that exists there and not here). So read this
    # file as the ORIGIN of the design and NFL as the second repo carrying it --
    # not as a description of NFL's shape, and not as licence to port NFL's
    # identifiers here. Copy the construction, and in particular keep the
    # subtraction: a plain intersection, or a single row, still depends on the
    # sample happening to be multi-position, which is a fact about the sample
    # and not about the code.
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


def _verify_models_current() -> None:
    """Raise unless the committed models match the code about to publish them.

    A published snapshot is a promise that every number in it came from the
    code in this repo. Nothing at write time used to check that: `build_snapshot`
    swallows per-week exceptions (`! skipped prediction for ...`) and reuses
    previously-published weeks wholesale, so a build against stale models
    produces a full, well-formed, wrong file. The site then serves the old
    label's numbers exactly as NFL's did.

    Deliberately in the writing code and not in the workflow YAML. The
    refresh-public-snapshot workflow can be dispatched by hand, and the deploy
    workflow can run from a different ref, so a YAML-only check is bypassed by
    exactly the runs nobody re-reads.

    It calls the same `_verify_artifact_fingerprint` that `load_models` calls,
    rather than `load_models` itself: the check is the point, and calling it
    directly means no pickle is unpickled to decide whether the snapshot may be
    written. Sharing the one function means the snapshot cannot be published
    from a model the API would have rejected.
    """
    manifest._verify_artifact_fingerprint(manifest.load_manifest())


def main() -> None:
    _verify_models_current()
    previous = json.loads(config.PUBLIC_SNAPSHOT_PATH.read_text()) if config.PUBLIC_SNAPSHOT_PATH.exists() else None
    snapshot = jsonable_encoder(build_snapshot(previous))
    config.PUBLIC_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {config.PUBLIC_SNAPSHOT_PATH} ({config.PUBLIC_SNAPSHOT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
