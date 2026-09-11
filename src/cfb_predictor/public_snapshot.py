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
    for week in range(1, MAX_WEEK + 1):
        key = str(week)
        if rebuild_from <= week <= rebuild_to or key not in previous_weeks:
            print(f"  week {week}")
            weeks[key] = _build_week(season, week)
        else:
            weeks[key] = previous_weeks[key]

    print("Building season standings projection (this predicts every remaining FBS game -- slow)...")
    try:
        standings = routes._get_standings_live(season)
    except Exception as exc:
        print(f"  ! skipped standings: {exc}")
        standings = previous.get("standings", []) if previous.get("season") == season else []

    import pandas as pd

    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "season": season,
        "current_week": current_week,
        "weeks": weeks,
        "standings": standings,
    }


def main() -> None:
    previous = json.loads(config.PUBLIC_SNAPSHOT_PATH.read_text()) if config.PUBLIC_SNAPSHOT_PATH.exists() else None
    snapshot = jsonable_encoder(build_snapshot(previous))
    config.PUBLIC_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {config.PUBLIC_SNAPSHOT_PATH} ({config.PUBLIC_SNAPSHOT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
