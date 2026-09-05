"""odds_api.py — The Odds API bulk h2h/totals/spreads feed for CFB.

Same account/key already configured for PL_Predictor/NFL_Predictor — only
ODDS_API_SPORT_KEY differs (americanfootball_ncaaf). Best-effort: any
failure (missing key, network error, bad response) returns an empty
DataFrame rather than raising, matching every other best-effort data source
across these projects. Odds coverage for FBS games outside Power-conference
matchups is expected to be sparser than NFL's -- this module's job is just
returning whatever the API has; api/routes.py and odds/value_bets.py already
degrade gracefully to "no recommendation" for any game with no quoted line.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from ..config import ODDS_API_BASE_URL, ODDS_API_KEY, ODDS_API_SPORT_KEY


def _fetch_raw_odds() -> list[dict]:
    url = f"{ODDS_API_BASE_URL}/{ODDS_API_SPORT_KEY}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "decimal",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_game_odds() -> pd.DataFrame:
    if not ODDS_API_KEY:
        return pd.DataFrame()

    try:
        events = _fetch_raw_odds()
    except Exception:
        return pd.DataFrame()

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for event in events:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "event_id": event["id"],
                            "commence_time": event["commence_time"],
                            "home_team": event["home_team"],
                            "away_team": event["away_team"],
                            "bookmaker": bookmaker["key"],
                            "market": market["key"],
                            "outcome_name": outcome["name"],
                            "price": outcome["price"],
                            "point": outcome.get("point"),
                            "odds_fetched_at": fetched_at,
                        }
                    )
    return pd.DataFrame(rows)
