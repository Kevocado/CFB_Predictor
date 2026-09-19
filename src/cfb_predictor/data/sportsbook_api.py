"""sportsbook_api.py — live CFB (NCAAF) odds via RapidAPI's Sportsbook API
(https://rapidapi.com/sportsbook-api-sportsbook-api-default/api/sportsbook-api2),
the odds pipeline's primary source since 2026-09 -- same switch
PL_Predictor made and for the same reason: The Odds API's shared key
(config.py's ODDS_API_KEY, used across several of this account's projects)
confirmed live at 500/500 credits used, 0 remaining, so it couldn't serve
as a day-to-day source. odds_api.py is left in place, unused by default,
in case it's ever worth reverting to or supplementing with.

Produces the exact same long-format shape odds_api.fetch_game_odds()
already does (event_id, commence_time, home_team, away_team, bookmaker,
market, outcome_name, price, point, odds_fetched_at) -- a drop-in
replacement api/routes.py's _lines_for_game/_team_name_matches need no
changes to consume; Sportsbook API's own NCAAF participant names (e.g.
"Wake Forest", "Miami (FL)") are already close enough to CFBD's own short
names that the existing prefix-match logic works unchanged.

One event = one request for real prices -- confirmed live (same as
PL_Predictor's own module): no batch/bulk odds endpoint exists despite
trying comma-separated eventKeys, repeated eventKeys= params, and an
eventKeys[] array. /v0/competitions/{key}/events lists fixtures with
market *keys* attached but never outcomes/prices. CFB has far more games
per week (60-70+) than NFL/PL, against the same confirmed 150/day cap
(the x-ratelimit-requests-limit response header) -- both event-list and
per-event responses are cached to disk per SPORTSBOOK_CACHE_TTL_SECONDS
(20h, well above refresh-public-snapshot.yml's 4h cron gap -- a shorter
TTL guarantees every scheduled run re-fetches from scratch instead of
reusing the prior run's cache, confirmed live to blow the daily cap on
its own before spreads ever populate), and only games actually being
predicted right now ever get their odds fetched (never the whole season
at once), which is what keeps this inside budget.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from ..config import (
    SPORTSBOOK_API_BASE_URL,
    SPORTSBOOK_API_HOST,
    SPORTSBOOK_API_KEY,
    SPORTSBOOK_CACHE_DIR,
    SPORTSBOOK_CACHE_TTL_SECONDS,
    SPORTSBOOK_NCAAF_COMPETITION_KEY,
)

# FULL_MATCH is this provider's segment for the whole game (as opposed to
# HALF_1, QUARTER_3, etc.) -- the only segment this project's model prices
# against.
_SEGMENT = "FULL_MATCH"


def _headers() -> dict:
    return {"X-RapidAPI-Key": SPORTSBOOK_API_KEY, "X-RapidAPI-Host": SPORTSBOOK_API_HOST}


def _cache_path(name: str) -> Path:
    return SPORTSBOOK_CACHE_DIR / f"{name}.json"


def _is_fresh(path: Path, ttl_seconds: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds


def fetch_ncaaf_events_raw(force_refresh: bool = False) -> list[dict]:
    """Every event this provider has for the season, with market *keys*
    attached (no prices yet -- see fetch_event_odds_raw). One request;
    cached to disk."""
    cache_path = _cache_path("ncaaf_events")
    if not force_refresh and _is_fresh(cache_path, SPORTSBOOK_CACHE_TTL_SECONDS):
        return json.loads(cache_path.read_text())

    resp = requests.get(
        f"{SPORTSBOOK_API_BASE_URL}/competitions/{SPORTSBOOK_NCAAF_COMPETITION_KEY}/events",
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    events = resp.json().get("events", [])
    cache_path.write_text(json.dumps(events))
    return events


def fetch_event_odds_raw(event_key: str, force_refresh: bool = False) -> dict | None:
    """Real prices for one event. One request per call; cached to disk per
    event so a partial refresh doesn't re-spend budget on events that
    haven't gone stale yet."""
    cache_path = _cache_path(f"event_{event_key}")
    if not force_refresh and _is_fresh(cache_path, SPORTSBOOK_CACHE_TTL_SECONDS):
        return json.loads(cache_path.read_text())

    resp = requests.get(
        f"{SPORTSBOOK_API_BASE_URL}/events",
        headers=_headers(),
        params={"eventKeys": event_key},
        timeout=15,
    )
    resp.raise_for_status()
    # Nested as a one-element list per requested event (confirmed live,
    # same shape PL_Predictor's module documents):
    # {"events": [[{...}]]}, not flattened.
    groups = resp.json().get("events", [])
    event = groups[0][0] if groups and groups[0] else None
    if event is not None:
        cache_path.write_text(json.dumps(event))
    return event


def _best_quote_per_outcome(event: dict, market_type: str) -> dict[str, dict]:
    """One representative (participantKey or OVER/UNDER type) -> quote for
    the FULL_MATCH segment of market_type, picking the first bookmaker that
    has it (this project only ever reads a single point/price per side, the
    same way odds_api.py's downstream consumers already do with The Odds
    API's own first-bookmaker-wins behavior)."""
    best: dict[str, dict] = {}
    for market in event.get("markets", []):
        if market.get("type") != market_type or market.get("segment") != _SEGMENT:
            continue
        for rows in market.get("outcomes", {}).values():
            for row in rows:
                key = row.get("participantKey") or row.get("type")
                if key not in best:
                    best[key] = row
    return best


def event_to_rows(event: dict, fetched_at: str) -> list[dict]:
    """Normalizes one Sportsbook API event's FULL_MATCH markets into
    odds_api.py's own long-format row shape.

    Unlike the event-LIST response (fetch_ncaaf_events_raw, which has a
    top-level `participants` array), the priced per-event response from
    /v0/events (fetch_event_odds_raw) has no `participants` key at all
    (confirmed live) -- team names only exist nested inside each
    WIN-type outcome row's own `participant` object. Collected from
    whichever market has them, keyed by participantKey."""
    home_key = event.get("homeParticipantKey")
    participants: dict[str, str] = {}
    for market in event.get("markets", []):
        if market.get("segment") != _SEGMENT:
            continue
        for outcome_rows in market.get("outcomes", {}).values():
            for outcome_row in outcome_rows:
                pk, p = outcome_row.get("participantKey"), outcome_row.get("participant")
                if pk and p and pk not in participants:
                    participants[pk] = p.get("name")
    if home_key not in participants or len(participants) != 2:
        return []
    away_key = next(k for k in participants if k != home_key)
    home_team, away_team = participants[home_key], participants[away_key]

    def row(market: str, outcome_name: str, quote: dict, point: float | None) -> dict:
        return {
            "event_id": event["key"],
            "commence_time": event.get("startTime"),
            "home_team": home_team,
            "away_team": away_team,
            "bookmaker": quote["source"],
            "market": market,
            "outcome_name": outcome_name,
            "price": quote["payout"],
            "point": point,
            "odds_fetched_at": fetched_at,
        }

    rows: list[dict] = []

    moneyline = _best_quote_per_outcome(event, "MONEYLINE")
    if home_key in moneyline:
        rows.append(row("h2h", home_team, moneyline[home_key], None))
    if away_key in moneyline:
        rows.append(row("h2h", away_team, moneyline[away_key], None))

    spreads = _best_quote_per_outcome(event, "POINT_SPREAD")
    if home_key in spreads:
        rows.append(row("spreads", home_team, spreads[home_key], spreads[home_key]["modifier"]))
    if away_key in spreads:
        rows.append(row("spreads", away_team, spreads[away_key], spreads[away_key]["modifier"]))

    totals = _best_quote_per_outcome(event, "POINT_TOTAL")
    if "OVER" in totals:
        rows.append(row("totals", "Over", totals["OVER"], totals["OVER"]["modifier"]))
    if "UNDER" in totals:
        rows.append(row("totals", "Under", totals["UNDER"], totals["UNDER"]["modifier"]))

    return rows


def _team_name_matches(cfbd_name: str, sportsbook_name: str) -> bool:
    """Same prefix-match tolerance api/routes.py's own _team_name_matches
    uses against The Odds API's names -- Sportsbook API's NCAAF
    participant names (e.g. "Wake Forest", "Miami (FL)") are already close
    to CFBD's own short names, so this rarely needs the full tolerance,
    but stays consistent with the rest of this project's odds matching."""
    cfbd_words = cfbd_name.lower().split()
    sportsbook_words = sportsbook_name.lower().split()
    return sportsbook_words[: len(cfbd_words)] == cfbd_words


def match_events_to_games(events: list[dict], games_df: pd.DataFrame) -> list[dict]:
    """Only the event *summaries* (no prices yet) for games actually
    present in games_df -- callers then fetch real prices for just these,
    one request each. Matching the whole season's worth of events instead
    of scoping to games_df would cost one request per event regardless of
    whether this project even asked about that game, and CFB's full
    season is ~180 events against a 150/day cap."""
    matched = []
    for _, game in games_df.iterrows():
        home, away = game["home_team"], game["away_team"]
        for event in events:
            participants = {p["key"]: p["name"] for p in event.get("participants", [])}
            home_key = event.get("homeParticipantKey")
            if home_key not in participants or len(participants) != 2:
                continue
            away_name = next(name for key, name in participants.items() if key != home_key)
            if _team_name_matches(home, participants[home_key]) and _team_name_matches(away, away_name):
                matched.append(event)
                break
    return matched


def fetch_game_odds(games_df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """Drop-in replacement for odds_api.fetch_game_odds() -- same empty
    DataFrame on any failure (missing key, network error, bad response,
    empty games_df), matching every other best-effort data source across
    these projects. Unlike odds_api.fetch_game_odds() (no arguments, one
    bulk call gets every game at once), this one REQUIRES games_df
    (home_team/away_team columns, e.g. a single week's worth of games) to
    scope which events actually get a real per-event price request --
    calling this with the whole season loaded, or from a tight per-game
    loop across many weeks, would still be a request-per-event regardless
    of games_df, just wastefully repeated. See api/routes.py's callers:
    each already only fetches odds for the current week, never a future
    one (sportsbooks don't post real lines that far out anyway -- confirmed
    live, a game 4 months out came back with zero priced markets)."""
    if not SPORTSBOOK_API_KEY or games_df is None or games_df.empty:
        return pd.DataFrame()

    try:
        events = fetch_ncaaf_events_raw(force_refresh=force_refresh)
    except Exception:
        return pd.DataFrame()

    matched_events = match_events_to_games(events, games_df)

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for event_summary in matched_events:
        try:
            event = fetch_event_odds_raw(event_summary["key"], force_refresh=force_refresh)
        except Exception:
            continue
        if event is None:
            continue
        rows.extend(event_to_rows(event, fetched_at))

    return pd.DataFrame(rows)
