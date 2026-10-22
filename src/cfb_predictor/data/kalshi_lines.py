"""Collect Kalshi's college football game, spread and total markets as a free, public record of what the market thought.

Kalshi lists three binary series per college football game: one market per team to win ("Miami (OH) wins"), a ladder of
spreads ("Dartmouth wins by over 6.5 points") and a ladder of totals ("Full Game: Over 66.5 points scored"). Every rung is
a binary contract whose price is the market's probability of that outcome. Reading them needs no account, key or credits.
Snapshots taken before kickoff are the "lines" we had no historical source for; the same markets settle afterwards with a
yes/no `result`, so each snapshot is its own labelled outcome and needs no box-score join to be scored.

Prices are dollars in [0, 1] (0.62 = 62%); the dollar-denominated quote fields (`yes_ask_dollars` and friends) are what
`normalize` reads, so no cent conversion is needed. A missing bid/ask means no quote, not a 0% market.

`floor_strike` is the real threshold on a spread or total: "over 6.5 points" has floor_strike 6.5. A moneyline market has
no floor_strike, so its `line` is None.

`occurrence_datetime` is the kickoff; `close_time` is when Kalshi finishes settling, which is later. The no-hindsight rule in
the paper ledger compares against occurrence, not close_time.

Volume is recorded as Kalshi reports it, including zero: most of these markets have never traded, and the ledger's own
liquidity filter is where that gets decided.
"""
from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
#: Kalshi series ticker -> the market it prices.
SERIES = {
    "KXNCAAFGAME": "moneyline",
    "KXNCAAFSPREAD": "spread",
    "KXNCAAFTOTAL": "total",
}
#: The stats that price a game rather than a player. Their titles are not "<player>: N+ <stat>",
#: so the player slot stays None -- a game market must never look like a prop.
GAME_LEVEL_STATS = ("moneyline", "spread", "total")
LINES_DIR = Path("data/lines")
PAGE_LIMIT = 1000
SETTLED_LOOKBACK_DAYS = 4


def _dollars(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v


def normalize(market: dict, series: str, snapshot_ts: str) -> dict:
    """One flat row per market rung. Pure; the unit of everything downstream."""
    title = market.get("title") or ""
    # Same rule as NFL's normalizer: derive the player from the title first, then drop it for the
    # game-level stats. A total's title ("Full Game: Over 66.5 points scored") has a colon, so the
    # split would otherwise hand the ledger a player named "Full Game".
    player = title.split(":", 1)[0].strip() if ":" in title else None
    if SERIES.get(series) in GAME_LEVEL_STATS:
        player = None
    return {
        "snapshot_ts": snapshot_ts,
        "series": series,
        "stat": SERIES.get(series),
        "ticker": market.get("ticker"),
        "event_ticker": market.get("event_ticker"),
        "player": player,
        "title": title,
        "line": _dollars(market.get("floor_strike")),
        "yes_bid": _dollars(market.get("yes_bid_dollars")),
        "yes_ask": _dollars(market.get("yes_ask_dollars")),
        "no_bid": _dollars(market.get("no_bid_dollars")),
        "no_ask": _dollars(market.get("no_ask_dollars")),
        "last": _dollars(market.get("last_price_dollars")),
        "volume": _dollars(market.get("volume_fp")),
        "open_interest": _dollars(market.get("open_interest_fp")),
        "status": market.get("status"),
        "result": market.get("result") or None,
        "settlement_value": market.get("expiration_value") or None,
        "occurrence": market.get("occurrence_datetime") or market.get("expected_expiration_time"),
        "close_time": market.get("close_time"),
    }


def _get(session, path: str, params: dict) -> dict:
    for attempt in range(4):
        r = session.get(f"{BASE_URL}{path}", params=params, timeout=30)
        if r.status_code == 429:  # public API rate limit: back off, never hammer
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Kalshi rate-limited {path} after retries")


def fetch_series(session, series: str, status: str, min_close_ts: int | None = None) -> list[dict]:
    markets: list[dict] = []
    cursor = None
    while True:
        params = {"series_ticker": series, "status": status, "limit": PAGE_LIMIT}
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        if cursor:
            params["cursor"] = cursor
        body = _get(session, "/markets", params)
        markets.extend(body.get("markets") or [])
        cursor = body.get("cursor")
        if not cursor:
            return markets


def collect(session=None, now: datetime | None = None) -> list[dict]:
    """Open markets (the pre-game lines) plus the last few days of settled ones (their outcomes)."""
    session = session or requests.Session()
    now = now or datetime.now(timezone.utc)
    ts = now.isoformat(timespec="seconds")
    since = int((now - timedelta(days=SETTLED_LOOKBACK_DAYS)).timestamp())
    rows: list[dict] = []
    for series in SERIES:
        for status, min_close in (("open", None), ("settled", since)):
            for m in fetch_series(session, series, status, min_close):
                rows.append(normalize(m, series, ts))
    return rows


def write_snapshot(rows: list[dict], out_dir: Path = LINES_DIR, now: datetime | None = None) -> Path | None:
    """Append to today's gzip JSONL file (one file per UTC day, so the repo grows by one small file a day)."""
    if not rows:
        return None
    now = now or datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"kalshi_cfb_lines_{now:%Y-%m-%d}.jsonl.gz"
    with gzip.open(path, "at", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def main() -> int:
    rows = collect()
    path = write_snapshot(rows)
    open_rows = sum(1 for r in rows if r["status"] == "active")
    print(f"kalshi lines: {len(rows)} rows ({open_rows} open) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
