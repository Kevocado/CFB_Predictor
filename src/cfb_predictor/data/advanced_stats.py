"""CFB Data Hub: team advanced stats (CFBD PPA, the college equivalent of
EPA) and player season tables. CFBD allows 1,000 calls a month, so each
season's advanced stats and player PPA are fetched at most once a day and
cached on disk. Without a key, or when CFBD fails, the hub renders without
advanced numbers rather than erroring."""
from __future__ import annotations

import json
import os
import time

import cfbd
import pandas as pd

from ..config import CACHE_DIR, CFBD_API_KEY

MAX_AGE_SECONDS = 24 * 3600
POSITIONS = ["QB", "RB", "WR", "TE"]
SUM_COLS = ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds", "receiving_yards",
            "receiving_tds", "receptions", "carries"]


def _client() -> cfbd.ApiClient:
    client = cfbd.ApiClient(cfbd.Configuration())
    key = (CFBD_API_KEY or os.getenv("CFBD_API_KEY", "")).strip()
    if key:
        client.default_headers["Authorization"] = key if key.startswith("Bearer ") else f"Bearer {key}"
    return client


def _call_advanced(season: int) -> list[dict]:
    with _client() as api_client:
        rows = cfbd.StatsApi(api_client).get_advanced_season_stats(year=season, exclude_garbage_time=True)
    out = []
    for r in rows:
        off, dfn = r.offense, r.defense
        out.append({"team": r.team, "off_ppa": off.ppa, "def_ppa": dfn.ppa,
                    "off_success_rate": off.success_rate, "def_success_rate": dfn.success_rate,
                    "off_explosiveness": off.explosiveness, "def_explosiveness": dfn.explosiveness})
    return out


def _call_player_ppa(season: int) -> list[dict]:
    with _client() as api_client:
        rows = cfbd.MetricsApi(api_client).get_predicted_points_added_by_player_season(year=season)
    return [{"player_id": str(r.id), "name": r.name, "team": r.team, "position": r.position,
             "ppa_total": getattr(r.total_ppa, "all", None)} for r in rows]


def _cached(kind: str, season: int, call) -> list[dict]:
    if not (CFBD_API_KEY or "").strip():
        return []
    path = CACHE_DIR / "cfbd_advanced" / f"{kind}_{season}.json"
    if path.exists():
        body = json.loads(path.read_text())
        if time.time() - body.get("fetched_at", 0) < MAX_AGE_SECONDS:
            return body["rows"]
    try:
        rows = call(season)
    except Exception:
        # Quota, network or schema trouble: serve the last copy if any.
        return json.loads(path.read_text())["rows"] if path.exists() else []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched_at": time.time(), "rows": rows}))
    return rows


def fetch_advanced(season: int) -> list[dict]:
    return _cached("teams", season, _call_advanced)


def fetch_player_ppa(season: int) -> list[dict]:
    return _cached("players", season, _call_player_ppa)


def _r(x, nd: int = 3):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def _results(team: str, played: pd.DataFrame) -> list[dict]:
    out = []
    for _, g in played.sort_values("gameday").iterrows():
        home = g["home_team"] == team
        ts, os_ = (g["home_score"], g["away_score"]) if home else (g["away_score"], g["home_score"])
        out.append({"gameday": str(g["gameday"])[:10], "opponent": g["away_team"] if home else g["home_team"],
                    "is_home": bool(home), "team_score": int(ts), "opponent_score": int(os_),
                    "result": "W" if ts > os_ else "L" if ts < os_ else "T"})
    return out


def _streak(results: list[dict]) -> int:
    if not results:
        return 0
    last = results[-1]["result"]
    n = 0
    for r in reversed(results):
        if r["result"] != last:
            break
        n += 1
    return n if last == "W" else -n if last == "L" else 0


def _trend(results: list[dict]) -> str:
    if len(results) < 3:
        return "new"
    nets = [r["team_score"] - r["opponent_score"] for r in results]
    diff = sum(nets[-3:]) / 3 - sum(nets) / len(nets)
    return "up" if diff > 3 else "down" if diff < -3 else "steady"


def team_hub(advanced: list[dict], games: pd.DataFrame, season: int) -> list[dict]:
    by_team = {a["team"]: a for a in advanced}
    season_games = games[games["season"] == season]
    teams = sorted(set(season_games["home_team"]) | set(season_games["away_team"]))
    played = season_games[season_games["home_score"].notna() & season_games["away_score"].notna()]
    rows = []
    for team in teams:
        results = _results(team, played[(played["home_team"] == team) | (played["away_team"] == team)])
        n = len(results)
        a = by_team.get(team, {})
        rows.append({
            "team": team, "games": n,
            "wins": sum(r["result"] == "W" for r in results),
            "losses": sum(r["result"] == "L" for r in results),
            "ties": sum(r["result"] == "T" for r in results),
            "points_for_pg": _r(sum(r["team_score"] for r in results) / n, 1) if n else None,
            "points_against_pg": _r(sum(r["opponent_score"] for r in results) / n, 1) if n else None,
            "off_epa_play": _r(a.get("off_ppa")),
            "def_epa_play": _r(a.get("def_ppa")),
            "off_success_rate": _r(a.get("off_success_rate")),
            "def_success_rate": _r(a.get("def_success_rate")),
            "yards_per_play": None, "pass_rate": None, "turnover_margin": None,
            "streak": _streak(results),
            "form": [r["result"] for r in results[-5:]],
            "form_trend": _trend(results),
            "recent_games": list(reversed(results[-5:])),
        })
    return rows


def player_hub(weekly: pd.DataFrame, ppa: list[dict], season: int) -> dict:
    ppa_by = {(p["name"], p["team"]): p["ppa_total"] for p in ppa}
    df = weekly[(weekly["season"] == season) & weekly["position"].isin(POSITIONS)].sort_values("week")
    players = []
    for pid, g in df.groupby("player_id"):
        last = g.iloc[-1]
        row = {"player_id": str(pid), "name": last["player_name"], "team": last["recent_team"],
               "position": last["position"], "games": int(g["week"].nunique())}
        row.update({c: int(pd.to_numeric(g[c], errors="coerce").fillna(0).sum()) for c in SUM_COLS})
        row.update({"completions": 0, "attempts": 0, "interceptions": 0, "targets": 0})
        epa = ppa_by.get((row["name"], row["team"]))
        row["epa_total"] = _r(epa, 2)
        row["target_share"] = None
        row["air_yards_share"] = None
        row["fantasy_ppr_pg"] = None
        players.append(row)
    boards = {}
    for pos in POSITIONS:
        ranked = [p for p in players if p["position"] == pos and p["epa_total"] is not None]
        boards[pos] = sorted(ranked, key=lambda p: p["epa_total"], reverse=True)[:5]
    return {"players": players, "leaderboards": boards}
