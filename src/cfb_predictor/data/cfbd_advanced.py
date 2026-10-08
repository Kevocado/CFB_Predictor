"""CFBD game-level advanced stats as a committed, resumable, budgeted batch."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class BudgetExceeded(RuntimeError):
    pass


def _side(block: dict) -> dict:
    return {
        "epa": block.get("ppa"),
        "epa_pass": (block.get("passingPlays") or {}).get("ppa"),
        "epa_rush": (block.get("rushingPlays") or {}).get("ppa"),
        "success": block.get("successRate"),
    }


def to_team_game_frame(rows: list[dict]) -> pd.DataFrame:
    by_game: dict = {}
    for r in rows:
        by_game.setdefault(r["gameId"], []).append(r)
    out = []
    for gid, pair in by_game.items():
        if len(pair) != 2:
            continue  # a game with one side missing cannot give defence columns; skip, do not guess
        for me, opp in (pair, pair[::-1]):
            o, d_ = _side(me["offense"]), _side(opp["offense"])  # my defence = what the opponent's offence did to me
            out.append({
                "game_id": str(gid), "team": me["team"],
                "epa_off": o["epa"], "epa_off_pass": o["epa_pass"], "epa_off_rush": o["epa_rush"], "success_off": o["success"],
                "epa_def": d_["epa"], "epa_def_pass": d_["epa_pass"], "epa_def_rush": d_["epa_rush"], "success_def": d_["success"],
            })
    return pd.DataFrame(out)


def fetch_season_advanced(client, year: int) -> list[dict]:
    return client.get("/stats/game/advanced", {"year": year})


def pull_all(client, years, out_dir, budget: int = 40) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used = 0
    for year in years:
        target = out_dir / f"advanced_{year}.json"
        if target.exists():
            continue
        if used + 1 > budget:
            raise BudgetExceeded(f"{used} calls used; budget {budget}")
        target.write_text(json.dumps(fetch_season_advanced(client, year)))
        used += 1
    return used