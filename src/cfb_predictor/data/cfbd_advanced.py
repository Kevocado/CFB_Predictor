"""CFBD game-level advanced stats as a committed, resumable, budgeted batch."""
from __future__ import annotations

import json
import os
import tempfile
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
            # `season` and `week` are carried through from the CFBD row so a caller can
            # subset by season without re-deriving it from the game id (CFBD ids like
            # 400547640 do NOT begin with the calendar year -- the 2014 file's first id
            # is 400547640 and the 2025 file's is 401752665, so `game_id[:4]` is wrong).
            out.append({
                "game_id": str(gid), "team": me["team"],
                "season": me.get("season"), "week": me.get("week"),
                "epa_off": o["epa"], "epa_off_pass": o["epa_pass"], "epa_off_rush": o["epa_rush"], "success_off": o["success"],
                "epa_def": d_["epa"], "epa_def_pass": d_["epa_pass"], "epa_def_rush": d_["epa_rush"], "success_def": d_["success"],
            })
    return pd.DataFrame(out, columns=[
        "game_id", "team", "season", "week",
        "epa_off", "epa_def", "epa_off_pass", "epa_off_rush",
        "epa_def_pass", "epa_def_rush", "success_off", "success_def",
    ])


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
        data = json.dumps(fetch_season_advanced(client, year))
        # Atomic write: write to temp file in same directory, then replace, so an interrupted
        # write never leaves an advanced_{year}.json the next pull treats as done. The unlink is
        # a no-op after a successful replace (the temp name is gone) and cleans up the temp file
        # when the publication itself fails.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=out_dir, prefix=f"advanced_{year}.", suffix=".tmp", delete=False
            ) as tmp:
                tmp_path = tmp.name  # before the write: a failed write still needs cleaning up
                tmp.write(data)
            os.replace(tmp_path, target)
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)
        used += 1
    return used

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "cfbd"
DEFAULT_YEARS = list(range(2014, 2026))


def _client(key: str):
    """A CFBD ApiClient configured with the key. The key is read from the environment
    only; it is never logged, echoed or written to a file."""
    import cfbd

    config = cfbd.Configuration(access_token=key)
    return cfbd.ApiClient(config)


def _load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def efficiency_frame_from_disk(out_dir: Path = None, years=None) -> pd.DataFrame:
    """to_team_game_frame() over the already-pulled JSON files, for callers that must
    not spend API calls (the block_eval run reads these, never the network)."""
    out_dir = Path(out_dir or DEFAULT_OUT_DIR)
    years = list(years or DEFAULT_YEARS)
    frames = []
    for year in years:
        path = out_dir / f"advanced_{year}.json"
        if not path.exists():
            continue
        frames.append(to_team_game_frame(_load_rows(path)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def current_season_efficiency(season: int, cache_dir: Path, max_age_days: float = 6.0, fetch=None) -> pd.DataFrame:
    """Team-game efficiency for an in-progress season, refreshed AT MOST once per `max_age_days` (1 CFBD call).

    Only the snapshot job calls this (the request path reads the snapshot). A fresh cache file is reused with no
    call; a stale or missing one is refetched via `fetch(season) -> rows` (default: CFBD with CFBD_API_KEY); a
    failed refetch falls back to the stale cache, then to the committed data/cfbd file, then to an empty frame."""
    import time

    cache = Path(cache_dir) / f"advanced_{season}.json"
    rows = None
    if cache.exists() and time.time() - cache.stat().st_mtime < max_age_days * 86400:
        rows = _load_rows(cache)
    else:
        try:
            if fetch is None:
                key = os.environ.get("CFBD_API_KEY")
                if not key:
                    raise RuntimeError("CFBD_API_KEY is not set")
                with _client(key) as client:
                    rows = fetch_season_advanced(client, season)
            else:
                rows = fetch(season)
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(rows))
            os.replace(tmp, cache)
        except Exception as exc:  # noqa: BLE001 - fail open: stale data beats none, none beats a crashed snapshot
            print(f"    ! CFBD advanced pull for {season} failed ({exc}); using what is on disk")
            for path in (cache, DEFAULT_OUT_DIR / f"advanced_{season}.json"):
                if path.exists():
                    rows = _load_rows(path)
                    break
    return to_team_game_frame(rows) if rows else pd.DataFrame()


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Pull CFBD game-level advanced stats for a set of years into committed JSON."
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--budget", type=int, default=40)
    args = parser.parse_args()

    key = os.environ.get("CFBD_API_KEY")
    if not key:
        # Never print the key or any part of it; the absence is the only thing reportable.
        print("CFBD_API_KEY is not set; nothing to do", file=sys.stderr)
        return 2

    already = sum(1 for y in args.years if (Path(args.out_dir) / f"advanced_{y}.json").exists())
    print(f"requested years: {sorted(args.years)}")
    print(f"already on disk: {already} of {len(args.years)}")
    with _client(key) as client:
        used = pull_all(client, args.years, args.out_dir, budget=args.budget)
    calls_this_run = used
    print(f"API calls made this run: {calls_this_run}")
    print(f"API calls saved by existing files: {already}")
    print(f"total API calls for the whole pull (this run + saved): {calls_this_run + already}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
