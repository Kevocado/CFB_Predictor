# CFB Predictor v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working FBS college football prediction dashboard — pre-kickoff moneyline/spread/total probabilities for every game and anytime-TD/passing/rushing/receiving-yardage player props — served through a FastAPI backend and React frontend, with every prediction snapshotted before kickoff and honestly reconciled after.

**Architecture:** Fourth sibling project to `PL_Predictor`, `F1_Predictor`, and `NFL_Predictor`, reusing `NFL_Predictor`'s architecture near-verbatim: `src/cfb_predictor/{data,features,models,evaluate,tracking,odds,api}`, React+TS+Vite frontend, SQLite snapshot-then-reconcile tracking. Game-outcome models are a candidate race (Elo baseline, ridge margin regression, XGBoost margin regression) walk-forward validated on held-out log-loss, exactly like `NFL_Predictor`'s own race. Player props are a separate anytime-TD classifier plus passing/rushing/receiving yardage regressors. The only architecturally new piece is the data layer: CollegeFootballData.com (CFBD) replaces `nfl_data_py`, and it is free but API-keyed and rate-capped at 1,000 calls/month, so every data module here is cache-first and fetches per-season, never per-team or per-game.

**Tech Stack:** Python 3.10+, `cfbd` (CollegeFootballData.com official client), FastAPI, uvicorn, pandas, numpy, scipy, scikit-learn, xgboost, requests, python-dotenv, SQLite; React 19 + TypeScript + Vite for the frontend.

**Spec:** `/Users/sigey/Documents/Projects/CFB_Predictor/docs/superpowers/specs/2026-09-04-cfb-predictor-design.md`

## Global Constraints

- One new free API key required: `CFBD_API_KEY` (Bearer token, from collegefootballdata.com/key) — no paid keys anywhere in this plan.
- CFBD free tier is rate-capped at 1,000 calls/month — every data-module task must fetch per-season (or per-season+week where the endpoint requires it), never per-team/per-game, and cache to `data/cache/` after first fetch. Tests must NEVER hit the real CFBD API (monkeypatch the `cfbd` client calls).
- The Odds API reuses the existing key already configured for `PL_Predictor`/`NFL_Predictor`, sport key `americanfootball_ncaaf`.
- Every feature-construction consumer goes through one `features/build.py` entry point.
- Every game/player prediction is snapshotted to SQLite before kickoff and never overwritten.
- Value-bet detection compares model probability to Shin-de-vigged market probability, at most one recommended market per game, never a parlay.
- Candidate models are selected by walk-forward held-out log-loss.
- FCS-opponent games are excluded from `build_training_frame`'s training rows but still counted toward the FBS team's rest-days/rolling-form bookkeeping.
- `pip install -e ".[dev]"` may silently skip editable installs on this machine (confirmed true for NFL_Predictor's environment) — use `export PYTHONPATH=$(pwd)/src` in every shell running python/uvicorn/pytest for this project, same as NFL_Predictor.

## Implementation notes carried from research (read before starting)

- **`cfbd` package shape, confirmed live against the package's own generated docs** (github.com/CFBD/cfbd-python): `cfbd.GamesApi.get_games(year=...)` (one call per season, `week` optional), `cfbd.GamesApi.get_player_game_stats(year=...)` (one call per season, returns every played game's box score for that whole season), `cfbd.TeamsApi.get_fbs_teams(year=...)`. The `Game` model already carries `conference_game: bool`, `home_conference`/`away_conference`, and `home_division`/`away_division` (values like `"fbs"`/`"fcs"`) directly — **this means `conference_game` does not need to be derived from a team→conference join; CFBD already computes it.** Task 8 uses it directly, with `fetch_fbs_teams` kept as the authoritative FCS-exclusion source and the per-game division fields as a fallback, matching the spec's "cross-check" instruction.
- **`cfbd.GamesApi.get_player_game_stats` returns a nested box-score tree**, not a flat per-player-per-week row like `nfl_data_py`: each entry is `{id: <game_id>, teams: [{school, categories: [{name: "passing"|"rushing"|"receiving"|..., types: [{name: "YDS"|"TD"|"CAR"|"REC"|..., athletes: [{id, name, stat}]}]}]}]}`. Task 3 flattens this. Two real data-source gaps follow from this shape and are handled explicitly rather than silently: (1) CFBD's box score does not track **targets**, so that column is always `NaN` for CFB (harmless — `features/player_usage.py`'s rolling mean and the model-fit `.fillna(0)` calls both already tolerate an all-NaN column); (2) CFBD's box score carries **no roster position**, so position is inferred from which stat category dominates a player's game (passing → QB, rushing → RB, receiving → WR), collapsing WR/TE into one bucket — harmless downstream because `models/player_props.py`'s `POSITION_YARDAGE_MARKET` already maps both WR and TE to `receiving_yards`.
- **CFBD's `Game` model has no `spread_line`/`total_line` fields** (unlike `nfl_data_py`'s schedule frame, which includes Vegas lines as a convenience column). Task 15 pulls spread/total lines from The Odds API's own `spreads`/`totals` markets instead, matched by team name — the same team-name-literal matching `odds/value_bets.py` already uses for `h2h`, carried over unchanged rather than fixed here (out of scope for this plan, see that task's note).
- **Spot-check before Task 1 is done:** the exact `cfbd.Configuration`/`cfbd.ApiClient` construction shown in this plan's code (`cfbd.Configuration(access_token=...)` used as a context manager) matches the modern generated-client convention this plan assumes, but the package's own README (last checked during this plan's research) still documents an older `configuration.api_key['Authorization'] = ...` style. Run `pip show cfbd` and skim the installed version's own `README.md`/`docs/GamesApi.md` during Task 1 Step 1 and adjust `_cfbd_configuration()` in `data/games.py`/`data/player_stats.py` if the installed version differs — isolated to one small helper function per file specifically so this is a one-line fix if needed.
- **No `injuries.py` / ESPN scoreboard module** — the spec's own project layout only lists `cfbd_client.py` and `odds_api.py` under `data/`; unlike NFL_Predictor, CFB v1 has no injuries data source in scope.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/cfb_predictor/__init__.py`
- Create: `src/cfb_predictor/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.PROJECT_ROOT`, `config.DATA_DIR`, `config.CACHE_DIR`, `config.MODELS_DIR`, `config.FRONTEND_DIST_DIR`, `config.TRACKING_DB_PATH`, `config.GAMES_CACHE_DIR`, `config.TEAMS_CACHE_DIR`, `config.PLAYER_STATS_CACHE_DIR`, `config.ODDS_CACHE_DIR`, `config.CURRENT_SEASON` (int), `config.CFBD_API_KEY`, `config.ODDS_API_KEY`, `config.ODDS_API_SPORT_KEY`, `config.ODDS_API_BASE_URL`, `config.PUBLIC_MODE` (bool).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "cfb_predictor"
version = "0.1.0"
description = "College football (FBS) game outcome and player prop predictor"
requires-python = ">=3.10"
dependencies = [
    "cfbd>=5.0.0",
    "pandas",
    "numpy",
    "scipy",
    "xgboost",
    "scikit-learn",
    "requests",
    "python-dotenv",
    "fastapi",
    "uvicorn[standard]",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-mock", "httpx"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Write `.env.example` and `.gitignore`**

```bash
# .env.example
CFBD_API_KEY=
ODDS_API_KEY=
PUBLIC_MODE=false
```

```gitignore
.venv/
__pycache__/
*.egg-info/
data/cache/
data/tracking.db
data/tracking.db-shm
data/tracking.db-wal
models/*.json
models/*.pkl
.env
frontend/node_modules/
frontend/dist/
.pytest_cache/
```

- [ ] **Step 3: Write the failing test for `config.py`**

```python
# tests/test_config.py
from pathlib import Path

from cfb_predictor import config


def test_paths_are_absolute_and_created():
    assert config.PROJECT_ROOT.is_absolute()
    assert config.CACHE_DIR.is_dir()
    assert config.MODELS_DIR.is_dir()
    assert config.GAMES_CACHE_DIR.is_dir()
    assert config.TEAMS_CACHE_DIR.is_dir()
    assert config.PLAYER_STATS_CACHE_DIR.is_dir()
    assert config.ODDS_CACHE_DIR.is_dir()


def test_current_season_is_reasonable():
    assert 2020 <= config.CURRENT_SEASON <= 2100


def test_odds_api_sport_key():
    assert config.ODDS_API_SPORT_KEY == "americanfootball_ncaaf"


def test_cfbd_api_key_attribute_exists():
    # No real key is required for tests to import config -- CFBD_API_KEY may
    # be None in CI/dev, but the attribute itself must always be defined so
    # data/games.py and data/player_stats.py can import it unconditionally.
    assert hasattr(config, "CFBD_API_KEY")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor'`

- [ ] **Step 5: Write `src/cfb_predictor/config.py`**

```python
"""config.py — paths, env loading, and shared constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"

PUBLIC_MODE = os.getenv("PUBLIC_MODE", "false").lower() == "true"

GAMES_CACHE_DIR = CACHE_DIR / "games"
TEAMS_CACHE_DIR = CACHE_DIR / "teams"
PLAYER_STATS_CACHE_DIR = CACHE_DIR / "player_stats"
ODDS_CACHE_DIR = CACHE_DIR / "odds"

CURRENT_SEASON = 2026  # bump each new CFB season (typically August)

# CollegeFootballData.com — this project's nfl_data_py equivalent, free but
# API-keyed and rate-capped at 1,000 calls/month on the free tier. Request a
# key once at collegefootballdata.com/key.
CFBD_API_KEY = os.getenv("CFBD_API_KEY")

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_SPORT_KEY = "americanfootball_ncaaf"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports"

TRACKING_DB_PATH = DATA_DIR / "tracking.db"

for _d in (
    DATA_DIR,
    CACHE_DIR,
    MODELS_DIR,
    GAMES_CACHE_DIR,
    TEAMS_CACHE_DIR,
    PLAYER_STATS_CACHE_DIR,
    ODDS_CACHE_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 6: Create the package `__init__.py`**

```python
# src/cfb_predictor/__init__.py
```

- [ ] **Step 7: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .env.example .gitignore src/cfb_predictor/__init__.py src/cfb_predictor/config.py tests/test_config.py
git commit -m "feat: project scaffolding and config"
```

---

## Task 2: Data — games/schedules and FBS team universe

**Files:**
- Create: `src/cfb_predictor/data/__init__.py`
- Create: `src/cfb_predictor/data/games.py`
- Test: `tests/test_games.py`

**Interfaces:**
- Consumes: `config.CFBD_API_KEY`, `config.GAMES_CACHE_DIR`, `config.TEAMS_CACHE_DIR`, `config.CURRENT_SEASON`.
- Produces: `games.fetch_schedules(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame` (columns: `game_id, season, week, gameday, home_team, away_team, home_score, away_score, home_conference, away_conference, home_division, away_division, conference_game, neutral_site`), `games.fetch_fbs_teams(season: int, force_refresh: bool = False) -> pd.DataFrame` (columns: `team, conference, division, classification`), `games.default_completed_seasons(n: int = 8) -> list[int]`, `games.load_training_data(seasons: list[int]) -> pd.DataFrame`, `games.fetch_current_season_partial() -> pd.DataFrame`, `games.fetch_upcoming_games(season: int, week: int) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_games.py
import pandas as pd
import pytest

from cfb_predictor.data import games


def _raw_games_frame():
    return pd.DataFrame(
        [
            {
                "id": 401520145, "season": 2025, "week": 1, "start_date": "2025-08-30T16:00:00.000Z",
                "home_team": "Texas", "away_team": "Ohio State",
                "home_points": 7, "away_points": 14,
                "home_conference": "SEC", "away_conference": "Big Ten",
                "home_division": "fbs", "away_division": "fbs",
                "conference_game": False, "neutral_site": True,
            },
            {
                "id": 401520200, "season": 2025, "week": 1, "start_date": "2025-08-30T19:30:00.000Z",
                "home_team": "Alabama", "away_team": "Western Carolina",
                "home_points": None, "away_points": None,
                "home_conference": "SEC", "away_conference": None,
                "home_division": "fbs", "away_division": "fcs",
                "conference_game": False, "neutral_site": False,
            },
        ]
    )


def _raw_teams_frame():
    return pd.DataFrame(
        [
            {"school": "Texas", "conference": "SEC", "division": "fbs", "classification": "fbs"},
            {"school": "Ohio State", "conference": "Big Ten", "division": "fbs", "classification": "fbs"},
            {"school": "Alabama", "conference": "SEC", "division": "fbs", "classification": "fbs"},
        ]
    )


def test_fetch_schedules_caches_per_season(monkeypatch, tmp_path):
    monkeypatch.setattr(games, "GAMES_CACHE_DIR", tmp_path)
    calls = []

    def fake_import(season):
        calls.append(season)
        return _raw_games_frame()

    monkeypatch.setattr(games, "_import_games", fake_import)

    first = games.fetch_schedules([2025])
    second = games.fetch_schedules([2025])

    assert calls == [2025]  # second call hit the cache, not the network
    assert len(first) == 2
    assert "conference_game" in first.columns
    assert second.equals(first)


def test_fetch_schedules_calls_once_per_missing_season(monkeypatch, tmp_path):
    monkeypatch.setattr(games, "GAMES_CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(games, "_import_games", lambda season: calls.append(season) or _raw_games_frame())

    games.fetch_schedules([2024, 2025])

    assert calls == [2024, 2025]  # exactly one call per season, never per game/team


def test_load_training_data_drops_unplayed_games(monkeypatch, tmp_path):
    monkeypatch.setattr(games, "GAMES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(games, "_import_games", lambda season: _raw_games_frame())

    df = games.load_training_data([2025])

    assert len(df) == 1
    assert df.iloc[0]["game_id"] == "401520145"


def test_default_completed_seasons_excludes_current_season(monkeypatch):
    monkeypatch.setattr(games, "CURRENT_SEASON", 2026)
    seasons = games.default_completed_seasons(n=3)
    assert seasons == [2023, 2024, 2025]


def test_fetch_upcoming_games_filters_season_week(monkeypatch, tmp_path):
    monkeypatch.setattr(games, "GAMES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(games, "_import_games", lambda season: _raw_games_frame())

    upcoming = games.fetch_upcoming_games(2025, 1)

    assert list(upcoming["game_id"]) == ["401520200"]


def test_fetch_fbs_teams_caches_and_returns_conference_and_division(monkeypatch, tmp_path):
    monkeypatch.setattr(games, "TEAMS_CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(games, "_import_fbs_teams", lambda season: calls.append(season) or _raw_teams_frame())

    first = games.fetch_fbs_teams(2025)
    second = games.fetch_fbs_teams(2025)

    assert calls == [2025]
    assert set(first["team"]) == {"Texas", "Ohio State", "Alabama"}
    assert set(first["conference"]) == {"SEC", "Big Ten"}
    assert second.equals(first)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_games.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.data'`

- [ ] **Step 3: Write `src/cfb_predictor/data/__init__.py` and `games.py`**

```python
# src/cfb_predictor/data/__init__.py
```

```python
"""games.py — CFB schedule/results and FBS team universe, cache-or-fetch
from CollegeFootballData (CFBD) via the cfbd Python package.

One parquet file per season under GAMES_CACHE_DIR/TEAMS_CACHE_DIR, mirroring
nfl_predictor's data/schedules.py per-season cache-or-fetch pattern — a
season's results never change once played, and a season's FBS membership
never changes once the season starts, so per-season caching is safe. This
caching is *load-bearing* here, not just a latency optimization: CFBD's free
tier caps API access at 1,000 calls/month, so every fetch below is exactly
one call per season (never per-team or per-game).

CFBD's Game model already exposes `conference_game` as a real boolean and
`home_division`/`away_division` directly — unlike NFL's div_game (which
nfl_data_py also supplies pre-computed), no team-conference join is needed
to populate `conference_game` here either. `fetch_fbs_teams` is kept as the
authoritative FBS team-universe source (features/build.py uses it to exclude
FCS-opponent games from training rows), with the per-game division fields
available as a fallback/cross-check when a team-universe lookup isn't
supplied.
"""

from __future__ import annotations

import pandas as pd

from ..config import CFBD_API_KEY, CURRENT_SEASON, GAMES_CACHE_DIR, TEAMS_CACHE_DIR

KEEP_COLUMNS = [
    "game_id", "season", "week", "gameday", "home_team", "away_team",
    "home_score", "away_score", "home_conference", "away_conference",
    "home_division", "away_division", "conference_game", "neutral_site",
]

TEAM_KEEP_COLUMNS = ["team", "conference", "division", "classification"]


def _cfbd_configuration():
    """Isolated in its own function so a mismatch between this plan's
    assumed cfbd package shape and the installed version (see this plan's
    "Implementation notes" section) is a one-line fix, not a scattered one."""
    import cfbd

    return cfbd.Configuration(access_token=CFBD_API_KEY)


def _import_games(season: int) -> pd.DataFrame:
    """One call per season — cfbd.GamesApi.get_games(year=season) returns
    every week of that season's games in a single response, so this never
    needs a per-week loop. Thin wrapper so tests can monkeypatch just this
    one function rather than the whole cfbd client."""
    import cfbd

    with cfbd.ApiClient(_cfbd_configuration()) as api_client:
        games_api = cfbd.GamesApi(api_client)
        fetched = games_api.get_games(year=season)
    return pd.DataFrame([g.to_dict() for g in fetched])


def _import_fbs_teams(season: int) -> pd.DataFrame:
    """One call per season — cfbd.TeamsApi.get_fbs_teams(year=season) is
    this project's team-universe source. FBS teams move conferences (and
    occasionally divisions) year to year, so hardcoding a list would go
    stale; this is the "one additional data-module function" the design
    spec calls for rather than a separate task."""
    import cfbd

    with cfbd.ApiClient(_cfbd_configuration()) as api_client:
        teams_api = cfbd.TeamsApi(api_client)
        fetched = teams_api.get_fbs_teams(year=season)
    return pd.DataFrame([t.to_dict() for t in fetched])


def _games_cache_path(season: int):
    return GAMES_CACHE_DIR / f"{season}.parquet"


def _teams_cache_path(season: int):
    return TEAMS_CACHE_DIR / f"{season}.parquet"


def _normalize_games(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(
        columns={
            "id": "game_id",
            "start_date": "gameday",
            "home_points": "home_score",
            "away_points": "away_score",
        }
    )
    for col in KEEP_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[KEEP_COLUMNS].copy()
    df["gameday"] = pd.to_datetime(df["gameday"])
    df["game_id"] = df["game_id"].astype(str)
    return df


def fetch_schedules(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """One row per game across every requested season. Seasons already
    cached on disk are read from cache; anything missing (or force_refresh)
    is fetched from CFBD one season at a time — CFBD's get_games only takes
    a single `year`, so a per-season loop is inherent to the endpoint, not a
    violation of the one-call-per-season budget."""
    frames = []
    for season in seasons:
        path = _games_cache_path(season)
        if not force_refresh and path.exists():
            frames.append(pd.read_parquet(path))
            continue
        fetched = _normalize_games(_import_games(season))
        fetched.to_parquet(path)
        frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def fetch_fbs_teams(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """This season's FBS team universe (team, conference, division,
    classification) — used to cross-check conference_game and to exclude
    FCS-opponent games from training. Cached per season since conference
    realignment only happens between seasons, not mid-season."""
    path = _teams_cache_path(season)
    if not force_refresh and path.exists():
        return pd.read_parquet(path)

    raw = _import_fbs_teams(season)
    df = raw.rename(columns={"school": "team"})
    for col in TEAM_KEEP_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[TEAM_KEEP_COLUMNS].copy()
    df.to_parquet(path)
    return df


def default_completed_seasons(n: int = 8) -> list[int]:
    return list(range(CURRENT_SEASON - n, CURRENT_SEASON))


def load_training_data(seasons: list[int]) -> pd.DataFrame:
    """Only games with a final score — excludes future/postponed games from
    the same fetch_schedules() call."""
    df = fetch_schedules(seasons)
    return df[df["home_score"].notna() & df["away_score"].notna()].reset_index(drop=True)


def fetch_current_season_partial() -> pd.DataFrame:
    """Completed games so far in CURRENT_SEASON, refetched every call (no
    per-season cache for the still-in-progress season, since its cache file
    would go stale after every week's games)."""
    df = fetch_schedules([CURRENT_SEASON], force_refresh=True)
    return df[df["home_score"].notna() & df["away_score"].notna()].reset_index(drop=True)


def fetch_upcoming_games(season: int, week: int) -> pd.DataFrame:
    """Games in a given season/week that haven't been played yet."""
    df = fetch_schedules([season], force_refresh=(season == CURRENT_SEASON))
    week_df = df[df["week"] == week]
    return week_df[week_df["home_score"].isna()].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_games.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cfb_predictor/data/__init__.py src/cfb_predictor/data/games.py tests/test_games.py
git commit -m "feat: games/schedules data module with FBS team universe"
```

---

## Task 3: Data — weekly player stats (CFBD box-score flatten)

**Files:**
- Create: `src/cfb_predictor/data/player_stats.py`
- Test: `tests/test_player_stats.py`

**Interfaces:**
- Consumes: `config.CFBD_API_KEY`, `config.PLAYER_STATS_CACHE_DIR`.
- Produces: `player_stats.fetch_weekly_player_stats(seasons: list[int], games_df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame` with columns `player_id, player_name, position, recent_team, season, week, passing_yards, passing_tds, rushing_yards, rushing_tds, receiving_yards, receiving_tds, receptions, targets, carries` (`targets` is always `NaN` — CFBD's box score does not track it).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_player_stats.py
import pandas as pd
import pytest

from cfb_predictor.data import player_stats


def _games_df():
    return pd.DataFrame(
        [
            {"game_id": "401520145", "season": 2025, "week": 1, "gameday": pd.Timestamp("2025-08-30")},
        ]
    )


def _raw_player_game_stats():
    return [
        {
            "id": 401520145,
            "teams": [
                {
                    "school": "Ohio State",
                    "categories": [
                        {
                            "name": "passing",
                            "types": [
                                {"name": "YDS", "athletes": [{"id": 4567, "name": "J. Doe", "stat": "291"}]},
                                {"name": "TD", "athletes": [{"id": 4567, "name": "J. Doe", "stat": "2"}]},
                            ],
                        },
                        {
                            "name": "rushing",
                            "types": [
                                {"name": "YDS", "athletes": [{"id": 4568, "name": "R. Back", "stat": "112"}]},
                                {"name": "CAR", "athletes": [{"id": 4568, "name": "R. Back", "stat": "18"}]},
                            ],
                        },
                        {
                            "name": "receiving",
                            "types": [
                                {"name": "YDS", "athletes": [{"id": 4569, "name": "W. Out", "stat": "85"}]},
                                {"name": "REC", "athletes": [{"id": 4569, "name": "W. Out", "stat": "6"}]},
                            ],
                        },
                    ],
                }
            ],
        }
    ]


def test_fetch_weekly_player_stats_caches_per_season(monkeypatch, tmp_path):
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    calls = []

    def fake_import(season):
        calls.append(season)
        return _raw_player_game_stats()

    monkeypatch.setattr(player_stats, "_import_player_game_stats", fake_import)

    first = player_stats.fetch_weekly_player_stats([2025], _games_df())
    second = player_stats.fetch_weekly_player_stats([2025], _games_df())

    assert calls == [2025]
    assert len(first) == 3  # one row each for the QB, RB, and WR
    assert second.equals(first)


def test_flattened_stats_attach_week_from_games_df(monkeypatch, tmp_path):
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(player_stats, "_import_player_game_stats", lambda season: _raw_player_game_stats())

    df = player_stats.fetch_weekly_player_stats([2025], _games_df())

    assert (df["week"] == 1).all()
    assert (df["season"] == 2025).all()


def test_flattened_stats_map_category_type_pairs_to_flat_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(player_stats, "_import_player_game_stats", lambda season: _raw_player_game_stats())

    df = player_stats.fetch_weekly_player_stats([2025], _games_df())

    qb = df[df["player_id"] == "4567"].iloc[0]
    assert qb["passing_yards"] == 291.0
    assert qb["passing_tds"] == 2.0
    assert qb["position"] == "QB"

    rb = df[df["player_id"] == "4568"].iloc[0]
    assert rb["rushing_yards"] == 112.0
    assert rb["carries"] == 18.0
    assert rb["position"] == "RB"

    wr = df[df["player_id"] == "4569"].iloc[0]
    assert wr["receiving_yards"] == 85.0
    assert wr["receptions"] == 6.0
    assert wr["position"] == "WR"


def test_targets_column_is_always_present_but_nan(monkeypatch, tmp_path):
    # CFBD's box score does not track targets -- the column must still exist
    # (features/player_usage.py's ROLL_STATS references it) but stays NaN.
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(player_stats, "_import_player_game_stats", lambda season: _raw_player_game_stats())

    df = player_stats.fetch_weekly_player_stats([2025], _games_df())

    assert "targets" in df.columns
    assert df["targets"].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_player_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.data.player_stats'`

- [ ] **Step 3: Write `src/cfb_predictor/data/player_stats.py`**

```python
"""player_stats.py — weekly player-level stats, cache-or-fetch from CFBD.

CFBD's per-game player stats (cfbd.GamesApi.get_player_game_stats) return a
deeply nested box-score tree (game -> team -> stat category -> stat type ->
athlete), not nfl_data_py's flat per-player-per-week row -- this module's job
is fetching that tree (one call per season) and flattening it into the same
flat shape nfl_predictor's data/player_stats.py already produces downstream
(features/player_usage.py, models/player_props.py both expect one row per
player-week with named stat columns). Needs the season's games frame (from
data/games.py) to attach a week/gameday to each stat row, since the raw CFBD
payload only carries a game id per top-level entry, not a week number.
"""

from __future__ import annotations

import pandas as pd

from ..config import CFBD_API_KEY, PLAYER_STATS_CACHE_DIR

KEEP_COLUMNS = [
    "player_id", "player_name", "position", "recent_team", "season", "week",
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "receptions", "targets", "carries",
]

# CFBD's box-score category/stat-type names -> this project's flat column
# names. NOTE: CFBD's official box score does not track targets (only
# receptions/yards/tds), so the "targets" column is always NaN here -- a
# real data-source gap, not a bug: features/player_usage.py's rolling mean
# over an all-NaN column degrades to NaN, and every downstream consumer
# already .fillna(0)s before feeding a model.
_CATEGORY_TYPE_TO_COLUMN = {
    ("passing", "YDS"): "passing_yards",
    ("passing", "TD"): "passing_tds",
    ("rushing", "YDS"): "rushing_yards",
    ("rushing", "TD"): "rushing_tds",
    ("rushing", "CAR"): "carries",
    ("receiving", "YDS"): "receiving_yards",
    ("receiving", "TD"): "receiving_tds",
    ("receiving", "REC"): "receptions",
}

_STAT_COLUMNS = [
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "receptions", "carries",
]


def _cfbd_configuration():
    import cfbd

    return cfbd.Configuration(access_token=CFBD_API_KEY)


def _import_player_game_stats(season: int) -> list[dict]:
    """One call per season -- cfbd.GamesApi.get_player_game_stats(year=season)
    returns every played game's box score for the whole season at once."""
    import cfbd

    with cfbd.ApiClient(_cfbd_configuration()) as api_client:
        games_api = cfbd.GamesApi(api_client)
        player_games = games_api.get_player_game_stats(year=season)
    return [g.to_dict() for g in player_games]


def _infer_position(row: dict) -> str:
    """CFBD's box score carries no roster position -- infer one from which
    stat category dominates this player-game (a real limitation, not a
    placeholder: fetching per-team rosters for a real position would cost
    one CFBD call per team per season, which the 1,000-calls/month budget
    can't absorb). WR and TE are indistinguishable from box-score stats
    alone and both collapse to "WR" -- see models/player_props.py's
    POSITION_YARDAGE_MARKET, which already maps WR and TE to the same
    receiving_yards market, so this collapse costs nothing downstream."""
    totals = {
        "QB": row.get("passing_yards") or 0.0,
        "RB": row.get("rushing_yards") or 0.0,
        "WR": row.get("receiving_yards") or 0.0,
    }
    return max(totals, key=totals.get)


def _flatten_player_game_stats(raw_games: list[dict], games_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Walk CFBD's nested game -> team -> category -> type -> athlete tree
    into one row per (game, player), with one column per stat."""
    game_lookup = games_df.set_index("game_id")[["week"]].to_dict("index")

    rows: dict[tuple, dict] = {}
    for game in raw_games:
        game_id = str(game.get("id"))
        game_meta = game_lookup.get(game_id)
        week = game_meta["week"] if game_meta else None
        for team in game.get("teams", []) or []:
            school = team.get("school")
            for category in team.get("categories", []) or []:
                cat_name = category.get("name")
                for stat_type in category.get("types", []) or []:
                    type_name = stat_type.get("name")
                    column = _CATEGORY_TYPE_TO_COLUMN.get((cat_name, type_name))
                    if column is None:
                        continue
                    for athlete in stat_type.get("athletes", []) or []:
                        player_id = str(athlete.get("id"))
                        key = (game_id, player_id)
                        if key not in rows:
                            rows[key] = {
                                "player_id": player_id,
                                "player_name": athlete.get("name"),
                                "recent_team": school,
                                "season": season,
                                "week": week,
                                **{col: 0.0 for col in _STAT_COLUMNS},
                                "targets": float("nan"),
                            }
                        try:
                            rows[key][column] = float(athlete.get("stat"))
                        except (TypeError, ValueError):
                            continue

    if not rows:
        return pd.DataFrame(columns=KEEP_COLUMNS)

    flat_rows = list(rows.values())
    for row in flat_rows:
        row["position"] = _infer_position(row)

    df = pd.DataFrame(flat_rows)
    return df[KEEP_COLUMNS]


def _season_cache_path(season: int):
    return PLAYER_STATS_CACHE_DIR / f"{season}.parquet"


def fetch_weekly_player_stats(
    seasons: list[int], games_df: pd.DataFrame, force_refresh: bool = False
) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = _season_cache_path(season)
        if not force_refresh and path.exists():
            frames.append(pd.read_parquet(path))
            continue
        raw = _import_player_game_stats(season)
        season_games = games_df[games_df["season"] == season]
        flattened = _flatten_player_game_stats(raw, season_games, season)
        flattened.to_parquet(path)
        frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["season", "week"]).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_player_stats.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cfb_predictor/data/player_stats.py tests/test_player_stats.py
git commit -m "feat: weekly player stats data module (CFBD box-score flatten)"
```

---

## Task 4: Data — Odds API (CFB h2h/spreads/totals)

**Files:**
- Create: `src/cfb_predictor/data/odds_api.py`
- Test: `tests/test_odds_api.py`

**Interfaces:**
- Consumes: `config.ODDS_API_KEY`, `config.ODDS_API_SPORT_KEY`, `config.ODDS_API_BASE_URL`.
- Produces: `odds_api.fetch_game_odds() -> pd.DataFrame` with columns `event_id, commence_time, home_team, away_team, bookmaker, market, outcome_name, price, point, odds_fetched_at`. Empty DataFrame (not an exception) when `ODDS_API_KEY` is unset or the request fails.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_odds_api.py
import pandas as pd
import pytest

from cfb_predictor.data import odds_api


def _raw_odds_response():
    return [
        {
            "id": "abc123",
            "commence_time": "2025-08-30T16:00:00Z",
            "home_team": "Texas Longhorns",
            "away_team": "Ohio State Buckeyes",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Texas Longhorns", "price": 2.10},
                                {"name": "Ohio State Buckeyes", "price": 1.80},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.91, "point": 51.5},
                                {"name": "Under", "price": 1.91, "point": 51.5},
                            ],
                        },
                    ],
                }
            ],
        }
    ]


def test_fetch_game_odds_flattens_bookmaker_markets(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "fake-key")
    monkeypatch.setattr(odds_api, "_fetch_raw_odds", lambda: _raw_odds_response())

    df = odds_api.fetch_game_odds()

    assert len(df) == 4  # 2 h2h outcomes + 2 totals outcomes
    assert set(df["market"]) == {"h2h", "totals"}
    assert df.iloc[0]["event_id"] == "abc123"


def test_fetch_game_odds_returns_empty_frame_without_api_key(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", None)

    df = odds_api.fetch_game_odds()

    assert df.empty


def test_fetch_game_odds_returns_empty_frame_on_request_error(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "fake-key")

    def _raise():
        raise RuntimeError("network error")

    monkeypatch.setattr(odds_api, "_fetch_raw_odds", _raise)

    df = odds_api.fetch_game_odds()

    assert df.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_odds_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.data.odds_api'`

- [ ] **Step 3: Write `src/cfb_predictor/data/odds_api.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_odds_api.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cfb_predictor/data/odds_api.py tests/test_odds_api.py
git commit -m "feat: Odds API data module for CFB h2h/spreads/totals"
```

---

## Task 5: Features — team power ratings (Elo)

**Files:**
- Create: `src/cfb_predictor/features/__init__.py`
- Create: `src/cfb_predictor/features/power_ratings.py`
- Test: `tests/test_power_ratings.py`

**Interfaces:**
- Produces: `power_ratings.compute_pregame_ratings(games_df: pd.DataFrame, k: float = 20.0, home_field: float = 65.0, start_rating: float = 1500.0) -> pd.DataFrame` — returns `games_df` with two new columns `home_pregame_rating`, `away_pregame_rating` (each game's rating *before* that game, chronological, no lookahead). `power_ratings.final_ratings(games_df: pd.DataFrame, **kwargs) -> dict[str, float]` — every team's rating after the last game in `games_df`, for live serving.
- `power_ratings.DEFAULT_START_RATING = 1500.0` is a public constant used by `features/build.py`.

Verbatim port of `nfl_predictor/features/power_ratings.py` — this Elo-style margin-of-victory update has no NFL-specific assumption (it operates purely on `home_score`/`away_score`/`gameday`/team names), so the design spec calls for reusing it unchanged. Confirmed while reading the real source for this task: no reference to `div_game`, positions, or any NFL-only column anywhere in the file.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_power_ratings.py
import pandas as pd

from cfb_predictor.features import power_ratings


def _games():
    return pd.DataFrame(
        [
            {"game_id": "g1", "season": 2025, "week": 1, "gameday": "2025-08-30",
             "home_team": "Texas", "away_team": "Ohio State", "home_score": 7, "away_score": 14},
            {"game_id": "g2", "season": 2025, "week": 2, "gameday": "2025-09-06",
             "home_team": "Ohio State", "away_team": "Texas", "home_score": 17, "away_score": 24},
        ]
    )


def test_first_game_uses_start_rating_for_both_teams():
    result = power_ratings.compute_pregame_ratings(_games(), start_rating=1500.0)

    first = result.iloc[0]
    assert first["home_pregame_rating"] == 1500.0
    assert first["away_pregame_rating"] == 1500.0


def test_rating_moves_after_a_result():
    result = power_ratings.compute_pregame_ratings(_games(), start_rating=1500.0)

    second = result.iloc[1]
    # Ohio State won game 1 as the away team, so its pregame rating for
    # game 2 (now at home) should have risen above 1500.
    assert second["home_pregame_rating"] > 1500.0
    # Texas lost game 1 at home, so its pregame rating for game 2 (now away)
    # should have dropped below 1500.
    assert second["away_pregame_rating"] < 1500.0


def test_final_ratings_reflects_every_game():
    ratings = power_ratings.final_ratings(_games(), start_rating=1500.0)

    assert set(ratings) == {"Texas", "Ohio State"}
    assert ratings["Ohio State"] > 1500.0
    assert ratings["Texas"] < 1500.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_power_ratings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.features'`

- [ ] **Step 3: Write `src/cfb_predictor/features/__init__.py` and `power_ratings.py`**

```python
# src/cfb_predictor/features/__init__.py
```

```python
"""power_ratings.py — Elo-style team power ratings, margin-of-victory
weighted.

Standard football Elo shape (à la 538's NFL model, equally applicable to
CFB): expected score from a logistic function of the rating gap plus a
home-field bonus, update scaled by both the surprise (actual - expected) and
a margin-of-victory multiplier so a 40-point win moves ratings more than a
3-point win. Chronological — every game's *pregame* rating only reflects
games played strictly before it, so this is safe to use as a training
feature with no lookahead. Verbatim port from nfl_predictor: nothing in this
file references anything NFL-specific.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_START_RATING = 1500.0
DEFAULT_K = 20.0
DEFAULT_HOME_FIELD = 65.0
ELO_SCALE = 400.0


def _expected_home_win_prob(home_rating: float, away_rating: float, home_field: float) -> float:
    diff = (home_rating + home_field) - away_rating
    return 1.0 / (1.0 + 10 ** (-diff / ELO_SCALE))


def _margin_multiplier(margin: float, rating_diff: float) -> float:
    """538's Elo margin-of-victory multiplier: log of the margin, damped
    when the favorite already led the ratings by a lot (an autocorrelation
    correction — a huge win over a much weaker team shouldn't move ratings
    as much as the same margin over an evenly matched one).

    `rating_diff` must be winner-relative (winner's pregame rating minus
    loser's), not home-minus-away — otherwise the damping inverts for away
    wins, damping upsets and amplifying expected outcomes backwards."""
    return np.log(max(abs(margin), 1) + 1) * (2.2 / ((rating_diff * 0.001) + 2.2))


def compute_pregame_ratings(
    games_df: pd.DataFrame,
    k: float = DEFAULT_K,
    home_field: float = DEFAULT_HOME_FIELD,
    start_rating: float = DEFAULT_START_RATING,
) -> pd.DataFrame:
    games_df = games_df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    home_pregame = []
    away_pregame = []

    for _, game in games_df.iterrows():
        home, away = game["home_team"], game["away_team"]
        home_rating = ratings.get(home, start_rating)
        away_rating = ratings.get(away, start_rating)
        home_pregame.append(home_rating)
        away_pregame.append(away_rating)

        if pd.isna(game["home_score"]) or pd.isna(game["away_score"]):
            continue  # unplayed game: record pregame rating, no update

        margin = game["home_score"] - game["away_score"]
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        expected = _expected_home_win_prob(home_rating, away_rating, home_field)
        winner_relative_diff = (home_rating - away_rating) if margin >= 0 else (away_rating - home_rating)
        multiplier = _margin_multiplier(margin, winner_relative_diff)
        delta = k * multiplier * (actual - expected)

        ratings[home] = home_rating + delta
        ratings[away] = away_rating - delta

    result = games_df.copy()
    result["home_pregame_rating"] = home_pregame
    result["away_pregame_rating"] = away_pregame
    return result


def final_ratings(
    games_df: pd.DataFrame,
    k: float = DEFAULT_K,
    home_field: float = DEFAULT_HOME_FIELD,
    start_rating: float = DEFAULT_START_RATING,
) -> dict[str, float]:
    """Every team's rating after the last played game in games_df — used to
    seed live predictions for upcoming games."""
    rated = compute_pregame_ratings(games_df, k=k, home_field=home_field, start_rating=start_rating)
    played = rated[rated["home_score"].notna() & rated["away_score"].notna()]
    if played.empty:
        return {}

    ratings: dict[str, float] = {}
    for _, game in played.iterrows():
        home, away = game["home_team"], game["away_team"]
        home_rating = ratings.get(home, game["home_pregame_rating"])
        away_rating = ratings.get(away, game["away_pregame_rating"])
        margin = game["home_score"] - game["away_score"]
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        expected = _expected_home_win_prob(home_rating, away_rating, home_field)
        winner_relative_diff = (home_rating - away_rating) if margin >= 0 else (away_rating - home_rating)
        multiplier = _margin_multiplier(margin, winner_relative_diff)
        delta = k * multiplier * (actual - expected)
        ratings[home] = home_rating + delta
        ratings[away] = away_rating - delta
    return ratings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_power_ratings.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cfb_predictor/features/__init__.py src/cfb_predictor/features/power_ratings.py tests/test_power_ratings.py
git commit -m "feat: Elo-style team power ratings feature"
```

---

## Task 6: Features — rolling form and rest days

**Files:**
- Create: `src/cfb_predictor/features/rolling_form.py`
- Create: `src/cfb_predictor/features/rest_days.py`
- Test: `tests/test_rolling_form.py`
- Test: `tests/test_rest_days.py`

**Interfaces:**
- Produces: `rolling_form.add_rolling_form(games_df: pd.DataFrame, window: int = 5) -> pd.DataFrame` — adds `home_points_scored_roll`, `home_points_allowed_roll`, `away_points_scored_roll`, `away_points_allowed_roll` (mean of each team's own last `window` played games, shifted so the current game is excluded — NaN until a team has at least one prior game). `rest_days.add_rest_days(games_df: pd.DataFrame) -> pd.DataFrame` — adds `home_rest_days`, `away_rest_days` (days since that team's previous game; NaN for a team's first game in the frame).

Both verbatim ports of `nfl_predictor`'s equivalents — the shift(1)/reshape-by-team-appearance logic operates only on `game_id`/`gameday`/`home_team`/`away_team`/`home_score`/`away_score`, with no NFL-specific assumption. Bye weeks are rarer in CFB (not every FBS team has one) but the no-lookahead logic is identical.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rolling_form.py
import pandas as pd

from cfb_predictor.features import rolling_form


def _games():
    return pd.DataFrame(
        [
            {"game_id": "g1", "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State",
             "home_score": 7, "away_score": 14},
            {"game_id": "g2", "gameday": "2025-09-06", "home_team": "Ohio State", "away_team": "Michigan",
             "home_score": 10, "away_score": 14},
            {"game_id": "g3", "gameday": "2025-09-13", "home_team": "Texas", "away_team": "Michigan",
             "home_score": 30, "away_score": 17},
        ]
    )


def test_first_appearance_has_no_rolling_form():
    result = rolling_form.add_rolling_form(_games(), window=5)

    g1 = result.iloc[0]
    assert pd.isna(g1["home_points_scored_roll"])
    assert pd.isna(g1["away_points_scored_roll"])


def test_second_game_reflects_only_the_prior_game():
    result = rolling_form.add_rolling_form(_games(), window=5)

    # Ohio State's second appearance (game g2, as home) should reflect only
    # its away-team performance in g1 (scored 14, allowed 7).
    g2 = result.iloc[1]
    assert g2["home_points_scored_roll"] == 14.0
    assert g2["home_points_allowed_roll"] == 7.0
```

```python
# tests/test_rest_days.py
import pandas as pd

from cfb_predictor.features import rest_days


def _games():
    return pd.DataFrame(
        [
            {"game_id": "g1", "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State"},
            {"game_id": "g2", "gameday": "2025-09-06", "home_team": "Ohio State", "away_team": "Michigan"},
            {"game_id": "g3", "gameday": "2025-09-20", "home_team": "Texas", "away_team": "Michigan"},
        ]
    )


def test_first_appearance_has_no_rest_days():
    result = rest_days.add_rest_days(_games())

    assert pd.isna(result.iloc[0]["home_rest_days"])
    assert pd.isna(result.iloc[0]["away_rest_days"])


def test_rest_days_counts_days_since_last_game():
    result = rest_days.add_rest_days(_games())

    g2 = result.iloc[1]
    assert g2["home_rest_days"] == 7  # Ohio State away in g1 (08-30) -> home in g2 (09-06)

    g3 = result.iloc[2]
    assert g3["home_rest_days"] == 21  # Texas home in g1 (08-30) -> home in g3 (09-20), a bye week
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_rolling_form.py tests/test_rest_days.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `src/cfb_predictor/features/rolling_form.py`**

```python
"""rolling_form.py — each team's own rolling scoring form, no lookahead.

Reshapes games into one row per team-appearance (home and away rows each
carry "points_scored"/"points_allowed" from that team's own perspective),
computes a shift(1) rolling mean per team so the current game is always
excluded, then reshapes back. Verbatim port from nfl_predictor — operates
only on game_id/gameday/home_team/away_team/home_score/away_score.
"""

from __future__ import annotations

import pandas as pd


def _team_appearances(games_df: pd.DataFrame) -> pd.DataFrame:
    home = games_df[["game_id", "gameday", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "points_scored", "away_score": "points_allowed"}
    )
    away = games_df[["game_id", "gameday", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "points_scored", "home_score": "points_allowed"}
    )
    appearances = pd.concat([home, away], ignore_index=True)
    return appearances.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def add_rolling_form(games_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    appearances = _team_appearances(games_df)
    grouped = appearances.groupby("team")
    appearances["points_scored_roll"] = grouped["points_scored"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    appearances["points_allowed_roll"] = grouped["points_allowed"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    home_form = appearances.rename(
        columns={
            "team": "home_team",
            "points_scored_roll": "home_points_scored_roll",
            "points_allowed_roll": "home_points_allowed_roll",
        }
    )[["game_id", "home_team", "home_points_scored_roll", "home_points_allowed_roll"]]
    away_form = appearances.rename(
        columns={
            "team": "away_team",
            "points_scored_roll": "away_points_scored_roll",
            "points_allowed_roll": "away_points_allowed_roll",
        }
    )[["game_id", "away_team", "away_points_scored_roll", "away_points_allowed_roll"]]

    result = games_df.merge(home_form, on=["game_id", "home_team"], how="left")
    result = result.merge(away_form, on=["game_id", "away_team"], how="left")
    return result
```

- [ ] **Step 4: Write `src/cfb_predictor/features/rest_days.py`**

```python
"""rest_days.py — days since each team's previous game, no lookahead.

Bye weeks are rarer and less uniform in CFB than in the NFL (not every FBS
team gets exactly one), but the shift(1)-since-last-appearance logic is
identical -- verbatim port from nfl_predictor.
"""

from __future__ import annotations

import pandas as pd


def _team_appearances(games_df: pd.DataFrame) -> pd.DataFrame:
    home = games_df[["game_id", "gameday", "home_team"]].rename(columns={"home_team": "team"})
    away = games_df[["game_id", "gameday", "away_team"]].rename(columns={"away_team": "team"})
    appearances = pd.concat([home, away], ignore_index=True)
    return appearances.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def add_rest_days(games_df: pd.DataFrame) -> pd.DataFrame:
    games_df = games_df.copy()
    games_df["gameday"] = pd.to_datetime(games_df["gameday"])
    appearances = _team_appearances(games_df)
    appearances["gameday"] = pd.to_datetime(appearances["gameday"])
    appearances["prior_gameday"] = appearances.groupby("team")["gameday"].shift(1)
    appearances["rest_days"] = (appearances["gameday"] - appearances["prior_gameday"]).dt.days

    home_rest = appearances.rename(columns={"team": "home_team", "rest_days": "home_rest_days"})[
        ["game_id", "home_team", "home_rest_days"]
    ]
    away_rest = appearances.rename(columns={"team": "away_team", "rest_days": "away_rest_days"})[
        ["game_id", "away_team", "away_rest_days"]
    ]

    result = games_df.merge(home_rest, on=["game_id", "home_team"], how="left")
    result = result.merge(away_rest, on=["game_id", "away_team"], how="left")
    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_rolling_form.py tests/test_rest_days.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/cfb_predictor/features/rolling_form.py src/cfb_predictor/features/rest_days.py tests/test_rolling_form.py tests/test_rest_days.py
git commit -m "feat: rolling form and rest days features"
```

---

## Task 7: Features — player usage

**Files:**
- Create: `src/cfb_predictor/features/player_usage.py`
- Test: `tests/test_player_usage.py`

**Interfaces:**
- Consumes: `data/player_stats.py`'s flat frame shape (`player_id, season, week, passing_yards, rushing_yards, receiving_yards, targets, carries, passing_tds, rushing_tds, receiving_tds`).
- Produces: `player_usage.PLAYER_FEATURE_COLUMNS: list[str]`, `player_usage.build_player_training_frame(player_stats_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]` (adds an `anytime_td` target column), `player_usage.build_features_for_player(player_id: str, player_stats_df: pd.DataFrame) -> pd.Series | None`.

Verbatim port from `nfl_predictor/features/player_usage.py` — same shift(1)-then-rolling discipline as `rolling_form.py`, applied per player instead of per team, on column names that are identical between the two projects' flat player-stats shape (Task 3 already normalizes CFBD's box score into these same names).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_player_usage.py
import pandas as pd
import pytest

from cfb_predictor.features import player_usage


def _player_stats():
    rows = []
    for week in range(1, 4):
        rows.append(
            {
                "player_id": "p1", "player_name": "Runner", "position": "RB", "recent_team": "Texas",
                "season": 2025, "week": week,
                "passing_yards": 0, "passing_tds": 0, "rushing_yards": 80 + week, "rushing_tds": 1,
                "receiving_yards": 10, "receiving_tds": 0, "receptions": 2, "targets": float("nan"), "carries": 18,
            }
        )
    return pd.DataFrame(rows)


def test_build_player_training_frame_adds_rolling_features_and_target():
    df, feature_cols = player_usage.build_player_training_frame(_player_stats())

    assert "anytime_td" in df.columns
    assert set(feature_cols).issubset(df.columns)
    week1 = df[df["week"] == 1].iloc[0]
    assert pd.isna(week1["rushing_yards_roll"])
    week3 = df[df["week"] == 3].iloc[0]
    assert week3["rushing_yards_roll"] == (81 + 82) / 2


def test_build_features_for_player_returns_none_with_no_history():
    row = player_usage.build_features_for_player("unknown", _player_stats())
    assert row is None


def test_build_features_for_player_returns_series_with_history():
    row = player_usage.build_features_for_player("p1", _player_stats())
    assert row is not None
    assert row["rushing_yards_roll"] == pytest.approx((81 + 82 + 83) / 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_player_usage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.features.player_usage'`

- [ ] **Step 3: Write `src/cfb_predictor/features/player_usage.py`**

```python
"""player_usage.py — player-level rolling usage/production features for
anytime-TD and yardage prop models. Same shift(1)-then-rolling discipline as
features/rolling_form.py, applied per player instead of per team. Verbatim
port from nfl_predictor: "targets" being always NaN for CFB (see
data/player_stats.py) degrades this file's rolling mean to NaN for that one
column, which fit_anytime_td_classifier/fit_yardage_regressor's .fillna(0)
already tolerates -- no change needed here."""

from __future__ import annotations

import pandas as pd

ROLL_STATS = ["passing_yards", "rushing_yards", "receiving_yards", "targets", "carries"]
PLAYER_FEATURE_COLUMNS = [f"{stat}_roll" for stat in ROLL_STATS]


def _add_rolling(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grouped = df.groupby("player_id")
    for stat in ROLL_STATS:
        df[f"{stat}_roll"] = grouped[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return df


def build_player_training_frame(player_stats_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = _add_rolling(player_stats_df)
    df["anytime_td"] = (
        (df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0) + df["passing_tds"].fillna(0)) > 0
    ).astype(int)
    return df, PLAYER_FEATURE_COLUMNS


def build_features_for_player(player_id: str, player_stats_df: pd.DataFrame) -> pd.Series | None:
    history = player_stats_df[player_stats_df["player_id"] == player_id].sort_values(["season", "week"])
    if history.empty:
        return None
    recent = history.tail(5)
    return pd.Series({f"{stat}_roll": float(recent[stat].mean()) for stat in ROLL_STATS})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_player_usage.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cfb_predictor/features/player_usage.py tests/test_player_usage.py
git commit -m "feat: player usage rolling features"
```

---

## Task 8: Features — single build entry point (`conference_game` + FCS exclusion)

**Files:**
- Create: `src/cfb_predictor/features/build.py`
- Test: `tests/test_build_features.py`

**Interfaces:**
- Consumes: `power_ratings.compute_pregame_ratings`, `power_ratings.final_ratings`, `power_ratings.DEFAULT_START_RATING`, `rolling_form.add_rolling_form`, `rest_days.add_rest_days`.
- Produces: `build.FEATURE_COLUMNS: list[str]` (`conference_game` replaces NFL's `div_game`), `build.build_training_frame(games_df: pd.DataFrame, fbs_teams: dict[int, set[str]] | None = None) -> tuple[pd.DataFrame, list[str]]` (adds `margin`, `total_points`, and `is_fbs_game` columns; excludes non-FBS-vs-FBS games from the returned training rows), `build.build_features_for_game(home_team: str, away_team: str, games_df: pd.DataFrame) -> pd.Series` (one live feature row for an upcoming game).

This is the one genuinely CFB-specific feature file. Two changes from `nfl_predictor/features/build.py`: (1) `conference_game` replaces `div_game` — CFBD's `Game` model already supplies `conference_game` as a real boolean (confirmed against the live `cfbd` package docs; see this plan's "Implementation notes"), so unlike NFL's `div_game` this needs no team-conference join, just a fillna/cast; (2) `build_training_frame` gains an `is_fbs_game` filter that excludes any game where either team isn't FBS from the *training* rows, while `power_ratings`/`rolling_form`/`rest_days` still run over the *entire* `games_df` (including FCS-opponent games) so an FBS team's rest-days/rolling-form bookkeeping stays complete — exactly the "features computed over full history, model trained on the relevant subset" split the design spec calls for.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_build_features.py
import pandas as pd

from cfb_predictor.features import build


def _games():
    rows = []
    teams = ["Texas", "Ohio State", "Michigan", "Alabama"]
    day = pd.Timestamp("2025-08-30")
    for week in range(1, 4):
        rows.append(
            {
                "game_id": f"g{week}a", "season": 2025, "week": week,
                "gameday": day + pd.Timedelta(days=7 * (week - 1)),
                "home_team": teams[0], "away_team": teams[1],
                "home_score": 24, "away_score": 20,
                "home_division": "fbs", "away_division": "fbs", "conference_game": False,
            }
        )
        rows.append(
            {
                "game_id": f"g{week}b", "season": 2025, "week": week,
                "gameday": day + pd.Timedelta(days=7 * (week - 1)),
                "home_team": teams[2], "away_team": teams[3],
                "home_score": 17, "away_score": 27,
                "home_division": "fbs", "away_division": "fbs", "conference_game": True,
            }
        )
    # An FCS-opponent game: counts for Texas's rest_days/rolling_form
    # bookkeeping but must be excluded from build_training_frame's rows.
    rows.append(
        {
            "game_id": "g4fcs", "season": 2025, "week": 4,
            "gameday": day + pd.Timedelta(days=21),
            "home_team": "Texas", "away_team": "Div II School",
            "home_score": 55, "away_score": 3,
            "home_division": "fbs", "away_division": "fcs", "conference_game": False,
        }
    )
    return pd.DataFrame(rows)


def test_build_training_frame_returns_feature_columns_and_targets():
    df, feature_cols = build.build_training_frame(_games())

    assert "margin" in df.columns
    assert "total_points" in df.columns
    assert "conference_game" in feature_cols
    assert set(feature_cols).issubset(df.columns)
    assert (df["margin"] == df["home_score"] - df["away_score"]).all()
    assert (df["total_points"] == df["home_score"] + df["away_score"]).all()


def test_build_training_frame_excludes_fcs_opponent_games_via_division_fallback():
    df, _ = build.build_training_frame(_games())

    assert "g4fcs" not in set(df["game_id"])
    assert len(df) == 6  # 3 weeks x 2 FBS-vs-FBS games each, the FCS game excluded


def test_build_training_frame_excludes_fcs_opponent_games_via_fbs_teams_override():
    fbs_teams = {2025: {"Texas", "Ohio State", "Michigan", "Alabama"}}  # "Div II School" absent
    df, _ = build.build_training_frame(_games(), fbs_teams=fbs_teams)

    assert "g4fcs" not in set(df["game_id"])
    assert len(df) == 6


def test_build_training_frame_preserves_conference_game_flag():
    df, _ = build.build_training_frame(_games())

    conf_games = df[df["game_id"].str.endswith("b")]
    assert (conf_games["conference_game"] == 1).all()
    non_conf_games = df[df["game_id"].str.endswith("a")]
    assert (non_conf_games["conference_game"] == 0).all()


def test_build_features_for_game_returns_series_with_feature_columns():
    games_df = _games()
    _, feature_cols = build.build_training_frame(games_df)

    row = build.build_features_for_game("Texas", "Ohio State", games_df)

    assert isinstance(row, pd.Series)
    for col in feature_cols:
        assert col in row.index
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_build_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.features.build'`

- [ ] **Step 3: Write `src/cfb_predictor/features/build.py`**

```python
"""build.py — the single feature-construction entry point for game-outcome
models. Every consumer (training, walk-forward evaluation, live serving)
must call build_training_frame / build_features_for_game rather than
reimplementing feature logic inline.

conference_game replaces NFL's div_game -- CFBD's Game model already
supplies conference_game as a real boolean, so (unlike NFL's div_game, which
nfl_data_py also supplies pre-computed) no team-conference join is needed
here either; this file only fills a default and casts to int.

FCS-opponent games are excluded from build_training_frame's *training* rows
(is_fbs_game filter below) but power_ratings/rolling_form/rest_days still
compute over every row in games_df, FCS-opponent games included -- an FBS
team's rest_days/rolling_form bookkeeping stays complete even for a week it
spent beating an FCS opponent.
"""

from __future__ import annotations

import pandas as pd

from . import power_ratings, rest_days, rolling_form

FEATURE_COLUMNS = [
    "home_pregame_rating", "away_pregame_rating", "rating_diff",
    "home_points_scored_roll", "home_points_allowed_roll",
    "away_points_scored_roll", "away_points_allowed_roll",
    "home_rest_days", "away_rest_days",
    "conference_game",
]


def _is_fbs_game(df: pd.DataFrame, fbs_teams: dict[int, set[str]] | None) -> pd.Series:
    """True when both teams in the game are FBS. Prefers the authoritative
    per-season FBS team-universe list (fbs_teams, from
    data.games.fetch_fbs_teams) when given -- the design spec's primary
    FCS-exclusion source, since it's CFBD's own official roster of FBS
    teams for that season. Falls back to the game row's own
    home_division/away_division columns (CFBD's per-game classification)
    when fbs_teams isn't supplied, as the spec's cross-check."""
    if fbs_teams:
        def _team_is_fbs(season, team) -> bool:
            teams_for_season = fbs_teams.get(int(season))
            return team in teams_for_season if teams_for_season else True

        return df.apply(
            lambda r: _team_is_fbs(r["season"], r["home_team"]) and _team_is_fbs(r["season"], r["away_team"]),
            axis=1,
        )
    if "home_division" in df.columns and "away_division" in df.columns:
        return (df["home_division"].fillna("fbs").str.lower() == "fbs") & (
            df["away_division"].fillna("fbs").str.lower() == "fbs"
        )
    return pd.Series(True, index=df.index)


def _assemble(games_df: pd.DataFrame) -> pd.DataFrame:
    df = power_ratings.compute_pregame_ratings(games_df)
    df = rolling_form.add_rolling_form(df)
    df = rest_days.add_rest_days(df)
    df["rating_diff"] = df["home_pregame_rating"] - df["away_pregame_rating"]
    df["home_rest_days"] = df["home_rest_days"].fillna(7)
    df["away_rest_days"] = df["away_rest_days"].fillna(7)
    if "conference_game" not in df.columns:
        df["conference_game"] = False
    df["conference_game"] = df["conference_game"].fillna(False).astype(int)
    return df


def build_training_frame(
    games_df: pd.DataFrame, fbs_teams: dict[int, set[str]] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    df = _assemble(games_df)
    df["is_fbs_game"] = _is_fbs_game(df, fbs_teams)
    played = df[
        df["home_score"].notna() & df["away_score"].notna() & df["is_fbs_game"]
    ].reset_index(drop=True)
    played["margin"] = played["home_score"] - played["away_score"]
    played["total_points"] = played["home_score"] + played["away_score"]
    return played, FEATURE_COLUMNS


def build_features_for_game(home_team: str, away_team: str, games_df: pd.DataFrame) -> pd.Series:
    """One live feature row for an upcoming home_team vs away_team game,
    computed from every played game in games_df (ratings/rolling form as of
    right now). conference_game defaults to 0 here -- api/routes.py knows
    the real value from the live schedule row for a specific upcoming game
    but this function only has the two team names, the same limitation
    NFL's build_features_for_game already had for div_game (not fixed here,
    to stay a near-verbatim port rather than a scope expansion)."""
    ratings = power_ratings.final_ratings(games_df)
    played = games_df[games_df["home_score"].notna() & games_df["away_score"].notna()]

    def _recent_form(team: str) -> tuple[float, float]:
        appearances = pd.concat(
            [
                played[played["home_team"] == team][["gameday", "home_score", "away_score"]].rename(
                    columns={"home_score": "scored", "away_score": "allowed"}
                ),
                played[played["away_team"] == team][["gameday", "away_score", "home_score"]].rename(
                    columns={"away_score": "scored", "home_score": "allowed"}
                ),
            ]
        ).sort_values("gameday")
        recent = appearances.tail(5)
        if recent.empty:
            return float("nan"), float("nan")
        return float(recent["scored"].mean()), float(recent["allowed"].mean())

    def _rest_days(team: str) -> float | None:
        appearances = pd.concat(
            [
                played[played["home_team"] == team][["gameday"]],
                played[played["away_team"] == team][["gameday"]],
            ]
        ).sort_values("gameday")
        if appearances.empty:
            return None
        last_game = pd.to_datetime(appearances.iloc[-1]["gameday"])
        return float((pd.Timestamp.now().normalize() - last_game).days)

    home_scored, home_allowed = _recent_form(home_team)
    away_scored, away_allowed = _recent_form(away_team)
    home_rating = ratings.get(home_team, power_ratings.DEFAULT_START_RATING)
    away_rating = ratings.get(away_team, power_ratings.DEFAULT_START_RATING)
    home_rest = _rest_days(home_team)
    away_rest = _rest_days(away_team)

    return pd.Series(
        {
            "home_pregame_rating": home_rating,
            "away_pregame_rating": away_rating,
            "rating_diff": home_rating - away_rating,
            "home_points_scored_roll": home_scored,
            "home_points_allowed_roll": home_allowed,
            "away_points_scored_roll": away_scored,
            "away_points_allowed_roll": away_allowed,
            "home_rest_days": home_rest if home_rest is not None else 7.0,
            "away_rest_days": away_rest if away_rest is not None else 7.0,
            "conference_game": 0,
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_build_features.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cfb_predictor/features/build.py tests/test_build_features.py
git commit -m "feat: feature build entry point with conference_game and FCS exclusion"
```

---

## Task 9: Models — game outcome candidates (Elo, ridge, XGBoost)

**Files:**
- Create: `src/cfb_predictor/models/__init__.py`
- Create: `src/cfb_predictor/models/game_outcome.py`
- Test: `tests/test_game_outcome.py`

**Interfaces:**
- Consumes: `features.build.FEATURE_COLUMNS` (specifically `rating_diff`, `home_rest_days`, `away_rest_days` for the Elo candidate; the full feature set for ridge/XGBoost).
- Produces: `game_outcome.ELO_POINTS_PER_RATING_POINT` (float constant), `game_outcome.fit_elo_candidate(train_df: pd.DataFrame) -> dict`, `game_outcome.predict_margin_elo(candidate: dict, rating_diff: float, home_rest_days: float, away_rest_days: float) -> float`, `game_outcome.fit_margin_regression(X_train, y_margin) -> Ridge`, `game_outcome.fit_xgb_margin(X_train, y_margin) -> XGBRegressor`, `game_outcome.residual_sigma(model, X_val, y_val) -> float`, `game_outcome.margin_to_probabilities(predicted_margin, sigma, spread_line=None, total_line=None, predicted_total=None, total_sigma=None) -> dict`.

Verbatim port of `nfl_predictor/models/game_outcome.py` — this file has zero NFL-specific logic (confirmed while reading the real source: no reference to `div_game`, team names, or any NFL-only column; it operates purely on the generic `rating_diff`/`home_rest_days`/`away_rest_days`/margin/total columns every project's `features/build.py` produces under the same names).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_game_outcome.py
import numpy as np
import pandas as pd
import pytest

from cfb_predictor.models import game_outcome


def _toy_frame():
    rng = np.random.default_rng(42)
    n = 60
    rating_diff = rng.normal(0, 100, n)
    margin = rating_diff * 0.05 + rng.normal(0, 10, n)
    return pd.DataFrame(
        {
            "home_pregame_rating": 1500 + rating_diff / 2,
            "away_pregame_rating": 1500 - rating_diff / 2,
            "rating_diff": rating_diff,
            "home_points_scored_roll": rng.normal(28, 6, n),
            "home_points_allowed_roll": rng.normal(24, 6, n),
            "away_points_scored_roll": rng.normal(27, 6, n),
            "away_points_allowed_roll": rng.normal(25, 6, n),
            "home_rest_days": 7,
            "away_rest_days": 7,
            "conference_game": 0,
            "margin": margin,
        }
    )


FEATURE_COLS = [
    "home_pregame_rating", "away_pregame_rating", "rating_diff",
    "home_points_scored_roll", "home_points_allowed_roll",
    "away_points_scored_roll", "away_points_allowed_roll",
    "home_rest_days", "away_rest_days", "conference_game",
]


def test_fit_margin_regression_predicts_signed_margin():
    df = _toy_frame()
    model = game_outcome.fit_margin_regression(df[FEATURE_COLS], df["margin"])

    preds = model.predict(df[FEATURE_COLS])
    assert np.corrcoef(preds, df["margin"])[0, 1] > 0.3


def test_fit_xgb_margin_predicts_signed_margin():
    df = _toy_frame()
    model = game_outcome.fit_xgb_margin(df[FEATURE_COLS], df["margin"])

    preds = model.predict(df[FEATURE_COLS])
    assert np.corrcoef(preds, df["margin"])[0, 1] > 0.3


def test_residual_sigma_is_positive():
    df = _toy_frame()
    model = game_outcome.fit_margin_regression(df[FEATURE_COLS], df["margin"])

    sigma = game_outcome.residual_sigma(model, df[FEATURE_COLS], df["margin"])

    assert sigma > 0


def test_margin_to_probabilities_favors_positive_margin():
    result = game_outcome.margin_to_probabilities(predicted_margin=7.0, sigma=13.0)

    assert result["home_win_prob"] > 0.5
    assert result["home_win_prob"] + result["away_win_prob"] == pytest.approx(1.0)


def test_margin_to_probabilities_includes_cover_and_total_when_lines_given():
    result = game_outcome.margin_to_probabilities(
        predicted_margin=7.0, sigma=13.0, spread_line=-3.0,
        total_line=45.0, predicted_total=48.0, total_sigma=10.0,
    )

    assert "home_cover_prob" in result
    assert "away_cover_prob" in result
    assert result["home_cover_prob"] + result["away_cover_prob"] == pytest.approx(1.0)
    assert "over_prob" in result
    assert "under_prob" in result
    assert result["over_prob"] + result["under_prob"] == pytest.approx(1.0)
    assert result["over_prob"] > 0.5


def test_margin_to_probabilities_spread_line_uses_home_expected_margin_convention():
    result = game_outcome.margin_to_probabilities(
        predicted_margin=3.0, sigma=13.0, spread_line=6.0,
    )

    assert result["home_cover_prob"] < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_game_outcome.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.models'`

- [ ] **Step 3: Write `src/cfb_predictor/models/__init__.py` and `game_outcome.py`**

```python
# src/cfb_predictor/models/__init__.py
```

```python
"""game_outcome.py — margin-of-victory candidates for the game-outcome
race, plus the shared margin -> win/cover/total-probability conversion.

CFB scoring, like NFL, isn't a low-count Poisson process -- this predicts a
continuous point margin (home_score - away_score) and total_points, then
converts each to probabilities via a fitted-Normal residual distribution.
Three candidates are raced in evaluate/walk_forward.py: Elo (implicit in
features.power_ratings' rating_diff, converted directly via
margin_to_probabilities with a fixed points-per-Elo-point scale), ridge
regression, and XGBoost -- whichever wins held-out log-loss is served.
Verbatim port from nfl_predictor: no NFL-specific column or assumption
anywhere in this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

ELO_POINTS_PER_RATING_POINT = 1.0 / 25.0


def fit_elo_candidate(train_df: pd.DataFrame) -> dict:
    return {"points_per_rating_point": ELO_POINTS_PER_RATING_POINT}


def predict_margin_elo(candidate: dict, rating_diff: float, home_rest_days: float, away_rest_days: float) -> float:
    return rating_diff * candidate["points_per_rating_point"] + 0.05 * (home_rest_days - away_rest_days)


def fit_margin_regression(X_train: pd.DataFrame, y_margin: pd.Series) -> Ridge:
    model = Ridge(alpha=1.0)
    model.fit(X_train.fillna(0), y_margin)
    return model


def fit_xgb_margin(X_train: pd.DataFrame, y_margin: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        reg_lambda=1.0, reg_alpha=0.0, random_state=42,
    )
    model.fit(X_train.fillna(0), y_margin)
    return model


def residual_sigma(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    preds = model.predict(X_val.fillna(0))
    residuals = np.asarray(y_val) - preds
    if len(residuals) == 0:
        return 1.0
    return float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.std(residuals) or 1.0)


def margin_to_probabilities(
    predicted_margin: float,
    sigma: float,
    spread_line: float | None = None,
    total_line: float | None = None,
    predicted_total: float | None = None,
    total_sigma: float | None = None,
) -> dict:
    """margin ~ Normal(predicted_margin, sigma). home_win_prob = P(margin > 0).
    spread_line follows the home team's expected margin (positive means home
    favored by that many points) -- the home team covers when
    margin > spread_line. total_points ~ Normal(predicted_total, total_sigma);
    over_prob = P(total > total_line)."""
    home_win_prob = float(1.0 - norm.cdf(0.0, loc=predicted_margin, scale=sigma))
    result = {"home_win_prob": home_win_prob, "away_win_prob": 1.0 - home_win_prob}

    if spread_line is not None:
        home_cover_prob = float(1.0 - norm.cdf(spread_line, loc=predicted_margin, scale=sigma))
        result["home_cover_prob"] = home_cover_prob
        result["away_cover_prob"] = 1.0 - home_cover_prob

    if total_line is not None and predicted_total is not None and total_sigma is not None:
        over_prob = float(1.0 - norm.cdf(total_line, loc=predicted_total, scale=total_sigma))
        result["over_prob"] = over_prob
        result["under_prob"] = 1.0 - over_prob

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_game_outcome.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cfb_predictor/models/__init__.py src/cfb_predictor/models/game_outcome.py tests/test_game_outcome.py
git commit -m "feat: game outcome candidate models (Elo, ridge, XGBoost)"
```

---

## Task 10: Evaluate — walk-forward validation and candidate race

**Files:**
- Create: `src/cfb_predictor/evaluate/__init__.py`
- Create: `src/cfb_predictor/evaluate/walk_forward.py`
- Test: `tests/test_walk_forward.py`

**Interfaces:**
- Consumes: `features.build.build_training_frame(games_df, fbs_teams=None)`, `models.game_outcome.fit_elo_candidate/predict_margin_elo/fit_margin_regression/fit_xgb_margin/residual_sigma/margin_to_probabilities`.
- Produces: `walk_forward.prepare_folds(games_df: pd.DataFrame, fbs_teams: dict[int, set[str]] | None = None, min_train_seasons: int = 2) -> list[dict]` (each fold: `val_season, train_df, val_df, feature_cols`), `walk_forward.evaluate_candidate(folds: list[dict], candidate: str) -> pd.DataFrame` (columns `val_season, n_games, log_loss, brier`).

Near-verbatim port of `nfl_predictor/evaluate/walk_forward.py` — the only change is threading an optional `fbs_teams` parameter through to `build_training_frame` so FCS-opponent games are excluded from every fold's train/validation rows exactly the way `models/manifest.py`'s single top-level call already does; `evaluate_candidate`'s candidate-fitting/scoring logic is unchanged (it never references `conference_game`/`div_game` or anything CFB/NFL-specific).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_walk_forward.py
import numpy as np
import pandas as pd
import pytest

from cfb_predictor.evaluate import walk_forward


def _multi_season_games():
    rng = np.random.default_rng(7)
    rows = []
    teams = [f"T{i}" for i in range(8)]
    for season in (2022, 2023, 2024):
        for week in range(1, 6):
            for i in range(0, len(teams), 2):
                home, away = teams[i], teams[i + 1]
                home_score = int(rng.integers(10, 45))
                away_score = int(rng.integers(10, 45))
                rows.append(
                    {
                        "game_id": f"{season}_{week}_{home}_{away}", "season": season, "week": week,
                        "gameday": pd.Timestamp(f"{season}-08-25") + pd.Timedelta(days=7 * (week - 1)),
                        "home_team": home, "away_team": away,
                        "home_score": home_score, "away_score": away_score,
                        "home_division": "fbs", "away_division": "fbs", "conference_game": False,
                    }
                )
    return pd.DataFrame(rows)


def test_prepare_folds_holds_out_each_season_after_minimum():
    folds = walk_forward.prepare_folds(_multi_season_games(), min_train_seasons=2)

    val_seasons = [fold["val_season"] for fold in folds]
    assert val_seasons == [2024]

    for fold in folds:
        assert fold["train_df"]["season"].max() < fold["val_season"]
        assert (fold["val_df"]["season"] == fold["val_season"]).all()


def test_prepare_folds_accepts_fbs_teams_and_still_holds_out_each_season():
    fbs_teams = {season: {f"T{i}" for i in range(8)} for season in (2022, 2023, 2024)}
    folds = walk_forward.prepare_folds(_multi_season_games(), fbs_teams=fbs_teams, min_train_seasons=2)

    assert [fold["val_season"] for fold in folds] == [2024]


def test_evaluate_candidate_returns_a_row_per_fold_for_each_candidate():
    folds = walk_forward.prepare_folds(_multi_season_games(), min_train_seasons=2)

    for candidate in ("elo", "ridge", "xgb"):
        result = walk_forward.evaluate_candidate(folds, candidate)
        assert len(result) == len(folds)
        assert (result["log_loss"] > 0).all()
        assert (result["brier"] >= 0).all()


def test_evaluate_candidate_rejects_unknown_candidate():
    folds = walk_forward.prepare_folds(_multi_season_games(), min_train_seasons=2)

    with pytest.raises(ValueError):
        walk_forward.evaluate_candidate(folds, "not-a-real-candidate")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_walk_forward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.evaluate'`

- [ ] **Step 3: Write `src/cfb_predictor/evaluate/__init__.py` and `walk_forward.py`**

```python
# src/cfb_predictor/evaluate/__init__.py
```

```python
"""walk_forward.py — season-by-season walk-forward validation for the three
models/game_outcome.py candidates. Builds the full feature frame ONCE (no
lookahead -- every feature is already shift(1)/expanding computed before any
slicing), then slices by season so evaluate_candidate can be called
repeatedly without redoing feature engineering. Near-verbatim port of
nfl_predictor's own evaluate/walk_forward.py -- the only change is threading
an optional fbs_teams mapping through to build_training_frame so every
fold's rows already exclude FCS-opponent games.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..features.build import build_training_frame
from ..models import game_outcome


def prepare_folds(
    games_df: pd.DataFrame,
    fbs_teams: dict[int, set[str]] | None = None,
    min_train_seasons: int = 2,
) -> list[dict]:
    df, feature_cols = build_training_frame(games_df, fbs_teams=fbs_teams)
    seasons = sorted(df["season"].unique())

    folds = []
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_seasons = seasons[:i]
        train_df = df[df["season"].isin(train_seasons)]
        val_df = df[df["season"] == val_season]
        if train_df.empty or val_df.empty:
            continue
        folds.append({"val_season": val_season, "train_df": train_df, "val_df": val_df, "feature_cols": feature_cols})
    return folds


def _predict_margin_elo_batch(candidate: dict, df: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            game_outcome.predict_margin_elo(candidate, r, hr, ar)
            for r, hr, ar in zip(df["rating_diff"], df["home_rest_days"], df["away_rest_days"])
        ]
    )


class _EloModelAdapter:
    """Adapts the elo candidate's dict + free function to the model.predict(X)
    interface residual_sigma expects, without ignoring the X it's given."""

    def __init__(self, candidate: dict):
        self.candidate = candidate

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return _predict_margin_elo_batch(self.candidate, X)


def _predict_margins(candidate: str, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]):
    X_train, y_train = train_df[feature_cols], train_df["margin"]
    X_val = val_df[feature_cols]

    if candidate == "elo":
        model = game_outcome.fit_elo_candidate(train_df)
        preds = _predict_margin_elo_batch(model, val_df)
        sigma = game_outcome.residual_sigma(_EloModelAdapter(model), X_train, y_train)
        return preds, sigma

    if candidate == "ridge":
        fit_fn = game_outcome.fit_margin_regression
    elif candidate == "xgb":
        fit_fn = game_outcome.fit_xgb_margin
    else:
        raise ValueError(f"Unknown candidate: {candidate!r}")

    model = fit_fn(X_train, y_train)
    preds = model.predict(X_val.fillna(0))
    sigma = game_outcome.residual_sigma(model, X_train, y_train)
    return preds, sigma


def evaluate_candidate(folds: list[dict], candidate: str) -> pd.DataFrame:
    rows = []
    for fold in folds:
        train_df, val_df, feature_cols = fold["train_df"], fold["val_df"], fold["feature_cols"]
        preds, sigma = _predict_margins(candidate, train_df, val_df, feature_cols)

        probs = np.array(
            [game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds]
        )
        actual = (val_df["margin"] > 0).astype(int).to_numpy()
        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        rows.append(
            {
                "val_season": fold["val_season"],
                "n_games": len(val_df),
                "log_loss": log_loss(actual, probs, labels=[0, 1]),
                "brier": brier_score_loss(actual, probs),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_walk_forward.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cfb_predictor/evaluate/__init__.py src/cfb_predictor/evaluate/walk_forward.py tests/test_walk_forward.py
git commit -m "feat: walk-forward validation and candidate race"
```

---

## Task 11: Models — player props (anytime-TD classifier + yardage regressors)

**Files:**
- Create: `src/cfb_predictor/models/player_props.py`
- Test: `tests/test_player_props.py`

**Interfaces:**
- Consumes: `features.player_usage.PLAYER_FEATURE_COLUMNS`.
- Produces: `player_props.YARDAGE_TARGETS: dict[str, str]`, `player_props.POSITION_YARDAGE_MARKET: dict[str, str]`, `player_props.fit_anytime_td_classifier(X_train, y_train) -> XGBClassifier`, `player_props.fit_yardage_regressor(X_train, y_train) -> XGBRegressor`, `player_props.predict_props(models: dict, feature_row: pd.Series, position: str) -> dict`.

Verbatim port of `nfl_predictor/models/player_props.py`. `POSITION_YARDAGE_MARKET` keeps its four-way `{QB, RB, WR, TE}` mapping unchanged even though `data/player_stats.py`'s CFBD-derived `_infer_position` can only ever emit `QB`/`RB`/`WR` (WR and TE collapse to `WR` — see Task 3's note) — the dict already maps both WR and TE to the same `receiving_yards` market, so keeping the fourth entry is harmless and future-proofs this file if a later CFB roster-position source is added.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_player_props.py
import numpy as np
import pandas as pd
import pytest

from cfb_predictor.models import player_props


def _toy_player_frame(n=80, seed=1):
    rng = np.random.default_rng(seed)
    passing_roll = rng.normal(240, 40, n)
    rushing_roll = rng.normal(90, 25, n)
    receiving_roll = rng.normal(55, 20, n)
    return pd.DataFrame(
        {
            "passing_yards_roll": passing_roll,
            "rushing_yards_roll": rushing_roll,
            "receiving_yards_roll": receiving_roll,
            "targets_roll": rng.normal(5, 2, n),
            "carries_roll": rng.normal(16, 5, n),
            "passing_yards": passing_roll + rng.normal(0, 15, n),
            "rushing_yards": rushing_roll + rng.normal(0, 15, n),
            "receiving_yards": receiving_roll + rng.normal(0, 15, n),
            "anytime_td": (rng.random(n) < (0.3 + rushing_roll / 500)).astype(int),
        }
    )


FEATURE_COLS = ["passing_yards_roll", "rushing_yards_roll", "receiving_yards_roll", "targets_roll", "carries_roll"]


def test_fit_anytime_td_classifier_predicts_probabilities():
    df = _toy_player_frame()
    model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])

    probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()


def test_fit_yardage_regressor_predicts_reasonable_values():
    df = _toy_player_frame()
    model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["rushing_yards"])

    preds = model.predict(df[FEATURE_COLS])
    assert np.corrcoef(preds, df["rushing_yards"])[0, 1] > 0.3


def test_predict_props_only_returns_relevant_yardage_market_for_position():
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    rushing_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["rushing_yards"])
    models = {"anytime_td": td_model, "rushing_yards": rushing_model, "feature_cols": FEATURE_COLS}

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="RB")

    assert "anytime_td_prob" in result
    assert "rushing_yards" in result
    assert "passing_yards" not in result
    assert "receiving_yards" not in result


def test_predict_props_only_returns_relevant_yardage_market_for_qb():
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    passing_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["passing_yards"])
    models = {"anytime_td": td_model, "passing_yards": passing_model, "feature_cols": FEATURE_COLS}

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="QB")

    assert "passing_yards" in result
    assert "rushing_yards" not in result
    assert "receiving_yards" not in result


def test_predict_props_collapses_wr_and_te_to_the_same_receiving_market():
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    receiving_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["receiving_yards"])
    models = {"anytime_td": td_model, "receiving_yards": receiving_model, "feature_cols": FEATURE_COLS}

    wr_result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="WR")
    te_result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="TE")

    assert "receiving_yards" in wr_result
    assert "receiving_yards" in te_result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_player_props.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.models.player_props'`

- [ ] **Step 3: Write `src/cfb_predictor/models/player_props.py`**

```python
"""player_props.py — anytime-TD classifier and per-position yardage
regressors. Verbatim port from nfl_predictor: yardage props are priced as a
continuous over/under line for both sports, and the fitting/prediction
shape has no NFL-specific assumption. POSITION_YARDAGE_MARKET's four-way
{QB, RB, WR, TE} mapping is kept as-is even though data/player_stats.py's
CFBD-derived position inference can only ever emit QB/RB/WR (see that
module's docstring) -- WR and TE already map to the same receiving_yards
market, so the extra entry costs nothing and future-proofs this file.
"""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

YARDAGE_TARGETS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
}

POSITION_YARDAGE_MARKET = {"QB": "passing_yards", "RB": "rushing_yards", "WR": "receiving_yards", "TE": "receiving_yards"}


def fit_anytime_td_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train.fillna(0), y_train)
    return model


def fit_yardage_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train.fillna(0), y_train)
    return model


def predict_props(models: dict, feature_row: pd.Series, position: str) -> dict:
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0).to_numpy().reshape(1, -1)

    result = {"anytime_td_prob": float(models["anytime_td"].predict_proba(X)[0, 1])}

    market = POSITION_YARDAGE_MARKET.get(position)
    if market and market in models:
        result[market] = float(models[market].predict(X)[0])

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_player_props.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cfb_predictor/models/player_props.py tests/test_player_props.py
git commit -m "feat: anytime-TD classifier and yardage regressors"
```

---

## Task 12: Models — manifest (train/save/load orchestration)

**Files:**
- Create: `src/cfb_predictor/models/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `data.games.default_completed_seasons/load_training_data/fetch_fbs_teams`, `data.player_stats.fetch_weekly_player_stats(seasons, games_df)`, `evaluate.walk_forward.prepare_folds/evaluate_candidate`, `features.build.build_training_frame`, `features.player_usage.build_player_training_frame`, `models.game_outcome.*`, `models.player_props.*`.
- Produces: `manifest.train_all(seasons: list[int] | None = None) -> dict`, `manifest.load_manifest() -> dict`, `manifest.load_models() -> dict` (returns `game_outcome_model, total_model, chosen_candidate, sigma, total_sigma, player_models, feature_cols, player_feature_cols` — this exact shape is relied on by `api/routes.py` in Task 15).

Near-verbatim port of `nfl_predictor/models/manifest.py`. Two adaptations: (1) `schedules` becomes `data.games`; (2) a `fbs_teams` dict (season -> set of FBS team names, from `games.fetch_fbs_teams`) is built once and threaded into both `feature_build.build_training_frame` and `walk_forward.prepare_folds` so FCS-opponent games are excluded from every candidate's training/validation rows; (3) `player_stats.fetch_weekly_player_stats` now takes `games_df` as a required second argument (see Task 3).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_manifest.py
import json

import numpy as np
import pandas as pd
import pytest

from cfb_predictor.models import manifest


def _fake_games(seasons):
    rng = np.random.default_rng(3)
    rows = []
    teams = [f"T{i}" for i in range(8)]
    for season in seasons:
        for week in range(1, 6):
            for i in range(0, len(teams), 2):
                home, away = teams[i], teams[i + 1]
                rows.append(
                    {
                        "game_id": f"{season}_{week}_{home}_{away}", "season": season, "week": week,
                        "gameday": pd.Timestamp(f"{season}-08-25") + pd.Timedelta(days=7 * (week - 1)),
                        "home_team": home, "away_team": away,
                        "home_score": int(rng.integers(10, 45)), "away_score": int(rng.integers(10, 45)),
                        "home_division": "fbs", "away_division": "fbs", "conference_game": False,
                    }
                )
    return pd.DataFrame(rows)


def _fake_fbs_teams(season):
    return pd.DataFrame({"team": [f"T{i}" for i in range(8)], "conference": ["X"] * 8})


def _fake_player_stats(seasons, games_df=None):
    rng = np.random.default_rng(4)
    rows = []
    for season in seasons:
        for week in range(1, 6):
            rows.append(
                {
                    "player_id": "p1", "player_name": "Runner", "position": "RB", "recent_team": "T0",
                    "season": season, "week": week,
                    "passing_yards": 0, "passing_tds": 0,
                    "rushing_yards": int(rng.integers(50, 140)), "rushing_tds": int(rng.integers(0, 2)),
                    "receiving_yards": int(rng.integers(0, 30)), "receiving_tds": 0,
                    "receptions": 2, "targets": float("nan"), "carries": 18,
                }
            )
    return pd.DataFrame(rows)


def test_train_all_writes_a_manifest_with_chosen_candidate(monkeypatch, tmp_path):
    from cfb_predictor import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")

    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.games_data, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.games_data, "fetch_fbs_teams", _fake_fbs_teams)
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", _fake_player_stats)

    result = manifest.train_all(seasons=seasons)

    assert result["chosen_candidate"] in ("elo", "ridge", "xgb")
    assert (tmp_path / "manifest.json").exists()
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["chosen_candidate"] == result["chosen_candidate"]
    assert (tmp_path / "game_outcome_model.pkl").exists()
    assert (tmp_path / "total_points_model.pkl").exists()
    assert (tmp_path / "anytime_td_model.pkl").exists()
    assert (tmp_path / "rushing_yards_model.pkl").exists()


def test_load_models_round_trips_after_train_all(monkeypatch, tmp_path):
    from cfb_predictor import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")

    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.games_data, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.games_data, "fetch_fbs_teams", _fake_fbs_teams)
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", _fake_player_stats)

    manifest.train_all(seasons=seasons)
    models = manifest.load_models()

    assert "game_outcome_model" in models
    assert "player_models" in models
    assert "anytime_td" in models["player_models"]


def test_train_all_rejects_training_data_without_walk_forward_fold(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest.games_data, "load_training_data", lambda s: _fake_games([2024]))
    monkeypatch.setattr(manifest.games_data, "fetch_fbs_teams", _fake_fbs_teams)

    with pytest.raises(ValueError, match="walk-forward validation fold"):
        manifest.train_all(seasons=[2024])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.models.manifest'`

- [ ] **Step 3: Write `src/cfb_predictor/models/manifest.py`**

```python
"""Train, save, and load the game-outcome and player-prop models."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import pandas as pd

from ..config import MODELS_DIR
from ..data import games as games_data
from ..data import player_stats
from ..evaluate import walk_forward
from ..features import build as feature_build
from ..features import player_usage
from . import game_outcome, player_props

MANIFEST_PATH = MODELS_DIR / "manifest.json"
GAME_MODEL_FILENAME = "game_outcome_model.pkl"
TOTAL_MODEL_FILENAME = "total_points_model.pkl"
ANYTIME_TD_MODEL_FILENAME = "anytime_td_model.pkl"

DEFAULT_TRAIN_SEASONS = 8


def _save_pickle(obj, path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _artifact_path(filename: str):
    return MODELS_DIR / filename


def _yardage_model_path(market: str):
    return _artifact_path(f"{market}_model.pkl")


def _fbs_teams_by_season(seasons: list[int]) -> dict[int, set[str]]:
    """One data.games.fetch_fbs_teams() call per season (each cached after
    first fetch) -- the authoritative FBS team universe build_training_frame
    and evaluate.walk_forward.prepare_folds use to exclude FCS-opponent
    games from every candidate's training rows."""
    return {season: set(games_data.fetch_fbs_teams(season)["team"]) for season in seasons}


def train_all(seasons: list[int] | None = None) -> dict:
    """Fit all models, persist their artifacts, and return their manifest."""
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    seasons = seasons or games_data.default_completed_seasons(n=DEFAULT_TRAIN_SEASONS)

    games_df = games_data.load_training_data(seasons)
    fbs_teams = _fbs_teams_by_season(seasons)
    train_df, feature_cols = feature_build.build_training_frame(games_df, fbs_teams=fbs_teams)

    folds = walk_forward.prepare_folds(games_df, fbs_teams=fbs_teams, min_train_seasons=max(1, len(seasons) - 2))
    if not folds:
        raise ValueError("Training requires at least one walk-forward validation fold.")
    candidate_scores = {}
    for candidate in ("elo", "ridge", "xgb"):
        scored = walk_forward.evaluate_candidate(folds, candidate) if folds else pd.DataFrame()
        candidate_scores[candidate] = float(scored["log_loss"].mean()) if not scored.empty else float("inf")
    chosen = min(candidate_scores, key=candidate_scores.get)

    X_train = train_df[feature_cols]
    y_margin = train_df["margin"]
    y_total = train_df["total_points"]

    if chosen == "elo":
        game_model = game_outcome.fit_elo_candidate(train_df)
    elif chosen == "ridge":
        game_model = game_outcome.fit_margin_regression(X_train, y_margin)
    else:
        game_model = game_outcome.fit_xgb_margin(X_train, y_margin)

    if chosen == "elo":
        margin_preds = train_df.apply(
            lambda r: game_outcome.predict_margin_elo(
                game_model, r["rating_diff"], r["home_rest_days"], r["away_rest_days"]
            ),
            axis=1,
        )
        sigma_model = type("_", (), {"predict": lambda self, X: margin_preds.to_numpy()})()
        sigma = game_outcome.residual_sigma(sigma_model, X_train, y_margin)
    else:
        sigma = game_outcome.residual_sigma(game_model, X_train, y_margin)

    total_model = game_outcome.fit_xgb_margin(X_train, y_total)
    total_sigma = game_outcome.residual_sigma(total_model, X_train, y_total)

    _save_pickle(game_model, _artifact_path(GAME_MODEL_FILENAME))
    _save_pickle(total_model, _artifact_path(TOTAL_MODEL_FILENAME))

    player_df_raw = player_stats.fetch_weekly_player_stats(seasons, games_df)
    player_train_df, player_feature_cols = player_usage.build_player_training_frame(player_df_raw)
    player_train_df = player_train_df.dropna(subset=player_feature_cols, how="all")
    X_player = player_train_df[player_feature_cols].fillna(0)

    anytime_td_model = player_props.fit_anytime_td_classifier(X_player, player_train_df["anytime_td"])
    _save_pickle(anytime_td_model, _artifact_path(ANYTIME_TD_MODEL_FILENAME))

    yardage_metrics = {}
    for market, target_col in player_props.YARDAGE_TARGETS.items():
        path = _yardage_model_path(market)
        subset = player_train_df[player_train_df[target_col] > 0]
        if subset.empty:
            path.unlink(missing_ok=True)
            continue
        model = player_props.fit_yardage_regressor(subset[player_feature_cols].fillna(0), subset[target_col])
        _save_pickle(model, path)
        yardage_metrics[market] = {"n_train": int(len(subset))}

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons": sorted(int(s) for s in seasons),
        "n_train": int(len(train_df)),
        "feature_cols": feature_cols,
        "player_feature_cols": player_feature_cols,
        "chosen_candidate": chosen,
        "candidate_scores": candidate_scores,
        "sigma": sigma,
        "total_sigma": total_sigma,
        "yardage_metrics": yardage_metrics,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("No trained models found. Run `python -m cfb_predictor.models.manifest` first.")
    return json.loads(MANIFEST_PATH.read_text())


def load_models() -> dict:
    manifest = load_manifest()
    player_models = {
        "feature_cols": manifest["player_feature_cols"],
        "anytime_td": _load_pickle(_artifact_path(ANYTIME_TD_MODEL_FILENAME)),
    }
    for market in manifest["yardage_metrics"]:
        player_models[market] = _load_pickle(_yardage_model_path(market))

    return {
        "game_outcome_model": _load_pickle(_artifact_path(GAME_MODEL_FILENAME)),
        "total_model": _load_pickle(_artifact_path(TOTAL_MODEL_FILENAME)),
        "chosen_candidate": manifest["chosen_candidate"],
        "sigma": manifest["sigma"],
        "total_sigma": manifest["total_sigma"],
        "player_models": player_models,
        "feature_cols": manifest["feature_cols"],
        "player_feature_cols": manifest["player_feature_cols"],
    }


if __name__ == "__main__":
    train_all()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_manifest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cfb_predictor/models/manifest.py tests/test_manifest.py
git commit -m "feat: model train/save/load orchestration manifest"
```

---

## Task 13: Tracking — SQLite snapshot-then-reconcile store

**Files:**
- Create: `src/cfb_predictor/tracking/__init__.py`
- Create: `src/cfb_predictor/tracking/store.py`
- Test: `tests/test_tracking_store.py`

**Interfaces:**
- Consumes: `config.TRACKING_DB_PATH`.
- Produces: `store.record_game_predictions(games: list[dict]) -> int`, `store.reconcile_game_predictions(results_df: pd.DataFrame) -> int`, `store.get_track_record() -> dict`, `store.record_player_prop_predictions(props: list[dict]) -> int`, `store.reconcile_player_prop_predictions(player_stats_df: pd.DataFrame) -> int`.

Verbatim port of `nfl_predictor/tracking/store.py`, including the `contextlib.closing` connection-handling fix from NFL_Predictor's final review — this SQLite snapshot-then-reconcile schema has zero sport-specific logic (it only ever touches `game_id`, team names, probabilities, and scores as opaque values).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tracking_store.py
import pandas as pd
import pytest

from cfb_predictor.tracking import store


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from cfb_predictor import config

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield


def _game():
    return {
        "game_id": "401520145", "home_team": "Texas", "away_team": "Ohio State",
        "commence_time": "2099-08-30T16:00:00", "home_win_prob": 0.42, "away_win_prob": 0.58,
        "home_cover_prob": 0.48, "away_cover_prob": 0.52, "over_prob": 0.55, "under_prob": 0.45,
    }


def test_record_game_predictions_is_idempotent():
    n1 = store.record_game_predictions([_game()])
    n2 = store.record_game_predictions([_game()])

    assert n1 == 1
    assert n2 == 0


def test_record_game_predictions_rejects_snapshots_after_kickoff():
    game = _game() | {"commence_time": "2000-08-30T16:00:00"}

    with pytest.raises(ValueError, match="before kickoff"):
        store.record_game_predictions([game])


def test_reconcile_game_predictions_fills_actual_outcome():
    store.record_game_predictions([_game()])
    results = pd.DataFrame([{"game_id": "401520145", "home_score": 7, "away_score": 14}])

    n = store.reconcile_game_predictions(results)

    assert n == 1
    record = store.get_track_record()
    assert record["n_resolved_games"] == 1
    assert record["pct_moneyline_correct"] == 1.0  # predicted away win, away won


def test_record_and_reconcile_player_prop_predictions():
    prop = {"game_id": "401520145", "player_id": "4568", "player_name": "R. Back",
            "market": "rushing_yards", "predicted_value": 95.0}
    store.record_player_prop_predictions([prop])

    player_stats_df = pd.DataFrame(
        [{"game_id": "401520145", "player_id": "4568", "rushing_yards": 112, "receiving_yards": 5,
          "passing_yards": 0, "rushing_tds": 1, "receiving_tds": 0, "passing_tds": 0}]
    )
    n = store.reconcile_player_prop_predictions(player_stats_df)

    assert n == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_tracking_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.tracking'`

- [ ] **Step 3: Write `src/cfb_predictor/tracking/__init__.py` and `store.py`**

```python
# src/cfb_predictor/tracking/__init__.py
```

```python
"""SQLite persistence for the live prediction track record.

Snapshots each game's core-market predictions and each tracked player prop
before kickoff, then reconciles them against actual results once games are
played. Snapshot rows are immutable (``INSERT OR IGNORE``); reconciliation
only fills outcome columns on existing, unresolved rows. Verbatim port of
nfl_predictor/tracking/store.py — this schema has zero sport-specific logic.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from ..config import TRACKING_DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(TRACKING_DB_PATH), timeout=15)
    conn.execute("PRAGMA busy_timeout = 15000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS game_predictions (
            game_id TEXT PRIMARY KEY,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            commence_time TEXT NOT NULL,
            snapshotted_at TEXT NOT NULL,
            home_win_prob REAL NOT NULL,
            away_win_prob REAL NOT NULL,
            home_cover_prob REAL,
            away_cover_prob REAL,
            over_prob REAL,
            under_prob REAL,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_home_score INTEGER,
            actual_away_score INTEGER,
            moneyline_hit INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_prop_predictions (
            game_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            market TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            snapshotted_at TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_value REAL,
            PRIMARY KEY (game_id, player_id, market)
        )
        """
    )
    return conn


def _require_pre_kickoff(commence_time: str) -> None:
    kickoff = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    if kickoff <= datetime.now(timezone.utc):
        raise ValueError("Game predictions must be snapshotted before kickoff")


def record_game_predictions(games: list[dict]) -> int:
    if not games:
        return 0
    for game in games:
        _require_pre_kickoff(game["commence_time"])
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            game["game_id"], game["home_team"], game["away_team"], game["commence_time"], now,
            float(game["home_win_prob"]), float(game["away_win_prob"]),
            game.get("home_cover_prob"), game.get("away_cover_prob"),
            game.get("over_prob"), game.get("under_prob"),
        )
        for game in games
    ]
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO game_predictions
                (game_id, home_team, away_team, commence_time, snapshotted_at,
                 home_win_prob, away_win_prob, home_cover_prob, away_cover_prob, over_prob, under_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


def reconcile_game_predictions(results_df: pd.DataFrame) -> int:
    if results_df.empty:
        return 0
    with contextlib.closing(_connect()) as conn, conn:
        unresolved = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 0", conn)
        if unresolved.empty:
            return 0

        merged = unresolved.merge(results_df, on="game_id", how="inner")
        resolved_count = 0
        for _, row in merged.iterrows():
            home_win = row["home_score"] > row["away_score"]
            predicted_home_win = row["home_win_prob"] >= row["away_win_prob"]
            moneyline_hit = int(predicted_home_win == home_win)
            cursor = conn.execute(
                """
                UPDATE game_predictions
                SET resolved = 1, actual_home_score = ?, actual_away_score = ?, moneyline_hit = ?
                WHERE game_id = ? AND resolved = 0
                """,
                (int(row["home_score"]), int(row["away_score"]), moneyline_hit, row["game_id"]),
            )
            resolved_count += cursor.rowcount
        return resolved_count


def get_track_record() -> dict:
    with contextlib.closing(_connect()) as conn, conn:
        resolved = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 1", conn)
    if resolved.empty:
        return {"n_resolved_games": 0, "pct_moneyline_correct": None}
    return {
        "n_resolved_games": int(len(resolved)),
        "pct_moneyline_correct": float(resolved["moneyline_hit"].mean()),
    }


def record_player_prop_predictions(props: list[dict]) -> int:
    if not props:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (prop["game_id"], prop["player_id"], prop["player_name"], prop["market"], float(prop["predicted_value"]), now)
        for prop in props
    ]
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO player_prop_predictions
                (game_id, player_id, player_name, market, predicted_value, snapshotted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


_MARKET_TO_STAT_COLUMN = {
    "anytime_td": None,
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
}


def reconcile_player_prop_predictions(player_stats_df: pd.DataFrame) -> int:
    if player_stats_df.empty:
        return 0
    with contextlib.closing(_connect()) as conn, conn:
        unresolved = pd.read_sql("SELECT * FROM player_prop_predictions WHERE resolved = 0", conn)
        if unresolved.empty:
            return 0

        merged = unresolved.merge(player_stats_df, on=["game_id", "player_id"], how="inner")
        resolved_count = 0
        for _, row in merged.iterrows():
            if row["market"] == "anytime_td":
                actual = float(
                    (row.get("rushing_tds", 0) or 0)
                    + (row.get("receiving_tds", 0) or 0)
                    + (row.get("passing_tds", 0) or 0)
                    > 0
                )
            else:
                stat_col = _MARKET_TO_STAT_COLUMN[row["market"]]
                if stat_col not in row or pd.isna(row[stat_col]):
                    continue
                actual = float(row[stat_col])
            cursor = conn.execute(
                """
                UPDATE player_prop_predictions
                SET resolved = 1, actual_value = ?
                WHERE game_id = ? AND player_id = ? AND market = ? AND resolved = 0
                """,
                (actual, row["game_id"], row["player_id"], row["market"]),
            )
            resolved_count += cursor.rowcount
        return resolved_count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_tracking_store.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cfb_predictor/tracking/__init__.py src/cfb_predictor/tracking/store.py tests/test_tracking_store.py
git commit -m "feat: SQLite snapshot-then-reconcile tracking store"
```

---

## Task 14: Odds — value bet detection (Shin de-vig)

**Files:**
- Create: `src/cfb_predictor/odds/__init__.py`
- Create: `src/cfb_predictor/odds/value_bets.py`
- Test: `tests/test_value_bets.py`

**Interfaces:**
- Produces: `value_bets.devig_h2h(home_price, away_price) -> dict | None`, `value_bets.devig_totals(over_price, under_price) -> dict | None`, `value_bets.build_value_bet_table(games_df: pd.DataFrame, odds_df: pd.DataFrame, predictions: dict[str, dict], edge_threshold: float = 0.05) -> pd.DataFrame` (adds `home_win_edge, away_win_edge, over_edge, under_edge, value_bet_flags` — `value_bet_flags` is a list with at most one entry, never a parlay).

Verbatim port of `nfl_predictor/odds/value_bets.py` — Shin de-vig math and same-bookmaker/same-point pairing are market-agnostic; nothing in this file references a sport.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_value_bets.py
import pandas as pd
import pytest

from cfb_predictor.odds import value_bets


def test_devig_h2h_removes_the_bookmaker_margin():
    result = value_bets.devig_h2h(home_price=1.91, away_price=1.91)

    assert result is not None
    assert result["home_win"] == pytest.approx(0.5, abs=0.01)
    assert result["home_win"] + result["away_win"] == pytest.approx(1.0, abs=1e-6)


def test_devig_totals_removes_the_bookmaker_margin():
    result = value_bets.devig_totals(over_price=1.91, under_price=1.91)

    assert result is not None
    assert result["over"] == pytest.approx(0.5, abs=0.01)


def test_build_value_bet_table_flags_a_positive_edge():
    games_df = pd.DataFrame(
        [{"game_id": "g1", "home_team": "Texas", "away_team": "Ohio State", "commence_time": "2025-08-30T16:00:00"}]
    )
    odds_df = pd.DataFrame(
        [
            {"event_id": "g1", "market": "h2h", "outcome_name": "Texas", "price": 2.20, "point": None, "bookmaker": "dk", "odds_fetched_at": pd.Timestamp.now(tz="UTC").isoformat()},
            {"event_id": "g1", "market": "h2h", "outcome_name": "Ohio State", "price": 1.75, "point": None, "bookmaker": "dk", "odds_fetched_at": pd.Timestamp.now(tz="UTC").isoformat()},
        ]
    )
    predictions = {"g1": {"home_win_prob": 0.60, "away_win_prob": 0.40}}

    table = value_bets.build_value_bet_table(games_df, odds_df, predictions)

    row = table.iloc[0]
    assert row["home_win_edge"] > 0
    assert "home_win" in row["value_bet_flags"]


def test_build_value_bet_table_recommends_only_the_largest_single_game_edge():
    games_df = pd.DataFrame(
        [{"game_id": "g1", "home_team": "Texas", "away_team": "Ohio State", "commence_time": "2025-08-30T16:00:00"}]
    )
    odds_df = pd.DataFrame(
        [
            {"event_id": "g1", "market": "h2h", "outcome_name": "Texas", "price": 2.20},
            {"event_id": "g1", "market": "h2h", "outcome_name": "Ohio State", "price": 1.75},
            {"event_id": "g1", "market": "totals", "outcome_name": "Over", "price": 2.20},
            {"event_id": "g1", "market": "totals", "outcome_name": "Under", "price": 1.75},
        ]
    )
    predictions = {
        "g1": {"home_win_prob": 0.60, "away_win_prob": 0.40, "over_prob": 0.70, "under_prob": 0.30}
    }

    table = value_bets.build_value_bet_table(games_df, odds_df, predictions)

    assert table.iloc[0]["value_bet_flags"] == ["over"]


def test_build_value_bet_table_only_pairs_h2h_prices_from_the_same_bookmaker():
    games_df = pd.DataFrame(
        [{"game_id": "g1", "home_team": "Texas", "away_team": "Ohio State", "commence_time": "2025-08-30T16:00:00"}]
    )
    odds_df = pd.DataFrame(
        [
            {"event_id": "g1", "market": "h2h", "outcome_name": "Texas", "price": 2.30, "bookmaker": "dk"},
            {"event_id": "g1", "market": "h2h", "outcome_name": "Ohio State", "price": 1.55, "bookmaker": "dk"},
            {"event_id": "g1", "market": "h2h", "outcome_name": "Texas", "price": 2.40, "bookmaker": "fd"},
            {"event_id": "g1", "market": "h2h", "outcome_name": "Ohio State", "price": 1.60, "bookmaker": "mgm"},
        ]
    )
    predictions = {"g1": {"home_win_prob": 0.55, "away_win_prob": 0.45}}

    table = value_bets.build_value_bet_table(games_df, odds_df, predictions)

    row = table.iloc[0]
    expected = value_bets.devig_h2h(2.30, 1.55)
    cross_book = value_bets.devig_h2h(2.40, 1.60)
    assert expected["home_win"] != pytest.approx(cross_book["home_win"], abs=1e-6)
    assert row["home_win_edge"] == pytest.approx(0.55 - expected["home_win"], abs=1e-6)


def test_build_value_bet_table_returns_no_edge_when_no_bookmaker_quotes_both_sides():
    games_df = pd.DataFrame(
        [{"game_id": "g1", "home_team": "Texas", "away_team": "Ohio State", "commence_time": "2025-08-30T16:00:00"}]
    )
    odds_df = pd.DataFrame(
        [
            {"event_id": "g1", "market": "h2h", "outcome_name": "Texas", "price": 2.30, "bookmaker": "dk"},
            {"event_id": "g1", "market": "h2h", "outcome_name": "Ohio State", "price": 1.60, "bookmaker": "fd"},
        ]
    )
    predictions = {"g1": {"home_win_prob": 0.55, "away_win_prob": 0.45}}

    table = value_bets.build_value_bet_table(games_df, odds_df, predictions)

    row = table.iloc[0]
    assert row["home_win_edge"] is None
    assert row["value_bet_flags"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_value_bets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.odds'`

- [ ] **Step 3: Write `src/cfb_predictor/odds/__init__.py` and `value_bets.py`**

```python
# src/cfb_predictor/odds/__init__.py
```

```python
"""Join model predictions to live odds and surface single-game value edges.
Verbatim port of nfl_predictor/odds/value_bets.py — Shin de-vig math is
market-agnostic."""

from __future__ import annotations

import math

import pandas as pd
from scipy.optimize import brentq

MAX_ODDS_AGE_SECONDS = 60 * 60


def _shin_two_way(price_a: float, price_b: float) -> tuple[float, float] | None:
    if not all(isinstance(price, (int, float)) and math.isfinite(price) and price > 1 for price in (price_a, price_b)):
        return None

    pi_a, pi_b = 1.0 / price_a, 1.0 / price_b
    overround = pi_a + pi_b
    if overround <= 1.0:
        return None

    def true_prob(pi: float, z: float) -> float:
        inside = z * z + 4 * (1 - z) * (pi * pi) / overround
        return (max(inside, 0.0) ** 0.5 - z) / (2 * (1 - z))

    def z_equation(z: float) -> float:
        return true_prob(pi_a, z) + true_prob(pi_b, z) - 1.0

    try:
        z = brentq(z_equation, 0.0, 0.2)
    except ValueError:
        return pi_a / overround, pi_b / overround

    p_a, p_b = true_prob(pi_a, z), true_prob(pi_b, z)
    total = p_a + p_b
    return p_a / total, p_b / total


def devig_h2h(home_price: float, away_price: float) -> dict | None:
    result = _shin_two_way(home_price, away_price)
    if result is None:
        return None
    home, away = result
    return {"home_win": home, "away_win": away}


def devig_totals(over_price: float, under_price: float) -> dict | None:
    result = _shin_two_way(over_price, under_price)
    if result is None:
        return None
    over, under = result
    return {"over": over, "under": under}


def _valid_price(price: object) -> bool:
    return (
        isinstance(price, (int, float))
        and not isinstance(price, bool)
        and pd.notna(price)
        and math.isfinite(price)
        and price > 1
    )


_NO_BOOKMAKER = "__no_bookmaker__"
_NO_POINT = "__no_point__"


def _grouping_series(odds_df: pd.DataFrame, column: str, sentinel: str) -> pd.Series:
    if column not in odds_df.columns:
        return pd.Series(sentinel, index=odds_df.index)
    return odds_df[column].fillna(sentinel)


def _best_same_book_pair(
    odds_df: pd.DataFrame,
    event_id: str,
    market: str,
    outcome_a: str,
    outcome_b: str,
    require_matching_point: bool = False,
) -> tuple[float | None, float | None]:
    required_columns = {"event_id", "market", "outcome_name", "price"}
    if odds_df is None or odds_df.empty or not required_columns.issubset(odds_df.columns):
        return None, None

    subset = odds_df[(odds_df["event_id"] == event_id) & (odds_df["market"] == market)]
    if subset.empty:
        return None, None

    subset = subset.assign(
        _bookmaker=_grouping_series(subset, "bookmaker", _NO_BOOKMAKER),
        _point=_grouping_series(subset, "point", _NO_POINT),
    )

    rows_a = subset[subset["outcome_name"] == outcome_a]
    rows_b = subset[subset["outcome_name"] == outcome_b]
    if rows_a.empty or rows_b.empty:
        return None, None

    join_keys = ["_bookmaker", "_point"] if require_matching_point else ["_bookmaker"]
    candidates = rows_a.merge(rows_b, on=join_keys, suffixes=("_a", "_b"))
    if candidates.empty:
        return None, None

    best_pair: tuple[float, float] | None = None
    best_overround: float | None = None
    for price_a, price_b in zip(candidates["price_a"], candidates["price_b"]):
        if not (_valid_price(price_a) and _valid_price(price_b)):
            continue
        overround = 1.0 / price_a + 1.0 / price_b
        if overround <= 1.0:
            continue
        if best_overround is None or overround < best_overround:
            best_overround = overround
            best_pair = (float(price_a), float(price_b))

    return best_pair if best_pair is not None else (None, None)


def _recommended_side(row: dict, edge_threshold: float) -> list[str]:
    candidates = [
        (side, row.get(f"{side}_edge"))
        for side in ("home_win", "away_win", "over", "under")
        if row.get(f"{side}_edge") is not None and row[f"{side}_edge"] > edge_threshold
    ]
    if not candidates:
        return []
    return [max(candidates, key=lambda candidate: candidate[1])[0]]


def build_value_bet_table(
    games_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    predictions: dict[str, dict],
    edge_threshold: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for _, game in games_df.iterrows():
        game_id, home, away = game["game_id"], game["home_team"], game["away_team"]
        prediction = predictions.get(game_id, {})
        row = {
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "commence_time": game["commence_time"],
            **prediction,
        }

        home_price, away_price = _best_same_book_pair(odds_df, game_id, "h2h", home, away)
        h2h_implied = (
            devig_h2h(home_price, away_price) if home_price is not None and away_price is not None else None
        )

        over_price, under_price = _best_same_book_pair(
            odds_df, game_id, "totals", "Over", "Under", require_matching_point=True
        )
        totals_implied = (
            devig_totals(over_price, under_price) if over_price is not None and under_price is not None else None
        )

        row["home_win_edge"] = prediction.get("home_win_prob", 0) - h2h_implied["home_win"] if h2h_implied else None
        row["away_win_edge"] = prediction.get("away_win_prob", 0) - h2h_implied["away_win"] if h2h_implied else None
        row["over_edge"] = prediction["over_prob"] - totals_implied["over"] if totals_implied and "over_prob" in prediction else None
        row["under_edge"] = prediction["under_prob"] - totals_implied["under"] if totals_implied and "under_prob" in prediction else None
        row["value_bet_flags"] = _recommended_side(row, edge_threshold)
        rows.append(row)

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_value_bets.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cfb_predictor/odds/__init__.py src/cfb_predictor/odds/value_bets.py tests/test_value_bets.py
git commit -m "feat: Shin de-vig value bet detection"
```

---

## Task 15: API — schemas and routes

**Files:**
- Create: `src/cfb_predictor/api/__init__.py`
- Create: `src/cfb_predictor/api/schemas.py`
- Create: `src/cfb_predictor/api/routes.py`
- Test: `tests/test_api_routes.py`

**Interfaces:**
- Consumes: `data.games.*`, `data.player_stats.fetch_weekly_player_stats`, `data.odds_api.fetch_game_odds`, `features.build.build_features_for_game`, `features.player_usage.build_features_for_player`, `models.game_outcome.predict_margin_elo/margin_to_probabilities/ELO_POINTS_PER_RATING_POINT`, `models.manifest.load_models/train_all`, `models.player_props.predict_props`, `tracking.store.get_track_record/record_game_predictions/reconcile_game_predictions`.
- Produces: `router` (a `fastapi.APIRouter` mounted at `/api`), `_load_models_cached() -> dict`, `_predict_game_from_models(models, home, away, games_df, spread_line=None, total_line=None) -> dict`, `_lines_for_game(odds_df, home_team, away_team) -> tuple[float | None, float | None]`, `background_tracking_tick(season: int, week: int) -> None`, `warm_caches() -> None`.

One genuinely CFB-specific change beyond a `schedules` → `games` rename: **CFBD's `Game` model carries no `spread_line`/`total_line` fields** (unlike `nfl_data_py`'s schedule frame, which nflverse already merges Vegas lines into). `_lines_for_game` below pulls them from The Odds API's own `spreads`/`totals` markets instead, matched by team name — the same team-name-literal matching `odds/value_bets.py` already relies on for `h2h` (not fixed here; a real team-name-normalization layer between CFBD's names and The Odds API's names is out of scope for this plan, consistent with that pre-existing characteristic of `NFL_Predictor`'s own shipped `value_bets.py`). `api/schemas.py` itself needs no changes at all — its Pydantic models describe probabilities and scores generically.

- [ ] **Step 1: Write `src/cfb_predictor/api/__init__.py` and `schemas.py`**

```python
# src/cfb_predictor/api/__init__.py
```

```python
"""schemas.py — Pydantic response models for the API. Verbatim port of
nfl_predictor/api/schemas.py — every field here is sport-agnostic."""

from __future__ import annotations

from pydantic import BaseModel


class GameSummary(BaseModel):
    game_id: str
    season: int
    week: int
    gameday: str
    home_team: str
    away_team: str
    home_score: int | None = None
    away_score: int | None = None


class GamePrediction(BaseModel):
    home_win_prob: float
    away_win_prob: float
    home_cover_prob: float | None = None
    away_cover_prob: float | None = None
    over_prob: float | None = None
    under_prob: float | None = None


class TrackRecord(BaseModel):
    n_resolved_games: int
    pct_moneyline_correct: float | None = None


class RetrainResponse(BaseModel):
    trained_at: str
    chosen_candidate: str
```

- [ ] **Step 2: Write the failing tests for `routes.py`**

```python
# tests/test_api_routes.py
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cfb_predictor.api.main import app
from cfb_predictor.api import routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        routes.games_data, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "401520145", "season": season, "week": week,
              "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State",
              "home_score": None, "away_score": None}]
        ),
    )
    monkeypatch.setattr(
        routes.games_data, "load_training_data",
        lambda seasons: pd.DataFrame(
            [{"game_id": "g0", "season": seasons[0], "week": 1, "gameday": "2025-08-01",
              "home_team": "Texas", "away_team": "Ohio State", "home_score": 24, "away_score": 17}]
        ),
    )
    monkeypatch.setattr(routes.odds_api, "fetch_game_odds", lambda: pd.DataFrame())
    monkeypatch.setattr(
        routes, "_load_models_cached",
        lambda: {
            "game_outcome_model": None, "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
            "total_model": None, "player_models": {"feature_cols": [], "anytime_td": None},
            "feature_cols": [], "player_feature_cols": [],
        },
    )
    monkeypatch.setattr(
        routes, "_predict_game_from_models",
        lambda models, home, away, games_df, spread_line=None, total_line=None: {
            "home_win_prob": 0.4, "away_win_prob": 0.6, "home_cover_prob": 0.45, "away_cover_prob": 0.55,
            "over_prob": 0.52, "under_prob": 0.48,
        },
    )
    monkeypatch.setattr(routes.store, "get_track_record", lambda: {"n_resolved_games": 0, "pct_moneyline_correct": None})
    return TestClient(app)


def test_get_games_returns_week_slate(client):
    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["game_id"] == "401520145"


def test_get_game_prediction(client):
    response = client.get("/api/games/2025/1/401520145/prediction")

    assert response.status_code == 200
    body = response.json()
    assert body["home_win_prob"] == 0.4


def test_get_game_prediction_404s_for_unknown_game(client):
    response = client.get("/api/games/2025/1/nonexistent/prediction")

    assert response.status_code == 404


def test_get_track_record(client):
    response = client.get("/api/track-record")

    assert response.status_code == 200
    assert response.json()["n_resolved_games"] == 0


def test_get_games_handles_nan_scores_for_unplayed_games(client, monkeypatch):
    monkeypatch.setattr(
        routes.games_data, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "401520145", "season": season, "week": week,
              "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State",
              "home_score": float("nan"), "away_score": float("nan")}]
        ),
    )

    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["home_score"] is None
    assert body[0]["away_score"] is None


def test_lines_for_game_reads_spreads_and_totals_from_odds_api():
    odds_df = pd.DataFrame(
        [
            {"home_team": "Texas", "away_team": "Ohio State", "market": "spreads", "outcome_name": "Texas", "point": -3.5},
            {"home_team": "Texas", "away_team": "Ohio State", "market": "totals", "outcome_name": "Over", "point": 51.5},
        ]
    )

    spread_line, total_line = routes._lines_for_game(odds_df, "Texas", "Ohio State")

    assert spread_line == -3.5
    assert total_line == 51.5


def test_lines_for_game_returns_none_when_odds_missing():
    spread_line, total_line = routes._lines_for_game(pd.DataFrame(), "Texas", "Ohio State")

    assert spread_line is None
    assert total_line is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_api_routes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_predictor.api'`

- [ ] **Step 4: Write `src/cfb_predictor/api/routes.py`**

```python
"""routes.py — game/player prop/track-record endpoints. Thin HTTP layer
over data/features/models/tracking, near-verbatim port of nfl_predictor's
own api/routes.py. Two CFB-specific adaptations: (1) schedules -> games
(data.games); (2) CFBD's Game model carries no spread_line/total_line, so
_lines_for_game pulls them from The Odds API's own spreads/totals markets
instead, matched by team name.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..config import CURRENT_SEASON, PUBLIC_MODE
from ..data import games as games_data
from ..data import odds_api, player_stats
from ..features import build as feature_build
from ..features import player_usage
from ..models import game_outcome, manifest, player_props
from ..odds import value_bets
from ..tracking import store

router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_models_cached() -> dict:
    return manifest.load_models()


def _lines_for_game(odds_df: pd.DataFrame, home_team: str, away_team: str) -> tuple[float | None, float | None]:
    """CFBD's Game model (unlike nflverse's schedule frame) carries no
    market spread/total -- pull them from The Odds API's own spreads/totals
    markets instead, matched by team name. Best-effort: returns (None, None)
    for either line the odds feed doesn't have for this matchup (a common
    case outside Power-conference games, per the design spec)."""
    if odds_df is None or odds_df.empty:
        return None, None
    matches = odds_df[(odds_df["home_team"] == home_team) & (odds_df["away_team"] == away_team)]
    if matches.empty:
        return None, None

    spread_line = None
    spread_rows = matches[(matches["market"] == "spreads") & (matches["outcome_name"] == home_team)]
    if not spread_rows.empty:
        spread_line = float(spread_rows.iloc[0]["point"])

    total_line = None
    total_rows = matches[(matches["market"] == "totals") & (matches["outcome_name"] == "Over")]
    if not total_rows.empty:
        total_line = float(total_rows.iloc[0]["point"])

    return spread_line, total_line


def _predict_game_from_models(
    models: dict, home: str, away: str, games_df: pd.DataFrame,
    spread_line: float | None = None, total_line: float | None = None,
) -> dict:
    feature_row = feature_build.build_features_for_game(home, away, games_df)
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0)

    if models["chosen_candidate"] == "elo":
        predicted_margin = game_outcome.predict_margin_elo(
            {"points_per_rating_point": game_outcome.ELO_POINTS_PER_RATING_POINT},
            feature_row["rating_diff"], feature_row["home_rest_days"], feature_row["away_rest_days"],
        )
    else:
        predicted_margin = float(models["game_outcome_model"].predict(X.to_numpy().reshape(1, -1))[0])

    predicted_total = float(models["total_model"].predict(X.to_numpy().reshape(1, -1))[0])

    return game_outcome.margin_to_probabilities(
        predicted_margin, models["sigma"],
        spread_line=spread_line, total_line=total_line,
        predicted_total=predicted_total, total_sigma=models["total_sigma"],
    )


def _load_game_history(season: int) -> pd.DataFrame:
    """Historical + requested-season game data for feature building. A
    still-in-progress CURRENT_SEASON needs the always-refetch
    fetch_current_season_partial() instead of load_training_data()'s
    per-season cache, or every request that season would return the same
    stale games-so-far snapshot forever."""
    history_seasons = games_data.default_completed_seasons(n=8)
    if season == CURRENT_SEASON:
        historical_df = games_data.load_training_data(history_seasons)
        current_df = games_data.fetch_current_season_partial()
        return pd.concat([historical_df, current_df], ignore_index=True)
    return games_data.load_training_data(history_seasons + [season])


def _load_player_history(season: int) -> pd.DataFrame:
    """Historical + requested-season player stats for feature building.
    player_stats.fetch_weekly_player_stats needs each season's games_df to
    attach week numbers to CFBD's raw per-game box scores (see
    data/player_stats.py), so this loads game history first."""
    history_seasons = games_data.default_completed_seasons(n=8)
    if season == CURRENT_SEASON:
        historical_games = games_data.load_training_data(history_seasons)
        current_games = games_data.fetch_current_season_partial()
        games_df = pd.concat([historical_games, current_games], ignore_index=True)
        historical_df = player_stats.fetch_weekly_player_stats(history_seasons, games_df)
        current_df = player_stats.fetch_weekly_player_stats([season], games_df, force_refresh=True)
        return pd.concat([historical_df, current_df], ignore_index=True)
    all_seasons = history_seasons + [season]
    games_df = games_data.load_training_data(all_seasons)
    return player_stats.fetch_weekly_player_stats(all_seasons, games_df)


@router.get("/games")
def get_games(season: int, week: int):
    games = games_data.fetch_upcoming_games(season, week)
    games = games.astype(object).where(pd.notna(games), None)
    return games.to_dict("records")


@router.get("/games/{season}/{week}/{game_id}/prediction")
def get_game_prediction(season: int, week: int, game_id: str):
    games = games_data.fetch_upcoming_games(season, week)
    matches = games[games["game_id"] == game_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    game = matches.iloc[0]

    models = _load_models_cached()
    history = _load_game_history(season)
    odds_df = odds_api.fetch_game_odds()
    spread_line, total_line = _lines_for_game(odds_df, game["home_team"], game["away_team"])
    prediction = _predict_game_from_models(
        models, game["home_team"], game["away_team"], history,
        spread_line=spread_line, total_line=total_line,
    )
    return prediction


@router.get("/players/{season}/{week}/props")
def get_player_props(season: int, week: int):
    models = _load_models_cached()
    player_history = _load_player_history(season)

    latest_players = (
        player_history[player_history["season"] == season]
        [["player_id", "player_name", "position", "recent_team"]]
        .drop_duplicates("player_id")
    )
    results = []
    for _, player in latest_players.iterrows():
        feature_row = player_usage.build_features_for_player(player["player_id"], player_history)
        if feature_row is None:
            continue
        props = player_props.predict_props(models["player_models"], feature_row, position=player["position"])
        results.append({"player_id": player["player_id"], "player_name": player["player_name"], **props})
    return results


@router.get("/track-record")
def get_track_record():
    return store.get_track_record()


@router.post("/retrain")
def retrain():
    if PUBLIC_MODE:
        raise HTTPException(status_code=403, detail="Retraining is disabled in public mode")
    result = manifest.train_all()
    _load_models_cached.cache_clear()
    return {"trained_at": result["trained_at"], "chosen_candidate": result["chosen_candidate"]}


def background_tracking_tick(season: int, week: int) -> None:
    """Snapshot this week's upcoming-game predictions, then reconcile
    anything now resolved. Called on a timer from api/main.py's lifespan."""
    games = games_data.fetch_upcoming_games(season, week)
    if not games.empty:
        models = _load_models_cached()
        history = _load_game_history(season)
        odds_df = odds_api.fetch_game_odds()
        predictions = []
        for _, game in games.iterrows():
            try:
                spread_line, total_line = _lines_for_game(odds_df, game["home_team"], game["away_team"])
                pred = _predict_game_from_models(
                    models, game["home_team"], game["away_team"], history,
                    spread_line=spread_line, total_line=total_line,
                )
                predictions.append(
                    {
                        "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                        "commence_time": str(game["gameday"]), **pred,
                    }
                )
            except Exception:
                logger.exception("prediction failed for game_id=%s", game.get("game_id"))
                continue
        store.record_game_predictions(predictions)

    completed = games_data.fetch_current_season_partial()
    store.reconcile_game_predictions(completed[["game_id", "home_score", "away_score"]])


def warm_caches() -> None:
    """Pre-fetch schedules/player stats/odds so the first real request
    after startup isn't slow — best-effort, never raises."""
    try:
        seasons = games_data.default_completed_seasons(n=8)
        games_df = games_data.fetch_schedules(seasons)
        player_stats.fetch_weekly_player_stats(seasons, games_df)
        odds_api.fetch_game_odds()
    except Exception:
        pass
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_api_routes.py -v`
Expected: PASS (7 tests) — this task depends on Task 16's `api/main.py` existing too, since the test imports `cfb_predictor.api.main`; write Task 16 before running this step if executing tasks strictly in order.

- [ ] **Step 6: Commit**

```bash
git add src/cfb_predictor/api/__init__.py src/cfb_predictor/api/schemas.py src/cfb_predictor/api/routes.py tests/test_api_routes.py
git commit -m "feat: API schemas and routes"
```

---

## Task 16: API — FastAPI app entry point

**Files:**
- Create: `src/cfb_predictor/api/main.py`

**Interfaces:**
- Consumes: `config.FRONTEND_DIST_DIR`, `api.routes.router/warm_caches/background_tracking_tick`.
- Produces: `main.app` (the FastAPI instance Task 15's tests import as `cfb_predictor.api.main.app`, and `uvicorn cfb_predictor.api.main:app` serves).

Near-verbatim port of `nfl_predictor/api/main.py` — same lifespan/background-tracking-tick pattern and hardened per-tick `try/except`, carried over from day one per the design spec (rather than re-discovered after a bug report, the way NFL_Predictor's own review found it). Only change: the title string and `_current_season_and_week`'s calendar anchor moves from September 1 (NFL's opening week) to August 20 (FBS regular-season openers typically start the last week of August, a week or so before the NFL's).

- [ ] **Step 1: Write `src/cfb_predictor/api/main.py`**

```python
"""main.py — FastAPI app entry point.

Run with:
    PYTHONPATH=$(pwd)/src uvicorn cfb_predictor.api.main:app --reload --host 0.0.0.0 --port 8003
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import FRONTEND_DIST_DIR
from .routes import router, warm_caches, background_tracking_tick

logger = logging.getLogger(__name__)

_TRACKING_INTERVAL_SECONDS = 300


def _current_season_and_week() -> tuple[int, int]:
    """A simple calendar-based estimate: CFB seasons are named by the year
    they start (late August) and run through the January bowl season --
    good enough for the background tracking tick to know which week to
    snapshot without hardcoding a schedule. Off by a week or two around the
    very start/end of a season doesn't matter here since
    fetch_upcoming_games just returns an empty frame for a week with
    nothing unplayed."""
    today = date.today()
    season = today.year if today.month >= 2 else today.year - 1
    week = max(1, min(20, ((today - date(season, 8, 20)).days // 7) + 1))
    return season, week


async def _tracking_loop():
    while True:
        await asyncio.sleep(_TRACKING_INTERVAL_SECONDS)
        try:
            season, week = _current_season_and_week()
            await asyncio.to_thread(background_tracking_tick, season, week)
        except Exception:
            logger.exception("background_tracking_tick failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    asyncio.create_task(asyncio.to_thread(warm_caches))
    tracking_task = asyncio.create_task(_tracking_loop())
    yield
    tracking_task.cancel()


app = FastAPI(title="CFB Predictor API", lifespan=lifespan)

# Same wide-open CORS as the other three sibling projects: this server is
# only ever reached over a private network or this project's own public
# read-only deployment, never with a login to protect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if FRONTEND_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
else:

    @app.get("/")
    def root():
        return {"status": "ok", "docs": "/docs"}
```

- [ ] **Step 2: Run Task 15's route tests now that `main.py` exists**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_api_routes.py -v`
Expected: PASS (7 tests)

- [ ] **Step 3: Write the failing test for background tracking**

```python
# tests/test_background_tracking.py
import pandas as pd
import pytest

from cfb_predictor.api import routes


def test_background_tracking_tick_records_and_reconciles(monkeypatch):
    calls = {"recorded": 0, "reconciled": 0}

    monkeypatch.setattr(
        routes.games_data, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "g1", "home_team": "Texas", "away_team": "Ohio State", "gameday": "2025-08-30"}]
        ),
    )
    monkeypatch.setattr(
        routes.games_data, "load_training_data",
        lambda seasons: pd.DataFrame(
            [{"game_id": "g0", "season": seasons[0], "week": 1, "gameday": "2025-08-01",
              "home_team": "Texas", "away_team": "Ohio State", "home_score": 24, "away_score": 17}]
        ),
    )
    monkeypatch.setattr(routes.odds_api, "fetch_game_odds", lambda: pd.DataFrame())
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {
        "game_outcome_model": None, "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
        "total_model": None, "player_models": {"feature_cols": [], "anytime_td": None},
        "feature_cols": [], "player_feature_cols": [],
    })
    monkeypatch.setattr(routes, "_predict_game_from_models", lambda *a, **k: {
        "home_win_prob": 0.4, "away_win_prob": 0.6, "home_cover_prob": 0.45,
        "away_cover_prob": 0.55, "over_prob": 0.52, "under_prob": 0.48,
    })
    monkeypatch.setattr(
        routes.store, "record_game_predictions",
        lambda games: calls.__setitem__("recorded", calls["recorded"] + len(games)) or len(games),
    )
    monkeypatch.setattr(
        routes.store, "reconcile_game_predictions",
        lambda results_df: calls.__setitem__("reconciled", calls["reconciled"] + 1) or 0,
    )
    monkeypatch.setattr(routes.games_data, "fetch_current_season_partial", lambda: pd.DataFrame(columns=["game_id", "home_score", "away_score"]))

    routes.background_tracking_tick(season=2025, week=1)

    assert calls["recorded"] == 1
    assert calls["reconciled"] == 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_background_tracking.py -v`
Expected: PASS

- [ ] **Step 5: Run the full backend suite**

Run: `PYTHONPATH=$(pwd)/src pytest tests/ -v`
Expected: PASS (all tests so far)

- [ ] **Step 6: Commit**

```bash
git add src/cfb_predictor/api/main.py tests/test_background_tracking.py
git commit -m "feat: FastAPI app entry point with background tracking loop"
```

---

## Task 17: Train the first real manifest and verify end-to-end

**Files:**
- Modify: none (verification task — runs the real pipeline against real CFBD/Odds API data for the first time)

**Rate-cap discipline for this task:** every test so far (Tasks 1-16) monkeypatches `_import_games`/`_import_fbs_teams`/`_import_player_game_stats`/`_fetch_raw_odds` and never touches the network. This task makes the **one** real training run this plan performs. With `DEFAULT_TRAIN_SEASONS = 8`, `manifest.train_all()` calls CFBD roughly 24 times total (`get_games` + `get_player_game_stats` + `get_fbs_teams`, once each per season, 8 seasons) — comfortably under the 1,000/month cap, but run it once and let the per-season parquet caches under `data/cache/` absorb any re-runs while debugging (only newly-added or `force_refresh`-ed seasons trigger a fresh call).

- [ ] **Step 1: Install the package and dependencies, and spot-check the `cfbd` package shape**

```bash
cd /Users/sigey/Documents/Projects/CFB_Predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export PYTHONPATH=$(pwd)/src
pip show cfbd
```

Compare the installed `cfbd` version's own `Configuration`/`ApiClient` construction against this plan's `_cfbd_configuration()` helpers in `data/games.py` and `data/player_stats.py` (see this plan's "Implementation notes" section for the specific ambiguity to check — `Configuration(access_token=...)` vs. the older `configuration.api_key['Authorization'] = ...` style). Adjust those two functions if the installed version differs; nothing else in this plan depends on the exact construction syntax.

- [ ] **Step 2: Set the CFBD and Odds API keys**

```bash
cp .env.example .env
# Edit .env and set CFBD_API_KEY (from collegefootballdata.com/key) and
# ODDS_API_KEY (the existing key already used by PL_Predictor/NFL_Predictor).
```

- [ ] **Step 3: Run the full test suite against the real environment**

Run: `pytest tests/ -v`
Expected: PASS (all tests — these use monkeypatched data, so this doesn't hit the network yet)

- [ ] **Step 4: Train the real manifest against live CFBD data**

Run: `python -m cfb_predictor.models.manifest`

Expected: prints training progress, writes `models/manifest.json`, `models/game_outcome_model.pkl`, `models/total_points_model.pkl`, `models/anytime_td_model.pkl`, and the yardage model files. This is the first real network call to CFBD — if it fails, check the installed `cfbd` version's actual method signatures against `data/games.py`'s `_import_games`/`_import_fbs_teams` and `data/player_stats.py`'s `_import_player_game_stats`, and check `KEEP_COLUMNS`/`TEAM_KEEP_COLUMNS`/`_CATEGORY_TYPE_TO_COLUMN` in each file against the real response shape (CFBD's box-score category/type names are a documented but occasionally-revised part of their API).

- [ ] **Step 5: Sanity-check the manifest**

```bash
python -c "import json; m = json.load(open('models/manifest.json')); print(m['chosen_candidate'], m['candidate_scores'])"
```

Expected: a real `chosen_candidate` and three finite `candidate_scores` (elo/ridge/xgb), confirming the walk-forward race actually ran against real historical FBS data.

- [ ] **Step 6: Start the API and confirm a real prediction**

```bash
uvicorn cfb_predictor.api.main:app --host 0.0.0.0 --port 8003 &
sleep 2
curl "http://localhost:8003/api/games?season=2026&week=1"
```

Expected: a JSON list of real week-1 2026 FBS games (or the nearest upcoming week if week 1 has already started/finished by the time this runs).

- [ ] **Step 7: Empirically check Odds API coverage for CFB** (per the design spec's "open items to confirm during implementation")

```bash
curl "http://localhost:8003/api/games/2026/1/<a real game_id from Step 6>/prediction"
```

Note whether `home_cover_prob`/`over_prob` are present (a quoted spread/total line was found via `_lines_for_game`) or absent (no line for that matchup) for a few different games — Power-conference vs. non-Power-conference matchups are expected to differ here per the spec. This is an observation, not a blocking check; `_lines_for_game`/`value_bets.py` already degrade gracefully either way.

- [ ] **Step 8: No commit needed** — this task verifies the pipeline works against real data; `models/*.pkl`/`manifest.json`/`data/cache/*` are gitignored (Task 1's `.gitignore`).

---

## Task 18: Frontend — scaffold, API client, and types

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tsconfig.app.json`, `frontend/tsconfig.node.json`, `frontend/index.html`
- Create: `frontend/src/main.tsx`, `frontend/src/types.ts`, `frontend/src/api/client.ts`, `frontend/src/index.css`

**Interfaces:**
- Produces: `types.GameSummary`, `types.GamePrediction`, `types.PlayerPropPrediction`, `types.TrackRecord`, `types.RetrainResponse`; `api.games`, `api.gamePrediction`, `api.playerProps`, `api.trackRecord`, `api.retrain` from `api/client.ts`.

Verbatim port of `nfl_predictor`'s frontend scaffold — none of these files (types, API client, global CSS) contain any NFL-specific text or field names; the field names already match this plan's `api/schemas.py` (Task 15) exactly. The only real change is `vite.config.ts`'s dev-proxy target port (`8003`, matching Task 16's `api/main.py`).

- [ ] **Step 1: Scaffold with Vite**

```bash
cd /Users/sigey/Documents/Projects/CFB_Predictor
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

- [ ] **Step 2: Write `frontend/vite.config.ts`**

```typescript
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8003",
    },
  },
})
```

- [ ] **Step 3: Write `frontend/src/types.ts`**

```typescript
export interface GameSummary {
  game_id: string;
  season: number;
  week: number;
  gameday: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
}

export interface GamePrediction {
  home_win_prob: number;
  away_win_prob: number;
  home_cover_prob: number | null;
  away_cover_prob: number | null;
  over_prob: number | null;
  under_prob: number | null;
}

export interface PlayerPropPrediction {
  player_id: string;
  player_name: string;
  anytime_td_prob: number;
  passing_yards?: number;
  rushing_yards?: number;
  receiving_yards?: number;
}

export interface TrackRecord {
  n_resolved_games: number;
  pct_moneyline_correct: number | null;
}

export interface RetrainResponse {
  trained_at: string;
  chosen_candidate: string;
}
```

- [ ] **Step 4: Write `frontend/src/api/client.ts`**

```typescript
import type {
  GamePrediction,
  GameSummary,
  PlayerPropPrediction,
  RetrainResponse,
  TrackRecord,
} from "../types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function post<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  games: (season: number, week: number) => get<GameSummary[]>(`/games?season=${season}&week=${week}`),
  gamePrediction: (season: number, week: number, gameId: string) =>
    get<GamePrediction>(`/games/${season}/${week}/${gameId}/prediction`),
  playerProps: (season: number, week: number) => get<PlayerPropPrediction[]>(`/players/${season}/${week}/props`),
  trackRecord: () => get<TrackRecord>("/track-record"),
  retrain: () => post<RetrainResponse>("/retrain"),
};
```

- [ ] **Step 5: Write `frontend/src/main.tsx`**

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 6: Write `frontend/src/index.css`**

```css
:root {
  --text: #6b6375;
  --text-h: #08060d;
  --bg: #fff;
  --border: #e5e4e7;
  --code-bg: #f4f3ec;
  --accent: #aa3bff;
  --accent-bg: rgba(170, 59, 255, 0.1);
  --accent-border: rgba(170, 59, 255, 0.5);
  --social-bg: rgba(244, 243, 236, 0.5);
  --shadow:
    rgba(0, 0, 0, 0.1) 0 10px 15px -3px, rgba(0, 0, 0, 0.05) 0 4px 6px -2px;

  --sans: system-ui, 'Segoe UI', Roboto, sans-serif;
  --heading: system-ui, 'Segoe UI', Roboto, sans-serif;
  --mono: ui-monospace, Consolas, monospace;

  font: 18px/145% var(--sans);
  letter-spacing: 0.18px;
  color-scheme: light dark;
  color: var(--text);
  background: var(--bg);
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;

  @media (max-width: 1024px) {
    font-size: 16px;
  }
}

@media (prefers-color-scheme: dark) {
  :root {
    --text: #9ca3af;
    --text-h: #f3f4f6;
    --bg: #16171d;
    --border: #2e303a;
    --code-bg: #1f2028;
    --accent: #c084fc;
    --accent-bg: rgba(192, 132, 252, 0.15);
    --accent-border: rgba(192, 132, 252, 0.5);
    --social-bg: rgba(47, 48, 58, 0.5);
    --shadow:
      rgba(0, 0, 0, 0.4) 0 10px 15px -3px, rgba(0, 0, 0, 0.25) 0 4px 6px -2px;
  }

  #social .button-icon {
    filter: invert(1) brightness(2);
  }
}

#root {
  width: 1126px;
  max-width: 100%;
  margin: 0 auto;
  text-align: center;
  border-inline: 1px solid var(--border);
  min-height: 100svh;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
}

body {
  margin: 0;
}

h1,
h2 {
  font-family: var(--heading);
  font-weight: 500;
  color: var(--text-h);
}

h1 {
  font-size: 56px;
  letter-spacing: -1.68px;
  margin: 32px 0;
  @media (max-width: 1024px) {
    font-size: 36px;
    margin: 20px 0;
  }
}
h2 {
  font-size: 24px;
  line-height: 118%;
  letter-spacing: -0.24px;
  margin: 0 0 8px;
  @media (max-width: 1024px) {
    font-size: 20px;
  }
}
p {
  margin: 0;
}

code,
.counter {
  font-family: var(--mono);
  display: inline-flex;
  border-radius: 4px;
  color: var(--text-h);
}

code {
  font-size: 15px;
  line-height: 135%;
  padding: 4px 8px;
  background: var(--code-bg);
}
```

- [ ] **Step 7: Verify the scaffold builds**

Run (from `frontend/`): `npm run build`
Expected: builds successfully with no TypeScript errors (the default Vite template's `App.tsx` is still in place at this point — Task 19 replaces it).

- [ ] **Step 8: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vite.config.ts frontend/tsconfig*.json frontend/index.html frontend/src/main.tsx frontend/src/types.ts frontend/src/api/client.ts frontend/src/index.css
git commit -m "feat: frontend scaffold with API client and types"
```

---

## Task 19: Frontend — Games page

**Files:**
- Create: `frontend/src/pages/GamesPage.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `api.games`, `api.gamePrediction` from `api/client.ts`; `GameSummary`, `GamePrediction` from `types.ts`.

Verbatim port of `nfl_predictor/frontend/src/pages/GamesPage.tsx` — no NFL-specific text anywhere in the component (team names render generically from whatever `GameSummary.home_team`/`away_team` contain).

- [ ] **Step 1: Write `frontend/src/pages/GamesPage.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { GamePrediction, GameSummary } from "../types";

export function GamesPage() {
  const [season, setSeason] = useState(2026);
  const [week, setWeek] = useState(1);
  const [games, setGames] = useState<GameSummary[]>([]);
  const [predictions, setPredictions] = useState<Record<string, GamePrediction>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api
      .games(season, week)
      .then(async (fetchedGames) => {
        setGames(fetchedGames);
        const entries = await Promise.all(
          fetchedGames.map(async (g) => {
            try {
              const prediction = await api.gamePrediction(season, week, g.game_id);
              return [g.game_id, prediction] as const;
            } catch {
              return null;
            }
          }),
        );
        setPredictions(Object.fromEntries(entries.filter((e): e is [string, GamePrediction] => e !== null)));
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [season, week]);

  return (
    <div>
      <h1>Week {week} Games</h1>
      <div>
        <label>
          Season:{" "}
          <input type="number" value={season} onChange={(e) => setSeason(Number(e.target.value))} />
        </label>
        <label>
          Week:{" "}
          <input type="number" min={1} max={20} value={week} onChange={(e) => setWeek(Number(e.target.value))} />
        </label>
      </div>
      {loading && <p>Loading…</p>}
      {error && <p role="alert">{error}</p>}
      <ul>
        {games.map((game) => {
          const prediction = predictions[game.game_id];
          return (
            <li key={game.game_id}>
              <strong>{game.away_team} @ {game.home_team}</strong> — {new Date(game.gameday).toLocaleDateString()}
              {prediction && (
                <span>
                  {" "}— Home win {Math.round(prediction.home_win_prob * 100)}%
                  {prediction.over_prob != null && ` · Over ${Math.round(prediction.over_prob * 100)}%`}
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
```

- [ ] **Step 2: Wire it into `frontend/src/App.tsx`**

```tsx
import { GamesPage } from "./pages/GamesPage";

function App() {
  return (
    <div>
      <GamesPage />
    </div>
  );
}

export default App;
```

- [ ] **Step 3: Verify the build succeeds**

Run (from `frontend/`): `npm run build`
Expected: no TypeScript errors.

- [ ] **Step 4: Manual browser check**

Run (from `frontend/`): `npm run dev`, with the backend running (`uvicorn cfb_predictor.api.main:app --port 8003` in another shell). Open the printed local URL and confirm the week's games render with win probabilities once real training data has been fetched (Task 17).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/GamesPage.tsx frontend/src/App.tsx
git commit -m "feat: Games page showing weekly slate and win/total probabilities"
```

---

## Task 20: Frontend — Player Props page

**Files:**
- Create: `frontend/src/pages/PlayerPropsPage.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `api.playerProps` from `api/client.ts`; `PlayerPropPrediction` from `types.ts`.

Verbatim port of `nfl_predictor/frontend/src/pages/PlayerPropsPage.tsx` — the four prop markets (anytime TD, passing/rushing/receiving yards) are exactly this project's v1 scope too (per the design spec), so no column changes are needed.

- [ ] **Step 1: Write `frontend/src/pages/PlayerPropsPage.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { PlayerPropPrediction } from "../types";

export function PlayerPropsPage({ season, week }: { season: number; week: number }) {
  const [props, setProps] = useState<PlayerPropPrediction[]>([]);
  const [sortBy, setSortBy] = useState<"anytime_td_prob" | "rushing_yards" | "receiving_yards" | "passing_yards">(
    "anytime_td_prob",
  );

  useEffect(() => {
    api.playerProps(season, week).then(setProps);
  }, [season, week]);

  const sorted = [...props].sort((a, b) => (b[sortBy] ?? 0) - (a[sortBy] ?? 0));

  return (
    <div>
      <h1>Player Props — Week {week}</h1>
      <label>
        Sort by:{" "}
        <select value={sortBy} onChange={(e) => setSortBy(e.target.value as typeof sortBy)}>
          <option value="anytime_td_prob">Anytime TD</option>
          <option value="passing_yards">Passing Yards</option>
          <option value="rushing_yards">Rushing Yards</option>
          <option value="receiving_yards">Receiving Yards</option>
        </select>
      </label>
      <table>
        <thead>
          <tr>
            <th>Player</th>
            <th>Anytime TD</th>
            <th>Passing Yds</th>
            <th>Rushing Yds</th>
            <th>Receiving Yds</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((player) => (
            <tr key={player.player_id}>
              <td>{player.player_name}</td>
              <td>{Math.round(player.anytime_td_prob * 100)}%</td>
              <td>{player.passing_yards != null ? Math.round(player.passing_yards) : "—"}</td>
              <td>{player.rushing_yards != null ? Math.round(player.rushing_yards) : "—"}</td>
              <td>{player.receiving_yards != null ? Math.round(player.receiving_yards) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 2: Wire it into `frontend/src/App.tsx` with a tab switcher**

```tsx
import { useState } from "react";
import { GamesPage } from "./pages/GamesPage";
import { PlayerPropsPage } from "./pages/PlayerPropsPage";
import { TrackRecordPage } from "./pages/TrackRecordPage";

type Tab = "games" | "props" | "track-record";

function App() {
  const [tab, setTab] = useState<Tab>("games");
  const season = 2026;
  const week = 1;

  return (
    <div>
      <nav>
        <button onClick={() => setTab("games")}>Games</button>
        <button onClick={() => setTab("props")}>Player Props</button>
        <button onClick={() => setTab("track-record")}>Track Record</button>
      </nav>
      {tab === "games" && <GamesPage />}
      {tab === "props" && <PlayerPropsPage season={season} week={week} />}
      {tab === "track-record" && <TrackRecordPage />}
    </div>
  );
}

export default App;
```

(`TrackRecordPage` is created in Task 21 — this file references it ahead of time; Task 21's first step creates that file so the app compiles.)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/PlayerPropsPage.tsx frontend/src/App.tsx
git commit -m "feat: Player Props page with sortable prediction table"
```

---

## Task 21: Frontend — Track Record page and app shell

**Files:**
- Create: `frontend/src/pages/TrackRecordPage.tsx`

**Interfaces:**
- Consumes: `api.trackRecord` from `api/client.ts`; `TrackRecord` from `types.ts`.

Verbatim port of `nfl_predictor/frontend/src/pages/TrackRecordPage.tsx`.

- [ ] **Step 1: Write `frontend/src/pages/TrackRecordPage.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { TrackRecord } from "../types";

export function TrackRecordPage() {
  const [record, setRecord] = useState<TrackRecord | null>(null);

  useEffect(() => {
    api.trackRecord().then(setRecord);
  }, []);

  if (!record) {
    return <p>Loading…</p>;
  }

  return (
    <div>
      <h1>Track Record</h1>
      <p>Resolved games: {record.n_resolved_games}</p>
      <p>
        Moneyline accuracy:{" "}
        {record.pct_moneyline_correct != null ? `${Math.round(record.pct_moneyline_correct * 100)}%` : "No resolved games yet"}
      </p>
    </div>
  );
}
```

- [ ] **Step 2: Run the full build**

Run (from `frontend/`): `npm run build`
Expected: no TypeScript errors — `App.tsx` from Task 20 now compiles cleanly against this file.

- [ ] **Step 3: Manual browser check of all three tabs**

Run (from `frontend/`): `npm run dev`, with the backend running. Click through Games / Player Props / Track Record and confirm each renders without a console error (Track Record legitimately shows "No resolved games yet" until real games have been played and reconciled).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/TrackRecordPage.tsx
git commit -m "feat: Track Record page"
```

---

## Task 22: Deploy — Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

Near-verbatim port of `nfl_predictor/Dockerfile`, including the `$PORT`-binding fix carried over from day one (per the design spec, rather than re-discovered on first deploy). Only the module name (`cfb_predictor`) and default port (`8003`, matching Tasks 16/18) differ.

- [ ] **Step 1: Write `.dockerignore`**

```
.venv/
__pycache__/
*.egg-info/
data/cache/
data/tracking.db*
.pytest_cache/
frontend/node_modules/
.git/
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV PYTHONPATH=/app/src
ENV PUBLIC_MODE=true

EXPOSE 8003
CMD ["sh", "-c", "uvicorn cfb_predictor.api.main:app --host 0.0.0.0 --port ${PORT:-8003}"]
```

- [ ] **Step 3: Verify the image builds locally**

Run: `docker build -t cfb-predictor .`
Expected: builds successfully (may take several minutes for the first `npm ci`/`pip install`).

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: Dockerfile for Render deployment"
```

- [ ] **Step 5: Deploy to Render** — this is a visible/shared-state action (creates a live public service, and requires setting the real `CFBD_API_KEY`/`ODDS_API_KEY` env vars on the Render service). Confirm with the user before running `render` CLI commands or connecting the GitHub repo in the Render dashboard; not automated as part of this plan.

---

## Task 23: Hub integration — add CFB card to predictor-hub

**Files:**
- Modify: `/Users/sigey/Documents/Projects/predictor-hub/index.html`

**Interfaces:** none (static HTML edit).

- [ ] **Step 1: Add the CFB card**

In `/Users/sigey/Documents/Projects/predictor-hub/index.html`, inside the `<div class="grid">` block, add a new card after the last existing card (read the file first — at the time this plan was written it held PL Predictor and F1 Predictor; NFL Predictor's own hub task may or may not have landed yet, so append after whatever is currently last rather than assuming a fixed position). Adjust the `href` once the real Render URL from Task 22 is known — placeholder shown, must be updated to the actual deployed URL before committing:

```html
    <a class="card" href="https://cfb-predictor-REPLACE.onrender.com" target="_blank" rel="noopener">
      <span class="badge live">Live</span>
      <span class="card-icon">🎓</span>
      <h2>CFB Predictor</h2>
      <p>FBS college football game winner, spread, and total predictions, plus anytime-touchdown and yardage player props.</p>
    </a>
```

- [ ] **Step 2: Verify the page still renders correctly**

Open `/Users/sigey/Documents/Projects/predictor-hub/index.html` directly in a browser and confirm every card displays in the grid with consistent styling.

- [ ] **Step 3: Commit (in the predictor-hub repo)**

```bash
cd /Users/sigey/Documents/Projects/predictor-hub
git add index.html
git commit -m "Add CFB Predictor card"
```

- [ ] **Step 4: Push** — requires the user's explicit go-ahead (this repo deploys from its GitHub remote; pushing makes the hub change publicly visible). Do not push automatically.

---

## Self-Review Notes

**1. Spec coverage** — walked every section of the design spec against the tasks above:
- Purpose/scope (moneyline/spread/total, four player prop categories, value-bet detection, snapshot-reconcile tracking, three-tab frontend, Docker/Render deploy, hub card): Tasks 9-23 cover each, matching NFL_Predictor's own shipped shape.
- "What actually changes vs. NFL_Predictor" table: every "transfers unchanged" file (Tasks 5-7, 9-11, 13-14, 18-21) is reproduced with only the import-path rename plus, where the spec's own analysis called for it, a small confirmed adaptation (conference_game in Task 8/9, fbs_teams threading in Task 10/12). Every "genuinely new" item (data layer: Tasks 2-4; team universe: folded into Task 2 as an additional function, not its own task, per the brief; odds coverage uncertainty: Task 4's docstring + Task 17 Step 7's empirical check; new API key: Task 1/17) has a task.
- "Deliberately not changing scope" (poll rankings, live in-game engine, playoff Monte Carlo): no task adds these, matching the spec's explicit deferral.
- Data layer call-budget constraint: every fetch function in Tasks 2-4 is one-call-per-season, tests never hit the real network (Global Constraints + each task's tests), and Task 17 isolates the one real run with an explicit call-count estimate.
- FCS-exclusion-but-count-for-bookkeeping: Task 8's `_assemble`/`build_training_frame` split, tested explicitly (`test_build_training_frame_excludes_fcs_opponent_games_via_division_fallback` / `..._via_fbs_teams_override`).
- Open items to confirm during implementation (Odds API CFB coverage, CFBD field names, transfer-portal player-id stability): Task 17 Steps 1 and 7 are where these get checked against real data, matching the spec's own "verify against real data" posture; transfer-portal id stability isn't designed around speculatively, per the spec's own instruction, and isn't blocking (CFBD player ids are athlete ids, stable within CollegeFootballData's own system regardless of team).

**2. Placeholder scan** — searched this plan for "similar to", "copy from", "TODO", "same as X" used as a substitute for code. Every reference to `NFL_Predictor`/`nfl_predictor` in this plan is either (a) rationale in prose ("verbatim port from nfl_predictor, confirmed while reading the real source") accompanying a task whose code block is fully written out, or (b) a specific behavioral pointer (e.g. "the `contextlib.closing` connection-handling fix from NFL_Predictor's final review") describing code that is itself reproduced in full in that same task. No task tells an implementer to go copy a file instead of showing its content.

**3. Type/interface consistency** — cross-checked signatures used across tasks:
- `features.build.build_training_frame(games_df, fbs_teams=None) -> tuple[pd.DataFrame, list[str]]` (Task 8) is called identically in `evaluate.walk_forward.prepare_folds` (Task 10) and `models.manifest.train_all` (Task 12).
- `data.games.fetch_fbs_teams(season) -> pd.DataFrame` (Task 2, columns `team, conference, division, classification`) is consumed by `models.manifest._fbs_teams_by_season` (Task 12) via `set(... ["team"])`, matching the column name.
- `data.player_stats.fetch_weekly_player_stats(seasons, games_df, force_refresh=False) -> pd.DataFrame` (Task 3) is called with this exact two-required-argument shape in `models.manifest.train_all` (Task 12) and `api.routes._load_player_history`/`warm_caches` (Task 15) — no caller uses the old NFL one-argument shape.
- `models.manifest.load_models()`'s returned dict shape (`game_outcome_model, total_model, chosen_candidate, sigma, total_sigma, player_models, feature_cols, player_feature_cols`) is used identically in `api/routes.py`'s `_predict_game_from_models` and in `tests/test_api_routes.py`'s/`tests/test_background_tracking.py`'s monkeypatched fixtures (Tasks 12, 15, 16).
- `FEATURE_COLUMNS` ends in `"conference_game"` (Task 8) and every test frame across Tasks 8-10 that builds a feature row includes a `conference_game` column, never the old `div_game` name.
- `_lines_for_game(odds_df, home_team, away_team) -> tuple[float | None, float | None]` (Task 15) is defined and tested in the same task, and called with this exact signature from both `get_game_prediction` and `background_tracking_tick` in that file.
