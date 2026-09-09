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
ROSTER_CACHE_DIR = CACHE_DIR / "rosters"

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
    ROSTER_CACHE_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)
