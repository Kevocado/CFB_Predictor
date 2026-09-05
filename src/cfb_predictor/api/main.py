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
