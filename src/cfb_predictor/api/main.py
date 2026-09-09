"""main.py — FastAPI app entry point."""

import asyncio
import logging
import traceback
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # This loads variables from your .env file into os.environ

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from ..config import PUBLIC_MODE, PUBLIC_SNAPSHOT_POLL_SECONDS
from .routes import (
    current_season_and_week,
    refresh_public_snapshot_from_remote,
    router,
    background_tracking_tick,
)

logger = logging.getLogger(__name__)

_TRACKING_INTERVAL_SECONDS = 3600


async def _tracking_loop():
    while True:
        await asyncio.sleep(_TRACKING_INTERVAL_SECONDS)
        try:
            season, week = current_season_and_week()
            await asyncio.to_thread(background_tracking_tick, season, week)
        except Exception:
            logger.exception("background_tracking_tick failed")


async def _public_snapshot_poll_loop():
    while True:
        await asyncio.to_thread(refresh_public_snapshot_from_remote)
        await asyncio.sleep(PUBLIC_SNAPSHOT_POLL_SECONDS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # The public deployment serves games/predictions/player-props from
    # public_snapshot.py's precomputed file (see routes.py's PUBLIC_MODE
    # branches). The tracking loop still runs regardless -- it's what
    # feeds the track-record/verdict history, is already throttled by its
    # own TTLs (see data/games.py's _CURRENT_SEASON_TTL_SECONDS), and skipping
    # it would leave that feature permanently empty on the public deployment.
    tasks = [asyncio.create_task(_tracking_loop())]
    if PUBLIC_MODE:
        tasks.append(asyncio.create_task(_public_snapshot_poll_loop()))
    yield
    for task in tasks:
        task.cancel()


app = FastAPI(title="CFB Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    # Always logged server-side, regardless of mode. The raw traceback is
    # only ever returned to the caller outside PUBLIC_MODE (local/private
    # dev) -- the public deployment is internet-reachable, and a raw
    # traceback in an HTTP response is a real information-disclosure risk
    # (file paths, code structure, and whatever local values happened to be
    # in scope when the exception was raised), not just a debugging nicety.
    logger.exception("Unhandled exception during request %s %s", request.method, request.url.path)
    if PUBLIC_MODE:
        return PlainTextResponse(status_code=500, content="Internal Server Error")
    error_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return PlainTextResponse(status_code=500, content=f"DEBUG_TRACEBACK:\n{error_trace}")


app.include_router(router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "CFB Predictor API",
        "docs": "/docs",
    }
