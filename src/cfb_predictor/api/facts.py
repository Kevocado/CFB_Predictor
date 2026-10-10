"""facts.py — the read-only CFB /facts bundle the match explainer consumes.

Read-only and suggest-only: this router never writes, never trains and
never places anything. It assembles the spec's facts contract from data
this API already has, and in PUBLIC_MODE it reads the precomputed public
snapshot only — no live model is ever computed there.

The honesty rules are the same as NFL's, because they are properties of the
contract rather than of a sport:

* a pick for a game that has started comes only from the pre-start stored
  snapshot, never from today's model;
* no stored pick means ``pick`` is null and ``pick_timing`` is ``none``;
* a snapshot taken at/after kickoff is ``rebuilt`` and is never counted;
* ``result.pick_won`` appears only for a ``pre_kickoff`` pick;
* spread/total markets are skipped for a started game unless they come
  from the pre-start snapshot, and are skipped entirely when no line
  exists (the common CFB case — see routes._lines_for_game).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..config import PUBLIC_MODE
from ..tracking import store
from . import routes

router = APIRouter()
logger = logging.getLogger(__name__)

_YARDS_BY_POSITION = {
    "QB": "passing_yards",
    "RB": "rushing_yards",
}
_YARD_KEYS = ("passing_yards", "rushing_yards", "receiving_yards")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _public_snapshot() -> dict:
    return routes._public_snapshot()


# --- small helpers ------------------------------------------------------

def _as_utc(value: Any) -> pd.Timestamp | None:
    """CFBD's gameday carries an explicit offset; a naive one is read as UTC."""
    if value in (None, ""):
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _iso_utc(value: Any) -> str:
    stamp = _as_utc(value)
    return "" if stamp is None else stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _spread_line(home_team: str, spread_line: Any) -> str | None:
    """spread_line is the home team's expected margin (positive means home
    is favoured), the same convention NFL uses; a favourite is worded as
    giving points."""
    line = _num(spread_line)
    if line is None:
        return None
    return f"{home_team} {-line:+.1f}"


def _pick(home_team: str, away_team: str, home_prob: Any, away_prob: Any) -> dict | None:
    home = _num(home_prob)
    away = _num(away_prob)
    if home is None or away is None:
        return None
    if home >= away:
        return {"label": home_team, "prob": home}
    return {"label": away_team, "prob": away}


def _projected_yards(prop: dict) -> float:
    key = _YARDS_BY_POSITION.get(str(prop.get("position") or "").upper(), "receiving_yards")
    value = _num(prop.get(key))
    if value is None:
        present = [v for v in (_num(prop.get(k)) for k in _YARD_KEYS) if v is not None]
        return max(present) if present else 0.0
    return value


def _players(props: list[dict], teams: set[str]) -> list[dict]:
    ranked = [p for p in props if p.get("recent_team") in teams]
    ranked.sort(key=_projected_yards, reverse=True)
    out = []
    for prop in ranked[:3]:
        yards = _projected_yards(prop)
        key = _YARDS_BY_POSITION.get(str(prop.get("position") or "").upper(), "receiving_yards")
        out.append({
            "name": prop.get("player_name"),
            "team": prop.get("recent_team"),
            "projection": f"{yards:.0f} {str(key).replace('_yards', '')} yds",
        })
    return out


def _record() -> dict | None:
    """The pre-kickoff record, under a label that says so.

    Since 2026-10-01 the track record's `games` headline counts every recorded pick,
    including ones the model made after kickoff (`track-record-counts-every-pick`, merged as
    predictor-hub #66). So this must read the `pre_kickoff` sub-record, not the headline:
    the number on this block is labelled "Picks made before kickoff", and putting an
    all-picks figure under that label is precisely the mislabelling the spec still forbids
    ("nothing computed after the start may be labelled 'made before kickoff'").

    The pre-kickoff subset is what this block showed before the rule changed, so the figures
    on the page are unchanged. A payload with no `pre_kickoff` at all yields None rather than
    a mislabelled record -- an omission is recoverable, a wrong label is not.
    """
    games = (store.get_track_record() or {}).get("games") or {}
    before_kickoff = games.get("pre_kickoff")
    if not isinstance(before_kickoff, dict):
        return None
    settled = int(before_kickoff.get("n_resolved") or 0)
    if settled <= 0:
        return None
    pct = _num(before_kickoff.get("pct_moneyline_correct"))
    return {
        "label": "Picks made before kickoff",
        "hits": None if pct is None else int(round(pct * settled)),
        "settled": settled,
    }


# --- data access --------------------------------------------------------

def _snapshot_game(game_id: str) -> tuple[int, int, dict] | None:
    snap = _public_snapshot()
    season = snap.get("season")
    for week_key, week in (snap.get("weeks") or {}).items():
        for game in week.get("games") or []:
            if str(game.get("game_id")) == str(game_id):
                return int(game.get("season") or season), int(week_key), game
    return None


def _snapshot_week(season: int, week: int) -> dict | None:
    snap = _public_snapshot()
    if snap.get("season") != season:
        return None
    return (snap.get("weeks") or {}).get(str(week))


def _load_game(game_id: str) -> tuple[int, int, dict]:
    """The game row. CFB ids are CFBD's own (no season/week encoded), so
    public mode searches the snapshot and live mode searches the season."""
    if PUBLIC_MODE:
        found = _snapshot_game(game_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
        return found
    season = routes.current_season_and_week()[0]
    games = routes.games_data.fetch_schedules([season])
    if games is None or games.empty:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    matches = games[games["game_id"].astype(str) == str(game_id)]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    game = matches.iloc[0].to_dict()
    return int(game.get("season") or season), int(game.get("week") or 0), game


def _snapshot_prediction(season: int, week: int, game_id: str) -> dict | None:
    """The prediction baked into the public snapshot. Only for games that
    haven't started: the snapshot rebuilds recent rounds after kickoff, so
    for a started game this is today's model, not a pre-start read."""
    snap = _snapshot_week(season, week) or {}
    predictions = snap.get("predictions") or {}
    for key, value in predictions.items():
        if str(key) == str(game_id):
            return value
    return None


def _current_prediction(season: int, week: int, game_id: str) -> dict | None:
    if PUBLIC_MODE:
        return _snapshot_prediction(season, week, game_id)
    return routes.get_game_prediction(season, week, game_id)


def _week_row(season: int, week: int, game_id: str) -> dict | None:
    """The stored pick row: the authority on whether a pick was snapshotted
    before kickoff, and whether it was rebuilt."""
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week) or {}
        games = snap.get("games") or []
    else:
        games = routes.get_games(season, week)
    if not games:
        return None
    frame = pd.DataFrame(games)
    frame = frame[frame["game_id"].astype(str) == str(game_id)]
    if frame.empty:
        return None
    for row in routes.store.get_predictions_for_week(season, week, frame) or []:
        if str(row.get("game_id")) == str(game_id):
            return row
    return None


def _props(season: int, week: int) -> list[dict]:
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week) or {}
        return snap.get("player_props") or []
    return routes.get_player_props(season, week)


# --- bundle assembly ----------------------------------------------------

def _status(game: dict, now: datetime) -> str:
    home_score = _num(game.get("home_score"))
    away_score = _num(game.get("away_score"))
    if home_score is not None and away_score is not None:
        return "final"
    gameday = _as_utc(game.get("gameday"))
    if gameday is not None and gameday <= pd.Timestamp(now):
        return "live"
    return "upcoming"


def _markets(game: dict, prediction: dict | None, moneyline_from: dict | None = None) -> list[dict]:
    """The markets, from `prediction` — except the moneyline, which may come from
    `moneyline_from`.

    The bundle's `pick` is deliberately the number snapshotted before kickoff,
    because that is the record which will be judged. For an UPCOMING game the
    snapshot also holds today's model, recomputed, and the two genuinely differ.
    Emitting the moneyline from the recomputation stated the same pick twice at
    two probabilities — measured here as `pick: TCU 0.67` beside
    `markets: [moneyline {TCU: 0.55}]` — and the v2 panel draws its figure from
    the market while computing its confidence band from the pick, so it would have
    shown both with nothing to tell the reader two numbers were in play.

    The spread and the total stay on `prediction`. That is a real trade-off, not
    a proof: they are different claims with their own numbers, the row carries
    neither, and nothing in the bundle dates them. Taking the whole set from the
    row would be self-consistent and would drop the spread and total for every
    snapshotted pick; that is a product call, and it is not this bug.
    """
    """Moneyline always; spread/total only where a line actually exists.
    CFB's snapshot carries no lines at all (routes._get_games_live attaches
    them from the odds feed for the current week only), so the usual public
    bundle is the pick market on its own."""
    if not prediction:
        return []
    home_team = game["home_team"]
    away_team = game["away_team"]
    out: list[dict] = []

    source = moneyline_from if moneyline_from is not None else prediction
    home_prob = _num(source.get("home_win_prob"))
    away_prob = _num(source.get("away_win_prob"))
    if home_prob is not None and away_prob is not None:
        out.append({
            "market": "moneyline",
            "model": {home_team: home_prob, away_team: away_prob},
        })

    margin = _num(prediction.get("predicted_margin"))
    line = _spread_line(home_team, game.get("spread_line"))
    if margin is not None and line is not None:
        market = {"market": "spread", "model_margin": margin, "line": line}
        cover = _num(prediction.get("home_cover_prob"))
        if cover is not None:
            market["model_cover_prob"] = cover
        out.append(market)

    total = _num(prediction.get("predicted_total"))
    total_line = _num(game.get("total_line"))
    if total is not None and total_line is not None:
        market = {"market": "total", "model_total": total, "line": total_line}
        over = _num(prediction.get("over_prob"))
        if over is not None:
            market["model_over_prob"] = over
        out.append(market)

    return out


def _drivers(game: dict, season: int, home_team: str, away_team: str, live_ok: bool = True) -> list[dict]:
    """Why the model leans the way it does. In public mode only what the
    snapshot carries — a rating gap would mean computing live.

    The rating gap is built live from game history, so `live_ok` is False for
    a game that has already started: that history now contains the game
    itself, and a "rating gap" would be a post-kickoff number presented as the
    pre-kickoff reason. Fixed facts (home field, conference game) still show."""
    drivers: list[dict] = []
    rating_diff = None

    if not PUBLIC_MODE and live_ok:
        try:
            history = routes._load_game_history(season)
            history = history[history["game_id"].astype(str) != str(game["game_id"])]
            row = routes.feature_build.build_features_for_game(
                home_team, away_team, history, gameday=game.get("gameday"), conference_game=game.get("conference_game"),
            )
            rating_diff = _num(row.get("rating_diff"))
        except Exception:
            logger.info("driver features unavailable for game_id=%s", game.get("game_id"))

    if rating_diff is not None:
        drivers.append({
            "name": "Rating gap",
            "value": f"{rating_diff:+.1f} pts",
            "direction": home_team if rating_diff >= 0 else away_team,
        })
    drivers.append({
        "name": "Home field",
        "value": "neutral site" if game.get("neutral_site") else "advantage",
        "direction": "neutral" if game.get("neutral_site") else home_team,
    })
    if game.get("conference_game"):
        drivers.append({"name": "Conference game", "value": "cross-conference", "direction": "both"})
    return drivers


def _stored_matchup_rows(season: int, week: int, game_id: str) -> list[dict]:
    """The snapshot's stored FBS duels for this game as facts rows. `toward_pick` is always null (neutral): no
    lift gate has proven a CFB duel type, so none may claim a direction."""
    from ..signals.matchups import rows_to_duels, to_context
    rows = ((_snapshot_week(season, week) or {}).get("matchups") or {}).get(str(game_id)) or []
    return to_context(rows_to_duels(rows), None)


def _context(game: dict, matchups: list[dict] | None = None) -> dict:
    context: dict[str, Any] = {}
    if matchups:
        context["matchups"] = matchups
    home_conf = game.get("home_conference")
    away_conf = game.get("away_conference")
    if home_conf and away_conf:
        context["conferences"] = f"{away_conf} at {home_conf}"
    # CFB has no injuries source; nothing is invented here.
    return context


def _result(game: dict, status: str, pick_timing: str, stored: dict | None) -> dict | None:
    if status != "final":
        return None
    home_score = _num(game.get("home_score"))
    away_score = _num(game.get("away_score"))
    if home_score is None or away_score is None:
        return None
    result: dict[str, Any] = {"score": f"{game['home_team']} {home_score:.0f}-{away_score:.0f}"}
    if pick_timing != "pre_kickoff" or not stored:
        return result
    home_prob = _num(stored.get("home_win_prob"))
    away_prob = _num(stored.get("away_win_prob"))
    if home_prob is None or away_prob is None:
        return result
    predicted_home = home_prob >= away_prob
    actual_home = home_score > away_score
    result["pick_won"] = bool(predicted_home == actual_home)
    return result


@router.get("/facts/upcoming")
def get_facts_upcoming(hours: int = 72) -> dict:
    """Ids of games starting within the window, for the pre-generation loop."""
    if hours < 0:
        raise HTTPException(status_code=422, detail="hours must be >= 0")
    now = _now()
    cutoff = pd.Timestamp(now) + timedelta(hours=hours)

    games: list[dict] = []
    if PUBLIC_MODE:
        snap = _public_snapshot()
        for week in (snap.get("weeks") or {}).values():
            games.extend(week.get("games") or [])
    else:
        season, week = routes.current_season_and_week()
        games = routes.get_games(season, week) or []

    ids = []
    for game in games:
        gameday = _as_utc(game.get("gameday"))
        if gameday is None or gameday <= pd.Timestamp(now) or gameday > cutoff:
            continue
        game_id = game.get("game_id")
        if game_id is not None and str(game_id) not in ids:
            ids.append(str(game_id))
    return {"ids": ids}


def game_pick(game_id: str) -> dict:
    """The game's headline pick and everything the decision to make it was taken
    from. Raises 404 when the game is unknown.

    Split out of `get_facts` so `GET /api/signals/{game_id}` can quote the SAME
    pick. That is the whole point of the split: a signal that recomputed the pick
    could put a different probability on the page than the facts block beside it,
    and the two would both be defensible and inconsistent. The rule below is the
    one that decides which number a started game shows, so it has exactly one
    implementation.

    Returns the pick, its timing label, the row-sourced probabilities, and the
    started flag — the four things the bundle and the signals endpoint both need.
    """
    now = _now()
    season, week, game = _load_game(game_id)
    home_team, away_team = game["home_team"], game["away_team"]
    status = _status(game, now)
    started = status in ("live", "final")

    row = _week_row(season, week, game_id)
    row_probs = (
        {"home_win_prob": row.get("home_win_prob"), "away_win_prob": row.get("away_win_prob")}
        if row is not None and row.get("home_win_prob") is not None
        else None
    )

    # THE RULE: the pick number and its timing label always come from the SAME
    # source, so a number is never described by another source's honesty.
    #
    # * STARTED -> the tracking row stored before kickoff, in both modes. The
    #   public snapshot is NOT that: it rebuilds recent rounds every few hours,
    #   so its prediction for a started game is today's model, recomputed after
    #   kickoff. The row carries no margin/total, so those are simply absent.
    #   No row at all -> no pick, and "none".
    # * UPCOMING -> the row when one exists, so the number shown is the very
    #   record that will be judged and a late-written row still reports itself
    #   as 'rebuilt'. With no row yet (weeks out) the live forecast is used and
    #   labelled 'pre_kickoff': a game that has not been played cannot have
    #   been predicted after it, so that label is always true of the number.
    if started:
        stored = row_probs
    else:
        stored = _current_prediction(season, week, game_id)

    pick_source = row_probs if row_probs is not None else (None if started else stored)
    pick = (
        _pick(home_team, away_team, pick_source.get("home_win_prob"), pick_source.get("away_win_prob"))
        if pick_source is not None
        else None
    )
    if pick is None:
        pick_timing = "none"
    elif pick_source is row_probs:
        pick_timing = "rebuilt" if row.get("rebuilt") else "pre_kickoff"
    else:
        pick_timing = "pre_kickoff"

    return {
        "season": season,
        "week": week,
        "game": game,
        "home_team": home_team,
        "away_team": away_team,
        "status": status,
        "started": started,
        "stored": stored,
        "pick_source": pick_source,
        "pick": pick,
        "pick_timing": pick_timing,
    }


@router.get("/facts/{game_id}")
def get_facts(game_id: str) -> dict:
    ctx = game_pick(game_id)
    season, week, game = ctx["season"], ctx["week"], ctx["game"]
    home_team, away_team = ctx["home_team"], ctx["away_team"]
    status, started = ctx["status"], ctx["started"]
    stored, pick_source = ctx["stored"], ctx["pick_source"]
    pick, pick_timing = ctx["pick"], ctx["pick_timing"]

    # A started game with no stored pick has no honest line to quote.
    # The moneyline comes from `pick_source` so the bundle states the pick
    # once, at one probability -- see _markets' docstring for why that is not
    # the same thing as taking every market from the row.
    markets = [] if (started and stored is None) else _markets(game, stored, moneyline_from=pick_source)

    matchups_rows: list[dict] = []
    if PUBLIC_MODE and not started:  # request path reads the snapshot only; never ranks or fetches
        try:
            matchups_rows = _stored_matchup_rows(season, week, game_id)
        except Exception:
            logger.info("stored matchups unavailable for game_id=%s", game_id)

    return {
        "sport": "cfb",
        "id": str(game_id),
        "title": f"{away_team} at {home_team}",
        "starts_at": _iso_utc(game.get("gameday")),
        "status": status,
        "pick_timing": pick_timing,
        "pick": pick,
        "markets": markets,
        # The rating gap and player props are both built/fetched now, so a
        # started game quotes neither; fixed pre-match facts still show.
        "drivers": _drivers(game, season, home_team, away_team, live_ok=not started),
        "context": _context(game, matchups_rows),
        "players": [] if started else _players(_props(season, week), {home_team, away_team}),
        "record": _record(),
        "result": _result(game, status, pick_timing, stored),
    }
