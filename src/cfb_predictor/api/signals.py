"""signals.py — `GET /api/signals/{game_id}`, the fixture's signal payloads.

Spec `2026-10-01-fixture-signals-design.md` §3 (the contract) and §2 ("No data,
no row"). One adapter ships, `trust`; the others arrive with their phases, and a
game with no honest signal gets an empty list rather than a placeholder.

The shape of every payload is dictated by the shared component at
`frontend/src/predictor-ui/components/SignalRows.tsx`, which REFUSES several
things a payload could plausibly carry — a rate outside [0, 1], a
`reliability_bar` with no `figures.rate`, a headline that does not state the
figure its bar draws (`HeadlineFigureMismatchError`), a visual it cannot draw.
Read that file before adding a field here.

The game id and the pick are NOT reimplemented here: they come from
`facts.game_pick`, so a signal can never quote a different probability than the
facts block on the same page.

## The `/api` prefix is load-bearing.

Every route this frontend calls is served under `/api` — `routes.py` declares
`APIRouter(prefix="/api")` — and the frontend reaches the backend through exactly
that prefix in both environments (`client.ts`'s `BASE_URL` is `/api`, and
`vite.config.ts` proxies only `'/api'`). A router mounted at the root would be
unreachable from the page, which is exactly the defect F1 shipped and fixed in
its own #36: `tests/test_trust_signal.py` drove the app through `TestClient`,
which calls paths directly and so never exercised the prefix a browser uses.
The OpenAPI assertion at the bottom of this module's tests pins it.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from . import facts
from ..signals import trust

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

#: Spec §2: "a fixed rule (not the model) ranks them by `strength` and keeps the
#: top 2-3". One adapter, so this never truncates anything today; it is here
#: because the rule belongs to the endpoint and not to a component, per
#: `SignalRows`'s own "Rejected: ranking and capping here" section.
MAX_SIGNALS = 3


def signals_for_game(game_id: str) -> list[dict]:
    """Every signal this game can honestly carry, strongest first.

    Each adapter is asked independently and may return nothing; a game with no
    signal returns `[]`, which the client renders as no rows at all (spec §2: no
    empty states, no filler). An adapter that raises is treated as "no signal"
    rather than being allowed to take down the endpoint — a signal is an
    enhancement on a fixture page, and the page itself must survive its absence.
    """
    ctx = facts.game_pick(game_id)
    pick = ctx.get("pick") or {}
    found: list[dict] = []

    try:
        signal = trust.trust_signal(game_id, pick.get("prob"))
    except Exception:
        logger.exception("trust signal unavailable for %s", game_id)
        signal = None
    if signal is not None:
        found.append(signal)

    return sorted(found, key=lambda s: s["strength"], reverse=True)[:MAX_SIGNALS]


@router.get("/signals/{game_id}")
def get_signals(game_id: str) -> dict:
    """The signals for one fixture. `{"signals": []}` is a valid, complete answer.

    An unknown game is a 404, matching `/facts/{game_id}`: the id grammar belongs
    to `facts` and this router does not own a second one.
    """
    try:
        signals = signals_for_game(game_id)
    except HTTPException:
        raise
    except Exception:
        # The game itself is unreadable, or the feature pipeline is down.
        # `facts.game_pick` answers the first of those with a 404, which passes
        # through above; anything else is a server fault this endpoint absorbs so
        # a missing row cannot become a broken fixture page.
        logger.exception("signals unavailable for %s", game_id)
        # The SAME shape as the success path, so a client reading `sport` or
        # `id` does not get a different object only when the backend is failing —
        # the one moment a client is least able to cope with a special case.
        return {"sport": "cfb", "id": game_id, "signals": []}

    return {"sport": "cfb", "id": game_id, "signals": signals}