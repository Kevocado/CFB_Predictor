"""trust.py — the `trust` signal: how often this model has been right at the
confidence it is quoting.

Spec `2026-10-01-fixture-signals-design.md` §4: *"When this model says ~59% on
a home favourite it has been right X% of the time (n games)."* The probability
is the game's own headline pick; `X`, `n` and the band come from the committed
`data/tracking.db`, bucketed on the `home_win_prob` the pick was stored with.

## Why the moneyline and nothing else.

Kevin, 2026-10-04: *"only add the trust badge to proven markets then since its a
trust badge."* Measured on the production store that day
(`predictor-hub/docs/superpowers/plans/2026-10-04-production-measurements.md`),
only one CFB market clears spec §4's floor:

| market | graded pairs | bands over the floor |
|---|---|---|
| **moneyline** | **390** | **3 of 4** |
| ATS | 59 | 0 |
| total | 59 | 0 |

So this adapter reads `get_calibration()["winner"]` and nothing else. The ATS and
total buckets exist in the same call and are **not** a matter of a flag: 59
graded pairs cannot fill a band of 30 in more than one or two bands, and a rate
from 12 games is the thing §4 forbids in as many words. When those stores grow
past the floor, adding them is a one-line change to the market read below.

## THE FLOOR, and which layer owns it. Read this before changing `TRUST_MIN_N`.

Spec §4: *"Renders only at n >= 30; below that the row says nothing, never a
rate from a handful of games."* **This adapter owns it: a band under the floor
produces NO signal, not a signal with a small `n`.**

The shared component also enforces a floor — `SPEC_MIN_N` in `SignalRows.tsx`,
and `signalIsDrawn` DROPS a sub-floor rate rather than throwing it. That is not
a second opinion on the same question, because the two answer different ones:

  * The component's floor answers **"may this be DRAWN?"** Its own file header
    says so: *"the rule is about what may be drawn, not about what may be
    computed. An adapter can honestly hold a bucket of 12 games and report it;
    what it may not do is put a rate built from it on the page."*
  * This adapter's floor answers **"may this be SENT?"** — and this endpoint is
    read by more than the component. Phase 4 feeds the payloads to the AI "so
    what" writer, and its validator admits a figure that appears in a signal
    payload. A sub-floor rate emitted here would therefore be quotable as a
    claim about 12 games, which is the thing §4 forbids, and the component's
    floor would not catch it, because by then the payload is already gone.

Effective floor is `max(TRUST_MIN_N, SPEC_MIN_N)` = 30: the component clamps
`floorOf = max(SPEC_MIN_N, minN)`, so a caller can only ever RAISE it.

## `strength` — the formula and WHY it points that way.

    gap      = |hit_rate - mean_prob|          the reliability gap
    se       = sqrt(mean_prob * (1 - mean_prob) / n)
    z        = gap / se                        miscalibration in SEs
    strength = min(1.0, z / 3.0)

Identical to F1's `signals/trust.py`, and deliberately so: spec §9 decision 4
ranks by adapter `strength`, and two adapters that disagreed on what a number
means could not be compared on one page. The same reasoning applies — it RISES
with the reliability gap, because the page already shows the model saying its
probability and what it cannot show is whether that probability has ever been
worth anything. Divided by the standard error so a gap is scored against how
surprising it is rather than how big it looks, which also means that **at a
fixed gap, `strength` rises as `n` grows**: a five-point gap on 35 picks is
noise and the same gap on 500 is a fact.

`z / 3.0` puts a full-strength row at three standard errors. It is a scale, not
a threshold: nothing is gated on it, and it saturates rather than clipping, so
two bands can never tie at an arbitrary ceiling.

## THE HEADLINE, and why its wording is load-bearing.

`SignalRows` THROWS `HeadlineFigureMismatchError` when a row's words do not state
the figure its bar draws, so an over-precise headline ("74.4%") is a 500 on the
page rather than a warning. The bar draws `headline.figures.rate` at
`fmt.pct`'s whole-percent rounding, so `headline()` states that same rounded
percent and nothing finer.

It also does NOT restate the model's current pick, for spec §4's own reason: the
page already shows that number, and a signal that repeats it adds nothing.
"""

from __future__ import annotations

import math

from ..tracking import store

#: Spec §4's floor, per sport (spec §9 decision 3; plan Phase 1). See the module
#: docstring for which layer owns it and how it interacts with the component's
#: own `SPEC_MIN_N`.
TRUST_MIN_N = 30

#: `visual` for this signal, and therefore which figure in `headline.figures`
#: the component draws. `reliability_bar` draws `figures.rate`, which
#: `SignalRows.signalFigure` requires to be a share in [0, 1].
VISUAL = "reliability_bar"

#: Figure name `VISUAL` draws. Read out of the component's own `FIGURE` map by
#: the tests rather than trusted from here.
RATE_FIGURE = "rate"

#: Which of `get_calibration`'s markets this adapter reads, and the word the
#: signal uses for it. Only the moneyline is proven — see the module docstring's
#: table. Named rather than indexed so adding a market later cannot silently
#: repoint this one.
MARKET = "winner"

#: What the market is called in the signal's `source`, which is reader-facing.
MARKET_LABEL = "the moneyline"


def _is_probability(value) -> bool:
    """Whether `value` is a finite probability this adapter can place in a band.

    `bool` is rejected explicitly: it is an `int` in Python, so `isinstance(True,
    int)` is true and a stray flag would be read as a probability of 1.0 — which
    lands in the top band and reports that band's record for it.
    """
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _find_band(bands: list[dict], prob: float) -> dict | None:
    """The band containing `prob`, or None.

    The band edges come from `store.get_calibration` itself rather than from a
    second list written here. That is the opposite choice from F1's adapter,
    which carries its own `BUCKET_BOUNDS` because its `get_probability_buckets`
    GENERATES its SQL from that list — there, the list and the query cannot
    disagree because they are the same object. CFB's bucket edges are computed
    inside the store from `CALIBRATION_N_BUCKETS`, and that constant is load-
    bearing well beyond this adapter: the trade hub's edge gate needs every band
    to clear its own minimum before admitting an edge, which is why it is 4 and
    not 10. Duplicating the edges here would create a second place for that
    decision to live, and the whole failure this file guards against is a band
    that disagrees with the query that built it.

    `get_calibration`'s own convention is `[lo, hi)` for every band except the
    last, which is closed. Both edges are tested on every band rather than
    assuming the upper one — written as "closed only when last", a probability
    of -0.1 would satisfy `prob <= hi` and be reported from the top band.
    """
    if not _is_probability(prob):
        return None
    for index, band in enumerate(bands):
        lo, hi = band.get("lo"), band.get("hi")
        if lo is None or hi is None:
            continue
        last = index == len(bands) - 1
        if lo <= prob < hi or (last and lo <= prob <= hi):
            return band
    return None


def strength(rate: float, mean_predicted: float, n: int) -> float:
    """How much a reader should care about this band's calibration. 0..1.

    The formula and its direction are in the module docstring. `se` is zero only
    when `mean_predicted` is exactly 0 or 1, which no recorded CFB probability
    is; it is handled rather than left to divide by zero.
    """
    if n <= 0:
        return 0.0
    gap = abs(rate - mean_predicted)
    variance = mean_predicted * (1.0 - mean_predicted)
    if variance <= 0.0:
        return 1.0 if gap > 0.0 else 0.0
    se = math.sqrt(variance / n)
    if se <= 0.0:
        return 1.0 if gap > 0.0 else 0.0
    return min(1.0, (gap / se) / 3.0)


def headline(rate: float, mean_predicted: float) -> str:
    """The one line, at most 12 words, stating the rate the bar draws.

    `SignalRows.assertFigureIsStated` compares NUMBERS: the stated figure must
    equal one of the percents `fmt.pct` can print for `rate`, at whole-percent
    precision and with a `%` sign. So the percent here is `round(rate * 100)`,
    never `rate * 100` unrounded.

    **`~` on the stated figure is load-bearing.** `mean_predicted` is the MEAN of
    a whole band a quarter wide, so "When the model says 50%" would claim picks
    of exactly 50% when the row covers everything from 0.5 to 0.75. The tilde is
    spec §4's own notation ("When this model says ~59%") and keeps the sentence
    true of the band.

    No sportsbook vocabulary, and no present-tense certainty: this is a record,
    not a forecast.
    """
    stated = f"{round(mean_predicted * 100):g}"
    landed = f"{round(rate * 100):g}"
    return f"When the model says ~{stated}%, its picks landed {landed}% of the time"


def trust_signal(game_id: str, prob: float) -> dict | None:
    """The `trust` signal for a game whose headline pick carries `prob`, or None
    when the honest answer is that there is no signal.

    None is returned, rather than a signal with a thin `n`, when:

      * `prob` is not a finite number, or falls outside every band;
      * the tracking store cannot be read at all (a missing file that
        `sqlite3.connect` would silently CREATE empty, a corrupt file, a
        `game_predictions` table that is not there); or
      * the band `prob` falls in holds fewer than `TRUST_MIN_N` resolved picks.

    That last one is the floor, and it is why a missing or emptied
    `tracking.db` yields no signal rather than a rate from two games: the bands
    come out empty, every one of them is under the floor, and the same guard
    covers both. A silent degradation to "n = 2, right 50% of the time" is the
    specific outcome §4 forbids, and it is unreachable from here.
    """
    if not _is_probability(prob):
        return None

    try:
        calibration = store.get_calibration()
    except Exception:
        # sqlite3.connect CREATES a missing file and `_connect` then creates the
        # tables, so a deleted or unreadable store does not raise — it arrives
        # here as empty bands, which the floor below already handles. This catch
        # is for the case that DOES raise: a file that is not a database, or one
        # locked by another writer. Either way there is no honest signal.
        return None

    band = _find_band(calibration.get(MARKET) or [], prob)
    if band is None:
        return None
    n = int(band.get("n") or 0)
    if n < TRUST_MIN_N:
        return None

    rate, mean_predicted = band.get("hit_rate"), band.get("mean_prob")
    if rate is None or mean_predicted is None:
        return None
    lo, hi = band["lo"], band["hi"]
    return {
        "kind": "trust",
        "sport": "cfb",
        "game_id": str(game_id),
        "headline": {
            "text": headline(rate, mean_predicted),
            # Only the figure the bar draws, plus the model-side probability the
            # sentence quotes. §3 leaves `figures` open and the component reads
            # the one its `visual` names.
            "figures": {RATE_FIGURE: rate, "stated_prob": mean_predicted},
        },
        "n": n,
        "source": (
            f"{n} resolved games in this project's tracking.db, "
            f"{lo:g}-{hi:g} probability band, {MARKET_LABEL}"
        ),
        # `get_calibration` reports no timestamp, and §3 says a row carries a
        # source and a date "or neither" — `SignalRows` renders a blank date as a
        # dash. A guessed date would be worse than none.
        "as_of": "",
        "strength": strength(rate, mean_predicted, n),
        "pre_kickoff_only": True,
        "visual": VISUAL,
    }