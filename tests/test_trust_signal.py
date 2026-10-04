"""The `trust` signal for CFB, and the endpoint that serves it.

Kevin, 2026-10-04: *"only add the trust badge to proven markets then since its a
trust badge."* So the first thing this file holds is that the adapter reads the
MONEYLINE and nothing else — CFB's ATS and total buckets exist in the same
`get_calibration` call and are deliberately left alone, because on the production
store they hold 59 graded pairs between them and cannot fill a band of 30.

What each test is defending, in order of how quietly it would fail:

  * the FLOOR. A band under `TRUST_MIN_N` produces no signal at all, not a
    signal with a small `n`, because this endpoint's payloads are read by the AI
    "so what" writer and a sub-floor rate would be quotable as a claim about a
    handful of games.
  * an EMPTY OR MISSING STORE produces no signal, rather than a rate from two
    games. `sqlite3.connect` CREATES a missing file, so the silent degradation is
    reachable and has to be shut off deliberately.
  * the `/api` PREFIX, which is the defect F1 shipped in its own Phase 1: the
    router at the root cannot be called by the page, and `TestClient` calls paths
    directly so it never noticed.
  * the HEADLINE states the figure the bar draws, at whole-percent precision.
    `SignalRows` raises `HeadlineFigureMismatchError` on a mismatch, so this is a
    500 on the page, not a warning.
"""

import math

import pytest

from cfb_predictor.api import facts
from cfb_predictor.api.signals import MAX_SIGNALS
from cfb_predictor.signals import trust


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from cfb_predictor import config
    from cfb_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield


def _resolved(prob: float, hit: int) -> dict:
    """One resolved game prediction at `prob`, where the home side did (`hit`)."""
    return {
        "home_team": "Texas", "away_team": "Ohio State",
        "commence_time": "2099-08-30T16:00:00",
        "home_win_prob": prob, "away_win_prob": 1.0 - prob,
        "home_cover_prob": prob, "away_cover_prob": 1.0 - prob,
        "over_prob": prob, "under_prob": 1.0 - prob,
        "actual_home_score": 34 if hit else 7,
        "actual_away_score": 7 if hit else 34,
    }


def _write(prob: float, hit: int, count: int) -> None:
    """Record `count` DISTINCT resolved games at `prob`.

    The distinct ids matter and cost a test to learn: `record_game_predictions`
    is `INSERT OR IGNORE` on `game_id`, so a helper that stamps every copy with
    the same id stores one row no matter how many it is handed. The band then
    comes back under the floor and the test fails as "no signal" — which reads
    like an adapter bug and is not one.
    """
    from cfb_predictor.tracking import store
    import pandas as pd

    rows = []
    for i in range(count):
        row = _resolved(prob, hit)
        row["game_id"] = f"g-{prob}-{hit}-{i}"
        rows.append(row)

    store.record_game_predictions(rows)
    # `record_game_predictions` only snapshots; grading is a separate call, and
    # the buckets read the graded columns.
    store.reconcile_game_predictions(pd.DataFrame([
        {"game_id": r["game_id"], "home_score": r["actual_home_score"],
         "away_score": r["actual_away_score"]}
        for r in rows
    ]))


# --------------------------------------------------------------------------
# the floor
# --------------------------------------------------------------------------

def test_a_band_under_the_floor_produces_no_signal():
    _write(0.65, 1, trust.TRUST_MIN_N - 1)
    assert trust.trust_signal("g", 0.65) is None


def test_exactly_at_the_floor_produces_a_signal():
    _write(0.65, 1, trust.TRUST_MIN_N)
    signal = trust.trust_signal("g", 0.65)
    assert signal is not None
    assert signal["n"] == trust.TRUST_MIN_N


def test_the_floor_is_thirty():
    # Spec §4's number, not this file's preference.
    assert trust.TRUST_MIN_N == 30


# --------------------------------------------------------------------------
# no data, no row
# --------------------------------------------------------------------------

def test_an_empty_store_produces_no_signal():
    # The case §2 calls out. `_connect` CREATEs the tables, so an absent store
    # arrives here as empty bands rather than an exception.
    assert trust.trust_signal("g", 0.65) is None


def test_a_store_that_cannot_be_read_produces_no_signal(monkeypatch):
    from cfb_predictor.tracking import store

    def _boom(*_a, **_k):
        raise RuntimeError("locked")

    monkeypatch.setattr(store, "get_calibration", _boom)
    assert trust.trust_signal("g", 0.65) is None


@pytest.mark.parametrize("prob", [None, True, False, float("nan"), float("inf"), "0.65", 1.5, -0.1])
def test_a_probability_that_is_not_one_produces_no_signal(prob):
    _write(0.65, 1, 60)
    assert trust.trust_signal("g", prob) is None


# --------------------------------------------------------------------------
# proven markets only
# --------------------------------------------------------------------------

def test_it_reads_the_moneyline_and_not_the_ats_or_total():
    # The decision under test. If this ever starts reporting a spread or total
    # record, it is reporting 59 graded pairs spread across bands of 30.
    assert trust.MARKET == "winner"
    assert "moneyline" in trust.MARKET_LABEL


# --------------------------------------------------------------------------
# strength
# --------------------------------------------------------------------------

def test_strength_rises_with_the_gap():
    assert trust.strength(0.90, 0.65, 100) > trust.strength(0.66, 0.65, 100)


def test_strength_rises_with_n_at_a_fixed_gap():
    # The direction that is easy to state backwards: the same discrepancy on more
    # picks is harder to write off as chance.
    assert trust.strength(0.70, 0.65, 400) > trust.strength(0.70, 0.65, 40)


def test_a_perfectly_calibrated_band_has_no_strength():
    assert trust.strength(0.65, 0.65, 500) == 0.0


def test_strength_is_bounded():
    for rate, mean, n in [(1.0, 0.0, 10**6), (0.0, 1.0, 5), (0.5, 0.5, 1)]:
        assert 0.0 <= trust.strength(rate, mean, n) <= 1.0


def test_strength_of_an_empty_band_is_zero():
    assert trust.strength(0.5, 0.5, 0) == 0.0


# --------------------------------------------------------------------------
# the headline
# --------------------------------------------------------------------------

def test_the_headline_states_the_rate_the_bar_draws():
    # `SignalRows.assertFigureIsStated` compares against the whole percents
    # `fmt.pct` can print for the rate, so the stated figure must be one of them.
    text = trust.headline(0.7556, 0.6549)
    assert str(round(0.7556 * 100)) in text


def test_the_headline_never_states_a_finer_percent_than_the_bar_draws():
    text = trust.headline(0.7556, 0.6549)
    assert "75.56" not in text and "74.4" not in text


def test_the_headline_keeps_the_stated_probability_a_band():
    # `mean_predicted` is the MEAN of a band a quarter wide, so without the tilde
    # the sentence would claim picks of exactly that probability.
    assert "~" in trust.headline(0.71, 0.65)


def test_the_headline_is_twelve_words_or_fewer():
    assert len(trust.headline(0.7556, 0.6549).split()) <= 12


def test_the_headline_is_a_record_not_a_forecast():
    # No present-tense promise of a future rate.
    assert "will be" not in trust.headline(0.7556, 0.6549)


# --------------------------------------------------------------------------
# the payload
# --------------------------------------------------------------------------

def test_the_payload_carries_the_figures_its_visual_draws():
    _write(0.65, 1, 60)
    signal = trust.trust_signal("g", 0.65)
    assert signal["visual"] == trust.VISUAL == "reliability_bar"
    assert trust.RATE_FIGURE in signal["headline"]["figures"]
    assert 0.0 <= signal["headline"]["figures"][trust.RATE_FIGURE] <= 1.0


def test_the_payload_is_pre_kickoff_only():
    _write(0.65, 1, 60)
    assert trust.trust_signal("g", 0.65)["pre_kickoff_only"] is True


def test_the_payload_names_its_sport_and_game():
    _write(0.65, 1, 60)
    signal = trust.trust_signal("401520145", 0.65)
    assert signal["sport"] == "cfb"
    assert signal["game_id"] == "401520145"


# --------------------------------------------------------------------------
# the endpoint
# --------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from cfb_predictor.api import routes

    game = {
        "game_id": "401520145", "season": 2099, "week": 1,
        "home_team": "Texas", "away_team": "Ohio State",
        "gameday": "2099-08-30T16:00:00", "status": "pre",
        "home_win_prob": 0.65, "away_win_prob": 0.35,
    }
    monkeypatch.setattr(facts, "_load_game", lambda gid: (2099, 1, game))
    monkeypatch.setattr(facts, "_week_row", lambda s, w, g: {
        "home_win_prob": 0.65, "away_win_prob": 0.35, "rebuilt": False,
    })
    monkeypatch.setattr(facts, "_snapshot_prediction", lambda s, w, g: {
        "home_win_prob": 0.65, "away_win_prob": 0.35,
        "predicted_margin": 3.1, "predicted_total": 55.0,
    })
    # `_current_prediction` is what `game_pick` calls for an UPCOMING game, and in
    # non-PUBLIC_MODE it delegates to `routes.get_game_prediction`, which 404s a
    # game it cannot find. Patched rather than seeded through the routes: the
    # point of these tests is the endpoint's shape, not how a forecast is built.
    monkeypatch.setattr(facts, "_current_prediction", lambda s, w, g: {
        "home_win_prob": 0.65, "away_win_prob": 0.35,
        "predicted_margin": 3.1, "predicted_total": 55.0,
    })
    monkeypatch.setattr(facts, "_props", lambda s, w: [])
    monkeypatch.setattr(routes, "current_season_and_week", lambda: (2099, 1))

    from cfb_predictor.api.main import app

    return TestClient(app)


def test_the_endpoint_is_served_under_api(client):
    # THE PREFIX. F1 shipped this router at the root: its own `TestClient` tests
    # passed, and the page could not call it, because the frontend reaches the
    # backend through `/api` in production (`client.ts`'s `BASE_URL`) and in
    # development (`vite.config.ts` proxies only '/api'). Read off the schema
    # rather than `app.routes`: FastAPI wraps an included router in an object with
    # no `.path`, so the route list does not show included endpoints at all.
    from cfb_predictor.api.main import app

    assert "/api/signals/{game_id}" in app.openapi()["paths"]


def test_an_unknown_game_is_a_404(client, monkeypatch):
    from fastapi import HTTPException

    from cfb_predictor.api import facts as facts_mod

    def _raise(_gid):
        raise HTTPException(status_code=404, detail="Unknown game")

    monkeypatch.setattr(facts_mod, "_load_game", _raise)
    assert client.get("/api/signals/nope").status_code == 404


def test_a_game_with_no_signal_answers_an_empty_list(client):
    # `{"signals": []}` is a valid, complete answer -- spec §2's "no data, no row".
    body = client.get("/api/signals/401520145").json()
    assert body["signals"] == []
    assert body["sport"] == "cfb"
    assert body["id"] == "401520145"


def test_the_success_and_failure_shapes_are_the_same(client, monkeypatch):
    # A client reading `sport` or `id` must not get a different object only when
    # the backend is failing, which is the moment it can least cope.
    from cfb_predictor.api import signals as signals_mod

    monkeypatch.setattr(
        signals_mod, "signals_for_game",
        lambda _g: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    failed = client.get("/api/signals/401520145").json()
    assert failed == {"sport": "cfb", "id": "401520145", "signals": []}


def test_a_broken_adapter_does_not_take_down_the_endpoint(client, monkeypatch):
    from cfb_predictor.signals import trust as trust_mod

    def _boom(*_a, **_k):
        raise RuntimeError("adapter down")

    monkeypatch.setattr(trust_mod, "trust_signal", _boom)
    body = client.get("/api/signals/401520145").json()
    assert body["signals"] == []


def test_the_endpoint_caps_at_max_signals():
    assert MAX_SIGNALS == 3