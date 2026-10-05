"""The last join for CFB's QB passing-TD market: `routes._get_player_props_live`.

`tests/test_qb_passing_td.py` covers the model, and
`tests/test_qb_passing_td_wiring.py` covers fit -> artefact -> manifest -> load.
This file covers the hop from a loaded artefact to a row a browser renders, and
the two ways that hop can go wrong while every other test still passes:

- **a NON-QB gets the market.** `passing_tds` is 0 for every running back, so an
  RB served a passing-TD call would be a real-looking over/under on a player who
  never threw. The position gate is the whole defence, so it is asserted from the
  outside rather than assumed from the model's own name.

- **a QB whose row cannot be scored gets NO call, not a zeroed one.** A `None` is
  the honest answer: `qb_passing_td_call` returns None for a QB with no history,
  and turning that into a 0.0 line would be the all-zero-region defect this
  function already skips players for.

Run: python -m pytest tests/test_qb_passing_td_serving.py -q
"""
from __future__ import annotations

import pandas as pd
import pytest

from cfb_predictor.api import routes
from cfb_predictor.models import qb_passing_td

# A minimal but REALISTIC fitted dict: `qb_passing_td_call` only needs `model`,
# `distribution` and optionally `alpha`, and the served key comes from
# `PASSING_TD_MARKET` so a rename cannot leave the route and the model disagreeing.
FITTED = {
    "distribution": "negative_binomial",
    "alpha": 7.36571588075843,
    "model": object(),
    "feature_cols": list(qb_passing_td.MU_FEATURE_COLUMNS)
    + ["passing_tds_roll"],
    "n_train": 26926,
}


class _FakeRegressor:
    """A `predict` that returns a fixed mu, so a call is producible without
    training. `qb_passing_td.expected_passing_tds` reindexes onto the fitted
    columns first, so the columns matter and are supplied by the fixture row.
    """

    def __init__(self, mu: float = 1.8):
        self.mu = mu

    def predict(self, X):
        return [self.mu] * len(X)


def _fitted(mu: float = 1.8) -> dict:
    return {**FITTED, "model": _FakeRegressor(mu)}


def _install(monkeypatch, history_rows, feature_row_for):
    monkeypatch.setattr(
        routes, "_load_models_cached",
        lambda: {"player_models": {qb_passing_td.PASSING_TD_MARKET: _fitted()}},
    )
    monkeypatch.setattr(routes, "_load_player_history", lambda season: pd.DataFrame(history_rows))
    monkeypatch.setattr(
        routes.games_data, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"home_team": "Alabama", "away_team": "Auburn", "week": week}]
        ),
    )
    monkeypatch.setattr(
        routes.games_data, "fetch_current_season_partial",
        lambda: pd.DataFrame(columns=["home_team", "away_team", "week"]),
    )
    # The yardage path needs a real regressor per market; the QB call is what is
    # under test, so the flat prop models are stubbed out of the way.
    monkeypatch.setattr(
        routes.player_props, "predict_props",
        lambda models, row, position=None: {"anytime_td_prob": 0.5},
    )
    monkeypatch.setattr(routes.player_usage, "build_features_for_player", feature_row_for)


FEATURE_ROW = pd.Series(
    {
        "passing_yards_roll": 240.0,
        "rushing_yards_roll": 5.0,
        "receiving_yards_roll": 10.0,
        "passing_tds_roll": 1.6,
        "rushing_yards_roll_alt": 0.0,
    }
)

QB = {"player_id": "qb1", "player_name": "A. QB", "position": "QB",
      "recent_team": "Alabama", "season": 2026}
RB = {"player_id": "rb1", "player_name": "A. Back", "position": "RB",
      "recent_team": "Alabama", "season": 2026}


def test_a_qb_row_carries_the_passing_td_call(monkeypatch):
    _install(monkeypatch, [QB], lambda pid, history: FEATURE_ROW)
    props = routes._get_player_props_live(2026, 3)
    assert len(props) == 1
    call = props[0]["qb_passing_td"]
    assert call is not None, "a fitted model, a QB and a servable row must produce a call"
    assert call["market"] == qb_passing_td.PASSING_TD_MARKET
    assert call["line_source"] == qb_passing_td.MODEL_LINE_SOURCE
    assert call["line"] >= qb_passing_td.MIN_MODEL_LINE


def test_a_non_qb_never_gets_the_passing_td_market(monkeypatch):
    """The gate. `passing_tds` is 0 for every RB, so an RB served an over/under
    would be a real-looking line on a player who never threw -- and unlike the
    other prop rows there is nothing else on the payload that would look wrong."""
    _install(monkeypatch, [RB], lambda pid, history: FEATURE_ROW)
    props = routes._get_player_props_live(2026, 3)
    assert len(props) == 1
    assert props[0].get("qb_passing_td") is None


def test_a_qb_with_no_history_gets_no_call_rather_than_a_zeroed_one(monkeypatch):
    """`build_features_for_player` returning None means the player is skipped
    entirely, which is this function's existing discipline -- the alternative is a
    flat distribution for every QB the store has no row for."""
    _install(monkeypatch, [QB], lambda pid, history: None)
    assert routes._get_player_props_live(2026, 3) == []


def test_no_fitted_model_means_no_call_for_anyone(monkeypatch):
    """A deployment whose manifest predates this market serves every OTHER player
    market and no passing-TD row at all. Not a 0.0 line."""
    _install(monkeypatch, [QB], lambda pid, history: FEATURE_ROW)
    # Re-stub AFTER `_install`, which points at a fitted model.
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {"player_models": {"anytime_td": object()}})
    props = routes._get_player_props_live(2026, 3)
    assert len(props) == 1
    assert props[0].get("qb_passing_td") is None


def test_a_qb_call_that_raises_does_not_cost_the_qb_their_other_markets(monkeypatch):
    """A prop feed that 500s because ONE market blew up would take the anytime-TD
    and yardage rows with it -- those are the rows the page has today. The QB call
    is additive, so its failure is contained to the QB call."""
    _install(monkeypatch, [QB], lambda pid, history: FEATURE_ROW)

    def boom(*a, **k):
        raise RuntimeError("passing-TD fit exploded")

    monkeypatch.setattr(routes.qb_passing_td, "qb_passing_td_call", boom)
    props = routes._get_player_props_live(2026, 3)
    assert len(props) == 1, "the player must still be served their other markets"
    assert props[0].get("qb_passing_td") is None
    assert props[0]["anytime_td_prob"] == 0.5


def test_the_key_is_stable_whatever_the_market_is_renamed(monkeypatch):
    """The route must publish the model's own `PASSING_TD_MARKET`, not a literal.
    A rename that changed the pickle key but not this string would ship a market
    the page silently never reads."""
    _install(monkeypatch, [QB], lambda pid, history: FEATURE_ROW)
    props = routes._get_player_props_live(2026, 3)
    assert props[0]["qb_passing_td"]["market"] == qb_passing_td.PASSING_TD_MARKET


def test_the_call_survives_a_json_round_trip(monkeypatch):
    """The row is cached into the public snapshot as JSON and read back by the
    frontend, so a numpy scalar anywhere in the call is a 500 at serialisation
    time -- long after this function returns."""
    import json

    _install(monkeypatch, [QB], lambda pid, history: FEATURE_ROW)
    props = routes._get_player_props_live(2026, 3)
    encoded = json.dumps(props[0]["qb_passing_td"])
    assert json.loads(encoded)["market"] == qb_passing_td.PASSING_TD_MARKET
