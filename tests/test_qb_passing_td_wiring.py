"""The QB passing-TD model is FITTED but was never WIRED: nothing called
`fit_qb_passing_td_model`, nothing persisted it, nothing loaded it, and nothing
served it. `tests/test_qb_passing_td.py` covers the model to the last branch — the
distribution choice, the line, the tail probabilities, the refusal to fill a
missing feature with zero — and all of it was reachable only from a test.

This file covers the four joins that make it a market instead of a module:

    train_all  -> artefact on disk      (a model nobody retrains goes stale
                                          against the label version, silently)
    manifest   -> the fit's own numbers (so a reader can see WHICH distribution
                                          won, rather than trusting a docstring
                                          that NFL already got wrong once)
    load_models-> a key routes can ask for
    routes     -> a prop row for a QB, and NO prop row for a non-QB

The test that matters most is `test_the_fitted_columns_are_all_served`. The model
is fitted on `passing_tds_roll`, which `player_usage` deliberately keeps OUT of
`PLAYER_FEATURE_COLUMNS` "because widening the list would change the feature count
of every already-committed artefact". That is the right call, and it is also
exactly how a model ends up fitted on a feature the served row never carries —
which `qb_passing_td.expected_passing_tds` raises on, per player, because it has
already happened once and served every QB from a constant-zero roll. So the join
is asserted from BOTH ends here: the columns the fit records, and the columns
`build_features_for_player` actually emits.

Run: python -m pytest tests/test_qb_passing_td_wiring.py -q
"""
from __future__ import annotations

import pickle
import pickle

import numpy as np
import pandas as pd
import pytest

from cfb_predictor.features import player_usage
from cfb_predictor.models import manifest, qb_passing_td

from test_manifest import _fake_fbs_teams, _fake_games, _fake_player_stats


def _qb_player_stats(seasons, games_df=None, n_qbs=3):
    """Weekly player stats with real QBs, so the passing-TD fit has something to
    fit on. Overdispersed on purpose: a constant passing-TD count would make the
    negative binomial trivially lose and the manifest assertion about recording
    the winner would pass on a fixture that cannot tell the two apart.
    """
    rng = np.random.default_rng(11)
    rows = []
    for season in seasons:
        for week in range(1, 6):
            for i in range(n_qbs):
                rows.append(
                    {
                        "player_id": f"qb{i}", "player_name": f"QB {i}", "position": "QB",
                        "recent_team": f"T{i}", "season": season, "week": week,
                        "passing_yards": int(rng.integers(120, 340)),
                        "passing_tds": int(rng.integers(0, 4)),
                        "rushing_yards": int(rng.integers(-20, 60)),
                        "rushing_tds": int(rng.integers(0, 2)),
                        "receiving_yards": int(rng.integers(0, 40)),
                        "receiving_tds": int(rng.integers(0, 1)),
                        "receptions": int(rng.integers(0, 5)),
                        "targets": float("nan"), "carries": int(rng.integers(0, 4)),
                    }
                )
            # One non-QB, so "only QBs get this market" is testable rather than
            # assumed -- the pool always contains a player who must NOT get a row.
            rows.append(
                {
                    "player_id": "rb1", "player_name": "Runner", "position": "RB",
                    "recent_team": "T0", "season": season, "week": week,
                    "passing_yards": 0, "passing_tds": 0,
                    "rushing_yards": int(rng.integers(50, 140)), "rushing_tds": int(rng.integers(0, 2)),
                    "receiving_yards": int(rng.integers(0, 30)), "receiving_tds": 0,
                    "receptions": 2, "targets": float("nan"), "carries": 18,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trained(monkeypatch, tmp_path):
    """`train_all` against fake data, with every artefact landed in `tmp_path`."""
    from cfb_predictor import config

    # Four seasons, not two: `train_all` refuses without a walk-forward fold and
    # tests/test_manifest.py gets real folds from the season count rather than
    # stubbing `prepare_folds`. Matching that keeps this fixture honest -- a
    # stubbed fold would train the QB model on a path no deployment takes.
    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest.games_data, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.games_data, "fetch_fbs_teams", _fake_fbs_teams)
    monkeypatch.setattr(
        manifest.player_stats, "fetch_weekly_player_stats", lambda s, g=None: _qb_player_stats(s, g)
    )
    written = manifest.train_all(seasons)
    return written, tmp_path


# --- the fit runs, and its artefact lands -------------------------------


def test_train_all_persists_the_qb_passing_td_artefact(trained):
    _, tmp_path = trained
    assert (tmp_path / manifest.QB_PASSING_TD_MODEL_FILENAME).exists(), (
        "the model is fitted in tests/test_qb_passing_td.py and never written anywhere, "
        "so nothing that ships can ever load it"
    )


def test_the_manifest_records_the_fit_not_just_its_existence(trained):
    """NFL's docstring described an artefact as it was BEFORE a retrain onto
    anytime-TD label v2, and nothing noticed, because prose is not asserted. The
    manifest is where a reader looks instead, so the winner goes there."""
    written, _ = trained
    entry = written["qb_passing_td"]
    assert entry["distribution"] in ("poisson", "negative_binomial")
    assert set(entry["log_loss"]) == {"poisson", "negative_binomial"}
    assert entry["n_train"] > 0
    assert entry["feature_cols"]
    # The two numbers that decide the choice, read back rather than asserted in prose.
    assert entry["variance_ratio"] > 0
    if entry["distribution"] == "negative_binomial":
        assert entry["alpha"] is not None and entry["alpha"] > 0
    else:
        assert entry["alpha"] is None


def test_no_qb_history_removes_the_artefact_rather_than_saving_an_empty_one(monkeypatch, tmp_path):
    """Same discipline as the yardage loop: a market with no rows has no artefact.
    A model fitted on nothing raises, and the alternative — an artefact that
    predicts a flat distribution for every QB forever — is worse."""
    from cfb_predictor import config

    # Four seasons, not two: `train_all` refuses without a walk-forward fold and
    # tests/test_manifest.py gets real folds from the season count rather than
    # stubbing `prepare_folds`. Matching that keeps this fixture honest -- a
    # stubbed fold would train the QB model on a path no deployment takes.
    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest.games_data, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.games_data, "fetch_fbs_teams", _fake_fbs_teams)
    monkeypatch.setattr(
        manifest.player_stats, "fetch_weekly_player_stats",
        # The RB-only fixture from tests/test_manifest.py: passing_tds is all
        # zeros and there is no QB at all.
        _fake_player_stats,
    )

    written = manifest.train_all(seasons)
    assert not (tmp_path / manifest.QB_PASSING_TD_MODEL_FILENAME).exists()
    # And the manifest says so, rather than omitting the key and leaving a reader
    # to infer "not trained" from an absence.
    assert "qb_passing_td" not in written


# --- the served feature row can carry what the fit was fitted on ---------


def test_the_fitted_columns_are_all_served(trained):
    """The load-bearing join. `passing_tds_roll` is deliberately NOT in
    `PLAYER_FEATURE_COLUMNS`, so the fit's columns and the served row's columns are
    maintained in two different places — and a model fitted on a feature the
    served row lacks scores every QB from a constant zero. `expected_passing_tds`
    raises per player because that has already happened once; this asserts it
    globally, before any player is scored."""
    written, _ = trained
    fitted_cols = set(written["qb_passing_td"]["feature_cols"])

    history = _qb_player_stats([2022], None)
    row = player_usage.build_features_for_player("qb0", history)
    assert row is not None, "the fixture must produce a servable row for this to mean anything"
    served = set(row.index)

    assert fitted_cols <= served, (
        f"fitted on {sorted(fitted_cols - served)} which the served row does not carry; "
        "every QB would be scored from a constant-zero feature"
    )
    # And the feature that caused the risk is genuinely part of the fit, so this
    # test cannot pass vacuously on a fit that never used it.
    assert "passing_tds_roll" in fitted_cols


def test_load_models_returns_the_fitted_model_under_its_own_key(trained):
    """Under `player_models`, keyed by `PASSING_TD_MARKET` -- the same place and
    the same kind of key as every other player market, so `routes` reaches it the
    way it reaches `passing_yards`. `passing_tds` is deliberately NOT a
    `YARDAGE_TARGETS` member, so nothing else in the manifest can collide with it.
    """
    _, _ = trained
    player_models = manifest.load_models()["player_models"]
    assert qb_passing_td.PASSING_TD_MARKET in player_models
    assert player_models[qb_passing_td.PASSING_TD_MARKET]["model"] is not None
    assert player_models[qb_passing_td.PASSING_TD_MARKET]["feature_cols"]


def test_load_models_refuses_a_payload_whose_qb_columns_are_not_served(trained):
    """The same property as the test above, enforced at LOAD time for the whole
    payload rather than per player at score time — so the deployment fails to come
    up rather than quietly serving zeros to every QB for a season.

    **The artefact is rewritten, not the manifest**, because the artefact is what
    `expected_passing_tds` reads its columns from at score time. An earlier version
    of this test edited `manifest.json` and failed to prove the guard fires: the
    guard reads the pickle, and a manifest edit simply never reached it. A guard
    tested against the wrong artefact is worse than no guard, because it reports
    the property as covered.
    """
    _, tmp_path = trained
    path = tmp_path / manifest.QB_PASSING_TD_MODEL_FILENAME
    fitted = pickle.loads(path.read_bytes())
    fitted["feature_cols"] = list(fitted["feature_cols"]) + ["not_a_column"]
    path.write_bytes(pickle.dumps(fitted))

    with pytest.raises(Exception) as exc:
        manifest.load_models()
    assert "not_a_column" in str(exc.value)


# --- the call a QB's row produces ---------------------------------------


def test_a_fitted_artefact_produces_a_call_for_a_qb_row(trained):
    """End to end through the artefact, not a hand-built dict: fit -> pickle ->
    load -> feature row -> call. Every stage above can pass on its own while the
    chain between them is broken, which is the shape this whole file exists for."""
    written, _ = trained
    history = _qb_player_stats([2022], None)
    row = player_usage.build_features_for_player("qb0", history)
    assert row is not None

    fitted = manifest.load_models()["player_models"][qb_passing_td.PASSING_TD_MARKET]
    call = qb_passing_td.qb_passing_td_call(fitted, row)
    assert call is not None
    assert call["market"] == qb_passing_td.PASSING_TD_MARKET
    assert call["line_source"] == qb_passing_td.MODEL_LINE_SOURCE
    assert 0.0 <= call["over_prob"] <= 1.0
    assert 0.0 <= call["under_prob"] <= 1.0
    # A half-point line against an integer count cannot tie, and the module
    # carries `push_prob` explicitly so a consumer never has to infer it.
    assert call["push_prob"] == 0.0
    assert call["side"] == ("over" if call["over_prob"] > call["under_prob"] else "under")


def test_the_call_is_none_when_no_model_was_trained():
    """A deployment whose manifest predates this market must serve NO passing-TD
    row, not a zeroed one."""
    assert qb_passing_td.qb_passing_td_call(None, pd.Series(dtype=float)) is None
    assert qb_passing_td.qb_passing_td_call({"model": None}, pd.Series(dtype=float)) is None


def test_the_line_is_a_half_point_because_that_is_a_sportsbook_line(trained):
    """Not the model's own expectation: a whole number would read as a real line
    on a market where every actual line is a half. The module asserts this; this
    asserts the SERVED artefact still does it, because the artefact is fitted in
    one process and read in another."""
    written, _ = trained
    history = _qb_player_stats([2022], None)
    row = player_usage.build_features_for_player("qb1", history)
    fitted = manifest.load_models()["player_models"][qb_passing_td.PASSING_TD_MARKET]
    call = qb_passing_td.qb_passing_td_call(fitted, row)
    assert call is not None
    assert float(call["line"]) % 1 == 0.5
    assert call["line"] >= qb_passing_td.MIN_MODEL_LINE
