"""The committed QB passing-TD artefact, and the manifest entry that describes it.

## Why this file exists

`models/qb_passing_td_model.pkl` and `models/manifest.json` are both committed, and
a retrain rewrites both. **Nothing asserted that they agree.** That gap is not
hypothetical: this module's own docstring carried the previous fit's figures --
26,926 rows, variance-to-mean 1.390346, alpha 7.36571588075843 -- and every one of
them was stale the moment the artefact was refitted, with nothing to notice. NFL
learned the same lesson: its docstring described its artefact as it was *before* a
retrain onto anytime-TD label v2 features.

So the artefact's numbers live in `manifest.json` (read by whoever retrains next),
and the assertions live here. Prose is not the mechanism; it is the cautionary
half of one.

## Why these are cheap

They read the two committed files and compare them. No fit, no fixtures, no
network -- so they run in milliseconds and cannot themselves go stale, which is the
failure they exist to catch.

Run: python -m pytest tests/test_qb_passing_td_artefact.py -q
"""
from __future__ import annotations

import json
import math
import pickle

import pytest

from cfb_predictor.models import manifest, qb_passing_td

ARTEFACT = manifest.MODELS_DIR / manifest.QB_PASSING_TD_MODEL_FILENAME
COMMITTED = json.loads((manifest.MODELS_DIR / "manifest.json").read_text())


@pytest.fixture(scope="module")
def fitted():
    """The committed pickle, loaded."""
    assert ARTEFACT.exists(), (
        f"{ARTEFACT.name} is not committed. `load_models` reads the manifest, so "
        "without the artefact the QB passing-TD market stays dark — correctly, "
        "and visibly."
    )
    return pickle.loads(ARTEFACT.read_bytes())


def test_the_manifest_declares_the_market(fitted):
    assert "qb_passing_td" in COMMITTED, (
        "the artefact is committed but the manifest does not declare it, so "
        "`load_models` never reads it and the market never serves"
    )


def test_the_manifest_entry_describes_the_committed_artefact(fitted):
    """THE assertion. Every field, from the pickle, not from a literal."""
    entry = COMMITTED["qb_passing_td"]
    assert entry["distribution"] == fitted["distribution"]
    assert entry["n_train"] == fitted["n_train"]
    assert entry["feature_cols"] == list(fitted["feature_cols"])
    assert entry["variance_ratio"] == pytest.approx(fitted["variance_ratio"])
    for name, value in fitted["log_loss"].items():
        assert entry["log_loss"][name] == pytest.approx(value)
    # `alpha` is None for a Poisson artefact and a real number for the NB2 one --
    # so "both None" and "one wrong" are distinguishable, which a truthiness check
    # could not tell.
    if fitted["distribution"] == "negative_binomial":
        assert entry["alpha"] == pytest.approx(fitted["alpha"])
        assert entry["alpha"] > 0
    else:
        assert entry["alpha"] is None


def test_the_committed_fit_actually_chose_a_distribution(fitted):
    """A manifest entry that recorded neither winner would satisfy every other
    test here."""
    assert fitted["distribution"] in ("poisson", "negative_binomial")


def test_the_recorded_winner_is_the_lower_log_loss(fitted):
    """The manifest is not just a copy -- it is a copy of a *decision*. A retrain
    that recorded a winner contradicting its own scores is the shape of "the
    docstring said POISSON WINS"."""
    entry = COMMITTED["qb_passing_td"]
    winner = min(entry["log_loss"], key=entry["log_loss"].get)
    assert winner == entry["distribution"], (
        f"the manifest names {entry['distribution']!r} but its own log losses say "
        f"{winner!r} is lower"
    )


def test_both_distributions_were_fitted_and_recorded(fitted):
    """Not one of them. A single score is not a comparison, and it is how a
    distribution 'wins' by default rather than by measurement."""
    assert set(fitted["log_loss"]) == {"poisson", "negative_binomial"}


def test_the_fit_saw_a_real_number_of_qb_seasons(fitted):
    """A handful of rows would fit, and would be worthless. NFL's was 5,179; a
    market fitted on 200 rows is not a market."""
    assert fitted["n_train"] >= 1000, (
        f"the committed artefact was fitted on {fitted['n_train']} rows, which is "
        "too few for the dispersion choice to mean anything"
    )


def test_every_fitted_column_is_one_the_served_feature_row_carries(fitted):
    """The load-time property, asserted against the COMMITTED artefact rather than
    a freshly fitted one. `load_models` raises on a mismatch; this says the
    committed pair is not that mismatch."""
    served = set(COMMITTED["player_feature_cols"]) | {player_usage_column()}
    missing = [c for c in fitted["feature_cols"] if c not in served]
    assert not missing, (
        f"the committed artefact is fitted on {fitted['feature_cols']}, and {missing} "
        "are not columns the served row carries -- every QB would score from a "
        "constant zero"
    )


def player_usage_column() -> str:
    from cfb_predictor.features import player_usage

    return player_usage.PASSING_TDS_ROLL_COLUMN


def test_the_fitted_columns_are_exactly_the_declared_ones(fitted):
    """Not a subset. `fit_qb_passing_td_model` reads whatever columns it is handed,
    so the artefact's own list IS the contract -- and it must be the four this
    market means to use, not whatever a caller happened to pass."""
    assert list(fitted["feature_cols"]) == [
        *qb_passing_td.MU_FEATURE_COLUMNS,
        "passing_tds_roll",
    ]


def test_the_artefact_is_not_empty_or_degenerate(fitted):
    """A pickled object of `None`s satisfies most field checks. This asserts the
    model can actually predict."""
    assert fitted.get("model") is not None
    assert hasattr(fitted["model"], "predict")
    assert fitted.get("distribution") in ("poisson", "negative_binomial")


def test_alpha_is_interior_not_pinned_to_a_bound(fitted):
    """The reason CFB's fit is interesting at all: NFL's alpha hit a bound because
    its counts were underdispersed. CFB's lands inside `_NB_LOG_ALPHA_BOUNDS`, and
    a committed artefact with alpha at a bound is the NFL failure mode arriving
    late."""
    lo, hi = -8.0, 12.0
    if fitted["distribution"] == "negative_binomial":
        assert lo < math.log(fitted["alpha"]) < hi, (
            f"alpha={fitted['alpha']} sits at or past a fit bound; the dispersion "
            "parameter was pinned rather than estimated, which is what NFL's "
            "underdispersed counts produced"
        )


def test_the_manifest_and_the_artefact_were_written_by_the_same_shape_of_run(fitted):
    """`load_models` gates the whole payload on `player_feature_cols` via
    `artifact_fingerprint`. A manifest whose QB entry was added by hand while the
    fingerprint still describes an older training frame is the shape that fails at
    load."""
    fingerprint = COMMITTED["artifact_fingerprint"]
    assert fingerprint["player_feature_cols"] == COMMITTED["player_feature_cols"]
    # The QB model reads a SUBSET of those plus its own column, so its entry must
    # not widen the shared list.
    assert set(fitted["feature_cols"]) - {player_usage_column()} <= set(
        fingerprint["player_feature_cols"]
    )
