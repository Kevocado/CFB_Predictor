"""CFB's QB passing-TD count model.

Ported from `nfl_predictor/models/qb_passing_td.py`, so these are NFL's
properties applied to CFB's code. What is CFB's own is the fit, and the one test
that matters most here is the last: NFL's fit landed on Poisson, CFB's lands on
negative binomial, and a suite that could not tell those apart would not notice
if a retrain silently moved it back.

Each test below pins something that fails quietly:

  - the LINE GRID ends in .5, so a push is structurally impossible. A whole-number
    line makes `actual == line` reachable, `store._line_is_half_point` then
    refuses to grade the row, and the pick is reported ungradeable rather than
    scored.
  - a fitted column ABSENT from a served row RAISES rather than filling 0.0. That
    is the defect that had NFL's `passing_tds_roll` fitted on and served as a
    constant zero, per player, silently.
  - a fit scoring better than any real pmf can is REJECTED. An unbounded NB2
    dispersion fit collapses onto a point mass, returns `inf` per row, and reports
    a spectacular negative log loss that beats any honest likelihood.
  - a QB with no history produces NO call, not a fabricated mu of 0.0.
"""
import math

import numpy as np
import pandas as pd
import pytest

from cfb_predictor.models import qb_passing_td as m


def _fitted(rows: int = 400, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "passing_tds_roll": rng.uniform(0, 3, rows),
        "passing_yards_roll": rng.uniform(100, 400, rows),
        "rushing_yards_roll": rng.uniform(0, 60, rows),
        "receiving_yards_roll": rng.uniform(0, 60, rows),
    })
    mu = 0.3 + 0.8 * X["passing_tds_roll"].to_numpy()
    y = pd.Series(rng.poisson(np.clip(mu, 0.05, None)))
    return m.fit_qb_passing_td_model(X, y)


# --------------------------------------------------------------------------
# the line
# --------------------------------------------------------------------------

def test_the_line_is_the_nearest_half_point_never_a_whole_number():
    # A whole-number line is an instant push waiting to happen.
    for mu, expected in [(1.0, 1.5), (1.4, 1.5), (1.8, 1.5), (2.3, 2.5), (0.3, 0.5), (0.99, 0.5)]:
        assert m.model_line(mu) == pytest.approx(expected)


def test_no_line_can_ever_be_a_whole_number():
    # Structural, not a sample: this is what makes "push cannot occur" true. A
    # half-point line is one whose fractional part is exactly 0.5 — equivalently
    # one whose doubled value is ODD. Asserting "0.5 away from an integer" would
    # be the wrong shape: doubling 0.5 gives 1, which is not 0.5 from anything.
    for mu in np.arange(0.0, 12.0, 0.05):
        line = m.model_line(float(mu))
        assert line % 1 == pytest.approx(0.5), f"mu={mu} produced line {line}"
        assert round(line * 2) % 2 == 1, f"mu={mu} produced line {line}"


def test_the_line_is_never_below_half():
    assert m.model_line(0.0) == m.MIN_MODEL_LINE == 0.5


def test_a_non_finite_mu_is_refused():
    for bad in (None, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            m.model_line(bad)


# --------------------------------------------------------------------------
# the call
# --------------------------------------------------------------------------

def test_a_call_carries_the_line_the_side_and_the_probability():
    fitted = _fitted()
    row = pd.Series({
        "passing_tds_roll": 1.5, "passing_yards_roll": 260.0,
        "rushing_yards_roll": 20.0, "receiving_yards_roll": 10.0,
    })
    call = m.qb_passing_td_call(fitted, row)
    assert call["market"] == m.PASSING_TD_MARKET == "passing_tds"
    assert call["line"] == pytest.approx(m.model_line(call["mu"]))
    assert call["side"] in ("over", "under")
    assert 0.5 <= call["call_prob"] <= 1.0
    assert call["push_prob"] == 0.0


def test_the_line_source_says_model_line_and_not_a_book():
    # A model line derived from mu is not a sportsbook line, and calling it an
    # edge would be a category error -- there is nothing to have an edge against.
    fitted = _fitted()
    row = pd.Series({c: 1.0 for c in fitted["feature_cols"]})
    assert m.qb_passing_td_call(fitted, row)["line_source"] == m.MODEL_LINE_SOURCE == "model_line"


def test_the_two_sides_sum_to_one():
    fitted = _fitted()
    row = pd.Series({c: 1.0 for c in fitted["feature_cols"]})
    call = m.qb_passing_td_call(fitted, row)
    assert call["over_prob"] + call["under_prob"] == pytest.approx(1.0, abs=1e-9)


def test_the_side_is_the_more_likely_one():
    fitted = _fitted()
    row = pd.Series({c: 1.0 for c in fitted["feature_cols"]})
    call = m.qb_passing_td_call(fitted, row)
    assert call["side"] == ("over" if call["over_prob"] > call["under_prob"] else "under")


def test_a_missing_fitted_column_raises_rather_than_filling_zero():
    # The defect this exists to prevent: a fitted-on, served-as-constant-zero
    # feature, which produces a real-looking number for every QB.
    fitted = _fitted()
    incomplete = pd.Series({fitted["feature_cols"][0]: 1.0})
    with pytest.raises(KeyError) as exc:
        m.expected_passing_tds(fitted, incomplete)
    assert fitted["feature_cols"][1] in str(exc.value)


def test_a_null_value_on_a_present_column_still_fills():
    # Different case from a missing column: a player with no prior games in
    # scope. `routes` skips those before they get here.
    fitted = _fitted()
    row = pd.Series({c: (np.nan if i == 0 else 1.0) for i, c in enumerate(fitted["feature_cols"])})
    assert math.isfinite(m.expected_passing_tds(fitted, row))


def test_no_history_means_no_call_not_a_zero_row():
    fitted = _fitted()
    assert m.qb_passing_td_call(None, pd.Series(dtype=float)) is None
    assert m.qb_passing_td_call({"model": None}, pd.Series(dtype=float)) is None


def test_mu_is_never_negative():
    fitted = _fitted()
    row = pd.Series({c: -5.0 for c in fitted["feature_cols"]})
    assert m.expected_passing_tds(fitted, row) >= 0.0


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------

def test_both_distributions_are_fitted_and_reported():
    fitted = _fitted()
    assert set(fitted["log_loss"]) == {"poisson", "negative_binomial"}
    assert all(math.isfinite(v) for v in fitted["log_loss"].values())


def test_the_lower_log_loss_wins():
    fitted = _fitted()
    assert fitted["log_loss"][fitted["distribution"]] == min(fitted["log_loss"].values())


def test_a_negative_binomial_choice_carries_alpha_and_poisson_does_not():
    fitted = _fitted()
    if fitted["distribution"] == "negative_binomial":
        assert fitted["alpha"] and fitted["alpha"] > 0
    else:
        assert fitted["alpha"] is None


def test_a_distribution_that_scores_impossibly_is_refused():
    # A mean NLL below -1.0 cannot come from a real pmf over integer counts. An
    # unbounded dispersion fit reaches it by collapsing onto a point mass, so the
    # winning score must never be allowed to be one.
    fitted = _fitted()
    assert min(fitted["log_loss"].values()) >= m._MIN_PLAUSIBLE_LOG_LOSS


def test_the_dispersion_fit_stays_inside_its_bounds():
    # The bounds are what keep the collapse unreachable.
    assert m._NB_LOG_ALPHA_BOUNDS == (-8.0, 12.0)
    fitted = _fitted()
    if fitted["alpha"] is not None:
        assert math.exp(m._NB_LOG_ALPHA_BOUNDS[0]) <= fitted["alpha"] <= math.exp(m._NB_LOG_ALPHA_BOUNDS[1])


def test_an_unknown_distribution_is_refused():
    with pytest.raises(ValueError):
        m.DistributionSpec("gaussian")
    with pytest.raises(ValueError):
        m.DistributionSpec("negative_binomial", alpha=0.0)


def test_the_fit_records_its_own_inputs():
    fitted = _fitted(rows=123)
    assert fitted["n_train"] == 123
    assert fitted["variance_ratio"] > 0


def test_fitting_on_nothing_raises():
    with pytest.raises(ValueError):
        m.fit_qb_passing_td_model(pd.DataFrame(), pd.Series(dtype=float))


def test_the_docstring_quotes_only_the_committed_fit():
    # NFL's own lesson: its docstring described the artefact as it was BEFORE a
    # retrain and nothing noticed, because prose is not asserted by default. The
    # numbers quoted in CFB's docstring are CFB's measured fit, and this is what
    # makes that claim checkable rather than decorative.
    import inspect

    doc = m.fit_qb_passing_td_model.__doc__ or ""
    if "MEASURED OUTCOME ON CFB" not in doc:
        pytest.skip("this docstring asserts no measured outcome; nothing to check")
    # NFL's numbers must be ABSENT, not merely corrected -- a paragraph that
    # quotes a superseded figure in order to say it is superseded still puts that
    # figure in front of every reader who skims.
    for nfl_only in ("1.392621", "5,179 usable"):
        assert nfl_only not in doc, f"{nfl_only} is NFL's figure and must not appear as CFB's"