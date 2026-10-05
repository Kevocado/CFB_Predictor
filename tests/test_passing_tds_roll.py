"""The rolling passing-TD feature CFB's QB passing-TD model is fitted on.

The port from `nfl_predictor` is deliberate and near-verbatim, because the two
sites must mean the same thing by the same arithmetic: a feature computed one way
in training and another at serving is how a model ends up fitted on a column it
never sees. NFL's own history is the cautionary tale — `passing_tds_roll` was
fitted on and served as a constant zero for a while, because the training column
and the serving column were written separately.

What this file pins, because each is a way the feature could be wrong:

  - IT IS SHIFTED. A pregame feature cannot know the game being predicted, so the
    rolling mean is over PREVIOUS weeks only (`shift(1)`). An unshifted rolling
    mean includes the target week, which is a lookahead leak and inflates every
    fit built on it.
  - IT IS NOT IN `PLAYER_FEATURE_COLUMNS`. That list is what the anytime-TD
    classifier and every yardage regressor are fitted on and what `predict_props`
    indexes by name, so adding to it changes the feature count of every
    already-committed model.
  - TRAINING AND SERVING CALL THE SAME FUNCTION. `with_passing_tds_roll` and
    `build_features_for_player` are what a test compares, so they cannot drift.
"""
import pandas as pd
import pytest

from cfb_predictor.features import player_usage


def _rows(n_weeks: int = 6) -> pd.DataFrame:
    """One QB's first `n_weeks` weeks, passing a TD every week."""
    return pd.DataFrame([
        {
            "player_id": "qb1", "season": 2024, "week": w + 1,
            "passing_yards": 200.0 + w, "passing_tds": 2.0,
            "rushing_yards": 10.0, "rushing_tds": 0.0,
            "receiving_yards": 0.0, "receiving_tds": 0.0,
            "targets": 30.0, "carries": 0.0, "receptions": 0.0,
        }
        for w in range(n_weeks)
    ])


def test_the_column_is_emitted_and_named():
    df = player_usage.with_passing_tds_roll(_rows())
    assert player_usage.PASSING_TDS_ROLL_COLUMN in df.columns
    assert player_usage.PASSING_TDS_ROLL_COLUMN == "passing_tds_roll"


def test_it_is_not_added_to_the_committed_feature_list():
    # Adding to this list changes the feature count of every already-committed
    # model. The passing-TD column is served as its own.
    assert player_usage.PASSING_TDS_ROLL_COLUMN not in player_usage.PLAYER_FEATURE_COLUMNS


def test_it_is_shifted_so_the_target_week_is_never_in_its_own_feature():
    # A leak check, not a value check. Week 1 has nothing before it, so its value
    # is NaN; from week 2 on, the mean is over the PREVIOUS weeks only. If the
    # shift were dropped, week 6 would read the mean of weeks 1-6 rather than 1-5
    # -- and with a constant 2.0 it would look identical, which is why this is
    # asserted against a varying series below instead.
    varying = _rows(6)
    varying["passing_tds"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    df = player_usage.with_passing_tds_roll(varying)
    roll = df[player_usage.PASSING_TDS_ROLL_COLUMN]
    assert pd.isna(roll.iloc[0]), "week 1 must have no history"
    # week 2 sees only week 1
    assert roll.iloc[1] == pytest.approx(1.0)
    # week 6 sees weeks 1-5
    assert roll.iloc[5] == pytest.approx((1 + 2 + 3 + 4 + 5) / 5)


def test_an_unshifted_rolling_mean_would_fail_that_check():
    # The test above is only meaningful if it actually distinguishes shifted from
    # unshifted, so assert the leak explicitly. On [1..6] with a 5-week window at
    # the last row: SHIFTED averages weeks 1-5 (3.0); UNSHIFTED averages weeks
    # 2-6 (4.0) -- because it includes the week being predicted. The leak is
    # worth one number per row, and on a monotone series it is plainly visible.
    varying = _rows(6)
    varying["passing_tds"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    leaked = varying.groupby("player_id")["passing_tds"].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    assert leaked.iloc[5] == pytest.approx((2 + 3 + 4 + 5 + 6) / 5)
    assert leaked.iloc[5] != pytest.approx((1 + 2 + 3 + 4 + 5) / 5)
    # And the value the real implementation produces at that row.
    df = player_usage.with_passing_tds_roll(varying)
    assert df[player_usage.PASSING_TDS_ROLL_COLUMN].iloc[5] == pytest.approx(3.0)


def test_the_window_is_five_and_declared_once():
    assert player_usage.DEFAULT_ROLL_WINDOW == 5


def test_the_helper_does_not_mutate_its_input():
    # It returns a copy; a helper that mutated its argument would make the
    # training frame and the serving frame the same object by accident.
    original = _rows(3)
    before = list(original.columns)
    player_usage.with_passing_tds_roll(original)
    assert list(original.columns) == before


def test_training_and_serving_derive_the_column_the_same_way():
    # The drift that matters: `build_features_for_player` emits the serving
    # value, `with_passing_tds_roll` produces the training column, and if those
    # two disagree every fit is fitted on a column serving never produces.
    df = player_usage.with_passing_tds_roll(_rows(6))
    served = player_usage.build_features_for_player("qb1", df)
    assert served is not None
    col = player_usage.PASSING_TDS_ROLL_COLUMN
    assert col in served.index
    # The last week's row is the one serving would use, and it must equal the
    # training column's value for that same row.
    assert served[col] == pytest.approx(df[col].iloc[-1], nan_ok=True)