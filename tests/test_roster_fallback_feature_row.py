"""The roster-fallback feature row used column names the model never heard of.

`player_props.predict_props` assembles its input as

    X = feature_row.reindex(feature_cols).fillna(0)

where `feature_cols` is the manifest's `player_feature_cols` --
`passing_yards_roll, rushing_yards_roll, receiving_yards_roll, targets_roll,
carries_roll` (and `receptions_roll` in current code). `reindex` on a Series whose
index does not contain those labels yields NaN for every one of them, and
`.fillna(0)` turns that into an all-zero vector.

The roster fallback in `routes.py` built its placeholder with un-suffixed keys
-- `passing_yards`, `rushing_yards`, `receiving_yards`, `targets`, `carries`,
`receptions` -- so **none** of them matched. Every roster-fallback player was
scored on the all-zero origin, which is a point in the feature space the model
was never trained on, so what came back was the intercept rather than a
prediction. On the Azure build that was 71% of the 2112 CFB rows sharing one
`anytime_td_prob`; the equivalent NFL row is the 62.592 passing yards and
0.111846 TD probability that appeared on 430 of 880 rows.

The same dict also carried `attempts` and `completions`, which are not in
`PLAYER_FEATURE_COLUMNS` at all -- dead weight that was silently discarded by
the same `reindex`.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from cfb_predictor.features import player_usage
from cfb_predictor.models import player_props


def _fallback_series() -> pd.Series:
    """The fallback as `routes.py` constructed it, kept verbatim."""
    return pd.Series(
        {
            "attempts": 5.0,
            "completions": 3.0,
            "passing_yards": 40.0,
            "carries": 2.0,
            "rushing_yards": 10.0,
            "receptions": 2.0,
            "receiving_yards": 20.0,
            "targets": 3.0,
        }
    )


def test_the_old_fallback_matched_none_of_the_model_feature_columns():
    """Why the bug existed, pinned so it cannot be reintroduced silently."""
    feature_cols = player_usage.PLAYER_FEATURE_COLUMNS
    reindexed = _fallback_series().reindex(feature_cols)
    assert reindexed.isna().all(), (
        "this test is meant to fail if the fallback keys are ever fixed; "
        "update it and the regression test alongside"
    )


def test_the_built_fallback_row_uses_the_real_feature_column_names():
    """The fix: the placeholder must be keyed by what the model actually reads."""
    from cfb_predictor.api.routes import _roster_fallback_feature_row

    row = _roster_fallback_feature_row()
    assert set(row.index) == set(player_usage.PLAYER_FEATURE_COLUMNS)
    assert not row.reindex(player_usage.PLAYER_FEATURE_COLUMNS).isna().any()


def test_the_built_fallback_row_reaches_predict_props_unfilled():
    """End to end: no NaN survives the reindex that `predict_props` performs."""
    from cfb_predictor.api.routes import _roster_fallback_feature_row

    row = _roster_fallback_feature_row()
    matrix = row.reindex(player_usage.PLAYER_FEATURE_COLUMNS).fillna(0)
    assert matrix.notna().all()
    assert not (matrix == 0).all(), (
        "a zero row would still be the origin; the placeholder must carry "
        "non-zero values so it is at least inside the trained region"
    )


def test_routes_no_longer_hand_builds_the_placeholder():
    """Guard against the literal dict being pasted back into routes.py."""
    source = inspect.getsource(__import__("cfb_predictor.api.routes", fromlist=["routes"]))
    assert '"passing_yards": 40.0' not in source, (
        "routes.py is constructing the fallback dict inline again; it must call "
        "_roster_fallback_feature_row() so the key set cannot drift from "
        "PLAYER_FEATURE_COLUMNS"
    )
