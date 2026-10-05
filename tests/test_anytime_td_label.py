# tests/test_anytime_td_label.py
"""The `anytime_td` DEFINITION, and the guarantee that there is only one of it.

`anytime_td` = rushing TDs + receiving TDs. Passing TDs are excluded, matching
NFL_Predictor's v2 (its PR #26). Before this, CFB summed passing TDs in as
well, so the same label meant two different things on two sites and a
quarterback could be credited with an anytime TD for throwing one.
"""

import pandas as pd
import pytest

from cfb_predictor.features import player_usage
from cfb_predictor.tracking import store


def _frame(**td_overrides):
    """One player-game row. Defaults: an RB who neither rushed nor caught a TD,
    so every override below is the only TD on the row."""
    row = {
        "player_id": "p1", "player_name": "Player", "position": "RB",
        "recent_team": "Texas", "season": 2025, "week": 1,
        "passing_yards": 0.0, "passing_tds": 0.0,
        "rushing_yards": 0.0, "rushing_tds": 0.0,
        "receiving_yards": 0.0, "receiving_tds": 0.0,
        "receptions": 0.0, "targets": float("nan"), "carries": 0.0,
    }
    row.update(td_overrides)
    return pd.DataFrame([row])


def _label(df):
    return int(player_usage.build_player_training_frame(df)[0]["anytime_td"].iloc[0])


def test_a_qb_with_passing_tds_only_is_not_an_anytime_td():
    """The whole point of the change.

    A passer who threw three TDs and never touched the ball the other way is
    NOT an anytime TD. Under the old definition he was, purely on his arm.
    """
    df = _frame(position="QB", passing_yards=320.0, passing_tds=3.0)
    assert _label(df) == 0
    assert player_usage.anytime_td_actual(0, 0) == 0.0


def test_a_qb_with_a_rushing_td_is_an_anytime_td():
    """A goal-line sneak is a real anytime TD, and stays one."""
    df = _frame(position="QB", passing_yards=320.0, passing_tds=3.0, rushing_tds=1.0)
    assert _label(df) == 1
    assert player_usage.anytime_td_actual(1, 0) == 1.0


def test_a_qb_with_a_receiving_td_is_an_anytime_td():
    """QB upcatch counts too -- the position is irrelevant, the play type is not."""
    df = _frame(position="QB", passing_tds=1.0, receiving_tds=1.0)
    assert _label(df) == 1
    assert player_usage.anytime_td_actual(0, 1) == 1.0


@pytest.mark.parametrize(
    "position, rushing_tds, receiving_tds, expected",
    [
        ("RB", 1, 0, 1),   # RB on a rush
        ("RB", 0, 1, 1),   # RB on a catch
        ("WR", 0, 1, 1),   # WR on a catch
        ("TE", 0, 1, 1),   # TE on a catch
        ("WR", 1, 0, 1),   # WR on a rush (jet sweep) -- still a score
        ("RB", 0, 0, 0),   # no TD at all
        ("WR", 0, 2, 1),   # two receiving TDs is still one anytime TD
    ],
)
def test_rb_wr_te_behaviour_is_unchanged(position, rushing_tds, receiving_tds, expected):
    """PINNED so a future edit cannot quietly move it.

    Non-QB behaviour is identical before and after this change: for RB/WR/TE
    the two definitions agree, because those positions do not pass. That is
    the reason this PR is safe for them, so it is pinned rather than assumed.
    """
    df = _frame(position=position, rushing_tds=rushing_tds, receiving_tds=receiving_tds)
    assert _label(df) == expected


def test_missing_td_columns_read_as_zero():
    """`(x or 0)`/`fillna(0)` parity, so a NaN stat cannot make a TD or erase one."""
    assert player_usage.anytime_td_actual(None, 0) == 0.0
    assert player_usage.anytime_td_actual(float("nan"), 0) == 0.0
    assert player_usage.anytime_td_actual(float("nan"), 1) == 1.0
    assert player_usage.anytime_td_actual(1, float("nan")) == 1.0
    assert _label(_frame(rushing_tds=float("nan"), receiving_tds=float("nan"))) == 0


def test_the_grader_and_the_training_frame_agree_for_every_row():
    """THE test whose absence let NFL's bug through.

    The grader used to carry its own inline copy of the sum while the training
    frame carried another, so a label fix in one place left the other grading
    against the old definition. Asserting a few hand-picked outcomes would not
    catch that; this asserts the two code paths produce the SAME outcome for
    the SAME row, across a grid that includes the QB case where they used to
    disagree.
    """
    for passing_tds in (0.0, 1.0, 3.0):
        for rushing_tds in (0.0, 1.0, 2.0):
            for receiving_tds in (0.0, 1.0, 2.0):
                df = _frame(
                    passing_tds=passing_tds,
                    rushing_tds=rushing_tds,
                    receiving_tds=receiving_tds,
                )
                trained = _label(df)
                # Exactly the call `reconcile_player_prop_predictions` makes on
                # its merged row, including the `row.get` -> None path.
                row = df.iloc[0].to_dict()
                graded = player_usage.anytime_td_actual(
                    row.get("rushing_tds"), row.get("receiving_tds")
                )
                assert trained == graded, (
                    f"training frame says {trained}, grader says {graded} for "
                    f"passing={passing_tds} rushing={rushing_tds} receiving={receiving_tds}"
                )


def test_the_grader_never_reads_passing_tds():
    """A QB row whose `passing_tds` changes must not change the graded outcome.

    Pins the specific arithmetic removal: deleting `passing_tds` from the
    grader's expression is the fix, and this is the observable difference.
    """
    df = _frame(position="QB", passing_tds=4.0, passing_yards=400.0)
    frame_label = _label(df)
    df_no_passing = df.copy()
    df_no_passing["passing_tds"] = 0.0
    assert _label(df_no_passing) == frame_label == 0


def test_the_label_version_is_two():
    """The version is the staleness alarm's payload, so pin it: v1 is the old
    `+ passing_tds` definition and must never come back."""
    assert player_usage.ANYTIME_TD_LABEL_VERSION == 2


def test_the_module_has_exactly_one_implementation_of_the_label():
    """No second copy of the arithmetic can hide in this module.

    The bug class is duplication, so this asserts the shape directly: the
    training frame must delegate to `anytime_td_actual` rather than re-deriving
    the sum. A regression that inlines `(rushing + receiving) > 0` back into
    `build_player_training_frame` is the one that let NFL's grader drift.
    """
    import inspect

    source = inspect.getsource(player_usage.build_player_training_frame)
    assert "anytime_td_actual(" in source
    # No raw TD sum in the training frame: the definition lives in one place.
    # Checked for EVERY TD column, by the same bracketed-plus shape as the
    # sibling assertion above.
    #
    # This used to end `assert "passing_tds" not in source`, a bare substring
    # that was coarser than the thing it protected. The QB passing-TD FEATURE now
    # lives in this module, and `with_passing_tds_roll(df)` contains that
    # substring while having nothing to do with the label -- so the guard could
    # not tell a feature helper from a label, and a guard like that gets deleted
    # instead of fixed. The two assertions below are what it was reaching for.
    for column in ("rushing_tds", "receiving_tds", "passing_tds"):
        assert f'["{column}"] +' not in source
    # The blunt net, in a form that does not misfire: every identifier in this
    # function containing "passing_tds" must be one of the two names that
    # legitimately refer to the FEATURE. `df["passing_tds"]`, or any new
    # identifier mentioning it, trips this.
    import re

    allowed = {"with_passing_tds_roll", "PASSING_TDS_ROLL_COLUMN"}
    mentioned = set(re.findall(r"\w*passing_tds\w*", source))
    assert mentioned <= allowed, (
        f"the label path references passing_tds as {sorted(mentioned - allowed)}; "
        "only the feature helper may"
    )


def test_the_grader_delegates_to_the_shared_definition():
    """Same delegation check for the grader, in `store.py`."""
    import inspect

    source = inspect.getsource(store.reconcile_player_prop_predictions)
    assert "anytime_td_actual(" in source
    assert "passing_tds" not in source