"""player_usage.py — player-level rolling usage/production features for
anytime-TD and yardage prop models. Same shift(1)-then-rolling discipline as
features/rolling_form.py, applied per player instead of per team. Ported from
nfl_predictor: "targets" being always NaN for CFB (see
data/player_stats.py) degrades this file's rolling mean to NaN for that one
column, which fit_anytime_td_classifier/fit_yardage_regressor's .fillna(0)
already tolerates -- no change needed here.

The anytime-TD label lives here as `anytime_td_actual` -- rushing TDs plus
receiving TDs, passing TDs excluded -- and is shared with the grader so the
two cannot drift apart. See its docstring."""

from __future__ import annotations

import pandas as pd

ROLL_STATS = ["passing_yards", "rushing_yards", "receiving_yards", "targets", "carries", "receptions"]
PLAYER_FEATURE_COLUMNS = [f"{stat}_roll" for stat in ROLL_STATS]

#: The window every rolling feature uses, in training and at serving alike.
#: Declared once because `with_passing_tds_roll` and `build_features_for_player`
#: have to agree on it, and two literal `5`s are two chances to disagree.
DEFAULT_ROLL_WINDOW = 5

#: `passing_tds_roll` -- the rolling passing-TD count the QB passing-TD model
#: (`models/qb_passing_td.py`) is fitted on.
#:
#: Deliberately NOT in `ROLL_STATS`, so not in `PLAYER_FEATURE_COLUMNS`: that
#: list is what the anytime-TD classifier and every yardage regressor are fitted
#: on and what `predict_props` indexes by name, so widening it would change the
#: feature count of every already-committed model. It is computed and served as
#: its own column.
PASSING_TDS_ROLL_COLUMN = "passing_tds_roll"

#: Version of the `anytime_td` DEFINITION (not of the code -- of the label).
#: Bump this whenever `anytime_td_actual`'s arithmetic changes. It is recorded
#: in the manifest at fit time and checked at load time by
#: `models/manifest._verify_artifact_fingerprint`, so an artefact fitted
#: against a different definition raises instead of serving quietly.
#:
#: 1 = `rushing_tds + receiving_tds + passing_tds > 0` (superseded).
#: 2 = `rushing_tds + receiving_tds > 0` (2026-10-04; passing TDs excluded).
#:     Matches NFL_Predictor's v2 so one label means one thing across both
#:     sites -- see `anytime_td_actual` for why.
ANYTIME_TD_LABEL_VERSION = 2


def with_passing_tds_roll(df: pd.DataFrame, window: int = DEFAULT_ROLL_WINDOW) -> pd.DataFrame:
    """`df` plus `passing_tds_roll`, on `_add_rolling`'s exact discipline.

    Same `shift(1).rolling(window, min_periods=1).mean()` per player, for the same
    reason every other rolled stat uses it: a pregame feature cannot know the game
    being predicted. It exists as one function so the training column and the
    column `build_features_for_player` emits cannot drift apart -- which is how
    NFL's own `passing_tds_roll` came to be fitted on and served as a constant
    zero, the two having been written separately.

    Returns a copy sorted by `["player_id", "season", "week"]` with a fresh index,
    like `_add_rolling`, so it never mutates its argument.
    """
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    df[PASSING_TDS_ROLL_COLUMN] = df.groupby("player_id")["passing_tds"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return df


def _add_rolling(df: pd.DataFrame, window: int = DEFAULT_ROLL_WINDOW) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grouped = df.groupby("player_id")
    for stat in ROLL_STATS:
        df[f"{stat}_roll"] = grouped[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return df


def anytime_td_actual(rushing_tds, receiving_tds) -> float:
    """Ground truth for the `anytime_td` market, as 0.0 or 1.0.

    THE definition, in one function, because this quantity has to be computed
    in two places that must never disagree:

    * `build_player_training_frame` below, to produce the classifier's target;
    * `tracking/store.reconcile_player_prop_predictions`, to grade a stored
      prediction. That grader carried its own inline copy of the sum and
      included `passing_tds` in it, exactly as NFL's did before its PR #26, so
      changing the label here alone would have left the grader resolving the
      market against the OLD definition while the code claimed the new one --
      every QB's pick scored against a truth the model was never fitted on. A
      grader that disagrees with the model is worse than either definition on
      its own.

    `anytime_td` = **rushing TDs + receiving TDs, and nothing else.** Passing
    TDs are deliberately EXCLUDED (2026-10-04; both call sites previously summed
    `passing_tds` in as well).

    The reason is that the two are not the same market. "Anytime TD" reads to a
    user as a rushing-or-receiving score, but with passing included it fired on
    passing alone, so a quarterback's anytime-TD was dominated by his arm --
    and in CFB, where a starter QB attempts 30-40 passes a game, that is a
    near-certainty rather than a signal. Quarterbacks sorted to the top of a
    category whose name never mentions passing, and the same label meant two
    different things here and on NFL_Predictor, which dropped passing in PR
    #26 (`e1c7d5e8`).

    CFB has no QB passing-TD model or market line, so excluding passing TDs
    costs a passer nothing that was being measured honestly: there is no line
    judging him on his arm to begin with. Blending them into "anytime" was
    papering over that gap, not solving it. Adding the model is separate scope
    with its own data validation.

    Scalars or a Series; missing values read as 0, matching the `fillna(0)` the
    training frame applies and the `(x or 0)` the grader used. Returns a float
    for scalars and a float Series for a Series, so the grader's per-row call
    and the frame-level call share one arithmetic expression.
    """
    if isinstance(rushing_tds, pd.Series):
        rushing, receiving = rushing_tds.fillna(0), receiving_tds.fillna(0)
    else:
        rushing = 0 if rushing_tds is None or pd.isna(rushing_tds) else rushing_tds
        receiving = 0 if receiving_tds is None or pd.isna(receiving_tds) else receiving_tds
    scored = (rushing + receiving) > 0
    return scored.astype(float) if isinstance(rushing_tds, pd.Series) else float(scored)


def build_player_training_frame(player_stats_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = _add_rolling(player_stats_df)

    # The LABEL only. `PLAYER_FEATURE_COLUMNS` is untouched, so no model changes
    # shape; and `predict_props` scores a live row from features alone, so the
    # only artefact affected is one refitted against this label.
    df["anytime_td"] = anytime_td_actual(df["rushing_tds"], df["receiving_tds"]).astype(int)

    # The passing-TD model's own feature, on the same shift(1) discipline. The
    # returned column list stays `PLAYER_FEATURE_COLUMNS` on purpose: this
    # column belongs to the passing-TD model alone and widening the list would
    # change the feature count of every already-committed artefact.
    df = with_passing_tds_roll(df)
    return df, PLAYER_FEATURE_COLUMNS


def build_features_for_player(player_id: str, player_stats_df: pd.DataFrame) -> pd.Series | None:
    history = player_stats_df[player_stats_df["player_id"] == player_id].sort_values(["season", "week"])
    if history.empty:
        return None
    # `tail(window)` is the serving-time equivalent of the shifted rolling mean:
    # the last `window` rows of a player's history are the weeks BEFORE the one
    # being predicted, because the store's row for that week does not exist yet.
    recent = history.tail(DEFAULT_ROLL_WINDOW)
    row = {f"{stat}_roll": float(recent[stat].mean()) for stat in ROLL_STATS}
    # Emitted for the passing-TD model, from the same window and so from the same
    # discipline as the training column.
    if "passing_tds" in recent.columns:
        row[PASSING_TDS_ROLL_COLUMN] = float(recent["passing_tds"].mean())
    return pd.Series(row)
