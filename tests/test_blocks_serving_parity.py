"""With the EPA block on, a game's served row still equals the row training builds for it.

build_features_for_game grew `blocks`/`aux` to mirror build_training_frame. Without
them it called `_assemble(frame)` bare, so a block-fitted model could never be served
its own columns -- the same train/serve split this repo keeps fixing.
"""
import numpy as np
import pandas as pd
import pytest

from cfb_predictor.features import build

CONF = {"T0": "A", "T1": "A", "T2": "A", "T3": "A", "T4": "B", "T5": "B", "T6": "B", "T7": "B"}


def _season_games(seed=3, seasons=(2023, 2024, 2025), weeks=12):
    rng = np.random.default_rng(seed)
    teams = list(CONF.keys())
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            order = list(rng.permutation(teams))
            for i in range(0, 8, 2):
                home, away = order[i], order[i + 1]
                rows.append({
                    "game_id": f"{season}_{week:02d}_{home}_{away}", "season": season, "week": week,
                    "gameday": pd.Timestamp(f"{season}-08-30") + pd.Timedelta(days=7 * (week - 1)),
                    "home_team": home, "away_team": away,
                    "home_score": int(rng.integers(3, 52)), "away_score": int(rng.integers(3, 52)),
                    "home_conference": CONF[home], "away_conference": CONF[away],
                    "conference_game": CONF[home] == CONF[away],
                    "home_division": "fbs", "away_division": "fbs",
                })
    return pd.DataFrame(rows)


def _aux(games: pd.DataFrame, upto: pd.Timestamp | None = None) -> build.Aux:
    """cfbd_advanced.to_team_game_frame()-shaped efficiency for the games played before `upto`.

    The block's EWMs are shift(1) over each team's own appearances, so the aux frame at
    serving must contain exactly the games that were history then -- the same rule the
    parity assertion below is checking.
    """
    rng = np.random.default_rng(42)
    rows = []
    for _, g in games.iterrows():
        if upto is not None and not pd.Timestamp(g["gameday"]) < upto:
            continue
        for team in (g["home_team"], g["away_team"]):
            rows.append({
                "game_id": g["game_id"], "team": team,
                "epa_off": float(rng.normal(0.05, 0.1)), "epa_def": float(rng.normal(-0.05, 0.1)),
                "epa_off_pass": float(rng.normal(0.05, 0.1)), "epa_off_rush": float(rng.normal(0.05, 0.1)),
                "epa_def_pass": float(rng.normal(-0.05, 0.1)), "epa_def_rush": float(rng.normal(-0.05, 0.1)),
                "success_off": float(rng.uniform(0.3, 0.5)), "success_def": float(rng.uniform(0.3, 0.5)),
            })
    return build.Aux(efficiency=pd.DataFrame(rows))


def test_the_served_row_equals_the_trained_row_with_the_epa_block():
    games = _season_games()
    trained, cols = build.build_training_frame(games, blocks=("epa",), aux=_aux(games))
    assert cols == build.feature_columns(("epa",))

    for _, g in trained.dropna(subset=cols).sample(10, random_state=2).iterrows():
        when = pd.Timestamp(g["gameday"])
        served = build.build_features_for_game(
            g["home_team"], g["away_team"],
            games[pd.to_datetime(games["gameday"]) < when], gameday=when,
            blocks=("epa",), aux=_aux(games, upto=when),
        )
        assert list(served.index) == cols, g["game_id"]
        np.testing.assert_allclose(
            served.to_numpy(float), g[cols].to_numpy(float),
            equal_nan=True, err_msg=g["game_id"],
        )


def test_a_legacy_call_with_no_blocks_is_unchanged():
    """blocks=() is the pre-block signature: same columns, no aux needed."""
    games = _season_games()
    served = build.build_features_for_game("T0", "T1", games, gameday="2025-09-05")
    assert list(served.index) == build.feature_columns(())


def test_the_epa_block_refuses_instead_of_defaulting_to_zeros():
    games = _season_games()
    with pytest.raises(ValueError, match="aux.efficiency"):
        build.build_training_frame(games, blocks=("epa",))
    with pytest.raises(ValueError, match="aux.efficiency"):
        build.build_features_for_game("T0", "T1", games, gameday="2025-09-05", blocks=("epa",))