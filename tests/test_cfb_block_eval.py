"""Tests for the CFB block_eval tool and the blocks/aux wiring it needed."""
import numpy as np
import pandas as pd
import pytest

from cfb_predictor.data.cfbd_advanced import to_team_game_frame
from cfb_predictor.evaluate import walk_forward
from cfb_predictor.features import build as feature_build
from cfb_predictor.tools import block_eval


def _adv_rows(n_games=6, season=2018, start_id=400547640):
    """Two CFBD advanced rows per game, shaped like the real API's response."""
    rows = []
    for i in range(n_games):
        gid = start_id + i
        pair = [
            {"gameId": gid, "season": season, "seasonType": "regular", "week": 1,
             "team": f"H{i}", "opponent": f"A{i}",
             "offense": {"ppa": 0.2, "successRate": 0.45,
                         "passingPlays": {"ppa": 0.3}, "rushingPlays": {"ppa": -0.1}}},
            {"gameId": gid, "season": season, "seasonType": "regular", "week": 1,
             "team": f"A{i}", "opponent": f"H{i}",
             "offense": {"ppa": -0.1, "successRate": 0.4,
                         "passingPlays": {"ppa": -0.2}, "rushingPlays": {"ppa": 0.05}}},
        ]
        rows.extend(pair)
    return rows


def _multi_season_games(seasons=(2018, 2019, 2020), games_per_season=6):
    rng = np.random.default_rng(11)
    rows = []
    for season in seasons:
        teams = [f"T{s}" for s in range(8)]
        for w in range(1, games_per_season + 1):
            order = list(rng.permutation(teams))
            for i in range(0, 8, 2):
                rows.append({
                    "game_id": f"{season}_{w:02d}_{order[i]}_{order[i+1]}",
                    "season": season, "week": w,
                    "gameday": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * (w - 1)),
                    "home_team": order[i], "away_team": order[i + 1],
                    "home_score": int(rng.integers(7, 45)), "away_score": int(rng.integers(7, 45)),
                    "conference_game": 0, "completed": True,
                })
    return pd.DataFrame(rows)


class TestToTeamGameFrame:
    def test_carries_season_and_week_from_the_row_not_the_game_id(self):
        """CFBD ids do NOT start with the calendar year: 2014's first id is 400547640.
        Deriving the season from game_id[:4] silently returns 4005 and breaks every join."""
        frame = to_team_game_frame(_adv_rows(n_games=3, season=2014, start_id=400547640))
        assert len(frame) == 6
        assert set(frame["season"]) == {2014}
        assert set(frame["week"]) == {1}

    def test_every_game_contributes_both_sides(self):
        frame = to_team_game_frame(_adv_rows(n_games=4, season=2019))
        assert len(frame) == 8
        assert frame.groupby("game_id").size().eq(2).all()

    def test_defence_is_the_opponents_offence(self):
        """epa_def for team X must equal epa_off for team Y in the same game."""
        frame = to_team_game_frame(_adv_rows(n_games=2, season=2020))
        for gid, grp in frame.groupby("game_id"):
            a, b = grp.iloc[0], grp.iloc[1]
            assert a["epa_def"] == pytest.approx(b["epa_off"])
            assert b["epa_def"] == pytest.approx(a["epa_off"])

    def test_a_game_with_one_side_missing_is_skipped_not_guessed(self):
        rows = _adv_rows(n_games=2, season=2021)
        rows = rows[:-1]  # drop one team's row of the second game
        assert len(to_team_game_frame(rows)) == 2  # only the complete game survives


class TestBlocksAuxWiring:
    def test_prepare_folds_forwards_blocks_and_aux(self):
        """Regression: prepare_folds() ignored blocks/aux, so the epa block was wired in
        _assemble but unreachable from the walk-forward loop."""
        games = _multi_season_games()
        eff = to_team_game_frame(_adv_rows(n_games=3, season=2018))
        aux = feature_build.Aux(efficiency=eff)
        folds = walk_forward.prepare_folds(games, blocks=("epa",), aux=aux)
        base = walk_forward.prepare_folds(games)
        assert folds, "blocked folds must not be empty"
        assert len(folds) == len(base)
        assert len(folds[0]["val_df"]) == len(base[0]["val_df"]), "same held-out games"

    def test_epa_block_adds_columns_over_base(self):
        games = _multi_season_games()
        eff = to_team_game_frame(_adv_rows(n_games=3, season=2018))
        aux = feature_build.Aux(efficiency=eff)
        _, base_cols = feature_build.build_training_frame(games)
        _, epa_cols = feature_build.build_training_frame(games, blocks=("epa",), aux=aux)
        assert set(epa_cols) - set(base_cols), "the epa block added no columns"

    def test_epa_refuses_to_default_aux_to_zeros(self):
        games = _multi_season_games()
        with pytest.raises(ValueError):
            feature_build.build_training_frame(games, blocks=("epa",), aux=None)

    def test_priors_block_is_wired_and_needs_no_aux(self):
        games = _multi_season_games()
        _, base_cols = feature_build.build_training_frame(games)
        frame, cols = feature_build.build_training_frame(games, blocks=("priors",))
        assert cols == base_cols + ["home_prior", "away_prior", "prior_diff"]
        assert not frame[["home_prior", "away_prior", "prior_diff"]].isna().any().any()


class TestBlockEvalTool:
    def test_pdi_gap_bootstrap_directions(self):
        assert block_eval.paired_bootstrap_gap([0.05, 0.06, 0.04], [0.02, 0.03, 0.01])[0] > 0
        assert block_eval.paired_bootstrap_gap([], []) == (0.0, 0.0)

    def test_calibration_gap_is_zero_for_perfect_buckets(self):
        p = np.repeat([0.1, 0.3, 0.5, 0.7, 0.9], 500)
        y = np.concatenate([np.r_[np.ones(int(500 * q)), np.zeros(500 - int(500 * q))]
                            for q in (0.1, 0.3, 0.5, 0.7, 0.9)])
        assert block_eval.calibration_gap(p, y) < 1e-9

    def test_evaluate_block_reports_not_evaluated_when_aux_is_missing(self):
        games = _multi_season_games()
        result, err = block_eval.evaluate_block(games, None, "epa")
        assert result is None and "aux" in err

    def test_evaluate_block_priors_runs_without_efficiency_aux(self):
        games = _multi_season_games(seasons=(2016, 2017, 2018, 2019))
        result, err = block_eval.evaluate_block(games, None, "priors")
        assert err is None and result["n_games"] > 0

    def test_evaluate_block_epa_runs_end_to_end_on_fixtures(self):
        games = _multi_season_games()
        eff = to_team_game_frame(_adv_rows(n_games=3, season=2018))
        aux = feature_build.Aux(efficiency=eff)
        result, err = block_eval.evaluate_block(games, aux, "epa")
        assert err is None
        assert result["n_games"] > 0
        assert set(result) >= {"mae_delta", "mae_ci", "brier_delta", "brier_ci", "gap_ci", "clears"}
        assert isinstance(result["clears"], bool)


def test_paired_bootstrap_auc_sign_and_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400).astype(float)
    good = y + rng.normal(0, 0.8, 400)
    bad = rng.normal(0, 1, 400)
    d, lo, hi = block_eval.paired_bootstrap_auc(bad, good, y)
    assert d > 0 and lo > 0
    assert block_eval.paired_bootstrap_auc(good, good, y)[0] == 0.0
