import numpy as np
import pandas as pd
import pytest

from cfb_predictor.evaluate import walk_forward


def _multi_season_games():
    rng = np.random.default_rng(7)
    rows = []
    teams = [f"T{i}" for i in range(8)]
    for season in (2022, 2023, 2024):
        for week in range(1, 6):
            for i in range(0, len(teams), 2):
                home, away = teams[i], teams[i + 1]
                home_score = int(rng.integers(10, 45))
                away_score = int(rng.integers(10, 45))
                rows.append(
                    {
                        "game_id": f"{season}_{week}_{home}_{away}", "season": season, "week": week,
                        "gameday": pd.Timestamp(f"{season}-08-25") + pd.Timedelta(days=7 * (week - 1)),
                        "home_team": home, "away_team": away,
                        "home_score": home_score, "away_score": away_score,
                        "home_division": "fbs", "away_division": "fbs", "conference_game": False,
                    }
                )
    return pd.DataFrame(rows)


def test_prepare_folds_holds_out_each_season_after_minimum():
    folds = walk_forward.prepare_folds(_multi_season_games(), min_train_seasons=2)

    val_seasons = [fold["val_season"] for fold in folds]
    assert val_seasons == [2024]

    for fold in folds:
        assert fold["train_df"]["season"].max() < fold["val_season"]
        assert (fold["val_df"]["season"] == fold["val_season"]).all()


def test_prepare_folds_accepts_fbs_teams_and_still_holds_out_each_season():
    fbs_teams = {season: {f"T{i}" for i in range(8)} for season in (2022, 2023, 2024)}
    folds = walk_forward.prepare_folds(_multi_season_games(), fbs_teams=fbs_teams, min_train_seasons=2)

    assert [fold["val_season"] for fold in folds] == [2024]


def test_evaluate_candidate_returns_a_row_per_fold_for_each_candidate():
    folds = walk_forward.prepare_folds(_multi_season_games(), min_train_seasons=2)

    for candidate in ("elo", "ridge", "xgb"):
        result = walk_forward.evaluate_candidate(folds, candidate)
        assert len(result) == len(folds)
        assert (result["log_loss"] > 0).all()
        assert (result["brier"] >= 0).all()


def test_evaluate_candidate_rejects_unknown_candidate():
    folds = walk_forward.prepare_folds(_multi_season_games(), min_train_seasons=2)

    with pytest.raises(ValueError):
        walk_forward.evaluate_candidate(folds, "not-a-real-candidate")
