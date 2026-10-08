import pandas as pd
import pytest
from cfb_predictor.features.priors import preseason_prior
from cfb_predictor.features.epa import add_epa_features, epa_columns


def test_prior_regresses_last_final_rating_toward_conference_mean():
    finals = pd.DataFrame({"team": ["A", "B", "C"], "season": [2023] * 3, "rating": [1800.0, 1500.0, 1200.0],
                           "conference": ["X", "X", "X"]})
    p = preseason_prior(finals, season=2024, regress=0.4).set_index("team")["prior"]
    mean = 1500.0
    assert abs(p["A"] - (mean + 0.6 * 300)) < 1e-9 and abs(p["C"] - (mean - 0.6 * 300)) < 1e-9


def test_new_team_gets_the_conference_mean_not_zero():
    finals = pd.DataFrame({"team": ["A", "B"], "season": [2023, 2023], "rating": [1600.0, 1400.0], "conference": ["X", "X"]})
    p = preseason_prior(finals, season=2024, regress=0.4, new_teams={"Z": "X"}).set_index("team")["prior"]
    assert p["Z"] == 1500.0


def test_only_prior_seasons_are_used():
    finals = pd.DataFrame({"team": ["A", "A"], "season": [2023, 2024], "rating": [1700.0, 1000.0], "conference": ["X", "X"]})
    p = preseason_prior(finals, season=2024, regress=0.0).set_index("team")["prior"]
    assert p["A"] == 1700.0  # the 2024 final must not leak into the 2024 preseason


def test_cfb_epa_columns_match_nfl():
    cols = epa_columns()
    assert len(cols) == 8 * 2 + 5  # 8 STATS * 2 sides + 5 EDGE_COLUMNS


def test_cfb_epa_adds_shrunk_features():
    games = pd.DataFrame([
        {"game_id": "g1", "gameday": "2024-09-01", "home_team": "A", "away_team": "B", "home_score": 28, "away_score": 14},
        {"game_id": "g2", "gameday": "2024-09-08", "home_team": "B", "away_team": "A", "home_score": 21, "away_score": 31},
    ])
    eff = pd.DataFrame([
        {"game_id": "g1", "team": "A", "epa_off": 0.3, "epa_def": 0.1, "epa_off_pass": 0.4, "epa_off_rush": 0.1,
         "epa_def_pass": 0.2, "epa_def_rush": 0.05, "success_off": 0.5, "success_def": 0.4},
        {"game_id": "g1", "team": "B", "epa_off": 0.1, "epa_def": 0.3, "epa_off_pass": 0.2, "epa_off_rush": 0.05,
         "epa_def_pass": 0.4, "epa_def_rush": 0.2, "success_off": 0.4, "success_def": 0.5},
        {"game_id": "g2", "team": "B", "epa_off": 0.2, "epa_def": 0.1, "epa_off_pass": 0.3, "epa_off_rush": 0.1,
         "epa_def_pass": 0.2, "epa_def_rush": 0.1, "success_off": 0.45, "success_def": 0.35},
        {"game_id": "g2", "team": "A", "epa_off": 0.15, "epa_def": 0.25, "epa_off_pass": 0.25, "epa_off_rush": 0.05,
         "epa_def_pass": 0.3, "epa_def_rush": 0.2, "success_off": 0.4, "success_def": 0.5},
    ])
    out = add_epa_features(games, eff)
    for c in epa_columns():
        assert c in out.columns
    # first game has no prior data -> NaN
    assert pd.isna(out.iloc[0]["home_epa_off_ewm"])
    # second game has one prior game -> shrunk value
    assert not pd.isna(out.iloc[1]["home_epa_off_ewm"])