import pandas as pd
import pytest
from cfb_predictor.features.priors import preseason_prior


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