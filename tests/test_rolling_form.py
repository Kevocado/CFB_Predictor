import pandas as pd

from cfb_predictor.features import rolling_form


def _games():
    return pd.DataFrame(
        [
            {"game_id": "g1", "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State",
             "home_score": 7, "away_score": 14},
            {"game_id": "g2", "gameday": "2025-09-06", "home_team": "Ohio State", "away_team": "Michigan",
             "home_score": 10, "away_score": 14},
            {"game_id": "g3", "gameday": "2025-09-13", "home_team": "Texas", "away_team": "Michigan",
             "home_score": 30, "away_score": 17},
        ]
    )


def test_first_appearance_has_no_rolling_form():
    result = rolling_form.add_rolling_form(_games(), window=5)

    g1 = result.iloc[0]
    assert pd.isna(g1["home_points_scored_roll"])
    assert pd.isna(g1["away_points_scored_roll"])


def test_second_game_reflects_only_the_prior_game():
    result = rolling_form.add_rolling_form(_games(), window=5)

    # Ohio State's second appearance (game g2, as home) should reflect only
    # its away-team performance in g1 (scored 14, allowed 7).
    g2 = result.iloc[1]
    assert g2["home_points_scored_roll"] == 14.0
    assert g2["home_points_allowed_roll"] == 7.0
