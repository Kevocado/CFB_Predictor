import pandas as pd

from cfb_predictor.features import power_ratings


def _games():
    return pd.DataFrame(
        [
            {"game_id": "g1", "season": 2025, "week": 1, "gameday": "2025-08-30",
             "home_team": "Texas", "away_team": "Ohio State", "home_score": 7, "away_score": 14},
            {"game_id": "g2", "season": 2025, "week": 2, "gameday": "2025-09-06",
             "home_team": "Ohio State", "away_team": "Texas", "home_score": 24, "away_score": 17},
        ]
    )


def test_first_game_uses_start_rating_for_both_teams():
    result = power_ratings.compute_pregame_ratings(_games(), start_rating=1500.0)

    first = result.iloc[0]
    assert first["home_pregame_rating"] == 1500.0
    assert first["away_pregame_rating"] == 1500.0


def test_rating_moves_after_a_result():
    result = power_ratings.compute_pregame_ratings(_games(), start_rating=1500.0)

    second = result.iloc[1]
    # Ohio State won game 1 as the away team, so its pregame rating for
    # game 2 (now at home) should have risen above 1500.
    assert second["home_pregame_rating"] > 1500.0
    # Texas lost game 1 at home, so its pregame rating for game 2 (now away)
    # should have dropped below 1500.
    assert second["away_pregame_rating"] < 1500.0


def test_final_ratings_reflects_every_game():
    ratings = power_ratings.final_ratings(_games(), start_rating=1500.0)

    assert set(ratings) == {"Texas", "Ohio State"}
    assert ratings["Ohio State"] > 1500.0
    assert ratings["Texas"] < 1500.0
