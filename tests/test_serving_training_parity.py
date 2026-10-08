"""A game's served feature row equals the row training builds for the same game, including its conference flag."""
import numpy as np
import pandas as pd

from cfb_predictor.features import build

CONF = {"T0": "A", "T1": "A", "T2": "A", "T3": "A", "T4": "B", "T5": "B", "T6": "B", "T7": "B"}


def _season_games(seed=3, seasons=(2023, 2024, 2025), weeks=12):
    rng = np.random.default_rng(seed)
    teams = list(CONF)
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


def test_the_served_row_equals_the_row_training_builds_for_the_same_game():
    games = _season_games()
    trained, cols = build.build_training_frame(games)
    sample = trained[trained["season"] == 2025].sample(15, random_state=1)
    for _, g in sample.iterrows():
        history = games[pd.to_datetime(games["gameday"]) < pd.Timestamp(g["gameday"])]
        served = build.build_features_for_game(g["home_team"], g["away_team"], history, gameday=g["gameday"], conference_game=g["conference_game"])
        np.testing.assert_allclose(served[cols].to_numpy(float), g[cols].to_numpy(float), equal_nan=True, err_msg=g["game_id"])


def test_conference_game_is_served_as_the_real_value_not_a_constant_zero():
    games = _season_games()
    same = build.build_features_for_game("T0", "T1", games, gameday="2026-09-05")
    cross = build.build_features_for_game("T0", "T5", games, gameday="2026-09-05")
    assert same["conference_game"] == 1 and cross["conference_game"] == 0


def test_an_explicit_schedule_value_overrides_the_derived_one():
    games = _season_games()
    assert build.build_features_for_game("T0", "T5", games, gameday="2026-09-05", conference_game=True)["conference_game"] == 1


def test_rest_days_are_measured_to_the_games_own_date_not_to_today():
    games = _season_games()
    last = pd.to_datetime(games["gameday"]).max()
    when = last + pd.Timedelta(days=9)
    team = games.sort_values("gameday").iloc[-1]["home_team"]
    other = "T7" if team != "T7" else "T6"
    served = build.build_features_for_game(team, other, games, gameday=when)
    last_played = pd.to_datetime(games[(games["home_team"] == team) | (games["away_team"] == team)]["gameday"]).max()
    assert served["home_rest_days"] == (when - last_played).days


def test_the_row_has_the_feature_columns_in_order():
    assert list(build.build_features_for_game("T0", "T1", _season_games(), gameday="2026-09-05").index) == build.FEATURE_COLUMNS


def test_nan_conference_game_for_different_conference_teams_serves_zero():
    """NaN from the schedule is treated as missing and derived; teams in different conferences → 0.

    This is the train/serve mismatch bug: NaN from the schedule is truthy (`bool(nan) == True`),
    but training fills NaN with False. The fix treats NaN the same as None and derives it.
    """
    games = _season_games()
    history = games[pd.to_datetime(games["gameday"]) < pd.Timestamp("2026-09-05")]
    served = build.build_features_for_game("T0", "T5", history, gameday="2026-09-05", conference_game=float("nan"))
    assert served["conference_game"] == 0, f"expected 0 for cross-conference teams, got {served['conference_game']}"


def test_nan_conference_game_for_same_conference_teams_serves_one():
    """NaN from the schedule is derived when teams share a conference; → 1."""
    games = _season_games()
    history = games[pd.to_datetime(games["gameday"]) < pd.Timestamp("2026-09-05")]
    served = build.build_features_for_game("T0", "T1", history, gameday="2026-09-05", conference_game=float("nan"))
    assert served["conference_game"] == 1, f"expected 1 for same-conference teams, got {served['conference_game']}"