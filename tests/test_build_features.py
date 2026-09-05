# tests/test_build_features.py
import pandas as pd

from cfb_predictor.features import build


def _games():
    rows = []
    teams = ["Texas", "Ohio State", "Michigan", "Alabama"]
    day = pd.Timestamp("2025-08-30")
    for week in range(1, 4):
        rows.append(
            {
                "game_id": f"g{week}a", "season": 2025, "week": week,
                "gameday": day + pd.Timedelta(days=7 * (week - 1)),
                "home_team": teams[0], "away_team": teams[1],
                "home_score": 24, "away_score": 20,
                "home_division": "fbs", "away_division": "fbs", "conference_game": False,
            }
        )
        rows.append(
            {
                "game_id": f"g{week}b", "season": 2025, "week": week,
                "gameday": day + pd.Timedelta(days=7 * (week - 1)),
                "home_team": teams[2], "away_team": teams[3],
                "home_score": 17, "away_score": 27,
                "home_division": "fbs", "away_division": "fbs", "conference_game": True,
            }
        )
    # An FCS-opponent game: counts for Texas's rest_days/rolling_form
    # bookkeeping but must be excluded from build_training_frame's rows.
    rows.append(
        {
            "game_id": "g4fcs", "season": 2025, "week": 4,
            "gameday": day + pd.Timedelta(days=21),
            "home_team": "Texas", "away_team": "Div II School",
            "home_score": 55, "away_score": 3,
            "home_division": "fbs", "away_division": "fcs", "conference_game": False,
        }
    )
    return pd.DataFrame(rows)


def test_build_training_frame_returns_feature_columns_and_targets():
    df, feature_cols = build.build_training_frame(_games())

    assert "margin" in df.columns
    assert "total_points" in df.columns
    assert "conference_game" in feature_cols
    assert set(feature_cols).issubset(df.columns)
    assert (df["margin"] == df["home_score"] - df["away_score"]).all()
    assert (df["total_points"] == df["home_score"] + df["away_score"]).all()


def test_build_training_frame_excludes_fcs_opponent_games_via_division_fallback():
    df, _ = build.build_training_frame(_games())

    assert "g4fcs" not in set(df["game_id"])
    assert len(df) == 6  # 3 weeks x 2 FBS-vs-FBS games each, the FCS game excluded


def test_build_training_frame_excludes_fcs_opponent_games_via_fbs_teams_override():
    fbs_teams = {2025: {"Texas", "Ohio State", "Michigan", "Alabama"}}  # "Div II School" absent
    df, _ = build.build_training_frame(_games(), fbs_teams=fbs_teams)

    assert "g4fcs" not in set(df["game_id"])
    assert len(df) == 6


def test_build_training_frame_preserves_conference_game_flag():
    df, _ = build.build_training_frame(_games())

    conf_games = df[df["game_id"].str.endswith("b")]
    assert (conf_games["conference_game"] == 1).all()
    non_conf_games = df[df["game_id"].str.endswith("a")]
    assert (non_conf_games["conference_game"] == 0).all()


def test_build_features_for_game_returns_series_with_feature_columns():
    games_df = _games()
    _, feature_cols = build.build_training_frame(games_df)

    row = build.build_features_for_game("Texas", "Ohio State", games_df)

    assert isinstance(row, pd.Series)
    for col in feature_cols:
        assert col in row.index
