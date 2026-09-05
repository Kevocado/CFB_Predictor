"""build.py — the single feature-construction entry point for game-outcome
models. Every consumer (training, walk-forward evaluation, live serving)
must call build_training_frame / build_features_for_game rather than
reimplementing feature logic inline.

conference_game replaces NFL's div_game -- CFBD's Game model already
supplies conference_game as a real boolean, so (unlike NFL's div_game, which
nfl_data_py also supplies pre-computed) no team-conference join is needed
here either; this file only fills a default and casts to int.

FCS-opponent games are excluded from build_training_frame's *training* rows
(is_fbs_game filter below) but power_ratings/rolling_form/rest_days still
compute over every row in games_df, FCS-opponent games included -- an FBS
team's rest_days/rolling_form bookkeeping stays complete even for a week it
spent beating an FCS opponent.
"""

from __future__ import annotations

import pandas as pd

from . import power_ratings, rest_days, rolling_form

FEATURE_COLUMNS = [
    "home_pregame_rating", "away_pregame_rating", "rating_diff",
    "home_points_scored_roll", "home_points_allowed_roll",
    "away_points_scored_roll", "away_points_allowed_roll",
    "home_rest_days", "away_rest_days",
    "conference_game",
]


def _is_fbs_game(df: pd.DataFrame, fbs_teams: dict[int, set[str]] | None) -> pd.Series:
    """True when both teams in the game are FBS. Prefers the authoritative
    per-season FBS team-universe list (fbs_teams, from
    data.games.fetch_fbs_teams) when given -- the design spec's primary
    FCS-exclusion source, since it's CFBD's own official roster of FBS
    teams for that season. Falls back to the game row's own
    home_division/away_division columns (CFBD's per-game classification)
    when fbs_teams isn't supplied, as the spec's cross-check."""
    if fbs_teams:
        def _team_is_fbs(season, team) -> bool:
            teams_for_season = fbs_teams.get(int(season))
            return team in teams_for_season if teams_for_season else True

        return df.apply(
            lambda r: _team_is_fbs(r["season"], r["home_team"]) and _team_is_fbs(r["season"], r["away_team"]),
            axis=1,
        )
    if "home_division" in df.columns and "away_division" in df.columns:
        return (df["home_division"].fillna("fbs").str.lower() == "fbs") & (
            df["away_division"].fillna("fbs").str.lower() == "fbs"
        )
    return pd.Series(True, index=df.index)


def _assemble(games_df: pd.DataFrame) -> pd.DataFrame:
    df = power_ratings.compute_pregame_ratings(games_df)
    df = rolling_form.add_rolling_form(df)
    df = rest_days.add_rest_days(df)
    df["rating_diff"] = df["home_pregame_rating"] - df["away_pregame_rating"]
    df["home_rest_days"] = df["home_rest_days"].fillna(7)
    df["away_rest_days"] = df["away_rest_days"].fillna(7)
    if "conference_game" not in df.columns:
        df["conference_game"] = False
    df["conference_game"] = df["conference_game"].fillna(False).astype(int)
    return df


def build_training_frame(
    games_df: pd.DataFrame, fbs_teams: dict[int, set[str]] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    df = _assemble(games_df)
    df["is_fbs_game"] = _is_fbs_game(df, fbs_teams)
    played = df[
        df["home_score"].notna() & df["away_score"].notna() & df["is_fbs_game"]
    ].reset_index(drop=True)
    played["margin"] = played["home_score"] - played["away_score"]
    played["total_points"] = played["home_score"] + played["away_score"]
    return played, FEATURE_COLUMNS


def build_features_for_game(home_team: str, away_team: str, games_df: pd.DataFrame) -> pd.Series:
    """One live feature row for an upcoming home_team vs away_team game,
    computed from every played game in games_df (ratings/rolling form as of
    right now). conference_game defaults to 0 here -- api/routes.py knows
    the real value from the live schedule row for a specific upcoming game
    but this function only has the two team names, the same limitation
    NFL's build_features_for_game already had for div_game (not fixed here,
    to stay a near-verbatim port rather than a scope expansion)."""
    ratings = power_ratings.final_ratings(games_df)
    played = games_df[games_df["home_score"].notna() & games_df["away_score"].notna()]

    def _recent_form(team: str) -> tuple[float, float]:
        appearances = pd.concat(
            [
                played[played["home_team"] == team][["gameday", "home_score", "away_score"]].rename(
                    columns={"home_score": "scored", "away_score": "allowed"}
                ),
                played[played["away_team"] == team][["gameday", "away_score", "home_score"]].rename(
                    columns={"away_score": "scored", "home_score": "allowed"}
                ),
            ]
        ).sort_values("gameday")
        recent = appearances.tail(5)
        if recent.empty:
            return float("nan"), float("nan")
        return float(recent["scored"].mean()), float(recent["allowed"].mean())

    def _rest_days(team: str) -> float | None:
        appearances = pd.concat(
            [
                played[played["home_team"] == team][["gameday"]],
                played[played["away_team"] == team][["gameday"]],
            ]
        ).sort_values("gameday")
        if appearances.empty:
            return None
        last_game = pd.to_datetime(appearances.iloc[-1]["gameday"])
        # Real CFBD gamedays parse as tz-aware (UTC) timestamps; test
        # fixtures use tz-naive ones. Strip tz so both compare cleanly
        # against the tz-naive "now" below -- confirmed against real data
        # in Task 17 (TypeError: Cannot subtract tz-naive and tz-aware).
        if last_game.tzinfo is not None:
            last_game = last_game.tz_localize(None)
        return float((pd.Timestamp.now().normalize() - last_game).days)

    home_scored, home_allowed = _recent_form(home_team)
    away_scored, away_allowed = _recent_form(away_team)
    home_rating = ratings.get(home_team, power_ratings.DEFAULT_START_RATING)
    away_rating = ratings.get(away_team, power_ratings.DEFAULT_START_RATING)
    home_rest = _rest_days(home_team)
    away_rest = _rest_days(away_team)

    return pd.Series(
        {
            "home_pregame_rating": home_rating,
            "away_pregame_rating": away_rating,
            "rating_diff": home_rating - away_rating,
            "home_points_scored_roll": home_scored,
            "home_points_allowed_roll": home_allowed,
            "away_points_scored_roll": away_scored,
            "away_points_allowed_roll": away_allowed,
            "home_rest_days": home_rest if home_rest is not None else 7.0,
            "away_rest_days": away_rest if away_rest is not None else 7.0,
            "conference_game": 0,
        }
    )
