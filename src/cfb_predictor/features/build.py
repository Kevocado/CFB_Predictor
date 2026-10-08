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

import numpy as np
import pandas as pd
from dataclasses import dataclass

from . import epa, power_ratings, priors, rest_days, rolling_form

#: Feature BLOCKS beyond the ten base columns. A block joins DEFAULT_BLOCKS only in the PR that shows it clears the
#: evaluation bar (paired-bootstrap intervals + calibration gap, on identical held-out games).
BLOCK_COLUMNS: dict[str, list[str]] = {"epa": epa.epa_columns(), "priors": []}
DEFAULT_BLOCKS: tuple[str, ...] = ()


def feature_columns(blocks: tuple[str, ...] = DEFAULT_BLOCKS) -> list[str]:
    unknown = [b for b in blocks if b not in BLOCK_COLUMNS]
    if unknown:
        raise ValueError(f"unknown feature blocks: {unknown}; known: {sorted(BLOCK_COLUMNS)}")
    cols = list(FEATURE_COLUMNS)
    for block in blocks:
        cols += BLOCK_COLUMNS[block]
    return cols


@dataclass
class Aux:
    """The extra inputs the blocks read. `efficiency` is cfbd_advanced.to_team_game_frame() output."""
    efficiency: pd.DataFrame | None = None


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


def _assemble_base(games_df: pd.DataFrame) -> pd.DataFrame:
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


def _assemble(games_df: pd.DataFrame, blocks: tuple[str, ...] = (), aux: Aux | None = None) -> pd.DataFrame:
    df = _assemble_base(games_df)
    if "epa" in blocks:
        if aux is None or aux.efficiency is None:
            raise ValueError("the epa block needs aux.efficiency; refusing to default it to zeros")
        df = epa.add_epa_features(df, aux.efficiency)
    if "priors" in blocks:
        raise ValueError("the priors block is registered but not wired into _assemble yet")
    return df


def build_training_frame(
    games_df: pd.DataFrame, fbs_teams: dict[int, set[str]] | None = None,
    blocks: tuple[str, ...] = DEFAULT_BLOCKS, aux: Aux | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    columns = feature_columns(blocks)
    df = _assemble(games_df, blocks, aux)
    df["is_fbs_game"] = _is_fbs_game(df, fbs_teams)
    played = df[
        df["home_score"].notna() & df["away_score"].notna() & df["is_fbs_game"]
    ].reset_index(drop=True)
    played["margin"] = played["home_score"] - played["away_score"]
    played["total_points"] = played["home_score"] + played["away_score"]
    return played, columns


def build_features_for_game(
    home_team: str, away_team: str, games_df: pd.DataFrame,
    gameday: str | pd.Timestamp | None = None, conference_game: bool | int | float | None = None,
    blocks: tuple[str, ...] = DEFAULT_BLOCKS, aux: Aux | None = None,
) -> pd.Series:
    """One feature row for an upcoming home_team vs away_team game, built by the SAME code that builds training rows.

    The upcoming game is appended to the PLAYED games (no result, its own date) and run through `_assemble`, so
    ratings, rolling form and rest are computed exactly as for a training row. This used to be a second, hand-written
    implementation with two defects: rest was measured from the last game to TODAY rather than to the game, and
    `conference_game` was hard-coded to 0 although training fits it from the schedule (a dead coefficient at inference).

    `gameday` is the game's date (None = today). `conference_game` is the schedule's value for this game; when it is
    not supplied it is derived from the two teams' conferences in `games_df`, and falls back to 0 only when neither
    is known. `NaN` is treated the same as missing and is also derived. `blocks` and `aux` mirror `build_training_frame`
    so the same feature blocks (e.g., epa) are assembled for serving as were used at training.
    """
    played = games_df[games_df["home_score"].notna() & games_df["away_score"].notna()].copy()
    when = pd.Timestamp(gameday) if gameday is not None else pd.Timestamp.now().normalize()
    if when.tzinfo is not None:
        when = when.tz_localize(None)
    if conference_game is None or pd.isna(conference_game):
        conference_game = _same_conference(home_team, away_team, games_df)
    upcoming = {c: np.nan for c in played.columns}
    upcoming.update({
        "game_id": "__upcoming__", "gameday": when, "home_team": home_team, "away_team": away_team,
        "home_score": np.nan, "away_score": np.nan, "conference_game": bool(conference_game),
    })
    if "season" in played.columns and played["season"].notna().any():
        upcoming["season"] = played["season"].max()
    frame = pd.concat([played, pd.DataFrame([upcoming])], ignore_index=True)
    frame["gameday"] = pd.to_datetime(frame["gameday"], utc=True).dt.tz_localize(None)
    row = _assemble(frame, blocks, aux)
    served = row[row["game_id"] == "__upcoming__"].iloc[0]
    # What `_assemble` produced, in the declared order; `manifest.load_models` refuses any model whose fitted
    # columns differ from this.
    return served[[c for c in feature_columns(blocks) if c in served.index]]


def _same_conference(home_team: str, away_team: str, games_df: pd.DataFrame) -> bool:
    """Whether the two teams share a conference, read from the most recent row that names each team's conference."""
    if not {"home_conference", "away_conference"} <= set(games_df.columns):
        return False

    def conference(team: str):
        rows = games_df[(games_df["home_team"] == team) | (games_df["away_team"] == team)]
        for _, r in rows.iloc[::-1].iterrows():
            c = r["home_conference"] if r["home_team"] == team else r["away_conference"]
            if isinstance(c, str) and c:
                return c
        return None

    h, a = conference(home_team), conference(away_team)
    return h is not None and h == a
