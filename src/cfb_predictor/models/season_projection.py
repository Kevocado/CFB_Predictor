"""season_projection.py — projected final standings for the current season.

Deterministic, not simulated (same reasoning as PL_Predictor's own
projected_table.py): every remaining game's predicted win probability IS
its expected-win contribution, so summing those over a team's remaining
schedule on top of their actual record so far is the projection -- no
Monte Carlo sampling needed.

Unlike NFL (fixed divisions from nfl_data_py's team_conf/team_division),
CFB conference realignment happens often enough, and not every conference
has real sub-divisions any more, that a hardcoded division map would go
stale. Conference alone (already present on every game row as
home_conference/away_conference) is what's grouped/ranked within here.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

PredictFn = Callable[[str, str], dict]


def compute_current_records(played_games: pd.DataFrame) -> dict[str, dict]:
    """wins/losses/ties/played/point_diff per team from games with a final score."""
    records: dict[str, dict] = {}
    for _, g in played_games.iterrows():
        home, away = g["home_team"], g["away_team"]
        home_score, away_score = g["home_score"], g["away_score"]
        for team in (home, away):
            records.setdefault(team, {"wins": 0, "losses": 0, "ties": 0, "played": 0, "point_diff": 0.0})
        records[home]["played"] += 1
        records[away]["played"] += 1
        records[home]["point_diff"] += float(home_score - away_score)
        records[away]["point_diff"] += float(away_score - home_score)
        if home_score > away_score:
            records[home]["wins"] += 1
            records[away]["losses"] += 1
        elif away_score > home_score:
            records[away]["wins"] += 1
            records[home]["losses"] += 1
        else:
            records[home]["ties"] += 1
            records[away]["ties"] += 1
    return records


def build_team_conferences(season_games: pd.DataFrame) -> dict[str, str]:
    """FBS team -> conference, read straight off the season's own schedule
    rows (home_conference/away_conference, home_division/away_division)
    rather than a separate fetch -- every FBS team appears as home or away
    at least once in a full-season schedule. CFBD supplies a conference
    for FCS teams too (their own FCS conferences, e.g. SWAC, Patriot), so
    the division check is what actually limits this to the ~130 FBS teams
    a "season standings" page should show, not the presence of a
    conference value alone."""
    conf_by_team: dict[str, str] = {}
    for _, g in season_games.iterrows():
        if g.get("home_division") == "fbs" and pd.notna(g.get("home_conference")):
            conf_by_team.setdefault(g["home_team"], g["home_conference"])
        if g.get("away_division") == "fbs" and pd.notna(g.get("away_conference")):
            conf_by_team.setdefault(g["away_team"], g["away_conference"])
    return conf_by_team


def _rank_within_group(rows: list[dict], wins_key: str, pd_key: str, group_key: str, position_key: str) -> None:
    groups: dict[object, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda r: (-r[wins_key], -r[pd_key]))
        for i, row in enumerate(group_rows, start=1):
            row[position_key] = i


def project_standings(
    remaining_games: pd.DataFrame,
    current_records: dict[str, dict],
    team_conferences: dict[str, str],
    predict_fn: PredictFn,
) -> list[dict]:
    """predict_fn(home, away) must return a dict with home_win_prob,
    away_win_prob, and (optionally) predicted_margin. Only teams with a
    known conference are ranked (an FBS team should always have one; an
    FCS "buy game" opponent legitimately has none and is excluded)."""
    projected_win_add: dict[str, float] = {}
    projected_pd_add: dict[str, float] = {}
    remaining_count: dict[str, int] = {}

    for _, g in remaining_games.iterrows():
        home, away = g["home_team"], g["away_team"]
        for team in (home, away):
            projected_win_add.setdefault(team, 0.0)
            projected_pd_add.setdefault(team, 0.0)
            remaining_count.setdefault(team, 0)
            remaining_count[team] += 1
        try:
            pred = predict_fn(home, away)
        except Exception:
            continue
        projected_win_add[home] += pred["home_win_prob"]
        projected_win_add[away] += pred["away_win_prob"]
        margin = pred.get("predicted_margin", 0.0) or 0.0
        projected_pd_add[home] += margin
        projected_pd_add[away] -= margin

    all_teams = (set(projected_win_add) | set(current_records)) & set(team_conferences)
    rows = []
    for team in all_teams:
        current = current_records.get(team, {"wins": 0, "losses": 0, "ties": 0, "played": 0, "point_diff": 0.0})
        total_games = current["played"] + remaining_count.get(team, 0)
        final_projected_wins = current["wins"] + projected_win_add.get(team, 0.0)
        rows.append({
            "team": team,
            "conference": team_conferences[team],
            "played": current["played"],
            "wins": current["wins"],
            "losses": current["losses"],
            "ties": current["ties"],
            "point_diff": round(current["point_diff"], 1),
            "projected_wins": round(final_projected_wins, 1),
            "projected_losses": round(max(0.0, total_games - final_projected_wins), 1),
            "projected_point_diff": round(current["point_diff"] + projected_pd_add.get(team, 0.0), 1),
        })

    _rank_within_group(rows, "wins", "point_diff", "conference", "current_conference_rank")
    _rank_within_group(rows, "projected_wins", "projected_point_diff", "conference", "projected_conference_rank")
    for row in rows:
        row["conference_rank_delta"] = row["current_conference_rank"] - row["projected_conference_rank"]

    rows.sort(key=lambda r: (r["conference"] or "", r["projected_conference_rank"]))
    return rows
