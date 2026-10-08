"""Preseason prior: last season's final rating, regressed toward its conference mean. Known before kickoff of week 1."""
from __future__ import annotations

import pandas as pd


def preseason_prior(finals: pd.DataFrame, season: int, regress: float = 0.4, new_teams: dict | None = None) -> pd.DataFrame:
    prev = finals[finals["season"] == season - 1]
    if prev.empty:
        prev = finals[finals["season"] < season].sort_values("season").groupby("team").tail(1)
    conf_mean = prev.groupby("conference")["rating"].mean()
    rows = []
    for _, r in prev.iterrows():
        mean = conf_mean[r["conference"]]
        rows.append({"team": r["team"], "prior": mean + (1 - regress) * (r["rating"] - mean)})
    for team, conf in (new_teams or {}).items():
        rows.append({"team": team, "prior": float(conf_mean.get(conf, prev["rating"].mean()))})
    return pd.DataFrame(rows)