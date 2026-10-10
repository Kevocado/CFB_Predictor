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


PRIOR_COLUMNS = ["home_prior", "away_prior", "prior_diff"]
NEUTRAL = 1500.0  # Elo start rating: what a team with no history gets, i.e. "no information"
UNKNOWN_CONF = "unknown"


def prior_columns() -> list[str]:
    return list(PRIOR_COLUMNS)


def add_priors(df: pd.DataFrame, regress: float = 0.4) -> pd.DataFrame:
    """Add home_prior / away_prior / prior_diff to a frame that already carries `*_pregame_rating` and `season`.

    A team's season-s prior is its Elo going into its LAST season-(s-1) game (so it omits that one game's update),
    regressed toward the mean of its season-(s-1) conference. The baseline is the COMPLETE season-(s-1) field, so a
    team's prior does not change as other teams play in season s (training sees the whole schedule; serving sees only
    the games played so far, and the two must agree). A team in season s with no season-(s-1) history gets its own
    conference's mean (`new_teams`). The first season in the data has no history and is neutral for everyone.
    """
    if "season" not in df.columns or df["season"].isna().all():
        raise ValueError("the priors block needs a `season` column")
    ordered = df.sort_values(["gameday", "game_id"])
    long = pd.concat([
        pd.DataFrame({"gameday": ordered["gameday"], "team": ordered[f"{side}_team"], "season": ordered["season"], "rating": ordered[f"{side}_pregame_rating"],
                      "conference": ordered[f"{side}_conference"] if f"{side}_conference" in ordered.columns else None})
        for side in ("home", "away")
    ], ignore_index=True).dropna(subset=["season"])
    long["conference"] = long["conference"].where(long["conference"].map(lambda c: isinstance(c, str) and bool(c)), UNKNOWN_CONF)
    long = long.sort_values("gameday", kind="stable")  # earliest first, home or away
    seasons = sorted(long["season"].unique())
    tables = []
    for i, s in enumerate(seasons):
        cur = long[long["season"] == s].groupby("team").first().reset_index()  # who plays in s, and their s conference
        if i == 0:
            tables.append(pd.DataFrame({"team": cur["team"], "season": s, "prior": NEUTRAL}))
            continue
        prev = long[long["season"] == seasons[i - 1]].groupby("team").last().reset_index()
        finals = prev.assign(season=s - 1)[["team", "season", "rating", "conference"]]
        new = cur[~cur["team"].isin(prev["team"])]
        t = preseason_prior(finals, season=s, regress=regress, new_teams=dict(zip(new["team"], new["conference"])))
        tables.append(t.assign(season=s))
    table = pd.concat(tables, ignore_index=True).drop_duplicates(["season", "team"]).set_index(["season", "team"])["prior"]
    out = df.copy()
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([out["season"], out[f"{side}_team"]])
        out[f"{side}_prior"] = table.reindex(idx).fillna(NEUTRAL).to_numpy()
    out["prior_diff"] = out["home_prior"] - out["away_prior"]
    return out
