"""CFB EPA block: same logic as NFL but over CFBD advanced stats (to_team_game_frame output)."""
from __future__ import annotations

import pandas as pd

EWM_HALFLIFE = 6.0
SHRINK_K = 4.0
STATS = [
    "epa_off", "epa_def", "epa_off_pass", "epa_off_rush", "epa_def_pass", "epa_def_rush", "success_off", "success_def",
]
EDGE_COLUMNS = ["home_pass_edge", "home_rush_edge", "away_pass_edge", "away_rush_edge", "epa_net_diff"]


def epa_columns() -> list[str]:
    return [f"{side}_{s}_ewm" for s in STATS for side in ("home", "away")] + EDGE_COLUMNS


def _shrunk_ewm(series: pd.Series, halflife: float, k: float) -> pd.Series:
    lagged = series.shift(1)
    n = lagged.notna().cumsum()
    return lagged.ewm(halflife=halflife, min_periods=1).mean() * (n / (n + k))


def add_epa_features(games_df: pd.DataFrame, efficiency: pd.DataFrame, halflife: float = EWM_HALFLIFE, k: float = SHRINK_K) -> pd.DataFrame:
    """Add CFB EPA features. `efficiency` is to_team_game_frame() output."""
    rows = []
    for side in ("home", "away"):
        rows.append(games_df[["game_id", "gameday", f"{side}_team"]].rename(columns={f"{side}_team": "team"}))
    long = pd.concat(rows, ignore_index=True).drop_duplicates(["game_id", "team"])
    long["gameday"] = pd.to_datetime(long["gameday"])
    long = long.merge(efficiency, on=["game_id", "team"], how="left").sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    for stat in STATS:
        long[f"{stat}_ewm"] = long.groupby("team")[stat].transform(lambda s: _shrunk_ewm(s, halflife, k))
    keyed = long.set_index(["game_id", "team"])[[f"{s}_ewm" for s in STATS]]
    out = games_df.copy()
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([out["game_id"], out[f"{side}_team"]])
        vals = keyed.reindex(idx)
        for s in STATS:
            out[f"{side}_{s}_ewm"] = vals[f"{s}_ewm"].to_numpy()
    out["home_pass_edge"] = out["home_epa_off_pass_ewm"] - out["away_epa_def_pass_ewm"]
    out["home_rush_edge"] = out["home_epa_off_rush_ewm"] - out["away_epa_def_rush_ewm"]
    out["away_pass_edge"] = out["away_epa_off_pass_ewm"] - out["home_epa_def_pass_ewm"]
    out["away_rush_edge"] = out["away_epa_off_rush_ewm"] - out["home_epa_def_rush_ewm"]
    out["epa_net_diff"] = (out["home_epa_off_ewm"] - out["home_epa_def_ewm"]) - (out["away_epa_off_ewm"] - out["away_epa_def_ewm"])
    return out


#: Opponent-adjusted EPA (the TOTAL-points model's block). Each game's offence/defence EPA minus what the opponent
#: usually gives up / produces going in (its own lagged shrunk EWM, 0 when unknown), then the same shrunk EWM, centred
#: on a FIXED league mean. The centre is a constant (mean of the adjusted values over the committed 2014-2015 games)
#: and not recomputed from whatever efficiency a caller passes: production trains on 2018+ and serves a rolling window,
#: so a data-derived centre would differ between the two and silently shift the shrinkage target.
ADJ_STATS = ["adj_off_epa", "adj_def_epa"]
ADJ_CENTER = {"adj_off_epa": 0.04321970229915947, "adj_def_epa": 0.025368536253321004}


def adj_epa_columns() -> list[str]:
    return [f"{side}_{s}_ewm" for s in ADJ_STATS for side in ("home", "away")]


def add_adj_epa_features(games_df: pd.DataFrame, efficiency: pd.DataFrame, halflife: float = EWM_HALFLIFE, k: float = SHRINK_K) -> pd.DataFrame:
    """Add the four opponent-adjusted EPA columns. `efficiency` is to_team_game_frame() output."""
    parts = [
        games_df[["game_id", "gameday", f"{side}_team", f"{other}_team"]].set_axis(["game_id", "gameday", "team", "opp"], axis=1)
        for side, other in (("home", "away"), ("away", "home"))
    ]
    long = pd.concat(parts, ignore_index=True).drop_duplicates(["game_id", "team"])
    long["gameday"] = pd.to_datetime(long["gameday"])
    eff = efficiency[["game_id", "team", "epa_off", "epa_def"]].assign(game_id=lambda d: d["game_id"].astype(str))
    long["game_id"] = long["game_id"].astype(str)
    long = long.merge(eff, on=["game_id", "team"], how="left")
    if long["epa_off"].notna().sum() == 0:
        raise ValueError("no efficiency row matches any game; refusing to build adj_epa from nothing")
    long = long.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    for c in ("epa_off", "epa_def"):
        long[c + "_pre"] = long.groupby("team")[c].transform(lambda s: _shrunk_ewm(s, halflife, k))
    opp = long[["game_id", "team", "epa_off_pre", "epa_def_pre"]].rename(
        columns={"team": "opp", "epa_off_pre": "opp_off_pre", "epa_def_pre": "opp_def_pre"})
    long = long.merge(opp, on=["game_id", "opp"], how="left")
    long["adj_off_epa"] = long["epa_off"] - long["opp_def_pre"].fillna(0)
    long["adj_def_epa"] = long["epa_def"] - long["opp_off_pre"].fillna(0)
    for s in ADJ_STATS:
        long[s + "_ewm"] = long.groupby("team")[s].transform(lambda x: _shrunk_ewm(x - ADJ_CENTER[s], halflife, k))
    keyed = long.set_index(["game_id", "team"])[[s + "_ewm" for s in ADJ_STATS]]
    out = games_df.copy()
    for side in ("home", "away"):
        vals = keyed.reindex(pd.MultiIndex.from_arrays([out["game_id"].astype(str), out[f"{side}_team"]]))
        for s in ADJ_STATS:
            out[f"{side}_{s}_ewm"] = vals[s + "_ewm"].to_numpy()
    return out
