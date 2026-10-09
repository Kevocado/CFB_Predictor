"""CFB offence-versus-defence duels for one game, from game-level efficiency known BEFORE the game.

Each duel is one side's attack (EPA/play, pass or rush) against the other side's matching defence, expressed as league
ranks. Only games of `season` strictly before `as_of`, and each team's last `WINDOW` of them, are used: early in a season
there are fewer than `MIN_GAMES` and the answer is no duels, which is better than ranking last year's roster.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .duel import Duel, make_duel, ranks

WINDOW = 8
MIN_GAMES = 3
#: (duel id, attack column, defence column, attack noun, defence noun). Defence columns are EPA ALLOWED: lower is better.
DUELS = [
    ("pass_off_vs_pass_def", "epa_off_pass", "epa_def_pass", "passing offence", "pass defence"),
    ("rush_off_vs_rush_def", "epa_off_rush", "epa_def_rush", "rushing offence", "run defence"),
]


def load_history_gaps(path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Past absolute rank gaps per duel type, from `data/duel_gaps.json` (written by the Task 10 tool
    from the walk-forward games). An absent file or an absent type yields {}, which sends
    `edge_strength` down the gap-scaled fallback (Task 1) instead of a percentile."""
    path = Path(path) if path is not None else Path(__file__).resolve().parents[3] / "data" / "duel_gaps.json"
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: np.asarray(v, dtype=float) for k, v in raw.items() if isinstance(v, list) and v}


def _recent_means(efficiency: pd.DataFrame, games_df: pd.DataFrame, as_of: pd.Timestamp, season: int) -> pd.DataFrame:
    """Last WINDOW games per team, from `season` strictly before `as_of`, with at least MIN_GAMES of data."""
    if efficiency.empty or games_df.empty:
        return pd.DataFrame(columns=efficiency.columns) if not efficiency.empty else pd.DataFrame()
    meta = games_df[["game_id", "gameday"]].copy()
    meta["gameday"] = pd.to_datetime(meta["gameday"])
    # Merge; if efficiency already has gameday, use suffixes to avoid collision
    efficiency_idxed = efficiency.merge(meta, on="game_id", how="left", suffixes=('', '_games'))
    # Use gameday_games if it exists (from meta), otherwise use gameday
    if "gameday_games" in efficiency_idxed.columns:
        efficiency_idxed = efficiency_idxed.drop(columns=["gameday"])
        efficiency_idxed = efficiency_idxed.rename(columns={"gameday_games": "gameday"})
    eff = efficiency_idxed.assign(
        gameday=pd.to_datetime(efficiency_idxed["gameday"]),
    )
    eff = eff[(eff["gameday"] < as_of) & (eff["season"] == season)].sort_values("gameday")
    last = eff.groupby("team").tail(WINDOW)
    counts = last.groupby("team").size()
    means = last.groupby("team").mean(numeric_only=True)
    return means[counts.reindex(means.index) >= MIN_GAMES]


def matchups_for_game(
    home: str, away: str, games_df: pd.DataFrame, efficiency: pd.DataFrame, as_of, season: int,
    fbs_teams: set[str], history_gaps: dict[str, np.ndarray] | None = None, min_gap: int = 15,
) -> list[Duel]:
    """Up to four duels (two kinds x two directions), strongest first. Empty when either team lacks data.

    `fbs_teams` is REQUIRED: the efficiency frame also carries FCS opponents, so ranking "everyone in the frame" gave
    "#15 of 264". Ranks are among the season's FBS teams only (the set `games_data.fetch_fbs_teams(season)` yields);
    a team outside the set gets no duel."""
    efficiency = efficiency[efficiency["team"].isin(fbs_teams)] if len(efficiency) else efficiency
    means = _recent_means(efficiency, games_df, pd.Timestamp(as_of), season)
    if home not in means.index or away not in means.index:
        return []
    history_gaps = history_gaps or {}
    out: list[Duel] = []
    for duel_id, attack_col, defence_col, attack_noun, defence_noun in DUELS:
        attack_ranks = ranks(means[attack_col].to_dict(), higher_is_better=True)
        defence_ranks = ranks(means[defence_col].to_dict(), higher_is_better=False)
        for attacker_side in ("home", "away"):
            d = make_duel(
                f"{duel_id}:{attacker_side}",
                home=home, away=away, attacker_side=attacker_side,
                attack_ranks=attack_ranks, defence_ranks=defence_ranks,
                history_gaps=history_gaps.get(duel_id, np.array([])), min_gap=min_gap,
                stat=attack_noun, foil=defence_noun,
            )
            if d is not None:
                out.append(d)
    out.sort(key=lambda d: d.strength, reverse=True)
    return out


def to_context(duels: list[Duel], pick_side: str | None, limit: int = 4,
               lift_gate: dict[str, bool] | None = None) -> list[dict]:
    """Facts-bundle form. `toward_pick` is True when the duel favours the pick
    AND the residual-lift gate (Task 10) has proven the duel's TYPE: `lift_gate`
    maps a duel type (\"pass_off_vs_pass_def\", id without the :side suffix) to
    whether it passes. Unproven (missing/failing) types ship `toward_pick` None
    -- neutral context, never Edge or Risk.

    The gate is FAIL CLOSED: `lift_gate=None` behaves exactly like `{}` (nothing
    is proven), so there is no "ungated" mode -- a caller that forgets to wire
    the gate can never emit Edge/Risk-capable rows before a type is proven.
    """
    out: list[dict] = []
    directed_types: set[str] = set()
    for d in duels[:limit]:
        duel_type = d.id.split(":")[0]
        proven = (lift_gate or {}).get(duel_type, False)
        # The lift test validates the STRONGEST duel of a type per game, so only that one (the first of its type in
        # the strongest-first order `matchups_for_game` returns) may carry a direction. The opposite-direction
        # duel of the same type was never tested and stays neutral context.
        eligible = proven and duel_type not in directed_types
        toward_pick = None if (pick_side is None or not eligible) else (d.toward == pick_side)
        if eligible and pick_side is not None:
            directed_types.add(duel_type)
        out.append({
            "id": d.id,
            "attacker": d.attacker,
            "defender": d.defender,
            "stat": d.stat,
            "foil": d.foil,
            "attacker_rank": d.attacker_rank,
            "defender_rank": d.defender_rank,
            "n_teams": d.n_teams,
            "toward_pick": toward_pick,
        })
    return out
