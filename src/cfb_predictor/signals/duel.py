"""Matchup duels: copied identically into NFL/CFB/PL/NBA repos."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Duel:
    id: str; attacker: str; defender: str; stat: str; foil: str
    attacker_rank: int; defender_rank: int; n_teams: int; toward: str; strength: float

def ranks(values: dict[str,float], higher_is_better: bool=True) -> dict[str,int]:
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    out, last_v, last_r = {}, None, 0
    for i,(team,v) in enumerate(ordered, start=1):
        last_r = last_r if v==last_v else i
        last_v = v
        out[team] = last_r
    return out

def edge_strength(gap: float, history: np.ndarray | list, n_teams: int|None=None) -> float:
    """Lower-tail percentile: how often a past |gap| was <= this |gap|.
    Bigger gap → higher strength. history = past absolute gaps."""
    h = np.abs(np.asarray(history, float)) if not isinstance(history, np.ndarray) else np.abs(history)
    if h.size == 0:
        n = n_teams or 32
        return min(1.0, abs(gap) / (n-1)) if n and n > 1 else 0.0
    return float((h <= abs(gap)).mean())

def make_duel(duel_id: str, *, home: str, away: str, attacker_side: str,
              attack_ranks: dict[str, int], defence_ranks: dict[str, int], history_gaps=None,
              min_gap: int = 8, stat: str = "", foil: str = "") -> Duel | None:
    """The duel of `attacker_side`'s attack against the other side's defence, or None when too close."""
    attacker = home if attacker_side == "home" else away
    defender = away if attacker_side == "home" else home
    a = attack_ranks.get(attacker)
    d = defence_ranks.get(defender)
    if a is None or d is None:
        return None
    gap = d - a  # positive: defence ranks WORSE than attack ranks → attack has the edge
    if abs(gap) < min_gap:
        return None
    toward = attacker_side if gap > 0 else ("away" if attacker_side == "home" else "home")
    history = np.array([], float) if history_gaps is None else np.asarray(history_gaps, float)
    return Duel(
        duel_id, attacker, defender, stat, foil,
        a, d, len(attack_ranks), toward,
        edge_strength(gap, history, len(attack_ranks))
    )
