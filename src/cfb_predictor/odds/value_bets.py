"""Join model predictions to live odds and surface single-game value edges.
Verbatim port of nfl_predictor/odds/value_bets.py — Shin de-vig math is
market-agnostic."""

from __future__ import annotations

import math

import pandas as pd
from scipy.optimize import brentq

MAX_ODDS_AGE_SECONDS = 60 * 60


def _shin_two_way(price_a: float, price_b: float) -> tuple[float, float] | None:
    if not all(isinstance(price, (int, float)) and math.isfinite(price) and price > 1 for price in (price_a, price_b)):
        return None

    pi_a, pi_b = 1.0 / price_a, 1.0 / price_b
    overround = pi_a + pi_b
    if overround <= 1.0:
        return None

    def true_prob(pi: float, z: float) -> float:
        inside = z * z + 4 * (1 - z) * (pi * pi) / overround
        return (max(inside, 0.0) ** 0.5 - z) / (2 * (1 - z))

    def z_equation(z: float) -> float:
        return true_prob(pi_a, z) + true_prob(pi_b, z) - 1.0

    try:
        z = brentq(z_equation, 0.0, 0.2)
    except ValueError:
        return pi_a / overround, pi_b / overround

    p_a, p_b = true_prob(pi_a, z), true_prob(pi_b, z)
    total = p_a + p_b
    return p_a / total, p_b / total


def devig_h2h(home_price: float, away_price: float) -> dict | None:
    result = _shin_two_way(home_price, away_price)
    if result is None:
        return None
    home, away = result
    return {"home_win": home, "away_win": away}


def devig_totals(over_price: float, under_price: float) -> dict | None:
    result = _shin_two_way(over_price, under_price)
    if result is None:
        return None
    over, under = result
    return {"over": over, "under": under}


def _valid_price(price: object) -> bool:
    return (
        isinstance(price, (int, float))
        and not isinstance(price, bool)
        and pd.notna(price)
        and math.isfinite(price)
        and price > 1
    )


_NO_BOOKMAKER = "__no_bookmaker__"
_NO_POINT = "__no_point__"


def _grouping_series(odds_df: pd.DataFrame, column: str, sentinel: str) -> pd.Series:
    if column not in odds_df.columns:
        return pd.Series(sentinel, index=odds_df.index)
    return odds_df[column].fillna(sentinel)


def _best_same_book_pair(
    odds_df: pd.DataFrame,
    event_id: str,
    market: str,
    outcome_a: str,
    outcome_b: str,
    require_matching_point: bool = False,
) -> tuple[float | None, float | None]:
    required_columns = {"event_id", "market", "outcome_name", "price"}
    if odds_df is None or odds_df.empty or not required_columns.issubset(odds_df.columns):
        return None, None

    subset = odds_df[(odds_df["event_id"] == event_id) & (odds_df["market"] == market)]
    if subset.empty:
        return None, None

    subset = subset.assign(
        _bookmaker=_grouping_series(subset, "bookmaker", _NO_BOOKMAKER),
        _point=_grouping_series(subset, "point", _NO_POINT),
    )

    rows_a = subset[subset["outcome_name"] == outcome_a]
    rows_b = subset[subset["outcome_name"] == outcome_b]
    if rows_a.empty or rows_b.empty:
        return None, None

    join_keys = ["_bookmaker", "_point"] if require_matching_point else ["_bookmaker"]
    candidates = rows_a.merge(rows_b, on=join_keys, suffixes=("_a", "_b"))
    if candidates.empty:
        return None, None

    best_pair: tuple[float, float] | None = None
    best_overround: float | None = None
    for price_a, price_b in zip(candidates["price_a"], candidates["price_b"]):
        if not (_valid_price(price_a) and _valid_price(price_b)):
            continue
        overround = 1.0 / price_a + 1.0 / price_b
        if overround <= 1.0:
            continue
        if best_overround is None or overround < best_overround:
            best_overround = overround
            best_pair = (float(price_a), float(price_b))

    return best_pair if best_pair is not None else (None, None)


def _recommended_side(row: dict, edge_threshold: float) -> list[str]:
    candidates = [
        (side, row.get(f"{side}_edge"))
        for side in ("home_win", "away_win", "over", "under")
        if row.get(f"{side}_edge") is not None and row[f"{side}_edge"] > edge_threshold
    ]
    if not candidates:
        return []
    return [max(candidates, key=lambda candidate: candidate[1])[0]]


def build_value_bet_table(
    games_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    predictions: dict[str, dict],
    edge_threshold: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for _, game in games_df.iterrows():
        game_id, home, away = game["game_id"], game["home_team"], game["away_team"]
        prediction = predictions.get(game_id, {})
        row = {
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "commence_time": game["commence_time"],
            **prediction,
        }

        home_price, away_price = _best_same_book_pair(odds_df, game_id, "h2h", home, away)
        h2h_implied = (
            devig_h2h(home_price, away_price) if home_price is not None and away_price is not None else None
        )

        over_price, under_price = _best_same_book_pair(
            odds_df, game_id, "totals", "Over", "Under", require_matching_point=True
        )
        totals_implied = (
            devig_totals(over_price, under_price) if over_price is not None and under_price is not None else None
        )

        row["home_win_edge"] = prediction.get("home_win_prob", 0) - h2h_implied["home_win"] if h2h_implied else None
        row["away_win_edge"] = prediction.get("away_win_prob", 0) - h2h_implied["away_win"] if h2h_implied else None
        row["over_edge"] = prediction["over_prob"] - totals_implied["over"] if totals_implied and "over_prob" in prediction else None
        row["under_edge"] = prediction["under_prob"] - totals_implied["under"] if totals_implied and "under_prob" in prediction else None
        row["value_bet_flags"] = _recommended_side(row, edge_threshold)
        rows.append(row)

    return pd.DataFrame(rows)
