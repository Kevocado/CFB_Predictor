"""Weekly player stats now carry their OWN `game_id` (KEEP_COLUMNS). `_attach_game_id` still merged a second `game_id`
in by season/week/team, which yields `game_id_x`/`game_id_y` and no `game_id`, so `reconcile_player_prop_predictions`
raised `KeyError: 'game_id'` on every tracker tick."""
import pandas as pd

from cfb_predictor.api.routes import _attach_game_id


def _games():
    return pd.DataFrame({"game_id": ["g1", "g2"], "season": [2026, 2026], "week": [6, 6],
                         "home_team": ["Troy", "Liberty"], "away_team": ["Southern Miss", "Sam Houston"]})


def test_stats_that_already_carry_a_game_id_keep_exactly_that_game_id():
    stats = pd.DataFrame({"player_id": ["p1"], "recent_team": ["Troy"], "season": [2026], "week": [6], "game_id": ["g1"]})
    out = _attach_game_id(stats, _games())
    assert "game_id" in out.columns and "game_id_x" not in out.columns and "game_id_y" not in out.columns
    assert list(out["game_id"]) == ["g1"]


def test_stats_without_a_game_id_still_get_one_from_season_week_team():
    stats = pd.DataFrame({"player_id": ["p1"], "recent_team": ["Liberty"], "season": [2026], "week": [6]})
    out = _attach_game_id(stats, _games())
    assert list(out["game_id"]) == ["g2"]
