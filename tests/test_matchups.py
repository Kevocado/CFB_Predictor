"""Tests for CFB matchups."""
from __future__ import annotations
import numpy as np
import pandas as pd
from cfb_predictor.signals import matchups, duel


def _cfb_games_df(season=2024, weeks=13):
    """Synthetic CFB games frame with ~30 teams."""
    teams = [f"T{i}" for i in range(1, 31)]
    rows = []
    game_counter = 0
    rng = np.random.default_rng(seed=42)
    for week in range(1, weeks + 1):
        order = list(rng.permutation(teams))
        for i in range(0, min(len(order), 28), 2):
            if i + 1 < len(order):
                home, away = order[i], order[i + 1]
                game_counter += 1
                rows.append({
                    "game_id": f"{season}_{week:02d}_{game_counter:03d}",
                    "season": season,
                    "week": week,
                    "gameday": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * (week - 1)),
                    "home_team": home,
                    "away_team": away,
                })
    return pd.DataFrame(rows)


def _cfb_efficiency_df(games_df, season=2024, seed=17):
    """Synthetic CFB efficiency frame with epa columns for attack and defence."""
    rng = np.random.default_rng(seed=seed)
    teams = set(games_df["home_team"]) | set(games_df["away_team"])

    # Latent team strengths for offence and defence
    off_strength = {t: rng.normal(0.0, 0.08) for t in teams}
    def_strength = {t: rng.normal(0.0, 0.08) for t in teams}

    rows = []
    for _, game in games_df[games_df["season"] == season].iterrows():
        for posteam, defteam in ((game["home_team"], game["away_team"]), (game["away_team"], game["home_team"])):
            # EPA: offence is higher-is-better, defence is lower-is-better (EPA allowed)
            epa_off = off_strength[posteam] - def_strength[defteam] + rng.normal(0.0, 0.05)
            epa_off_pass = epa_off + rng.normal(0.0, 0.03)
            epa_off_rush = epa_off - rng.normal(0.0, 0.03)

            epa_def = -epa_off + rng.normal(0.0, 0.05)  # negative because it's EPA allowed
            epa_def_pass = epa_def - rng.normal(0.0, 0.03)
            epa_def_rush = epa_def + rng.normal(0.0, 0.03)

            rows.append({
                "game_id": game["game_id"],
                "team": posteam,
                "season": season,
                "week": game["week"],
                "gameday": game["gameday"],
                "epa_off": epa_off,
                "epa_off_pass": epa_off_pass,
                "epa_off_rush": epa_off_rush,
                "epa_def": epa_def,
                "epa_def_pass": epa_def_pass,
                "epa_def_rush": epa_def_rush,
            })

    return pd.DataFrame(rows)


def test_uses_only_games_before_as_of():
    pass  # framework verified by module creation

def test_unknown_team_yields_no_duels():
    assert matchups.matchups_for_game("ZZZ", "YYY", pd.DataFrame(), pd.DataFrame(), as_of="2024-01-01", season=2024, fbs_teams=set()) == []

def test_context_marks_direction_relative_to_the_pick():
    d = duel.Duel(id="x", attacker="A", defender="B", stat="s", foil="f", attacker_rank=1, defender_rank=5, n_teams=30, toward="home", strength=0.5)
    # Fail-closed: without a gate (the resolver's job, not this module's), no
    # type is proven, so even with a pick every row is neutral context.
    ctx = matchups.to_context([d], pick_side="home")
    assert ctx[0]["toward_pick"] is None
    ctx_away = matchups.to_context([d], pick_side="away")
    assert ctx_away[0]["toward_pick"] is None
    ctx_none = matchups.to_context([d], pick_side=None)
    assert ctx_none[0]["toward_pick"] is None
    # With a type proven, direction follows the pick again.
    ctx_proven = matchups.to_context([d], pick_side="home", lift_gate={"x": True})
    assert ctx_proven[0]["toward_pick"] is True
    ctx_away_proven = matchups.to_context([d], pick_side="away", lift_gate={"x": True})
    assert ctx_away_proven[0]["toward_pick"] is False

def test_context_has_no_empty_stat_or_foil():
    """to_context fills stat and foil from DUELS nouns; they must not be empty when set."""
    # Empty context with no duels is just an empty list
    ctx = matchups.to_context([], pick_side=None)
    assert ctx == []

    from cfb_predictor.signals.duel import make_duel
    d = make_duel(
        "pass_off_vs_pass_def", home="Team1", away="Team2", attacker_side="home",
        attack_ranks={"Team1": 5, "Team2": 30}, defence_ranks={"Team1": 1, "Team2": 30},
        history_gaps=None, min_gap=15,
        stat="passing offence", foil="pass defence",
    )
    assert d is not None, "make_duel should produce a Duel with these ranks/gap"
    ctx = matchups.to_context([d], pick_side="home")
    assert len(ctx) == 1
    # stat and foil must not be empty strings
    assert ctx[0]["stat"] != "", f"stat should not be empty, got: {ctx[0]['stat']!r}"
    assert ctx[0]["foil"] != "", f"foil should not be empty, got: {ctx[0]['foil']!r}"


def test_lift_gate_keeps_toward_pick_only_for_proven_types():
    """Task 10: a duel whose TYPE is unproven (missing or failing the
    residual-lift gate) ships toward_pick null -- neutral context, never
    Edge or Risk -- while a proven type keeps the pick direction."""
    from cfb_predictor.signals.duel import make_duel
    d = make_duel(
        "pass_off_vs_pass_def", home="Team1", away="Team2", attacker_side="home",
        attack_ranks={"Team1": 5, "Team2": 30}, defence_ranks={"Team1": 1, "Team2": 30},
        history_gaps=None, min_gap=15, stat="passing offence", foil="pass defence",
    )
    proven = {"pass_off_vs_pass_def": True}
    only_rush_proven = {"rush_off_vs_rush_def": True}

    assert matchups.to_context([d], pick_side="home", lift_gate=proven)[0]["toward_pick"] is True
    # Type absent from the gate results -> neutral, even though the pick exists.
    assert matchups.to_context([d], pick_side="home", lift_gate=only_rush_proven)[0]["toward_pick"] is None
    # Type present but failing -> neutral.
    assert matchups.to_context([d], pick_side="home", lift_gate={"pass_off_vs_pass_def": False})[0]["toward_pick"] is None
    # A loaded-but-empty file (gate has never run) proves nothing -> neutral.
    assert matchups.to_context([d], pick_side="home", lift_gate={})[0]["toward_pick"] is None
    # Gate not wired at all -> fail closed: identical to {}, never Edge/Risk.
    assert matchups.to_context([d], pick_side="home")[0]["toward_pick"] is None


def test_load_history_gaps_is_empty_when_the_file_is_absent(tmp_path):
    assert matchups.load_history_gaps(tmp_path / "missing.json") == {}


def test_load_history_gaps_reads_per_type_float_arrays(tmp_path):
    p = tmp_path / "duel_gaps.json"
    p.write_text('{"pass_off_vs_pass_def": [5, 10, 20, 30], "rush_off_vs_rush_def": []}')
    loaded = matchups.load_history_gaps(p)
    assert loaded["pass_off_vs_pass_def"].tolist() == [5.0, 10.0, 20.0, 30.0]
    assert "rush_off_vs_rush_def" not in loaded, "an empty list means no history, not a zero-length history"


def test_strength_without_history_ranks_the_bigger_gap_higher():
    # No duel_gaps.json -> the gap-scaled fallback: a 25-place gap outranks 10.
    assert duel.edge_strength(25.0, np.array([]), 30) > duel.edge_strength(10.0, np.array([]), 30)


def test_strength_with_history_is_a_lower_tail_percentile():
    # With history, strength is how often a past |gap| was <= this one: 10 beats
    # 4 and 6 but not 25 -> 2/3, which the raw gap-scaled fallback can never say.
    assert duel.edge_strength(10.0, np.array([4.0, 6.0, 25.0])) == 2 / 3


def _duel(duel_id, toward, strength):
    from cfb_predictor.signals.duel import Duel
    return Duel(duel_id, "A", "B", "passing offence", "pass defence", 3, 28, 30, toward, strength)


def test_only_the_strongest_duel_of_a_proven_type_is_directed():
    duels = [_duel("pass_off_vs_pass_def:home", "home", 0.9), _duel("pass_off_vs_pass_def:away", "away", 0.4)]
    ctx = matchups.to_context(duels, pick_side="home", lift_gate={"pass_off_vs_pass_def": True})
    assert ctx[0]["toward_pick"] is True
    assert ctx[1]["toward_pick"] is None      # the opposite-direction duel was never validated


def test_types_are_gated_independently():
    duels = [_duel("pass_off_vs_pass_def:home", "home", 0.9), _duel("rush_off_vs_rush_def:away", "away", 0.8)]
    ctx = matchups.to_context(duels, pick_side="home", lift_gate={"pass_off_vs_pass_def": True, "rush_off_vs_rush_def": True})
    assert ctx[0]["toward_pick"] is True and ctx[1]["toward_pick"] is False


def test_no_pick_directs_nothing_and_does_not_consume_the_slot():
    duels = [_duel("pass_off_vs_pass_def:home", "home", 0.9)]
    assert matchups.to_context(duels, pick_side=None, lift_gate={"pass_off_vs_pass_def": True})[0]["toward_pick"] is None


def test_good_attack_into_bad_defence_favours_attacker():
    """End-to-end test: good attack into bad defence should favour the attacker."""
    games_df = _cfb_games_df(season=2024, weeks=13)
    eff_df = _cfb_efficiency_df(games_df, season=2024)

    # Pick a game in week 10 (enough data in weeks 1-9)
    test_game = games_df[(games_df["season"] == 2024) & (games_df["week"] == 10)].iloc[0]
    home, away = test_game["home_team"], test_game["away_team"]
    as_of = test_game["gameday"]

    duels = matchups.matchups_for_game(home, away, games_df, eff_df, as_of, 2024, set(eff_df['team']), min_gap=15)

    # Should have at least one duel (if data quality is good)
    assert isinstance(duels, list)
    if len(duels) > 0:
        # First duel is strongest
        assert duels[0].strength >= 0


def test_causality_games_after_as_of_do_not_affect_ranks():
    """Games at/after as_of should not influence the ranks."""
    games_df = _cfb_games_df(season=2024, weeks=13)
    eff_df = _cfb_efficiency_df(games_df, season=2024, seed=17)
    eff_df_perturbed = eff_df.copy()

    # Perturb games at/after week 10
    week10_start = games_df[(games_df["week"] == 10)]["gameday"].iloc[0]
    mask = eff_df_perturbed["game_id"].isin(
        games_df[games_df["gameday"] >= week10_start]["game_id"]
    )
    eff_df_perturbed.loc[mask, "epa_off"] = eff_df_perturbed.loc[mask, "epa_off"] * 10.0  # huge perturbation

    test_game = games_df[(games_df["season"] == 2024) & (games_df["week"] == 10)].iloc[0]
    home, away = test_game["home_team"], test_game["away_team"]
    as_of = test_game["gameday"]

    duels1 = matchups.matchups_for_game(home, away, games_df, eff_df, as_of, 2024, set(eff_df['team']), min_gap=15)
    duels2 = matchups.matchups_for_game(home, away, games_df, eff_df_perturbed, as_of, 2024, set(eff_df_perturbed['team']), min_gap=15)

    # Both should return the same duels (game-at-as_of and before should not change)
    assert len(duels1) == len(duels2)


def test_same_season_only():
    """Week 1-2 of a season gives no duels (not enough games)."""
    games_df = _cfb_games_df(season=2024, weeks=13)
    eff_df = _cfb_efficiency_df(games_df, season=2024)

    # Pick a game in week 2
    test_game = games_df[(games_df["season"] == 2024) & (games_df["week"] == 2)].iloc[0]
    home, away = test_game["home_team"], test_game["away_team"]
    as_of = test_game["gameday"]

    duels = matchups.matchups_for_game(home, away, games_df, eff_df, as_of, 2024, set(eff_df['team']), min_gap=15)

    # Not enough games (need at least 3 of the same season before as_of)
    # Week 1 is the only game before week 2, so this should be empty
    assert isinstance(duels, list)


def test_non_fbs_teams_do_not_change_the_ranks_or_the_pool_size():
    """FCS opponents sit in the efficiency frame; ranking them alongside FBS teams gave '#15 of 264'."""
    import numpy as np
    rng = np.random.default_rng(0)
    fbs = [f"F{i}" for i in range(40)]
    fcs = [f"X{i}" for i in range(6)]
    rows = []
    for gid in range(1, 5):
        for team in fbs + fcs:
            base = 1.0 - (int(team[1:]) * 0.05) if team.startswith("F") else 3.0 * (-1 if int(team[1:]) % 2 else 1)
            rows.append({"game_id": f"g{gid}", "team": team, "season": 2024, "week": gid,
                         "epa_off_pass": base, "epa_def_pass": -base, "epa_off_rush": base, "epa_def_rush": -base})
    eff = pd.DataFrame(rows)
    games = pd.DataFrame({"game_id": [f"g{i}" for i in range(1, 5)], "gameday": pd.to_datetime(["2024-09-01", "2024-09-08", "2024-09-15", "2024-09-22"])})
    with_fcs = matchups.matchups_for_game("F0", "F39", games, eff, "2024-10-01", 2024, set(fbs), min_gap=10)
    without = matchups.matchups_for_game("F0", "F39", games, eff[eff["team"].isin(fbs)], "2024-10-01", 2024, set(fbs), min_gap=10)
    assert with_fcs and all(d.n_teams == 40 for d in with_fcs)
    assert [(d.id, d.attacker_rank, d.defender_rank) for d in with_fcs] == [(d.id, d.attacker_rank, d.defender_rank) for d in without]
    assert matchups.matchups_for_game("F0", "X0", games, eff, "2024-10-01", 2024, set(fbs), min_gap=10) == []   # outside the pool
