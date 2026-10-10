import pandas as pd
import pytest
from cfb_predictor.features.priors import preseason_prior


def test_prior_regresses_last_final_rating_toward_conference_mean():
    finals = pd.DataFrame({"team": ["A", "B", "C"], "season": [2023] * 3, "rating": [1800.0, 1500.0, 1200.0],
                           "conference": ["X", "X", "X"]})
    p = preseason_prior(finals, season=2024, regress=0.4).set_index("team")["prior"]
    mean = 1500.0
    assert abs(p["A"] - (mean + 0.6 * 300)) < 1e-9 and abs(p["C"] - (mean - 0.6 * 300)) < 1e-9


def test_new_team_gets_the_conference_mean_not_zero():
    finals = pd.DataFrame({"team": ["A", "B"], "season": [2023, 2023], "rating": [1600.0, 1400.0], "conference": ["X", "X"]})
    p = preseason_prior(finals, season=2024, regress=0.4, new_teams={"Z": "X"}).set_index("team")["prior"]
    assert p["Z"] == 1500.0


def test_only_prior_seasons_are_used():
    finals = pd.DataFrame({"team": ["A", "A"], "season": [2023, 2024], "rating": [1700.0, 1000.0], "conference": ["X", "X"]})
    p = preseason_prior(finals, season=2024, regress=0.0).set_index("team")["prior"]
    assert p["A"] == 1700.0  # the 2024 final must not leak into the 2024 preseason

# --- the wired block, on frames shaped like data/cache/games/<season>.parquet (games.KEEP_COLUMNS) ---
import numpy as np
from cfb_predictor.features import build as feature_build
from cfb_predictor.features.priors import add_priors

CONFS = {"Alabama": "SEC", "Georgia": "SEC", "Ohio State": "Big Ten", "Michigan": "Big Ten", "Newcomer": "SEC",
         "Samford": None}  # FCS opponent: CFBD gives no FBS conference


def _games(seasons=(2018, 2019, 2020), newcomer_from=2020, seed=3):
    rng = np.random.default_rng(seed)
    teams = ["Alabama", "Georgia", "Ohio State", "Michigan", "Samford"]
    rows = []
    for s in seasons:
        ts = teams + (["Newcomer"] if s >= newcomer_from else [])
        for w in range(1, 6):
            order = list(rng.permutation(ts))
            for i in range(0, len(order) - 1, 2):
                h, a = order[i], order[i + 1]
                rows.append({
                    "game_id": str(401000000 + s * 100 + w * 10 + i), "season": s, "week": w,
                    "gameday": pd.Timestamp(f"{s}-09-01") + pd.Timedelta(days=7 * (w - 1)),
                    "home_team": h, "away_team": a,
                    "home_score": float(rng.integers(3, 50)), "away_score": float(rng.integers(3, 50)),
                    "home_conference": CONFS[h], "away_conference": CONFS[a],
                    "home_division": "fbs", "away_division": "fcs" if "Samford" in (h, a) else "fbs",
                    "conference_game": CONFS[h] == CONFS[a] and CONFS[h] is not None, "neutral_site": False,
                })
    return pd.DataFrame(rows)


def _priors_for(games, season):
    df = feature_build._assemble_base(games)
    out = add_priors(df)
    s = out[out["season"] == season]
    return {**dict(zip(s["home_team"], s["home_prior"])), **dict(zip(s["away_team"], s["away_prior"]))}


def test_first_season_is_neutral_and_none_conference_does_not_crash():
    p = _priors_for(_games(), 2018)
    assert set(p.values()) == {1500.0}


def test_season_prior_ignores_that_seasons_own_results():
    g = _games()
    base = _priors_for(g, 2020)
    g2 = g.copy()
    m = g2["season"] == 2020
    g2.loc[m, ["home_score", "away_score"]] = g2.loc[m, ["away_score", "home_score"]].to_numpy()  # flip every 2020 result
    assert _priors_for(g2, 2020) == base
    g3 = g.copy()
    m = g3["season"] == 2019
    g3.loc[m, ["home_score", "away_score"]] = g3.loc[m, ["away_score", "home_score"]].to_numpy()
    assert _priors_for(g3, 2020) != base  # but last season's results do move it


def test_team_new_to_the_data_gets_its_conference_mean():
    p = _priors_for(_games(), 2020)
    sec_old = [p["Alabama"], p["Georgia"]]
    # Newcomer's prior is the mean of the SEC rating-before-regression of returning SEC teams (regress toward it)
    assert min(sec_old) - 400 < p["Newcomer"] < max(sec_old) + 400
    assert p["Newcomer"] != 1500.0


def test_block_trains_serves_and_matches_row_for_row():
    g = _games()
    frame, cols = feature_build.build_training_frame(g, blocks=("priors",))
    assert not frame[["home_prior", "away_prior", "prior_diff"]].isna().any().any()
    played = g[g["season"] < 2020].copy()
    row = feature_build.build_features_for_game("Alabama", "Georgia", played, gameday="2020-09-02", conference_game=True,
                                                blocks=("priors",))
    assert list(row.index) == cols
    t2020 = frame[(frame["season"] == 2020) & (frame["home_team"] == "Alabama")]
    assert row["home_prior"] == pytest.approx(t2020.iloc[0]["home_prior"])
    # mid-season serving (2020 games already played) keeps the same season-2020 prior
    mid = g[(g["season"] < 2020) | (g["week"] <= 2)].copy()
    row2 = feature_build.build_features_for_game("Alabama", "Georgia", mid, gameday="2020-10-20", conference_game=True,
                                                 blocks=("priors",))
    assert row2["home_prior"] == pytest.approx(t2020.iloc[0]["home_prior"])


def _train_prior(frame, team, season):
    r = frame[(frame["season"] == season) & ((frame["home_team"] == team) | (frame["away_team"] == team))].iloc[0]
    return r["home_prior"] if r["home_team"] == team else r["away_prior"]


def test_serving_a_cross_conference_season_opener_matches_training():
    """The upcoming row has no conference; its teams must still regress toward their OWN conference means."""
    g = _games()
    frame, _ = feature_build.build_training_frame(g, blocks=("priors",))
    played = g[g["season"] < 2020].copy()
    row = feature_build.build_features_for_game("Alabama", "Ohio State", played, gameday="2020-09-02", conference_game=False,
                                                blocks=("priors",))
    assert row["home_prior"] == pytest.approx(_train_prior(frame, "Alabama", 2020))
    assert row["away_prior"] == pytest.approx(_train_prior(frame, "Ohio State", 2020))


def test_served_prior_does_not_depend_on_who_has_played_so_far():
    """Conference means come from the complete prior season, not from the teams already seen in the new one."""
    g = _games()
    frame, _ = feature_build.build_training_frame(g, blocks=("priors",))
    for upto_week in (0, 1, 2):
        played = g[(g["season"] < 2020) | ((g["season"] == 2020) & (g["week"] <= upto_week))].copy()
        row = feature_build.build_features_for_game("Michigan", "Alabama", played, gameday="2020-11-20", conference_game=False,
                                                    blocks=("priors",))
        assert row["home_prior"] == pytest.approx(_train_prior(frame, "Michigan", 2020))
        assert row["away_prior"] == pytest.approx(_train_prior(frame, "Alabama", 2020))
