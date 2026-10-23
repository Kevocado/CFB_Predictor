"""The TOTAL model has its own spec (ridge + adj_epa); margin and win must not move, and serving fails closed."""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from cfb_predictor.api import routes
from cfb_predictor.features import build
from cfb_predictor.models import manifest

CONF = {"T0": "A", "T1": "A", "T2": "A", "T3": "A", "T4": "B", "T5": "B", "T6": "B", "T7": "B"}
SEASONS = [2021, 2022, 2023, 2024]


def _games(seasons=SEASONS, weeks=8, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            order = list(rng.permutation(list(CONF)))
            for i in range(0, 8, 2):
                h, a = order[i], order[i + 1]
                rows.append({
                    "game_id": f"{season}_{week:02d}_{h}_{a}", "season": season, "week": week,
                    "gameday": pd.Timestamp(f"{season}-08-30") + pd.Timedelta(days=7 * (week - 1)),
                    "home_team": h, "away_team": a,
                    "home_score": int(rng.integers(10, 50)), "away_score": int(rng.integers(10, 50)),
                    "home_conference": CONF[h], "away_conference": CONF[a], "conference_game": CONF[h] == CONF[a],
                    "home_division": "fbs", "away_division": "fbs",
                })
    return pd.DataFrame(rows)


def _efficiency(games, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _, g in games.iterrows():
        for team in (g["home_team"], g["away_team"]):
            rows.append({"game_id": g["game_id"], "team": team, "season": g["season"], "week": g["week"],
                         "epa_off": float(rng.normal(0.05, 0.1)), "epa_def": float(rng.normal(-0.05, 0.1)),
                         "epa_off_pass": 0.0, "epa_off_rush": 0.0, "epa_def_pass": 0.0, "epa_def_rush": 0.0,
                         "success_off": 0.4, "success_def": 0.4})
    return pd.DataFrame(rows)


def _player_stats(seasons, games_df=None):
    rows = [{"player_id": "p1", "player_name": "Runner", "position": "RB", "recent_team": "T0", "season": s, "week": w,
             "passing_yards": 0, "passing_tds": 0, "rushing_yards": 60 + 7 * w, "rushing_tds": w % 2,
             "receiving_yards": 5 * w, "receiving_tds": 0, "receptions": 2, "targets": float("nan"), "carries": 18}
            for s in seasons for w in range(1, 6)]
    return pd.DataFrame(rows)


@pytest.fixture
def patched(monkeypatch):
    games = _games()
    monkeypatch.setattr(manifest.games_data, "load_training_data", lambda s: games)
    monkeypatch.setattr(manifest.games_data, "fetch_fbs_teams", lambda s: pd.DataFrame({"team": list(CONF), "conference": "X"}))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", _player_stats)
    monkeypatch.setattr(manifest.cfbd_advanced, "efficiency_frame_from_disk", lambda out_dir=None, years=None: _efficiency(games))
    return games


def _train_into(monkeypatch, tmp_path, **kw):
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    return manifest.train_all(seasons=SEASONS, **kw)


def test_train_all_records_the_total_spec_and_sigma_from_that_model(patched, monkeypatch, tmp_path):
    written = _train_into(monkeypatch, tmp_path)
    spec = written["total_spec"]
    assert spec["kind"] == "ridge" and spec["blocks"] == ["adj_epa"]
    assert spec["feature_cols"] == build.feature_columns(("adj_epa",))
    assert written["feature_cols"] == build.feature_columns(())  # margin/win columns unchanged
    assert written["feature_blocks"] == []
    (tmp_path / "legacy").mkdir()
    legacy = _train_into(monkeypatch, tmp_path / "legacy", total_spec=None)
    assert "total_spec" not in legacy
    assert written["total_sigma"] != legacy["total_sigma"]  # its own OOF residuals, not the xgb ones
    total = pickle.loads((tmp_path / "total_points_model.pkl").read_bytes())
    assert total.n_features_in_ == len(spec["feature_cols"])


def test_margin_and_win_artifacts_are_identical_with_or_without_a_total_spec(patched, monkeypatch, tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    a = _train_into(monkeypatch, tmp_path / "a")
    b = _train_into(monkeypatch, tmp_path / "b", total_spec=None)
    assert (tmp_path / "a" / "game_outcome_model.pkl").read_bytes() == (tmp_path / "b" / "game_outcome_model.pkl").read_bytes()
    for key in ("sigma", "chosen_candidate", "candidate_scores", "feature_cols", "n_train"):
        assert a[key] == b[key]
    assert (tmp_path / "a" / "total_points_model.pkl").read_bytes() != (tmp_path / "b" / "total_points_model.pkl").read_bytes()


def test_training_the_total_refuses_missing_efficiency(patched, monkeypatch, tmp_path):
    monkeypatch.setattr(manifest.cfbd_advanced, "efficiency_frame_from_disk", lambda out_dir=None, years=None: pd.DataFrame())
    with pytest.raises(ValueError, match="advanced stats"):
        _train_into(monkeypatch, tmp_path)


def test_a_manifest_whose_total_columns_differ_from_the_code_refuses_to_load(patched, monkeypatch, tmp_path):
    _train_into(monkeypatch, tmp_path)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    saved["total_spec"]["feature_cols"] = saved["total_spec"]["feature_cols"][:-1]
    (tmp_path / "manifest.json").write_text(json.dumps(saved))
    with pytest.raises(manifest.StaleArtifactError, match="total model"):
        manifest.load_models()


def test_a_manifest_with_no_total_spec_loads_with_none(patched, monkeypatch, tmp_path):
    _train_into(monkeypatch, tmp_path, total_spec=None)
    assert manifest.load_models()["total_spec"] is None


@pytest.fixture
def served(patched, monkeypatch, tmp_path):
    (tmp_path / "legacy").mkdir()
    _train_into(monkeypatch, tmp_path / "legacy", total_spec=None)
    legacy = manifest.load_models()
    _train_into(monkeypatch, tmp_path)
    models = {**manifest.load_models(), "legacy": legacy}
    monkeypatch.setattr(routes, "_efficiency_cache", {})
    return models, patched


def _call(models, games, **kw):
    return routes._predict_game_from_models(models, "T0", "T5", games, gameday="2025-09-06", **kw)


def test_serving_prices_the_total_with_efficiency_aux(served, monkeypatch):
    models, games = served
    monkeypatch.setattr(routes, "_efficiency_aux", lambda g: build.Aux(efficiency=_efficiency(g)))
    out = _call(models, games, total_line=45.0)
    assert np.isfinite(out["predicted_total"]) and out["total_sigma"] == models["total_sigma"]
    assert 0 < out["over_prob"] < 1


def test_serving_fails_closed_without_aux_and_margin_win_do_not_move(served, monkeypatch):
    models, games = served
    monkeypatch.setattr(routes, "_efficiency_aux", lambda g: build.Aux(efficiency=_efficiency(g)))
    with_aux = _call(models, games, spread_line=3.0, total_line=45.0)
    monkeypatch.setattr(routes, "_efficiency_aux", lambda g: None)
    without = _call(models, games, spread_line=3.0, total_line=45.0)
    assert without["predicted_total"] is None and without["total_sigma"] is None
    assert "over_prob" not in without
    legacy = _call(models["legacy"], games, spread_line=3.0, total_line=45.0)  # no total_spec: the old path
    for k in ("home_win_prob", "away_win_prob", "home_cover_prob", "predicted_margin", "sigma"):
        assert with_aux[k] == without[k] == legacy[k]


def test_efficiency_aux_is_none_when_nothing_is_on_disk(monkeypatch):
    monkeypatch.setattr(routes, "_efficiency_cache", {})
    monkeypatch.setattr(routes.cfbd_advanced, "efficiency_frame_from_disk", lambda out_dir=None, years=None: pd.DataFrame())
    assert routes._efficiency_aux(_games(seasons=[2021])) is None


def test_adj_epa_block_refuses_an_efficiency_frame_that_matches_no_game():
    games = _games(seasons=[2021])
    stranger = _efficiency(games).assign(game_id="nope")
    with pytest.raises(ValueError, match="no efficiency row"):
        build.build_training_frame(games, blocks=("adj_epa",), aux=build.Aux(efficiency=stranger))
    with pytest.raises(ValueError, match="needs aux"):
        build.build_training_frame(games, blocks=("adj_epa",))


def test_the_served_adj_epa_row_equals_the_trained_row():
    games = _games()
    aux = build.Aux(efficiency=_efficiency(games))
    trained, _ = build.build_training_frame(games, blocks=("adj_epa",), aux=aux)
    cols = build.feature_columns(("adj_epa",))
    for _, g in trained[trained["season"] == 2024].sample(10, random_state=1).iterrows():
        before = games[pd.to_datetime(games["gameday"]) < pd.Timestamp(g["gameday"])]
        eff = aux.efficiency[aux.efficiency["game_id"].isin(before["game_id"])]
        served = build.build_features_for_game(g["home_team"], g["away_team"], before, gameday=g["gameday"],
                                               conference_game=g["conference_game"], blocks=("adj_epa",), aux=build.Aux(efficiency=eff))
        np.testing.assert_allclose(served[cols].to_numpy(float), g[cols].to_numpy(float), equal_nan=True, err_msg=g["game_id"])
