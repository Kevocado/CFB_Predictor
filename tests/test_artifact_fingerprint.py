# tests/test_artifact_fingerprint.py
"""A stale artefact must REFUSE to load, not quietly predict the old market.

The label is never recomputed at serving -- `predict_props` scores the
committed pickle from features alone -- so a model fitted against the OLD
`anytime_td` definition serves with the right shape and the wrong meaning and
nothing records it. These tests cover the guard that makes that loud.

Measured on the artefact this branch started from: `models/manifest.json`
listed 5 `player_feature_cols` while the code had 6 (`receptions_roll` had been
added to `ROLL_STATS` without a retrain) and had no fingerprint at all.
"""

import json

import pytest

from cfb_predictor.features import player_usage
from cfb_predictor.models import manifest
from cfb_predictor.models import player_props
from cfb_predictor import public_snapshot


def _trained_manifest(monkeypatch, tmp_path, seasons=(2021, 2022, 2023, 2024)):
    """Train a real (tiny) artefact and return the manifest dict it wrote."""
    import numpy as np

    def _fake_games(seasons):
        rng = np.random.default_rng(3)
        rows = []
        teams = [f"T{i}" for i in range(8)]
        for season in seasons:
            for week in range(1, 6):
                for i in range(0, len(teams), 2):
                    home, away = teams[i], teams[i + 1]
                    rows.append({
                        "game_id": f"{season}_{week}_{home}_{away}", "season": season, "week": week,
                        "gameday": np.datetime64(f"{season}-08-25") + week * 7,
                        "home_team": home, "away_team": away,
                        "home_score": int(rng.integers(10, 45)), "away_score": int(rng.integers(10, 45)),
                        "home_division": "fbs", "away_division": "fbs", "conference_game": False,
                    })
        return __import__("pandas").DataFrame(rows)

    def _fake_player_stats(seasons, games_df=None):
        rng = np.random.default_rng(4)
        rows = []
        for season in seasons:
            for week in range(1, 6):
                for i in range(6):
                    rows.append({
                        "game_id": f"{season}_{week}_T0_T1", "player_id": f"p{i}",
                        "player_name": f"P{i}", "position": "RB", "season": season, "week": week,
                        "passing_yards": 0, "passing_tds": 0,
                        "rushing_yards": int(rng.integers(0, 200)), "rushing_tds": int(rng.integers(0, 3)),
                        "receiving_yards": int(rng.integers(0, 100)), "receiving_tds": int(rng.integers(0, 2)),
                        "receptions": int(rng.integers(0, 8)), "targets": int(rng.integers(0, 10)),
                        "carries": int(rng.integers(0, 25)),
                    })
        return __import__("pandas").DataFrame(rows)

    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest.games_data, "load_training_data", _fake_games)
    monkeypatch.setattr(manifest.games_data, "fetch_fbs_teams",
                        lambda s: __import__("pandas").DataFrame(
                            {"team": [f"T{i}" for i in range(8)], "conference": ["X"] * 8}))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", _fake_player_stats)
    return manifest.train_all(seasons=list(seasons))


def _rewrite(monkeypatch, tmp_path, mutate):
    """Train, then edit the written manifest, so the test fakes the recorded
    version rather than mocking the checker."""
    _trained_manifest(monkeypatch, tmp_path)
    path = tmp_path / "manifest.json"
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data, indent=2))
    return data


# --- the guard fires ---------------------------------------------------------

def test_an_artefact_fitted_against_the_old_label_version_refuses_to_load(monkeypatch, tmp_path):
    """THE red-checked case: an artefact trained when `anytime_td` still
    included passing TDs must not load under the v2 definition."""
    _rewrite(monkeypatch, tmp_path, lambda d: d["artifact_fingerprint"].update(
        {"anytime_td_label_version": 1}))

    with pytest.raises(ValueError, match="label definition v1"):
        manifest.load_models()


def test_a_missing_fingerprint_refuses_to_load(monkeypatch, tmp_path):
    """A manifest written before fingerprinting existed is the exact state
    this check exists to detect, so its absence must FAIL rather than pass."""
    _rewrite(monkeypatch, tmp_path, lambda d: d.pop("artifact_fingerprint"))

    with pytest.raises(ValueError, match="no 'artifact_fingerprint' key"):
        manifest.load_models()


def test_bumping_the_label_version_invalidates_every_committed_artefact(monkeypatch, tmp_path):
    """Red-checked by faking the version forward: the guard reads the CODE's
    version, so a code bump must reject an artefact recorded against the old
    one -- without any manifest edit at all."""
    _trained_manifest(monkeypatch, tmp_path)
    monkeypatch.setattr(manifest.player_usage, "ANYTIME_TD_LABEL_VERSION", 3)

    with pytest.raises(ValueError, match="label definition v2"):
        manifest.load_models()


def test_an_artefact_fitted_on_a_different_feature_list_refuses_to_load(monkeypatch, tmp_path):
    """The measured drift: `receptions_roll` was added to `ROLL_STATS` without
    a retrain, so the committed manifest listed 5 features and the code had 6."""
    _rewrite(monkeypatch, tmp_path, lambda d: d["artifact_fingerprint"].update(
        {"player_feature_cols": list(d["player_feature_cols"])[:-1]}))

    with pytest.raises(ValueError, match="different feature set"):
        manifest.load_models()


def test_the_feature_check_compares_against_the_code_not_the_manifest(monkeypatch, tmp_path):
    """NFL's first version built its expectation from the manifest, so the
    manifest was compared with itself and passed for exactly the artefact it
    exists to catch. Faking BOTH sides to agree must still fail, because the
    expectation comes from the code."""
    stale = list(player_usage.PLAYER_FEATURE_COLUMNS)[:-1]
    _rewrite(monkeypatch, tmp_path, lambda d: (
        d["artifact_fingerprint"].update({"player_feature_cols": stale}),
        d.update({"player_feature_cols": stale}),
    ))

    with pytest.raises(ValueError, match="different feature set"):
        manifest.load_models()


def test_a_manifest_whose_own_feature_list_drifted_refuses_to_load(monkeypatch, tmp_path):
    """`predict_props` reindexes live rows by the manifest's list, so a drift
    between the fingerprint and the manifest would score the model on features
    it was not fitted on while the feature check stayed green."""
    _rewrite(monkeypatch, tmp_path, lambda d: d.update(
        {"player_feature_cols": list(d["player_feature_cols"])[:-1]}))

    with pytest.raises(ValueError, match="reindexes every player row"):
        manifest.load_models()


def test_the_error_names_both_sides(monkeypatch, tmp_path):
    """"Your model is stale" is not actionable; the whole cost of this bug
    class was that nothing said anything at all."""
    _rewrite(monkeypatch, tmp_path, lambda d: d["artifact_fingerprint"].update(
        {"anytime_td_label_version": 1}))

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    message = str(excinfo.value)
    assert "v1" in message                       # what the artefact says
    assert f"v{player_usage.ANYTIME_TD_LABEL_VERSION}" in message  # what the code says
    assert "ANYTIME_TD_LABEL_VERSION" in message  # where to look
    assert "cfb_predictor.models.manifest" in message  # how to fix it


# --- the guard does not fire when things agree -------------------------------

def test_a_freshly_trained_artefact_loads(monkeypatch, tmp_path):
    """The guard must not be a gate that refuses everything."""
    _trained_manifest(monkeypatch, tmp_path)
    assert manifest.load_models()["player_models"]["anytime_td"] is not None


def test_train_all_records_the_codes_label_version_and_feature_list(monkeypatch, tmp_path):
    """The fit-time half: the fingerprint is written from the code's own
    values, so it is a fact about the fit rather than a constant."""
    written = _trained_manifest(monkeypatch, tmp_path)

    assert written["artifact_fingerprint"]["anytime_td_label_version"] == \
        player_usage.ANYTIME_TD_LABEL_VERSION
    assert written["artifact_fingerprint"]["player_feature_cols"] == \
        list(player_usage.PLAYER_FEATURE_COLUMNS)


# --- reachable from serving, not only from training --------------------------

def test_the_api_refuses_to_serve_a_stale_model_with_503(monkeypatch, tmp_path):
    """Serving must be able to hit this, not just `load_models`. Without this
    the guard would be a library-level check that a live request never meets.
    """
    from fastapi import HTTPException
    from cfb_predictor.api import routes

    _rewrite(monkeypatch, tmp_path, lambda d: d["artifact_fingerprint"].update(
        {"anytime_td_label_version": 1}))
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    routes._load_models_cached.cache_clear()

    with pytest.raises(HTTPException) as excinfo:
        routes._load_models_or_503()
    assert excinfo.value.status_code == 503
    # The reason survives: a generic 503 would throw away the actionable part.
    assert "label definition v1" in excinfo.value.detail
    routes._load_models_cached.cache_clear()


def test_publishing_the_snapshot_refuses_a_stale_model(monkeypatch, tmp_path):
    """The snapshot is committed and served, so a stale one publishes the old
    label's numbers exactly as NFL's did. The gate lives in the WRITING code so
    a hand-run or workflow_dispatch cannot bypass it."""
    _rewrite(monkeypatch, tmp_path, lambda d: d["artifact_fingerprint"].update(
        {"anytime_td_label_version": 1}))
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(public_snapshot.config, "PUBLIC_SNAPSHOT_PATH", tmp_path / "snap.json")
    # The build must never be reached: a stale model is refused before any
    # week is built or any file is written.
    monkeypatch.setattr(public_snapshot, "build_snapshot",
                        lambda *a, **k: pytest.fail("built a snapshot from a stale model"))

    with pytest.raises(ValueError, match="label definition v1"):
        public_snapshot.main()
    assert not (tmp_path / "snap.json").exists()


def test_the_snapshot_gate_is_not_only_in_workflow_yaml():
    """The gate must be in the module, not in CI config. Asserted structurally
    because `workflow_dispatch` and hand-runs bypass YAML."""
    import inspect

    source = inspect.getsource(public_snapshot.main)
    assert "_verify_models_current()" in source