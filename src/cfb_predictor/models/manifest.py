"""Train, save, and load the game-outcome and player-prop models."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import pandas as pd

from ..config import MODELS_DIR
from ..data import games as games_data
from ..data import player_stats
from ..evaluate import walk_forward
from ..features import build as feature_build
from ..features import player_usage
from . import game_outcome, player_props

MANIFEST_PATH = MODELS_DIR / "manifest.json"
GAME_MODEL_FILENAME = "game_outcome_model.pkl"
TOTAL_MODEL_FILENAME = "total_points_model.pkl"
ANYTIME_TD_MODEL_FILENAME = "anytime_td_model.pkl"

DEFAULT_TRAIN_SEASONS = 8


class StaleArtifactError(ValueError):
    """The committed models no longer match the code that would serve them.

    Its own type, rather than a bare `ValueError`, because callers need to tell
    this apart from every other `ValueError` on the path: `load_manifest`
    parses JSON, and a truncated or corrupt `manifest.json` raises
    `json.JSONDecodeError`, which IS a `ValueError`. Catching `ValueError` at
    the API edge would then report a corrupt file as "stale trained model" and
    send an operator to retrain when the actual fix is to restore the file.
    Subclasses `ValueError` so existing `pytest.raises(ValueError)` assertions
    keep holding.
    """


def _save_pickle(obj, path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _artifact_path(filename: str):
    return MODELS_DIR / filename


def _yardage_model_path(market: str):
    return _artifact_path(f"{market}_model.pkl")


def _fbs_teams_by_season(seasons: list[int]) -> dict[int, set[str]]:
    """One data.games.fetch_fbs_teams() call per season (each cached after
    first fetch) -- the authoritative FBS team universe build_training_frame
    and evaluate.walk_forward.prepare_folds use to exclude FCS-opponent
    games from every candidate's training rows."""
    return {season: set(games_data.fetch_fbs_teams(season)["team"]) for season in seasons}


def train_all(seasons: list[int] | None = None) -> dict:
    """Fit all models, persist their artifacts, and return their manifest."""
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    seasons = seasons or games_data.default_completed_seasons(n=DEFAULT_TRAIN_SEASONS)

    games_df = games_data.load_training_data(seasons)
    fbs_teams = _fbs_teams_by_season(seasons)
    train_df, feature_cols = feature_build.build_training_frame(games_df, fbs_teams=fbs_teams)

    folds = walk_forward.prepare_folds(games_df, fbs_teams=fbs_teams, min_train_seasons=max(1, len(seasons) - 2))
    if not folds:
        raise ValueError("Training requires at least one walk-forward validation fold.")
    candidate_scores = {}
    for candidate in ("elo", "ridge", "xgb"):
        scored = walk_forward.evaluate_candidate(folds, candidate) if folds else pd.DataFrame()
        candidate_scores[candidate] = float(scored["log_loss"].mean()) if not scored.empty else float("inf")
    chosen = min(candidate_scores, key=candidate_scores.get)

    X_train = train_df[feature_cols]
    y_margin = train_df["margin"]
    y_total = train_df["total_points"]

    if chosen == "elo":
        game_model = game_outcome.fit_elo_candidate(train_df)
    elif chosen == "ridge":
        game_model = game_outcome.fit_margin_regression(X_train, y_margin)
    else:
        game_model = game_outcome.fit_xgb_margin(X_train, y_margin)

    if chosen == "elo":
        margin_preds = train_df.apply(
            lambda r: game_outcome.predict_margin_elo(
                game_model, r["rating_diff"], r["home_rest_days"], r["away_rest_days"]
            ),
            axis=1,
        )
        sigma_model = type("_", (), {"predict": lambda self, X: margin_preds.to_numpy()})()
        sigma = game_outcome.residual_sigma(sigma_model, X_train, y_margin)
    else:
        sigma = game_outcome.residual_sigma(game_model, X_train, y_margin)

    total_model = game_outcome.fit_xgb_margin(X_train, y_total)
    total_sigma = game_outcome.residual_sigma(total_model, X_train, y_total)

    _save_pickle(game_model, _artifact_path(GAME_MODEL_FILENAME))
    _save_pickle(total_model, _artifact_path(TOTAL_MODEL_FILENAME))

    player_df_raw = player_stats.fetch_weekly_player_stats(seasons, games_df)
    player_train_df, player_feature_cols = player_usage.build_player_training_frame(player_df_raw)
    player_train_df = player_train_df.dropna(subset=player_feature_cols, how="all")
    X_player = player_train_df[player_feature_cols].fillna(0)

    anytime_td_model = player_props.fit_anytime_td_classifier(X_player, player_train_df["anytime_td"])
    _save_pickle(anytime_td_model, _artifact_path(ANYTIME_TD_MODEL_FILENAME))

    yardage_metrics = {}
    for market, target_col in player_props.YARDAGE_TARGETS.items():
        path = _yardage_model_path(market)
        subset = player_train_df[player_train_df[target_col] > 0]
        if subset.empty:
            path.unlink(missing_ok=True)
            continue
        model = player_props.fit_yardage_regressor(subset[player_feature_cols].fillna(0), subset[target_col])
        _save_pickle(model, path)
        yardage_metrics[market] = {"n_train": int(len(subset))}

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons": sorted(int(s) for s in seasons),
        "n_train": int(len(train_df)),
        "feature_cols": feature_cols,
        "player_feature_cols": player_feature_cols,
        "chosen_candidate": chosen,
        "candidate_scores": candidate_scores,
        "sigma": sigma,
        "total_sigma": total_sigma,
        "yardage_metrics": yardage_metrics,
        "artifact_fingerprint": artifact_fingerprint(player_feature_cols),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def artifact_fingerprint(player_feature_cols) -> dict:
    """What these committed pickles were fitted AGAINST, in a form the code can
    check itself against at load time.

    Two keys, and both are about meaning rather than shape:

    * `player_feature_cols` -- the exact list `train_all` fitted the player
      models on. Checked as an exact list, not a subset: a subset test cannot
      see an artefact that is missing a feature.
    * `anytime_td_label_version` from `player_usage`. Bump the version whenever
      the `anytime_td` definition changes and every artefact fitted before that
      bump stops loading loudly instead of quietly.

    The silent-degradation case this exists to close is measured, not
    hypothetical: on the artefact this branch started from,
    `models/manifest.json` listed 5 `player_feature_cols` while the code had 6
    (the 6th, `receptions_roll`, had been added to `ROLL_STATS` without a
    retrain), and `anytime_td_model.pkl` was fitted on 5. `load_models()`
    returned that payload without complaint and every anytime-TD prediction came
    out of a model that no longer matched the code.

    A subset check would have missed that, because the stale 5 columns ARE a
    subset of the 6 servable ones. The label half is the second reason: the
    label is never recomputed at serving -- `predict_props` scores the committed
    pickle from features alone -- so an artefact fitted against the OLD
    `anytime_td` definition serves exactly like one fitted against the new one.
    Right shape, wrong meaning, no error anywhere.
    """
    return {
        "player_feature_cols": list(player_feature_cols),
        "anytime_td_label_version": player_usage.ANYTIME_TD_LABEL_VERSION,
    }


def _verify_artifact_fingerprint(manifest: dict) -> None:
    """Raise when the committed artefacts disagree with the code they serve under.

    Called from `load_models`. A manifest written before this check existed has
    no `artifact_fingerprint` key at all, and that is treated as a FAILURE, not
    a pass -- the absence of a fingerprint is exactly the state this is
    detecting, so skipping the check when it is missing would defeat it.

    **Both halves are compared against the CODE, never against the manifest.**
    A check whose expectation comes from the artefact it is inspecting cannot
    detect that artefact drifting: NFL's first version of this built its
    expectation from `manifest["player_feature_cols"]`, so the manifest was
    compared with itself and passed for exactly the stale artefact it exists to
    catch. `player_usage.PLAYER_FEATURE_COLUMNS` is the authority instead,
    because it is the one thing training and serving both derive from:
    `train_all` fits the player models on the list `build_player_training_frame`
    returns (that constant), and `predict_props` reindexes a live row by
    `manifest["player_feature_cols"]`, which the check below pins to it.
    """
    recorded = manifest.get("artifact_fingerprint")
    if recorded is None:
        raise StaleArtifactError(
            f"{MANIFEST_PATH} has no 'artifact_fingerprint' key, so the committed "
            "models cannot be verified against the code that serves them. It was "
            "written before fingerprinting existed, and an unverified artefact is "
            "the case this check exists to catch. Re-run training "
            "(`python -m cfb_predictor.models.manifest`) to write one."
        )

    # The code's own fingerprint. Nothing here is read out of the manifest.
    current = artifact_fingerprint(player_usage.PLAYER_FEATURE_COLUMNS)
    expected_features = current["player_feature_cols"]
    fitted_features = list(recorded.get("player_feature_cols") or [])
    if fitted_features != expected_features:
        raise StaleArtifactError(
            f"{MANIFEST_PATH} records models fitted on {fitted_features} but the "
            f"code's player features are now {expected_features} "
            f"({player_usage.__name__}.PLAYER_FEATURE_COLUMNS). The committed "
            "pickles were fitted on a different feature set, so every prediction "
            "would come from a stale model. Re-run training "
            "(`python -m cfb_predictor.models.manifest`)."
        )

    # The fingerprint and the manifest's own `player_feature_cols` are written
    # from one value at fit time, so they must still agree -- and this is not
    # belt-and-braces. `load_models` hands `player_models["feature_cols"]` to
    # `player_props.predict_props`, which reindexes every live row by the
    # manifest's list, not the fingerprint's. A manifest whose top-level list had
    # drifted would score a 6-column model on a narrower feature set while the
    # check above stayed green: the same silent degradation from the other side.
    manifest_features = list(manifest.get("player_feature_cols") or [])
    if manifest_features != fitted_features:
        raise StaleArtifactError(
            f"{MANIFEST_PATH} records an artefact fingerprint fitted on "
            f"{fitted_features} but its own 'player_feature_cols' is "
            f"{manifest_features}. Serving reindexes every player row by the "
            "manifest's list, so the two disagreeing means the models would be "
            "scored on features they were not fitted on. Re-run training "
            "(`python -m cfb_predictor.models.manifest`)."
        )

    expected_version = current["anytime_td_label_version"]
    fitted_version = recorded.get("anytime_td_label_version")
    if fitted_version != expected_version:
        raise StaleArtifactError(
            f"{MANIFEST_PATH} records the anytime-TD model fitted against label "
            f"definition v{fitted_version}, but the code now defines "
            f"v{expected_version} ({player_usage.__name__}.ANYTIME_TD_LABEL_VERSION). "
            "The committed model predicts a different market than the one being "
            "graded: `anytime_td` is now rushing + receiving TDs only, excluding "
            "passing TDs. Serving it would score every prediction against a "
            "definition it was never fitted on. Re-run training "
            "(`python -m cfb_predictor.models.manifest`)."
        )


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("No trained models found. Run `python -m cfb_predictor.models.manifest` first.")
    return json.loads(MANIFEST_PATH.read_text())


def model_version(manifest: dict) -> str:
    """Identifies the trained model behind a prediction: candidate + training time."""
    return f"{manifest['chosen_candidate']}@{manifest['trained_at']}"


def load_models() -> dict:
    manifest = load_manifest()
    # Before the unpickles: an artefact that cannot be verified must not reach
    # the point of being loaded, let alone scored against.
    _verify_artifact_fingerprint(manifest)
    player_models = {
        "feature_cols": manifest["player_feature_cols"],
        "anytime_td": _load_pickle(_artifact_path(ANYTIME_TD_MODEL_FILENAME)),
    }
    for market in manifest["yardage_metrics"]:
        player_models[market] = _load_pickle(_yardage_model_path(market))

    return {
        "game_outcome_model": _load_pickle(_artifact_path(GAME_MODEL_FILENAME)),
        "total_model": _load_pickle(_artifact_path(TOTAL_MODEL_FILENAME)),
        "chosen_candidate": manifest["chosen_candidate"],
        "model_version": model_version(manifest),
        "sigma": manifest["sigma"],
        "total_sigma": manifest["total_sigma"],
        "player_models": player_models,
        "feature_cols": manifest["feature_cols"],
        "player_feature_cols": manifest["player_feature_cols"],
    }


if __name__ == "__main__":
    train_all()
