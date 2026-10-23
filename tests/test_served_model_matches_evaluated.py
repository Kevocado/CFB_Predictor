"""Test that the served game outcome model matches the walk-forward evaluated model.

For the ridge candidate, walk-forward evaluates a Pipeline(StandardScaler, Ridge),
so the served artifact must also be a Pipeline, not a bare Ridge.
"""
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def test_ridge_artifact_is_pipeline_with_scaler():
    """The committed ridge artifact must be a Pipeline with StandardScaler, not a bare Ridge."""
    model_path = Path("models/game_outcome_model.pkl")
    assert model_path.exists(), "game_outcome_model.pkl not found"
    
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    
    # The artifact should be a Pipeline with StandardScaler + Ridge
    assert isinstance(artifact, Pipeline), f"Expected Pipeline, got {type(artifact)}"
    assert "standardscaler" in artifact.named_steps, "Pipeline missing StandardScaler step"
    assert "ridge" in artifact.named_steps, "Pipeline missing Ridge step"
    assert isinstance(artifact.named_steps["standardscaler"], StandardScaler)
    assert isinstance(artifact.named_steps["ridge"], Ridge)


def test_manifest_chosen_candidate_matches_artifact():
    """The manifest's chosen_candidate must match the artifact type."""
    import json
    manifest_path = Path("models/manifest.json")
    assert manifest_path.exists(), "manifest.json not found"
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    chosen = manifest["chosen_candidate"]
    model_path = Path("models/game_outcome_model.pkl")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    
    if chosen == "ridge":
        assert isinstance(artifact, Pipeline), f"chosen_candidate=ridge but artifact is {type(artifact)}"
        assert "standardscaler" in artifact.named_steps
        assert "ridge" in artifact.named_steps
    elif chosen == "xgb":
        from xgboost import XGBRegressor
        assert isinstance(artifact, XGBRegressor), f"chosen_candidate=xgb but artifact is {type(artifact)}"
    elif chosen == "elo":
        assert isinstance(artifact, dict), f"chosen_candidate=elo but artifact is {type(artifact)}"
        assert "points_per_rating_point" in artifact
    else:
        raise ValueError(f"Unknown chosen_candidate: {chosen}")


if __name__ == "__main__":
    test_ridge_artifact_is_pipeline_with_scaler()
    test_manifest_chosen_candidate_matches_artifact()
    print("All tests passed!")
