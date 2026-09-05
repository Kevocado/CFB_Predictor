import pandas as pd
import pytest

from cfb_predictor.tracking import store


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from cfb_predictor import config

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield


def _game():
    return {
        "game_id": "401520145", "home_team": "Texas", "away_team": "Ohio State",
        "commence_time": "2099-08-30T16:00:00", "home_win_prob": 0.42, "away_win_prob": 0.58,
        "home_cover_prob": 0.48, "away_cover_prob": 0.52, "over_prob": 0.55, "under_prob": 0.45,
    }


def test_record_game_predictions_is_idempotent():
    n1 = store.record_game_predictions([_game()])
    n2 = store.record_game_predictions([_game()])

    assert n1 == 1
    assert n2 == 0


def test_record_game_predictions_rejects_snapshots_after_kickoff():
    game = _game() | {"commence_time": "2000-08-30T16:00:00"}

    with pytest.raises(ValueError, match="before kickoff"):
        store.record_game_predictions([game])


def test_reconcile_game_predictions_fills_actual_outcome():
    store.record_game_predictions([_game()])
    results = pd.DataFrame([{"game_id": "401520145", "home_score": 7, "away_score": 14}])

    n = store.reconcile_game_predictions(results)

    assert n == 1
    record = store.get_track_record()
    assert record["n_resolved_games"] == 1
    assert record["pct_moneyline_correct"] == 1.0  # predicted away win, away won


def test_record_and_reconcile_player_prop_predictions():
    prop = {"game_id": "401520145", "player_id": "4568", "player_name": "R. Back",
            "market": "rushing_yards", "predicted_value": 95.0}
    store.record_player_prop_predictions([prop])

    player_stats_df = pd.DataFrame(
        [{"game_id": "401520145", "player_id": "4568", "rushing_yards": 112, "receiving_yards": 5,
          "passing_yards": 0, "rushing_tds": 1, "receiving_tds": 0, "passing_tds": 0}]
    )
    n = store.reconcile_player_prop_predictions(player_stats_df)

    assert n == 1
