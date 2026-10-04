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


def test_record_game_predictions_skips_snapshots_after_kickoff():
    """A batch of only past-kickoff games is skipped entirely (returns 0)
    rather than raising -- record_game_predictions no longer raises for the
    whole batch over one bad game (see this plan's final-review fix,
    Task 23, finding I4; this replaces the old raise-based assertion)."""
    game = _game() | {"commence_time": "2000-08-30T16:00:00"}

    n = store.record_game_predictions([game])

    assert n == 0


def test_record_game_predictions_skips_past_kickoff_game_but_keeps_valid_one():
    """One malformed/past-kickoff game in a batch used to raise for the
    whole call, dropping every other valid game in the same tick along
    with it -- it should instead be skipped, recording only the valid
    game (see this plan's final-review fix, Task 23, finding I4)."""
    past_kickoff_game = _game() | {"game_id": "past123", "commence_time": "2000-08-30T16:00:00"}
    valid_game = _game()

    n = store.record_game_predictions([past_kickoff_game, valid_game])

    assert n == 1
    record = store.get_track_record()
    assert record is not None  # sanity: db is usable after the mixed batch


def test_reconcile_game_predictions_fills_actual_outcome():
    store.record_game_predictions([_game()])
    results = pd.DataFrame([{"game_id": "401520145", "home_score": 7, "away_score": 14}])

    n = store.reconcile_game_predictions(results)

    assert n == 1
    record = store.get_track_record()["games"]
    assert record["n_resolved"] == 1
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


def test_reconcile_grades_a_qb_with_passing_tds_only_as_a_miss():
    """End-to-end through the grader, not just the helper.

    A passer who threw TDs and never rushed or caught one is graded 0.0. Under
    the old inline sum in `reconcile_player_prop_predictions` he was graded 1.0
    -- a stored prediction scored against a truth the model was never fitted
    on, which is the bug NFL's PR #26 found.
    """
    store.record_player_prop_predictions([
        {"game_id": "g_qb", "player_id": "p1", "player_name": "Passer",
         "market": "anytime_td", "predicted_value": 0.62},
    ])
    player_stats_df = pd.DataFrame([
        {"game_id": "g_qb", "player_id": "p1", "rushing_tds": 0, "receiving_tds": 0, "passing_tds": 3},
    ])

    assert store.reconcile_player_prop_predictions(player_stats_df) == 1

    record = store.get_track_record()["player_props"]["anytime_td"]
    assert record["n_resolved"] == 1
    assert record["hit_rate_when_called"] == pytest.approx(0.0)


def test_reconcile_grades_a_qb_with_a_rushing_td_as_a_hit():
    """The other half of the QB case, so the test above cannot pass by the
    grader rejecting every QB row."""
    store.record_player_prop_predictions([
        {"game_id": "g_qb", "player_id": "p1", "player_name": "Passer",
         "market": "anytime_td", "predicted_value": 0.62},
    ])
    player_stats_df = pd.DataFrame([
        {"game_id": "g_qb", "player_id": "p1", "rushing_tds": 1, "receiving_tds": 0, "passing_tds": 3},
    ])

    store.reconcile_player_prop_predictions(player_stats_df)

    record = store.get_track_record()["player_props"]["anytime_td"]
    assert record["hit_rate_when_called"] == pytest.approx(1.0)


def test_anytime_td_confidence_buckets_have_agreed_keys():
    """Contract test: confidence bucket objects must carry exactly
    {label, n, hit_rate}. Renaming any key breaks the frontend
    guard b.n > 0 and the section silently never renders."""
    store.record_player_prop_predictions([
        {"game_id": "2025_01_BAL_KC", "player_id": "p1", "player_name": "Runner",
         "market": "anytime_td", "predicted_value": 0.55},
        {"game_id": "2025_01_BAL_KC", "player_id": "p2", "player_name": "Passer",
         "market": "anytime_td", "predicted_value": 0.65},
        {"game_id": "2025_01_BAL_KC", "player_id": "p3", "player_name": "Rusher",
         "market": "anytime_td", "predicted_value": 0.75},
    ])
    assert store.reconcile_player_prop_predictions(pd.DataFrame([
        {"game_id": "2025_01_BAL_KC", "player_id": "p1", "rushing_tds": 1, "receiving_tds": 0, "passing_tds": 0},
        {"game_id": "2025_01_BAL_KC", "player_id": "p2", "rushing_tds": 0, "receiving_tds": 0, "passing_tds": 0},
        {"game_id": "2025_01_BAL_KC", "player_id": "p3", "rushing_tds": 0, "receiving_tds": 1, "passing_tds": 0},
    ])) == 3
    buckets = store.get_track_record()["player_props"]["anytime_td"]["confidence_buckets"]
    agreed_keys = {"label", "n", "hit_rate"}
    for bucket in buckets:
        assert set(bucket.keys()) == agreed_keys, (
            f"bucket {bucket} has keys {set(bucket.keys())}, "
            f"expected exactly {agreed_keys}"
        )
    assert len(buckets) == 3
    assert buckets[0]["n"] >= 0
    assert buckets[0]["hit_rate"] is None or isinstance(buckets[0]["hit_rate"], float)
