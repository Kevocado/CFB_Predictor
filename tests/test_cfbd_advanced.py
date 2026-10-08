import json
from pathlib import Path
import pytest
from cfb_predictor.data.cfbd_advanced import BudgetExceeded, pull_all, to_team_game_frame

FIX = json.loads((Path(__file__).parent / "fixtures" / "cfbd_advanced_sample.json").read_text())


def test_frame_has_one_row_per_team_game_with_all_columns():
    df = to_team_game_frame(FIX)
    assert df.groupby(["game_id", "team"]).size().max() == 1
    assert {"epa_off", "epa_def", "epa_off_pass", "epa_off_rush", "success_off", "success_def"} <= set(df.columns)


def test_defence_columns_come_from_the_opponents_offence():
    df = to_team_game_frame(FIX).set_index(["game_id", "team"])
    (gid, a), (_, b) = list(df.index)[:2]
    assert df.loc[(gid, a), "epa_def"] == pytest.approx(df.loc[(gid, b), "epa_off"])


class FakeClient:
    def __init__(self):
        self.calls = 0

    def get(self, path, params):
        self.calls += 1
        return FIX


def test_budget_is_enforced_before_the_call(tmp_path):
    client = FakeClient()
    with pytest.raises(BudgetExceeded):
        pull_all(client, years=range(2014, 2026), out_dir=tmp_path, budget=3)
    assert client.calls <= 3


def test_pull_is_resumable(tmp_path):
    client = FakeClient()
    pull_all(client, years=[2023, 2024], out_dir=tmp_path, budget=10)
    first = client.calls
    pull_all(client, years=[2023, 2024], out_dir=tmp_path, budget=10)
    assert client.calls == first  # files exist, nothing refetched


def test_atomic_write_on_failure_leaves_no_partial_file(tmp_path):
    """A failed write (interrupted) leaves no advanced_{year}.json; next pull refetches."""
    call_count = 0

    class FailingClient:
        def __init__(self):
            self.should_fail = True

        def get(self, path, params):
            nonlocal call_count
            call_count += 1
            if self.should_fail:
                self.should_fail = False
                raise RuntimeError("simulated interruption")
            return FIX

    client = FailingClient()
    with pytest.raises(RuntimeError):
        pull_all(client, years=[2023], out_dir=tmp_path, budget=10)
    # No file should exist after failed write
    assert not (tmp_path / "advanced_2023.json").exists()
    # Next attempt should refetch and succeed
    pull_all(client, years=[2023], out_dir=tmp_path, budget=10)
    assert (tmp_path / "advanced_2023.json").exists()
    assert call_count == 2  # one failed, one succeeded