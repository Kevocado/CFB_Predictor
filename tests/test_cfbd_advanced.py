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