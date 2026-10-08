import json
import tempfile
from pathlib import Path
import pytest
from cfb_predictor.data import cfbd_advanced
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


def test_a_failed_write_publishes_nothing_and_leaves_no_temp_file(tmp_path, monkeypatch):
    """The failure is injected during the WRITE, after partial output exists.

    Failing the fetch instead would pass even with a non-atomic write_text: nothing was
    written yet, so there was no partial file to leave behind.
    """
    real_named_temp_file = tempfile.NamedTemporaryFile  # bound before the patch, else infinite recursion

    class SlowNamedTemporaryFile:
        """The real temp file, but writes only half the payload before failing."""

        def __init__(self, mode, dir, prefix, suffix, delete):
            self._real = real_named_temp_file(
                mode=mode, dir=dir, prefix=prefix, suffix=suffix, delete=delete
            )

        def __enter__(self):
            return self  # `as tmp` must bind this wrapper, or tmp.write is the real write

        @property
        def name(self):
            return self._real.name

        def __exit__(self, *exc):
            self._real.__exit__(*exc)

        def write(self, data):
            self._real.write(data[: len(data) // 2])
            self._real.flush()
            raise OSError("simulated interrupted write")

    monkeypatch.setattr(cfbd_advanced.tempfile, "NamedTemporaryFile", SlowNamedTemporaryFile)
    client = FakeClient()
    with pytest.raises(OSError):
        pull_all(client, years=[2023], out_dir=tmp_path, budget=10)

    # Undo the monkeypatch so the second call uses the real temp file
    monkeypatch.undo()

    assert not (tmp_path / "advanced_2023.json").exists()  # nothing published
    assert not list(tmp_path.glob("advanced_2023.*.tmp"))  # no temp file left behind

    monkeypatch.undo()  # the retry must write through the real NamedTemporaryFile
    pull_all(FakeClient(), years=[2023], out_dir=tmp_path, budget=10)
    assert json.loads((tmp_path / "advanced_2023.json").read_text()) == FIX