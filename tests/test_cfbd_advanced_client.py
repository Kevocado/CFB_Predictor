"""The CFBD client call must work with the real cfbd>=5 API (`ApiClient` has no `.get`), and must hand back
the same shape the committed advanced_<year>.json files hold. Offline: the HTTP method is stubbed, the models are real."""
import json
from pathlib import Path

import pytest

cfbd = pytest.importorskip("cfbd")
from cfbd.models import AdvancedGameStat

from cfb_predictor.data import cfbd_advanced

COMMITTED = Path(cfbd_advanced.__file__).resolve().parents[3] / "data" / "cfbd" / "advanced_2025.json"


def _strip(x):
    return {k: _strip(v) for k, v in x.items() if v is not None} if isinstance(x, dict) else x


def test_fetch_uses_the_typed_api_and_round_trips_real_rows(monkeypatch):
    rows = json.loads(COMMITTED.read_text())[:25]
    seen = {}

    def fake(self, year=None, **kw):
        seen["year"] = year
        return [AdvancedGameStat.from_dict(r) for r in rows]

    monkeypatch.setattr(cfbd.StatsApi, "get_advanced_game_stats", fake)
    client = cfbd.ApiClient(cfbd.Configuration(access_token="test-not-a-real-key"))
    out = cfbd_advanced.fetch_season_advanced(client, 2026)

    assert seen["year"] == 2026
    assert [_strip(r) for r in out] == [_strip(r) for r in rows]
    assert not hasattr(client, "get")  # the shape of the bug: callers must not rely on a generic .get
