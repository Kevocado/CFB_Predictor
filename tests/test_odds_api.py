import pandas as pd
import pytest

from cfb_predictor.data import odds_api


def _raw_odds_response():
    return [
        {
            "id": "abc123",
            "commence_time": "2025-08-30T16:00:00Z",
            "home_team": "Texas Longhorns",
            "away_team": "Ohio State Buckeyes",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Texas Longhorns", "price": 2.10},
                                {"name": "Ohio State Buckeyes", "price": 1.80},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.91, "point": 51.5},
                                {"name": "Under", "price": 1.91, "point": 51.5},
                            ],
                        },
                    ],
                }
            ],
        }
    ]


def test_fetch_game_odds_flattens_bookmaker_markets(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "fake-key")
    monkeypatch.setattr(odds_api, "_fetch_raw_odds", lambda: _raw_odds_response())

    df = odds_api.fetch_game_odds()

    assert len(df) == 4  # 2 h2h outcomes + 2 totals outcomes
    assert set(df["market"]) == {"h2h", "totals"}
    assert df.iloc[0]["event_id"] == "abc123"


def test_fetch_game_odds_returns_empty_frame_without_api_key(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", None)

    df = odds_api.fetch_game_odds()

    assert df.empty


def test_fetch_game_odds_returns_empty_frame_on_request_error(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "fake-key")

    def _raise():
        raise RuntimeError("network error")

    monkeypatch.setattr(odds_api, "_fetch_raw_odds", _raise)

    df = odds_api.fetch_game_odds()

    assert df.empty
