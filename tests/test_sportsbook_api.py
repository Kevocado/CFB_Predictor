import pandas as pd

from cfb_predictor.data import sportsbook_api


def test_team_name_matches_is_case_insensitive_prefix():
    assert sportsbook_api._team_name_matches("Texas", "Texas Longhorns")
    assert sportsbook_api._team_name_matches("miami", "Miami (FL)")
    # "Texas" IS a valid prefix-match of "Texas Tech" under this scheme --
    # the same short-name-prefix ambiguity api/routes.py's own
    # _team_name_matches has (and its caller, _lines_for_game, separately
    # guards against by refusing an ambiguous multi-game match). Pin the
    # actual boundary instead: a genuinely different first word never matches.
    assert not sportsbook_api._team_name_matches("Oklahoma", "Texas Tech")


def _event(key, home_key, participants):
    return {"key": key, "homeParticipantKey": home_key, "participants": participants}


def test_match_events_to_games_finds_the_right_event_by_team_name():
    events = [
        _event("e1", "h1", [
            {"key": "h1", "name": "Ohio State"},
            {"key": "a1", "name": "Michigan"},
        ]),
        _event("e2", "h2", [
            {"key": "h2", "name": "Texas"},
            {"key": "a2", "name": "Oklahoma"},
        ]),
    ]
    games_df = pd.DataFrame([{"home_team": "Texas", "away_team": "Oklahoma"}])

    matched = sportsbook_api.match_events_to_games(events, games_df)

    assert [e["key"] for e in matched] == ["e2"]


def test_fetch_game_odds_returns_empty_without_a_key_or_games(monkeypatch):
    monkeypatch.setattr(sportsbook_api, "SPORTSBOOK_API_KEY", "")
    result = sportsbook_api.fetch_game_odds(pd.DataFrame([{"home_team": "Texas", "away_team": "Oklahoma"}]))
    assert result.empty

    monkeypatch.setattr(sportsbook_api, "SPORTSBOOK_API_KEY", "fake-key")
    assert sportsbook_api.fetch_game_odds(pd.DataFrame()).empty
    assert sportsbook_api.fetch_game_odds(None).empty


def test_event_to_rows_produces_odds_api_compatible_shape():
    # No top-level "participants" array -- confirmed live that the priced
    # per-event response (fetch_event_odds_raw) never has one, unlike the
    # event-LIST response; team names only exist nested inside each
    # outcome row's own "participant" object.
    home = {"key": "h1", "slug": "chicago-bears", "name": "Chicago Bears", "shortName": "CHI", "sport": "AMERICAN_FOOTBALL"}
    away = {"key": "a1", "slug": "minnesota-vikings", "name": "Minnesota Vikings", "shortName": "MIN", "sport": "AMERICAN_FOOTBALL"}
    event = {
        "key": "e1",
        "startTime": "2026-09-20T17:00:00.000Z",
        "homeParticipantKey": "h1",
        "markets": [
            {
                "type": "MONEYLINE", "segment": "FULL_MATCH",
                "outcomes": {
                    "BET_PARX": [
                        {"participantKey": "h1", "participant": home, "payout": 1.44, "source": "BET_PARX"},
                        {"participantKey": "a1", "participant": away, "payout": 2.85, "source": "BET_PARX"},
                    ],
                },
            },
            {
                "type": "POINT_SPREAD", "segment": "FULL_MATCH",
                "outcomes": {
                    "BET_PARX": [
                        {"participantKey": "h1", "participant": home, "modifier": -4.5, "payout": 1.91, "source": "BET_PARX"},
                        {"participantKey": "a1", "participant": away, "modifier": 4.5, "payout": 1.91, "source": "BET_PARX"},
                    ],
                },
            },
            {
                "type": "POINT_TOTAL", "segment": "FULL_MATCH",
                "outcomes": {
                    "BET_PARX": [
                        {"type": "OVER", "modifier": 48.5, "payout": 1.92, "source": "BET_PARX"},
                        {"type": "UNDER", "modifier": 48.5, "payout": 1.9, "source": "BET_PARX"},
                    ],
                },
            },
            # A non-FULL_MATCH market must be ignored entirely.
            {
                "type": "MONEYLINE", "segment": "QUARTER_1",
                "outcomes": {"BET_PARX": [{"participantKey": "h1", "payout": 1.2, "source": "BET_PARX"}]},
            },
        ],
    }

    rows = sportsbook_api.event_to_rows(event, fetched_at="2026-09-18T00:00:00+00:00")
    by_market_outcome = {(r["market"], r["outcome_name"]): r for r in rows}

    assert len(rows) == 6
    assert by_market_outcome[("h2h", "Chicago Bears")]["price"] == 1.44
    assert by_market_outcome[("spreads", "Chicago Bears")]["point"] == -4.5
    assert by_market_outcome[("spreads", "Minnesota Vikings")]["point"] == 4.5
    assert by_market_outcome[("totals", "Over")]["point"] == 48.5
    assert all(r["home_team"] == "Chicago Bears" and r["away_team"] == "Minnesota Vikings" for r in rows)
