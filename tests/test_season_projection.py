import pandas as pd

from cfb_predictor.models import season_projection


def test_compute_current_records_tallies_wins_losses_and_point_diff():
    played = pd.DataFrame([
        {"home_team": "Ohio State", "away_team": "Michigan", "home_score": 24, "away_score": 10},
        {"home_team": "Iowa", "away_team": "Ohio State", "home_score": 14, "away_score": 21},
    ])
    records = season_projection.compute_current_records(played)

    assert records["Ohio State"] == {"wins": 2, "losses": 0, "ties": 0, "played": 2, "point_diff": 21.0}
    assert records["Michigan"] == {"wins": 0, "losses": 1, "ties": 0, "played": 1, "point_diff": -14.0}
    assert records["Iowa"] == {"wins": 0, "losses": 1, "ties": 0, "played": 1, "point_diff": -7.0}


def test_build_team_conferences_reads_conference_off_schedule_rows():
    schedule = pd.DataFrame([
        {"home_team": "Ohio State", "away_team": "Akron", "home_conference": "Big Ten", "away_conference": "MAC",
         "home_division": "fbs", "away_division": "fbs"},
        {"home_team": "Michigan", "away_team": "Ohio State", "home_conference": "Big Ten", "away_conference": "Big Ten",
         "home_division": "fbs", "away_division": "fbs"},
        # CFBD supplies a real conference for FCS teams too (their own FCS
        # conference) -- the division check, not conference presence, is
        # what excludes them from a "season standings" page.
        {"home_team": "Ohio State", "away_team": "Directional State", "home_conference": "Big Ten", "away_conference": "SWAC",
         "home_division": "fbs", "away_division": "fcs"},
    ])
    conferences = season_projection.build_team_conferences(schedule)

    assert conferences["Ohio State"] == "Big Ten"
    assert conferences["Michigan"] == "Big Ten"
    assert conferences["Akron"] == "MAC"
    assert "Directional State" not in conferences  # FCS -- excluded, not fabricated


def test_project_standings_adds_expected_wins_and_excludes_teams_without_a_conference():
    current_records = {
        "Ohio State": {"wins": 2, "losses": 0, "ties": 0, "played": 2, "point_diff": 21.0},
        "Michigan": {"wins": 0, "losses": 1, "ties": 0, "played": 1, "point_diff": -14.0},
        "FCS Directional": {"wins": 0, "losses": 1, "ties": 0, "played": 1, "point_diff": -30.0},
    }
    team_conferences = {"Ohio State": "Big Ten", "Michigan": "Big Ten"}
    remaining = pd.DataFrame([{"home_team": "Michigan", "away_team": "Ohio State"}])

    def predict_fn(home, away):
        assert (home, away) == ("Michigan", "Ohio State")
        return {"home_win_prob": 0.3, "away_win_prob": 0.7, "predicted_margin": -5.0}

    rows = season_projection.project_standings(remaining, current_records, team_conferences, predict_fn)
    by_team = {r["team"]: r for r in rows}

    assert "FCS Directional" not in by_team  # no known conference -- not ranked
    assert by_team["Michigan"]["projected_wins"] == 0.3
    assert by_team["Ohio State"]["projected_wins"] == 2.7
    assert by_team["Ohio State"]["current_conference_rank"] == 1
    assert by_team["Ohio State"]["projected_conference_rank"] == 1


def test_project_standings_predict_failure_is_skipped_not_fatal():
    current_records = {"Ohio State": {"wins": 0, "losses": 0, "ties": 0, "played": 0, "point_diff": 0.0}}
    team_conferences = {"Ohio State": "Big Ten", "Michigan": "Big Ten"}
    remaining = pd.DataFrame([{"home_team": "Ohio State", "away_team": "Michigan"}])

    def predict_fn(home, away):
        raise ValueError("boom")

    rows = season_projection.project_standings(remaining, current_records, team_conferences, predict_fn)
    by_team = {r["team"]: r for r in rows}
    assert by_team["Ohio State"]["projected_wins"] == 0.0
