import json
import pandas as pd
from cfb_predictor.data import advanced_stats as adv

def test_team_hub_maps_ppa_to_epa_fields_and_keeps_form():
    advanced = [dict(team="Georgia", off_ppa=0.31, def_ppa=-0.12, off_success_rate=0.48, def_success_rate=0.36,
                     off_explosiveness=1.2, def_explosiveness=1.1)]
    games = pd.DataFrame([dict(game_id=1, season=2026, week=1, gameday="2026-08-30", home_team="Georgia",
                               away_team="Clemson", home_score=34, away_score=3)])
    uga = next(r for r in adv.team_hub(advanced, games, 2026) if r["team"] == "Georgia")
    assert uga["off_epa_play"] == 0.31 and uga["def_epa_play"] == -0.12 and uga["turnover_margin"] is None
    assert uga["form"] == ["W"] and uga["streak"] == 1

def test_missing_key_means_no_advanced_and_no_call(monkeypatch, tmp_path):
    monkeypatch.setattr(adv, "CFBD_API_KEY", None)
    monkeypatch.setattr(adv, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(adv, "_call_advanced", lambda season: (_ for _ in ()).throw(AssertionError("called")))
    assert adv.fetch_advanced(2026) == []

def test_cache_is_reused_within_a_day(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(adv, "CFBD_API_KEY", "k")
    monkeypatch.setattr(adv, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(adv, "_call_advanced", lambda season: calls.append(season) or [dict(team="Georgia", off_ppa=0.3,
        def_ppa=-0.1, off_success_rate=0.5, def_success_rate=0.4, off_explosiveness=1.0, def_explosiveness=1.0)])
    adv.fetch_advanced(2026); adv.fetch_advanced(2026)
    assert calls == [2026]

def test_a_failing_cfbd_call_means_no_advanced_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(adv, "CFBD_API_KEY", "k")
    monkeypatch.setattr(adv, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(adv, "_call_advanced", lambda season: (_ for _ in ()).throw(RuntimeError("quota")))
    assert adv.fetch_advanced(2026) == []

def test_player_hub_joins_ppa_and_ranks_leaderboards():
    weekly = pd.DataFrame([
        dict(player_id="1", player_name="QB One", position="QB", recent_team="Georgia", season=2026, week=1,
             passing_yards=250, passing_tds=2, rushing_yards=10, rushing_tds=0, receiving_yards=0, receiving_tds=0,
             receptions=0, targets=None, carries=3),
        dict(player_id="2", player_name="WR Two", position="WR", recent_team="Georgia", season=2026, week=1,
             passing_yards=0, passing_tds=0, rushing_yards=0, rushing_tds=0, receiving_yards=90, receiving_tds=1,
             receptions=6, targets=None, carries=0),
    ])
    ppa = [dict(name="QB One", team="Georgia", ppa_total=12.4)]
    out = adv.player_hub(weekly, ppa, 2026)
    by = {p["player_id"]: p for p in out["players"]}
    assert by["1"]["epa_total"] == 12.4 and by["2"]["epa_total"] is None
    assert by["1"]["passing_yards"] == 250 and by["2"]["target_share"] is None
    assert out["leaderboards"]["QB"][0]["player_id"] == "1"


def test_a_failing_cfbd_call_backs_off_instead_of_retrying_every_request(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(adv, "CFBD_API_KEY", "k")
    monkeypatch.setattr(adv, "CACHE_DIR", tmp_path)

    def failing(season):
        calls.append(season)
        raise RuntimeError("429 quota")

    monkeypatch.setattr(adv, "_call_advanced", failing)
    assert adv.fetch_advanced(2026) == [] and adv.fetch_advanced(2026) == []
    assert calls == [2026]


def test_backoff_keeps_serving_the_last_good_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(adv, "CFBD_API_KEY", "k")
    monkeypatch.setattr(adv, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(adv, "_call_advanced", lambda season: [dict(team="Georgia", off_ppa=0.3)])
    adv.fetch_advanced(2026)
    path = tmp_path / "cfbd_advanced" / "teams_2026.json"
    body = json.loads(path.read_text()); body["fetched_at"] -= 2 * adv.MAX_AGE_SECONDS
    path.write_text(json.dumps(body))
    calls = []

    def down(season):
        calls.append(season)
        raise RuntimeError("down")

    monkeypatch.setattr(adv, "_call_advanced", down)
    assert adv.fetch_advanced(2026)[0]["team"] == "Georgia"
    assert adv.fetch_advanced(2026)[0]["team"] == "Georgia"
    assert calls == [2026]


def test_player_ppa_joins_on_athlete_id_before_name():
    weekly = pd.DataFrame([dict(player_id="77", player_name="Kendall Milton Jr.", position="RB", recent_team="Georgia",
                                season=2026, week=1, passing_yards=0, passing_tds=0, rushing_yards=80, rushing_tds=1,
                                receiving_yards=0, receiving_tds=0, receptions=0, carries=15)])
    ppa = [dict(player_id="77", name="Kendall Milton", team="Georgia", ppa_total=6.5)]
    assert adv.player_hub(weekly, ppa, 2026)["players"][0]["epa_total"] == 6.5
