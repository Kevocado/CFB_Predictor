from pathlib import Path

from cfb_predictor import config


def test_paths_are_absolute_and_created():
    assert config.PROJECT_ROOT.is_absolute()
    assert config.CACHE_DIR.is_dir()
    assert config.MODELS_DIR.is_dir()
    assert config.GAMES_CACHE_DIR.is_dir()
    assert config.TEAMS_CACHE_DIR.is_dir()
    assert config.PLAYER_STATS_CACHE_DIR.is_dir()
    assert config.ODDS_CACHE_DIR.is_dir()


def test_current_season_is_reasonable():
    assert 2020 <= config.CURRENT_SEASON <= 2100


def test_odds_api_sport_key():
    assert config.ODDS_API_SPORT_KEY == "americanfootball_ncaaf"


def test_cfbd_api_key_attribute_exists():
    # No real key is required for tests to import config -- CFBD_API_KEY may
    # be None in CI/dev, but the attribute itself must always be defined so
    # data/games.py and data/player_stats.py can import it unconditionally.
    assert hasattr(config, "CFBD_API_KEY")
