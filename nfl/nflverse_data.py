"""Downloads and caches the free nflverse data used for research: schedules with closing spreads, and weekly team stats."""

import os

import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
GAMES_URL = 'https://github.com/nflverse/nfldata/raw/master/data/games.csv'
TEAM_WEEK_URL = 'https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{year}.csv'
INJURIES_URL = 'https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.csv'
PLAYER_WEEK_URL = 'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.csv'


def _cached_download(url, filename, refresh=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if refresh or not os.path.exists(path):
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(path, 'wb') as f:
            f.write(response.content)
    return path


def load_games(refresh=False):
    """Every NFL game since 1999 with scores, closing spread_line (positive = home favored), rest days and starting QBs."""
    return pd.read_csv(_cached_download(GAMES_URL, 'games.csv', refresh))


def load_team_weeks(first_season, last_season, refresh=False):
    """One row per team per game with offensive box-score stats and EPA."""
    return _load_seasons(TEAM_WEEK_URL, 'stats_team_week_{year}.csv', first_season, last_season, refresh)


def load_injuries(first_season, last_season, refresh=False):
    """Weekly official injury reports (report_status: Out, Doubtful, Questionable) from 2009 on."""
    return _load_seasons(INJURIES_URL, 'injuries_{year}.csv', first_season, last_season, refresh)


def load_player_weeks(first_season, last_season, refresh=False):
    """One row per player per game with passing, rushing and receiving stats and EPA, from 2009 on."""
    return _load_seasons(PLAYER_WEEK_URL, 'stats_player_week_{year}.csv', first_season, last_season, refresh)


def _load_seasons(url_template, filename_template, first_season, last_season, refresh):
    frames = []
    for season in range(first_season, last_season + 1):
        try:
            path = _cached_download(url_template.format(year=season), filename_template.format(year=season), refresh)
        except requests.RequestException:
            continue
        frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True)
