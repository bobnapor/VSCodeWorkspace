"""Downloads and caches the free nflverse data used for research: schedules with closing spreads, and weekly team stats."""

import os

import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
GAMES_URL = 'https://github.com/nflverse/nfldata/raw/master/data/games.csv'
TEAM_WEEK_URL = 'https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{year}.csv'
INJURIES_URL = 'https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.csv'
PLAYER_WEEK_URL = 'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.csv'
PLAYERS_URL = 'https://github.com/nflverse/nflverse-data/releases/download/players/players.csv'


def _cached_download(url, filename, refresh=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if refresh or not os.path.exists(path):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
        except requests.RequestException:
            if os.path.exists(path):
                return path  # offline or the file is missing upstream: fall back to the last copy
            raise
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


def load_player_weeks(first_season, last_season, refresh=False, columns=None):
    """One row per player per game with passing, rushing and receiving stats and EPA, from 2009 on.

    `columns` limits the columns read (the files have about 150), which keeps many seasons small in memory.
    """
    return _load_seasons(PLAYER_WEEK_URL, 'stats_player_week_{year}.csv', first_season, last_season, refresh, columns)


def load_players(refresh=False):
    """Every player nflverse knows, with the ids that link the sources: gsis_id (the stats' player_id) and espn_id."""
    return pd.read_csv(_cached_download(PLAYERS_URL, 'players.csv', refresh), low_memory=False)


def _load_seasons(url_template, filename_template, first_season, last_season, refresh, columns=None):
    """Cached per-season files; `refresh` re-downloads only the latest season, `refresh='all'` every season."""
    wanted = None if columns is None else set(columns)
    frames = []
    for season in range(first_season, last_season + 1):
        force = refresh == 'all' or (bool(refresh) and season == last_season)
        try:
            path = _cached_download(url_template.format(year=season), filename_template.format(year=season), force)
        except requests.RequestException:
            continue
        frames.append(pd.read_csv(path, low_memory=False, usecols=None if wanted is None else (lambda c: c in wanted)))
    return pd.concat(frames, ignore_index=True)
