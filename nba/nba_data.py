"""Free NBA data: ESPN's NBA schedules and box scores as sportsdataverse publishes them (the NBA counterpart of
nflverse), one CSV per season, cached in nba/data_cache.

  schedule     every game: date and tip-off (UTC), season type, neutral site, venue, both teams and the final score
  team box     one row per team per game: shooting (2s, 3s, free throws), rebounds, assists, turnovers, steals,
               blocks, fouls, points in the paint and off turnovers, the opponent's score
  player box   one row per player per game: minutes, the box score, plus-minus, starter, did not play and why

sportsdataverse refreshes the files every morning (about 7:40 AM Eastern).  The latest season is downloaded again when
asked (refresh); older seasons come from the cache; a failed download falls back to the cached copy.  Teams are keyed
by ESPN's team id, which stays with a franchise when it moves or is renamed (New Jersey to Brooklyn, Charlotte Bobcats
to Hornets).
"""

import os

import pandas as pd
import requests

from nba_common import CACHE_DIR

BASE_URL = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download'
FILES = {'schedule': ('espn_nba_schedules', 'nba_schedule_{season}.csv'),
         'team_box': ('espn_nba_team_boxscores', 'team_box_{season}.csv'),
         'player_box': ('espn_nba_player_boxscores', 'player_box_{season}.csv')}
SCHEDULE_COLUMNS = ('game_id', 'season', 'season_type', 'start_date', 'game_date', 'neutral_site', 'notes_headline',
                    'type_abbreviation', 'venue_id', 'venue_full_name', 'venue_address_city', 'venue_address_state',
                    'status_type_completed', 'status_type_name', 'home_id', 'home_abbreviation', 'home_display_name',
                    'home_short_display_name', 'home_color', 'home_alternate_color', 'home_score', 'away_id',
                    'away_abbreviation', 'away_display_name', 'away_short_display_name', 'away_color',
                    'away_alternate_color', 'away_score')
TEAM_BOX_COLUMNS = ('game_id', 'season', 'season_type', 'game_date', 'team_id', 'team_abbreviation', 'team_home_away',
                    'team_score', 'opponent_team_id', 'opponent_team_score', 'field_goals_made',
                    'field_goals_attempted', 'three_point_field_goals_made', 'three_point_field_goals_attempted',
                    'free_throws_made', 'free_throws_attempted', 'offensive_rebounds', 'defensive_rebounds',
                    'total_rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'total_turnovers', 'fouls',
                    'points_in_paint', 'fast_break_points', 'turnover_points', 'largest_lead')
PLAYER_BOX_COLUMNS = ('game_id', 'season', 'season_type', 'game_date', 'athlete_id', 'athlete_display_name',
                      'athlete_position_abbreviation', 'team_id', 'home_away', 'minutes', 'field_goals_made',
                      'field_goals_attempted', 'three_point_field_goals_made', 'three_point_field_goals_attempted',
                      'free_throws_made', 'free_throws_attempted', 'offensive_rebounds', 'defensive_rebounds',
                      'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'fouls', 'plus_minus', 'points',
                      'starter', 'ejected', 'did_not_play', 'reason', 'active')


def cached_download(url, filename, refresh=False):
    """The cached copy of `url` in data_cache (downloaded when missing or when `refresh`); a failed download falls back
    to the cached copy when there is one."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if refresh or not os.path.exists(path):
        try:
            response = requests.get(url, timeout=180, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
        except requests.RequestException:
            if os.path.exists(path):
                return path
            raise
        temporary = path + '.download'
        with open(temporary, 'wb') as f:
            f.write(response.content)
        os.replace(temporary, path)
    return path


def season_file(kind, season, refresh=False):
    """The cached CSV of one season of `kind` ('schedule', 'team_box' or 'player_box')."""
    tag, name = FILES[kind]
    filename = name.format(season=season)
    return cached_download(f'{BASE_URL}/{tag}/{filename}', filename, refresh)


def _load(kind, seasons, columns, refresh):
    """`kind` for `seasons`, only `columns` (those a season's file has); `refresh` downloads the latest season again."""
    seasons = sorted(int(s) for s in seasons)
    wanted = set(columns)
    frames = []
    for season in seasons:
        try:
            path = season_file(kind, season, refresh=bool(refresh) and season == seasons[-1])
        except requests.RequestException:
            continue
        frames.append(pd.read_csv(path, low_memory=False, usecols=lambda c: c in wanted))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns))
    return frame.reindex(columns=[c for c in columns if c in frame.columns])


def load_schedule(seasons, refresh=False):
    """Every game of `seasons`, one row each, with tip-off as a UTC timestamp (`start`) and in Eastern time (`tipoff`)."""
    games = _load('schedule', seasons, SCHEDULE_COLUMNS, refresh)
    games = games.drop_duplicates('game_id')
    games['start'] = pd.to_datetime(games['start_date'], utc=True, errors='coerce')
    games['tipoff'] = games['start'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    games['neutral_site'] = games['neutral_site'].fillna(False).astype(bool)
    return games.sort_values(['start', 'game_id']).reset_index(drop=True)


def load_team_box(seasons, refresh=False):
    """One row per team per game for `seasons`."""
    return _load('team_box', seasons, TEAM_BOX_COLUMNS, refresh)


def load_player_box(seasons, refresh=False, columns=PLAYER_BOX_COLUMNS):
    """One row per player per game for `seasons` (only `columns`, to keep memory down)."""
    return _load('player_box', seasons, columns, refresh)
