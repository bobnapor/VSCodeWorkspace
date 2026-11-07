"""CollegeFootballData (api.collegefootballdata.com): this season's games and the sportsbooks' lines, which the free
archives do not have yet (cfbfastR's line archive ends with the last complete season and ESPN's play-by-play carries
no 2026 spreads).  The API key is read by cfb_common.cfbd_key (cfb/cfbd_key.txt).  The free tier allows 1,000 calls a
month; a season's games or lines are one call each, cached in data_cache and fetched again only when asked (refresh).

CollegeFootballData's game ids are ESPN's, and its lines name both teams by ESPN id, so nothing needs matching.
"""

import json
import os

import pandas as pd
import requests

from cfb_common import CACHE_DIR, POSTSEASON, REGULAR_SEASON, cfbd_key

BASE_URL = 'https://api.collegefootballdata.com'
SEASON_TYPES = {'regular': REGULAR_SEASON, 'postseason': POSTSEASON}


def _get(path, params, cache_name, refresh=False):
    """The API's JSON for `path`, cached in data_cache/`cache_name` (fetched when missing or on `refresh`); a failed
    call falls back to the cached copy when there is one."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, cache_name)
    if refresh or not os.path.exists(cache):
        try:
            response = requests.get(BASE_URL + path, params=params, timeout=60,
                                    headers={'Authorization': f'Bearer {cfbd_key()}', 'Accept': 'application/json'})
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            if os.path.exists(cache):
                with open(cache, encoding='utf-8') as f:
                    return json.load(f)
            raise
        with open(cache + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(cache + '.tmp', cache)
        return data
    with open(cache, encoding='utf-8') as f:
        return json.load(f)


def games(season, refresh=False):
    """Every game of `season` (regular season and postseason) in cfb_data.load_schedule's columns, with each team's
    classification ('fbs', 'fcs', ...)."""
    rows = []
    for kind in SEASON_TYPES:
        for g in _get('/games', {'year': season, 'seasonType': kind}, f'cfbd_games_{season}_{kind}.json', refresh):
            rows.append({
                'game_id': g['id'], 'season': g['season'], 'week': g['week'], 'season_type': SEASON_TYPES[kind],
                'game_date': g.get('startDate'), 'neutral_site': bool(g.get('neutralSite')),
                'conference_competition': bool(g.get('conferenceGame')), 'home_id': g.get('homeId'),
                'away_id': g.get('awayId'), 'home_team': g.get('homeTeam'), 'away_team': g.get('awayTeam'),
                'home_score': g.get('homePoints'), 'away_score': g.get('awayPoints'), 'venue': g.get('venue'),
                'status': 'STATUS_FINAL' if g.get('completed') else 'STATUS_SCHEDULED',
                'home_class': g.get('homeClassification'), 'away_class': g.get('awayClassification')})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame['start'] = pd.to_datetime(frame['game_date'], utc=True, errors='coerce')
    return frame.drop_duplicates('game_id').sort_values(['start', 'game_id']).reset_index(drop=True)


def teams(season, refresh=False):
    """{ESPN team id: (school, abbreviation, color, alternate color)} for the FBS teams of `season`, for the card's
    badges."""
    found = {}
    for t in _get('/teams/fbs', {'year': season}, f'cfbd_teams_{season}.json', refresh):
        found[t['id']] = (t.get('school'), t.get('abbreviation') or t.get('school'), t.get('color') or '#475569',
                          t.get('alternateColor') or '#94A3B8')
    return found


def lines(season, refresh=False):
    """Every book's line for `season`, one row per (game, provider): the home team's spread (negative when it is
    favored), the total and the opening lines, with the game's home team id.  For a game not yet played the spread is
    the current one; once played, the closing one."""
    rows = []
    for kind in SEASON_TYPES:
        for g in _get('/lines', {'year': season, 'seasonType': kind}, f'cfbd_lines_{season}_{kind}.json', refresh):
            for line in g.get('lines') or []:
                rows.append({'game_id': g['id'], 'season': g['season'], 'line_home_id': g.get('homeTeamId'),
                             'provider': line.get('provider'), 'home_spread': line.get('spread'),
                             'total': line.get('overUnder'), 'open_home_spread': line.get('spreadOpen'),
                             'open_total': line.get('overUnderOpen'), 'home_moneyline': line.get('homeMoneyline'),
                             'away_moneyline': line.get('awayMoneyline')})
    frame = pd.DataFrame(rows)
    for column in ('home_spread', 'total', 'open_home_spread', 'open_total', 'home_moneyline', 'away_moneyline'):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return frame
