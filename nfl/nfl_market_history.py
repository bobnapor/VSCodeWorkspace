"""Historical opening and closing lines from Sportsbook Reviews Online's free NFL archive (2007-08 through 2021-22).

Each season page lists every game as two rows (visitor, then home; "N" rows for neutral sites).  The Open and Close
columns hold the spread on the favorite's row and the total on the other row.  This module parses those pages into one
row per game, maps the team names to nflverse codes, and caches the result in data_cache/sbr_lines.csv.

Spreads follow the nflverse convention: positive = the home team is favored by that many points.
"""

import os
import re

import pandas as pd
import requests

from nflverse_data import CACHE_DIR

URL = 'https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl-odds-{start}-{end:02d}'
SEASONS = range(2007, 2022)
CACHE_FILE = os.path.join(CACHE_DIR, 'sbr_lines.csv')

TEAM_CODES = {
    'Arizona': 'ARI', 'Atlanta': 'ATL', 'Baltimore': 'BAL', 'Buffalo': 'BUF', 'BuffaloBills': 'BUF', 'Carolina': 'CAR', 'Chicago': 'CHI',
    'Cincinnati': 'CIN', 'Cleveland': 'CLE', 'Dallas': 'DAL', 'Denver': 'DEN', 'Detroit': 'DET', 'GreenBay': 'GB',
    'Houston': 'HOU', 'HoustonTexans': 'HOU', 'Indianapolis': 'IND', 'Jacksonville': 'JAX', 'KansasCity': 'KC', 'KCChiefs': 'KC',
    'Kansas': 'KC', 'LAChargers': 'LAC', 'LosAngelesChargers': 'LAC', 'SanDiego': 'LAC', 'LARams': 'LA', 'LosAngelesRams': 'LA',
    'StLouis': 'LA', 'St.Louis': 'LA', 'LosAngeles': 'LA', 'LVRaiders': 'LV', 'LasVegas': 'LV', 'LasVegasRaiders': 'LV',
    'Oakland': 'LV', 'Miami': 'MIA', 'Minnesota': 'MIN', 'NewEngland': 'NE', 'NewOrleans': 'NO', 'NYGiants': 'NYG',
    'NYJets': 'NYJ', 'Philadelphia': 'PHI', 'Pittsburgh': 'PIT', 'SanFrancisco': 'SF', 'Seattle': 'SEA', 'TampaBay': 'TB',
    'Tampa': 'TB', 'Tennessee': 'TEN', 'Washington': 'WAS', 'Washingtom': 'WAS', 'WashingtonFootballTeam': 'WAS',
    'Football Team': 'WAS', 'FootballTeam': 'WAS', 'WashingtonCom': 'WAS',
    'NewYork': 'NY?',   # Giants or Jets: resolved against the schedule by date and opponent
}


def _number(text):
    text = str(text).strip().lower()
    if text in ('pk', 'p', "pk'", '-0'):
        return 0.0
    text = text.replace('½', '.5')
    try:
        return float(text)
    except ValueError:
        return None


def _parse_season(html, season):
    rows = []
    for row in re.findall(r'<tr[^>]*>(.*?)</tr>', html, flags=re.S):
        cells = [re.sub(r'<[^>]+>', '', c).replace('&nbsp;', '').strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, flags=re.S)]
        if len(cells) >= 13 and cells[0].isdigit():
            rows.append(cells)
    games = []
    for visitor, home in zip(rows[0::2], rows[1::2]):
        if visitor[2] not in ('V', 'N') or home[2] not in ('H', 'N'):
            continue
        record = {'season': season, 'mmdd': int(visitor[0]), 'away_name': visitor[3], 'home_name': home[3],
                  'neutral': visitor[2] == 'N', 'away_score': _number(visitor[8]), 'home_score': _number(home[8]),
                  'away_ml': _number(visitor[11]), 'home_ml': _number(home[11])}
        for col, name in ((9, 'open'), (10, 'close')):
            v, h = _number(visitor[col]), _number(home[col])
            if v is None or h is None:
                record[f'{name}_spread'], record[f'{name}_total'] = None, None
                continue
            # the smaller number is the spread (on the favorite's row), the larger the total
            if h <= v:
                record[f'{name}_spread'], record[f'{name}_total'] = h, v       # home favored
            else:
                record[f'{name}_spread'], record[f'{name}_total'] = -v, h      # visitor favored
        games.append(record)
    frame = pd.DataFrame(games)
    month = frame['mmdd'] // 100
    year = season + (month < 8).astype(int)
    frame['gameday'] = pd.to_datetime(dict(year=year, month=month, day=frame['mmdd'] % 100), errors='coerce')
    return frame


def load_sbr_lines(refresh=False):
    """One row per game, 2007-2021: season, gameday, home/away nflverse codes, open/close spread (positive = home favored)
    and total, closing moneylines."""
    if os.path.exists(CACHE_FILE) and not refresh:
        return pd.read_csv(CACHE_FILE, parse_dates=['gameday'])
    frames = []
    for season in SEASONS:
        response = requests.get(URL.format(start=season, end=(season + 1) % 100), timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        frames.append(_parse_season(response.text, season))
    lines = pd.concat(frames, ignore_index=True)
    lines['home_team'] = lines['home_name'].map(TEAM_CODES)
    lines['away_team'] = lines['away_name'].map(TEAM_CODES)
    unknown = sorted(set(lines.loc[lines['home_team'].isna(), 'home_name']) | set(lines.loc[lines['away_team'].isna(), 'away_name']))
    if unknown:
        raise ValueError(f'Unmapped team names in the SBR archive: {unknown}')
    os.makedirs(CACHE_DIR, exist_ok=True)
    lines.to_csv(CACHE_FILE, index=False)
    return lines


def attach_open_lines(table):
    """Add sbr_open / sbr_close (spreads, positive = home favored) to an nflverse game table, matched by teams and date."""
    lines = load_sbr_lines()
    expanded = [lines]
    for side in ('home_team', 'away_team'):
        ambiguous = lines[lines[side] == 'NY?']
        expanded += [ambiguous.assign(**{side: code}) for code in ('NYG', 'NYJ')]
    lines = pd.concat(expanded, ignore_index=True)
    lines = lines[(lines['home_team'] != 'NY?') & (lines['away_team'] != 'NY?')]
    games = table[['game_id', 'season', 'gameday', 'home_team', 'away_team']].copy()
    games['gameday'] = pd.to_datetime(games['gameday'])
    merged = games.merge(lines[['home_team', 'away_team', 'gameday', 'open_spread', 'close_spread', 'open_total', 'close_total']],
                         on=['home_team', 'away_team'], how='inner', suffixes=('', '_sbr'))
    merged = merged[(merged['gameday'] - merged['gameday_sbr']).abs() <= pd.Timedelta(days=2)]
    merged = merged.drop_duplicates('game_id')
    out = table.merge(merged[['game_id', 'open_spread', 'close_spread', 'open_total', 'close_total']]
                      .rename(columns={'open_spread': 'sbr_open', 'close_spread': 'sbr_close'}), on='game_id', how='left')
    return out
