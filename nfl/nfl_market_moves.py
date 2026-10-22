"""Opening lines, line movement and public betting splits from Action Network (2023 on).

Past weeks return the final splits, so this builds a table of how the market moved and who was betting what
for every game since 2023, keyed by season, week and teams.
"""

import json
import os
import subprocess
import time

import pandas as pd

from nflverse_data import CACHE_DIR

NAMES = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL', 'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI', 'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL', 'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET', 'Green Bay Packers': 'GB', 'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC', 'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG', 'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT', 'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS', 'Washington Football Team': 'WAS',
}
URL = 'https://api.actionnetwork.com/web/v2/scoreboard/nfl?bookIds=15,30,68&season={season}&week={week}'


def _fetch(season, week):
    out = subprocess.run(['curl.exe', '-s', '-A', 'Mozilla/5.0', URL.format(season=season, week=week)], capture_output=True).stdout
    try:
        return json.loads(out).get('games', [])
    except ValueError:
        return []


def _book_line(game, book):
    spread = game.get('markets', {}).get(str(book), {}).get('event', {}).get('spread', [])
    # a game under way also carries live in-game lines; the pre-game line it keeps is the closing line
    home = next((o for o in spread if o.get('side') == 'home' and not o.get('is_live')), None)
    return home


def _rows_from_games(games, season):
    rows = []
    for game in games:
        teams = {t['id']: NAMES.get(t['full_name']) for t in game.get('teams', [])}
        home, away = teams.get(game.get('home_team_id')), teams.get(game.get('away_team_id'))
        consensus, opening, dk = _book_line(game, 15), _book_line(game, 30), _book_line(game, 68)
        if not (home and away and consensus and consensus.get('bet_info')):
            continue
        bet = consensus['bet_info']
        rows.append({
            'season': season, 'week': game.get('week'), 'home_team': home, 'away_team': away,
            'open_home_spread': opening['value'] if opening else None,
            'consensus_home_spread': consensus['value'],
            'dk_home_spread': dk['value'] if dk else None,
            'dk_home_odds': dk['odds'] if dk else None,
            'tickets_home': (bet.get('tickets') or {}).get('percent'),
            'money_home': (bet.get('money') or {}).get('percent'),
        })
    return rows


def build_market_table(seasons=(2023, 2024, 2025, 2026), refresh=False):
    path = os.path.join(CACHE_DIR, 'market_moves.csv')
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    rows = []
    for season in seasons:
        for week in range(1, 23):
            rows += _rows_from_games(_fetch(season, week), season)
            time.sleep(0.3)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def append_line_snapshot(season, week):
    """Append the lines and betting splits as they stand right now to NFL\\data\\line_log.csv, timestamped.

    The public history only has the opening and closing numbers, so logging every run builds the record that
    would show how much a line moves between the time you bet and the close.  Returns the rows added.
    """
    from datetime import datetime

    rows = _rows_from_games(_fetch(season, week), season)
    if not rows:
        return 0
    stamp = datetime.now().isoformat(timespec='seconds')
    frame = pd.DataFrame(rows).assign(logged_at=stamp)
    import nfl_sheet
    path = nfl_sheet.line_log_path()   # NFL\data\line_log.csv, with the rest of the daily run's files
    frame.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return len(frame)
