"""NBA spreads and totals from ESPN's archive of sportsbook lines: every game since the 2013-14 season (the backtests'
history), cached in nba/data_cache/espn_odds_<season>.csv.

ESPN keeps each game's lines from several books (5Dimes, BetOnline and Bovada in the early seasons; Caesars,
DraftKings and others later): the home team's spread (negative when the home team is favored), the total, their prices
and the moneylines, as they stood at tip-off, and since 2023-24 also where they opened.  A game is fetched once it is
final, so its lines are closing lines.  closing_lines() reduces the books to one line per game, the median across them,
which is steadier than any single book and close to the market's consensus.

Fill or update the cache (about 1,230 requests a season, a few minutes each with WORKERS threads):
    python nba_lines.py history            every season from FIRST_LINES_SEASON to the current one
    python nba_lines.py history 2026       one season
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests

import nba_data
from nba_common import CACHE_DIR, REGULAR_SEASON, current_season

ODDS_URL = ('https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game}/competitions/{game}/odds'
            '?limit=100')
FIRST_LINES_SEASON = 2014       # ESPN has no lines for earlier NBA games
WORKERS = 6
# ESPN's providers that are not sportsbooks: accuscore, teamrankings, numberfire (zeros as placeholders) and its own
# 'consensus' (the spread's sign is sometimes flipped); 'Opening' is the opening line (through 2016-17)
AGGREGATORS = (1001, 1002, 1003, 1004)
OPENING_PROVIDER, FALLBACK_PROVIDER = 23, 1002
SPREAD_LIMIT, TOTAL_RANGE = 25, (150, 280)
ODDS_COLUMNS = ['season', 'game_id', 'provider_id', 'provider', 'home_spread', 'total', 'home_spread_odds',
                'away_spread_odds', 'over_odds', 'under_odds', 'home_moneyline', 'away_moneyline', 'open_home_spread',
                'open_total', 'fetched_at']
_local = threading.local()


def _session():
    if not hasattr(_local, 'session'):
        _local.session = requests.Session()
        _local.session.headers['User-Agent'] = 'Mozilla/5.0'
    return _local.session


def _number(value):
    """6.0, '+6', '-110' or 'EVEN' as a float; None when blank or unreadable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace('+', '')
    if text.upper() in ('EVEN', 'EV', 'PK', "PICK'EM", 'PICK'):
        return 100.0 if text.upper().startswith('EV') else 0.0
    try:
        return float(text)
    except ValueError:
        return None


def _get_json(url, tries=4):
    for attempt in range(tries):
        try:
            response = _session().get(url, timeout=30)
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == tries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def game_odds(game_id, season):
    """One row per book ESPN has lines from for `game_id` (ODDS_COLUMNS), or one row with no book when it has none."""
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    rows = []
    for item in _get_json(ODDS_URL.format(game=game_id)).get('items', []):
        home, away = item.get('homeTeamOdds') or {}, item.get('awayTeamOdds') or {}
        opened = home.get('open') or {}
        total_open = ((item.get('open') or {}).get('total') or {}).get('american')
        rows.append({'season': season, 'game_id': game_id, 'provider_id': (item.get('provider') or {}).get('id'),
                     'provider': (item.get('provider') or {}).get('name'), 'home_spread': _number(item.get('spread')),
                     'total': _number(item.get('overUnder')), 'home_spread_odds': _number(home.get('spreadOdds')),
                     'away_spread_odds': _number(away.get('spreadOdds')), 'over_odds': _number(item.get('overOdds')),
                     'under_odds': _number(item.get('underOdds')), 'home_moneyline': _number(home.get('moneyLine')),
                     'away_moneyline': _number(away.get('moneyLine')),
                     'open_home_spread': _number(opened.get('pointSpread', {}).get('american')
                                                 if isinstance(opened.get('pointSpread'), dict) else opened.get('pointSpread')),
                     'open_total': _number(total_open), 'fetched_at': stamp})
    return rows or [dict({c: None for c in ODDS_COLUMNS}, season=season, game_id=game_id, fetched_at=stamp)]


def odds_path(season):
    return os.path.join(CACHE_DIR, f'espn_odds_{season}.csv')


def read_odds(season):
    path = odds_path(season)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=ODDS_COLUMNS)


def update_odds_history(seasons, workers=WORKERS, refresh_schedule=False, progress=True):
    """Fetch the lines of every final regular-season game of `seasons` not in the cache yet, saving as it goes.
    Returns the number of games fetched."""
    fetched_total = 0
    for season in seasons:
        schedule = nba_data.load_schedule([season], refresh=refresh_schedule or season == current_season())
        final = schedule[(schedule['season_type'] == REGULAR_SEASON) & schedule['status_type_completed'].fillna(False)
                         .astype(bool)]
        have = set(read_odds(season)['game_id'])
        missing = [int(g) for g in final['game_id'] if int(g) not in have]
        if progress:
            print(f'{season}: {len(final)} final regular-season games, {len(missing)} to fetch', flush=True)
        if not missing:
            continue
        path, done, started = odds_path(season), 0, time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(game_odds, game, season): game for game in missing}
            batch = []
            for future in as_completed(futures):
                try:
                    batch += future.result()
                except requests.RequestException as error:
                    print(f'  could not fetch game {futures[future]}: {error}', flush=True)
                done += 1
                if len(batch) >= 400 or done == len(missing):
                    frame = pd.DataFrame(batch, columns=ODDS_COLUMNS)
                    frame.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
                    batch = []
                if progress and (done % 200 == 0 or done == len(missing)):
                    print(f'  {done}/{len(missing)} games ({time.time() - started:.0f}s)', flush=True)
        fetched_total += done
    return fetched_total


def closing_lines(seasons):
    """One row per game with lines: the median across sportsbooks of the closing home spread and total, the median
    prices, how many books there were, and the opening spread and total where ESPN has them (its 'Opening' line through
    2016-17, the books' own openers from 2023-24).

    Only sportsbooks count: ESPN also lists line aggregators and projection sites (AGGREGATORS: numberfire posts zeros as
    placeholders, its 'consensus' entry sometimes has the spread's sign flipped) and in-game lines ('... - Live Odds').
    A game no sportsbook has falls back to teamrankings' line, which follows the books' median closely (line_source
    says which).  Values no NBA game has (a spread beyond SPREAD_LIMIT, a total outside TOTAL_RANGE) count as missing."""
    frames = [read_odds(season) for season in seasons]
    odds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ODDS_COLUMNS)
    odds = odds[odds['provider_id'].notna() & ~odds['provider'].astype(str).str.contains('live', case=False)].copy()
    numeric = ['home_spread', 'total', 'home_spread_odds', 'away_spread_odds', 'over_odds', 'under_odds',
               'home_moneyline', 'away_moneyline', 'open_home_spread', 'open_total']
    for column in numeric:
        odds[column] = pd.to_numeric(odds[column], errors='coerce')
    odds['provider_id'] = pd.to_numeric(odds['provider_id'], errors='coerce').astype('Int64')
    odds.loc[odds['home_spread'].abs() > SPREAD_LIMIT, 'home_spread'] = None
    odds.loc[~odds['total'].between(*TOTAL_RANGE), 'total'] = None
    for column in ('open_home_spread', 'open_total'):
        odds.loc[odds['provider_id'] == OPENING_PROVIDER, column] = odds[column.replace('open_', '')]
    odds.loc[odds['open_home_spread'].abs() > SPREAD_LIMIT, 'open_home_spread'] = None
    odds.loc[~odds['open_total'].between(*TOTAL_RANGE), 'open_total'] = None
    opening = odds.groupby('game_id')[['open_home_spread', 'open_total']].median()
    books = odds[~odds['provider_id'].isin(list(AGGREGATORS) + [OPENING_PROVIDER])]
    fallback = odds[(odds['provider_id'] == FALLBACK_PROVIDER) & ~odds['game_id'].isin(books['game_id'])]
    closing = pd.concat([books.assign(line_source='books'), fallback.assign(line_source='teamrankings')])
    lines = closing.groupby('game_id').agg(**{column: (column, 'median') for column in numeric[:8]},
                                           books=('provider_id', 'nunique'), line_source=('line_source', 'first'))
    lines = lines.rename(columns={'home_spread': 'close_home_spread', 'total': 'close_total'}).join(opening)
    return lines.reset_index()


# ---------------------------------------------------------------- the day's lines (Action Network)

ACTION_NETWORK_URL = 'https://api.actionnetwork.com/web/v2/scoreboard/nba?bookIds=15,30,68&date={day}'
DRAFTKINGS, CONSENSUS, OPENER = '68', '15', '30'
AN_NAMES = {'Los Angeles Clippers': 'LA Clippers'}     # Action Network's team names that differ from ESPN's
DAY_COLUMNS = ['game_id', 'dk_home_spread', 'dk_home_odds', 'dk_away_odds', 'dk_total', 'dk_over_odds', 'dk_under_odds',
               'consensus_home_spread', 'consensus_total', 'open_home_spread', 'open_total', 'tickets_home',
               'money_home', 'tickets_over', 'money_over']


def _pregame(outcomes, side):
    return next((o for o in outcomes if o.get('side') == side and not o.get('is_live')), None)


def day_lines(day, schedule):
    """DraftKings' spreads and totals with their prices (and the consensus, the opener and the betting splits) for the
    games of `day` (a date, Eastern) from Action Network, matched to ESPN's `schedule` rows by teams and tip-off.
    A game under way keeps its last pre-game line (in-game lines are skipped)."""
    try:
        response = requests.get(ACTION_NETWORK_URL.format(day=f'{day:%Y%m%d}'),
                                headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        response.raise_for_status()
        games = response.json().get('games', [])
    except (requests.RequestException, ValueError) as error:
        print(f'Could not load the day\'s lines from Action Network ({error})')
        return pd.DataFrame(columns=DAY_COLUMNS)
    rows = []
    for game in games:
        names = {t['id']: AN_NAMES.get(t.get('full_name'), t.get('full_name')) for t in game.get('teams', [])}
        home, away = names.get(game.get('home_team_id')), names.get(game.get('away_team_id'))
        start = pd.Timestamp(game.get('start_time'))
        match = schedule[(schedule['home_display_name'] == home) & (schedule['away_display_name'] == away)
                         & ((schedule['start'] - start).abs() <= pd.Timedelta(hours=6))]
        if match.empty:
            continue
        markets = game.get('markets', {})
        row = {'game_id': int(match['game_id'].iloc[0])}
        for book, prefix in ((DRAFTKINGS, 'dk'), (CONSENSUS, 'consensus'), (OPENER, 'open')):
            event = markets.get(book, {}).get('event', {})
            home_side = _pregame(event.get('spread', []), 'home')
            away_side = _pregame(event.get('spread', []), 'away')
            over, under = _pregame(event.get('total', []), 'over'), _pregame(event.get('total', []), 'under')
            row[f'{prefix}_home_spread'] = home_side.get('value') if home_side else None
            row[f'{prefix}_total'] = over.get('value') if over else None
            if prefix == 'dk':
                row.update(dk_home_odds=home_side.get('odds') if home_side else None,
                           dk_away_odds=away_side.get('odds') if away_side else None,
                           dk_over_odds=over.get('odds') if over else None,
                           dk_under_odds=under.get('odds') if under else None)
            if prefix == 'consensus':
                spread_bets = (home_side or {}).get('bet_info') or {}
                total_bets = (over or {}).get('bet_info') or {}
                row.update(tickets_home=(spread_bets.get('tickets') or {}).get('percent'),
                           money_home=(spread_bets.get('money') or {}).get('percent'),
                           tickets_over=(total_bets.get('tickets') or {}).get('percent'),
                           money_over=(total_bets.get('money') or {}).get('percent'))
        rows.append(row)
    return pd.DataFrame(rows, columns=DAY_COLUMNS)


def main(argv):
    if not argv or argv[0] != 'history':
        print(__doc__)
        return 2
    seasons = [int(a) for a in argv[1:]] or list(range(FIRST_LINES_SEASON, current_season() + 1))
    fetched = update_odds_history(seasons)
    print(f'Fetched the lines of {fetched} games.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
