"""NBA lines across sportsbooks and the betting splits, from Action Network's scoreboard (one request per day), cached
in nba/data_cache/an_lines_<season>.csv: one row per game and book with the closing spread, total and moneyline and
their prices, and the consensus book's betting splits (share of tickets and of money on the home side, the over and
the home moneyline).  Action Network keeps NBA lines back to the 2023-24 season.

Books (Action Network's ids): 15 consensus, 30 the opener, 68/1534/1548 DraftKings (NJ/PA/NY), 69 FanDuel, 71
BetRivers, 13/49/123 Caesars, 24/79 bet365, 74 Parx, 247 Unibet, 1903 Bally Bet, 12 Sports Interaction, 67 SI
consensus.  There is no sharp market-maker (Pinnacle, Circa) in the feed; the median of the books stands in for the
market.

    python nba_market.py history          fill or update the cache (about 0.5 seconds a day)
"""

import os
import sys
import time

import pandas as pd
import requests

import nba_data
from nba_common import CACHE_DIR, REGULAR_SEASON, current_season

BOOKS = (12, 13, 15, 24, 30, 49, 67, 68, 69, 71, 74, 79, 123, 247, 1534, 1548, 1903)
NAMES = {12: 'Sports Interaction', 13: 'Caesars', 15: 'Consensus', 24: 'bet365 CO', 30: 'Open', 49: 'Caesars NV',
         67: 'SI Consensus', 68: 'DraftKings NJ', 69: 'FanDuel NJ', 71: 'BetRivers NJ', 74: 'Parx', 79: 'bet365 NJ',
         123: 'Caesars NJ', 247: 'Unibet NJ', 1534: 'DraftKings PA', 1548: 'DraftKings NY', 1903: 'Bally Bet NY'}
URL = 'https://api.actionnetwork.com/web/v2/scoreboard/nba?bookIds={books}&date={day}'
FIRST_SEASON = 2024
AN_NAMES = {'Los Angeles Clippers': 'LA Clippers'}
COLUMNS = ['season', 'game_id', 'book', 'home_spread', 'home_spread_odds', 'away_spread_odds', 'total', 'over_odds',
           'under_odds', 'home_ml', 'away_ml', 'tickets_home', 'money_home', 'tickets_over', 'money_over',
           'tickets_home_ml', 'money_home_ml']


def _pregame(outcomes, side):
    return next((o for o in outcomes if o.get('side') == side and not o.get('is_live')), None)


def _value(outcome, key):
    return outcome.get(key) if outcome else None


def day_rows(day, schedule, season):
    response = requests.get(URL.format(books=','.join(map(str, BOOKS)), day=f'{day:%Y%m%d}'),
                            headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
    response.raise_for_status()
    rows = []
    for game in response.json().get('games', []):
        names = {t['id']: AN_NAMES.get(t.get('full_name'), t.get('full_name')) for t in game.get('teams', [])}
        home, away = names.get(game.get('home_team_id')), names.get(game.get('away_team_id'))
        start = pd.Timestamp(game.get('start_time'))
        match = schedule[(schedule['home_display_name'] == home) & (schedule['away_display_name'] == away)
                         & ((schedule['start'] - start).abs() <= pd.Timedelta(hours=6))]
        if match.empty:
            continue
        game_id = int(match['game_id'].iloc[0])
        for book, market in game.get('markets', {}).items():
            event = market.get('event', {})
            spread_home, spread_away = _pregame(event.get('spread', []), 'home'), _pregame(event.get('spread', []), 'away')
            over, under = _pregame(event.get('total', []), 'over'), _pregame(event.get('total', []), 'under')
            ml_home, ml_away = _pregame(event.get('moneyline', []), 'home'), _pregame(event.get('moneyline', []), 'away')
            row = {'season': season, 'game_id': game_id, 'book': int(book),
                   'home_spread': _value(spread_home, 'value'), 'home_spread_odds': _value(spread_home, 'odds'),
                   'away_spread_odds': _value(spread_away, 'odds'), 'total': _value(over, 'value'),
                   'over_odds': _value(over, 'odds'), 'under_odds': _value(under, 'odds'),
                   'home_ml': _value(ml_home, 'odds'), 'away_ml': _value(ml_away, 'odds')}
            if int(book) == 15:
                for key, outcome in (('home', spread_home), ('over', over), ('home_ml', ml_home)):
                    bets = (outcome or {}).get('bet_info') or {}
                    row[f'tickets_{key}'] = (bets.get('tickets') or {}).get('percent')
                    row[f'money_{key}'] = (bets.get('money') or {}).get('percent')
            rows.append(row)
    return rows


def history_path(season):
    return os.path.join(CACHE_DIR, f'an_lines_{season}.csv')


def load_history(seasons, progress=True):
    """The lines of every final regular-season game of `seasons` (from 2023-24), cached per season; days already
    fetched are skipped (a day is fetched again while the season is under way)."""
    frames = []
    for season in seasons:
        path = history_path(season)
        cached = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=COLUMNS)
        schedule = nba_data.load_schedule([season], refresh=season == current_season())
        final = schedule[(schedule['season_type'] == REGULAR_SEASON)
                         & schedule['status_type_completed'].fillna(False).astype(bool)]
        days = sorted(set(final.loc[~final['game_id'].isin(cached['game_id']), 'tipoff'].dt.normalize()))
        if days and progress:
            print(f'{season}: fetching {len(days)} days of lines', flush=True)
        new = []
        for i, day in enumerate(days, start=1):
            for attempt in range(3):
                try:
                    new += day_rows(day, final, season)
                    break
                except (requests.RequestException, ValueError):
                    time.sleep(2 * (attempt + 1))
            time.sleep(0.2)
            if progress and i % 50 == 0:
                print(f'  {i}/{len(days)} days', flush=True)
        if new:
            cached = pd.concat([cached, pd.DataFrame(new, columns=COLUMNS)], ignore_index=True)
            cached = cached.drop_duplicates(['game_id', 'book'], keep='last')
            cached.to_csv(path, index=False)
        frames.append(cached)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def main(argv):
    if not argv or argv[0] != 'history':
        print(__doc__)
        return 2
    seasons = [int(a) for a in argv[1:]] or list(range(FIRST_SEASON, current_season() + 1))
    lines = load_history(seasons)
    print(lines.groupby(['season', 'book'])['game_id'].nunique().unstack(fill_value=0).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
