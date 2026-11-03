"""NBA player prop lines from ESPN's public odds API: the day's lines and a cached history for the backtests.

ESPN's core API publishes its partner sportsbook's player props for every game (`.../odds/{provider}/propBets`): the
line now ('current') and where it opened ('open'), but not the prices.  The book is DraftKings (provider 100) since
the 2026-27 season and was ESPN BET (provider 58) before.  NBA history: ESPN BET from late December 2023 to the end of
November 2025, DraftKings from October 2026; ESPN kept nothing between (the switch of books).  ESPN BET's feed also
lists alternate lines, so the main line is the one listed most often (main_lines).  Lines of finished games stay in the
feed; a stored current line is the closing line when it was last updated before tip-off ('pregame'), and the opening
line is always a pre-game line.

    python nba_prop_lines.py history      fill or update the cache (data_cache/props_espn_<season>.csv)
"""

import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import requests

import nba_data
from nba_common import CACHE_DIR, REGULAR_SEASON, current_season, nba_folder

ODDS_URL = 'https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{event}/competitions/{event}/odds'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/130.0.0.0 Safari/537.36', 'Accept': 'application/json'}
PROVIDERS = {100: 'DraftKings', 58: 'ESPN BET'}
FIRST_PROP_SEASON = 2024
# market: (ESPN prop type id, label, player box columns summed for the result)
MARKETS = {
    'pts': ('1', 'Points', ('points',)),
    'reb': ('3', 'Rebounds', ('rebounds',)),
    'ast': ('2', 'Assists', ('assists',)),
    'fg3m': ('4', '3-Pointers Made', ('three_point_field_goals_made',)),
    'pra': ('90', 'Pts + Reb + Ast', ('points', 'rebounds', 'assists')),
    'pr': ('88', 'Pts + Reb', ('points', 'rebounds')),
    'pa': ('87', 'Pts + Ast', ('points', 'assists')),
    'ra': ('89', 'Reb + Ast', ('rebounds', 'assists')),
    'stl': ('5', 'Steals', ('steals',)),
    'blk': ('6', 'Blocks', ('blocks',)),
    'tov': ('7', 'Turnovers', ('turnovers',)),
}
TYPE_TO_MARKET = {type_id: market for market, (type_id, _, _) in MARKETS.items()}
LINE_COLUMNS = ['season', 'game_id', 'provider', 'athlete_id', 'market', 'line', 'open_line', 'over_odds', 'under_odds',
                'open_over_odds', 'open_under_odds', 'price_1', 'price_2', 'open_price_1', 'open_price_2',
                'last_updated', 'tipoff', 'pregame']
HISTORY_VERSION = 2         # 2: with prices (November 2, 2026)


def _get_json(url, params=None, tries=4):
    for attempt in range(tries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == tries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def fetch_event_props(event, provider):
    """Every propBets entry ESPN has for one game and book (all pages); [] when the book has none."""
    items, page = [], 1
    while True:
        data = _get_json(ODDS_URL.format(event=event) + f'/{provider}/propBets',
                         params={'lang': 'en', 'region': 'us', 'limit': 1000, 'page': page})
        if not data:
            return items
        items += data.get('items', [])
        if page >= data.get('pageCount', 1):
            return items
        page += 1


def _athlete(item):
    found = re.search(r'/athletes/(\d+)', (item.get('athlete') or {}).get('$ref', ''))
    return int(found.group(1)) if found else None


def _target(item, which):
    return ((item.get(which) or {}).get('target') or {}).get('value')


def _price(item, which):
    """The item's American price now ('value') or at the open ('open'), as a number, or None."""
    american = ((item.get('odds') or {}).get('american') or {}).get(which)
    try:
        return float(str(american).replace('+', '')) if american not in (None, '') else None
    except ValueError:
        return 100.0 if str(american).upper().startswith('EV') else None


def _side(item):
    """'over' or 'under' when the item says (ESPN BET's do; DraftKings' do not), else None."""
    current = item.get('current') or {}
    return 'over' if 'over' in current else 'under' if 'under' in current else None


def main_lines(items):
    """{(athlete, market): main line} with its line, open line, last update and prices: each over/under line is listed
    twice (the over and the under) and ESPN BET's alternate lines once each, so the main line is the (line, open line)
    pair listed most often; a player whose top count is shared by two pairs is skipped as ambiguous.  Prices: the over's
    and the under's now and at the open where the feed labels the sides (ESPN BET), and otherwise the two listed prices
    in the feed's order (price_1, price_2; DraftKings)."""
    counts, stamps, listed = defaultdict(Counter), defaultdict(dict), defaultdict(lambda: defaultdict(list))
    for item in items:
        market = TYPE_TO_MARKET.get(str((item.get('type') or {}).get('id')))
        athlete, line, opened = _athlete(item), _target(item, 'current'), _target(item, 'open')
        if market is None or athlete is None or line is None:
            continue
        key, pair = (athlete, market), (float(line), None if opened is None else float(opened))
        counts[key][pair] += 1
        stamps[key][pair] = max(stamps[key].get(pair, ''), item.get('lastUpdated') or '')
        listed[key][pair].append(item)
    lines = {}
    for key, pairs in counts.items():
        ranked = pairs.most_common()
        (line, opened), top = ranked[0]
        if top < 2 or (len(ranked) > 1 and ranked[1][1] == top):
            continue
        both = listed[key][(line, opened)]
        prices = {'over_odds': None, 'under_odds': None, 'open_over_odds': None, 'open_under_odds': None,
                  'price_1': _price(both[0], 'value'), 'price_2': _price(both[1], 'value'),
                  'open_price_1': _price(both[0], 'open'), 'open_price_2': _price(both[1], 'open')}
        for item in both:
            side = _side(item)
            if side:
                prices[f'{side}_odds'], prices[f'open_{side}_odds'] = _price(item, 'value'), _price(item, 'open')
        lines[key] = dict(line=line, open_line=opened, last_updated=stamps[key][(line, opened)], **prices)
    return lines


def game_lines(game_id, season, tipoff, providers=(100, 58)):
    """Main prop lines for one game from the first book in `providers` that has any, as rows of LINE_COLUMNS."""
    for provider in providers:
        items = fetch_event_props(game_id, provider)
        if items:
            break
    else:
        return []
    rows = []
    for (athlete, market), found in main_lines(items).items():
        updated = found['last_updated']
        stamp = pd.Timestamp(updated) if updated else None
        if stamp is not None and stamp.tzinfo is None:
            stamp = stamp.tz_localize('UTC')
        rows.append(dict(found, season=season, game_id=int(game_id), provider=provider, athlete_id=athlete,
                         market=market, tipoff=tipoff.isoformat(), pregame=bool(stamp is not None and stamp <= tipoff)))
    return rows


def _history_rows(game):
    """(rows, fetched) for one finished game; a failed download is retried on the next run."""
    try:
        return game_lines(int(game.game_id), int(game.season), game.start), True
    except (requests.RequestException, ValueError) as error:
        print(f'  could not fetch the prop lines of game {game.game_id} (retried next run): {error}', flush=True)
        return [], False


def history_path(season, done=False):
    return os.path.join(CACHE_DIR, f'props_espn_{season}_v{HISTORY_VERSION}{"_games" if done else ""}.csv')


def load_line_history(seasons, refresh=False, workers=6, progress=True):
    """Prop lines of every finished regular-season game of `seasons`, cached per season (only games not fetched before
    are downloaded).  Returns LINE_COLUMNS."""
    frames = []
    for season in seasons:
        path, done_path = history_path(season), history_path(season, done=True)
        cached = pd.read_csv(path) if os.path.exists(path) else None
        done = set(pd.read_csv(done_path)['game_id']) if os.path.exists(done_path) else set()
        schedule = nba_data.load_schedule([season], refresh=refresh or season == current_season())
        final = schedule[(schedule['season_type'] == REGULAR_SEASON)
                         & schedule['status_type_completed'].fillna(False).astype(bool)]
        todo = [g for g in final.itertuples() if int(g.game_id) not in done]
        if todo:
            if progress:
                print(f'{season}: fetching the prop lines of {len(todo)} games', flush=True)
            started = time.time()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fetched = list(pool.map(_history_rows, todo))
            new = pd.DataFrame([row for rows, _ in fetched for row in rows], columns=LINE_COLUMNS)
            cached = new if cached is None else pd.concat([cached, new], ignore_index=True)
            cached.to_csv(path, index=False)
            done |= {int(g.game_id) for g, (_, ok) in zip(todo, fetched) if ok}
            pd.DataFrame({'game_id': sorted(done)}).to_csv(done_path, index=False)
            if progress:
                print(f'  {len(new)} lines from {new["game_id"].nunique()} games ({time.time() - started:.0f}s)',
                      flush=True)
        if cached is not None:
            frames.append(cached)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LINE_COLUMNS)


def fetch_day_lines(games, now=None):
    """Current lines for the games of `games` (schedule rows) that have not tipped off: (lines, games skipped because
    they are under way or over, whose lines would be in-game lines)."""
    now = now or pd.Timestamp.now(tz='UTC')
    rows, skipped = [], []
    for game in games.itertuples():
        if game.start <= now:
            skipped.append(int(game.game_id))
            continue
        try:
            rows += game_lines(int(game.game_id), int(game.season), game.start)
        except (requests.RequestException, ValueError) as error:
            print(f'Could not fetch the prop lines of game {game.game_id}: {error}')
    return pd.DataFrame(rows, columns=LINE_COLUMNS), skipped


def snapshot_log_path():
    """NBA\\data\\prop_line_log.csv: the prop lines as each run saw them."""
    return os.path.join(nba_folder('data'), 'prop_line_log.csv')


def append_snapshot(lines):
    """Append the lines as they stand now to the snapshot log (timestamped); returns the rows added."""
    if lines.empty:
        return 0
    path = snapshot_log_path()
    lines.assign(logged_at=datetime.now().isoformat(timespec='seconds')).to_csv(
        path, mode='a', header=not os.path.exists(path), index=False)
    return len(lines)


def main(argv):
    if not argv or argv[0] != 'history':
        print(__doc__)
        return 2
    seasons = [int(a) for a in argv[1:]] or list(range(FIRST_PROP_SEASON, current_season() + 1))
    lines = load_line_history(seasons)
    print(lines.groupby(['season', 'provider', 'market']).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
