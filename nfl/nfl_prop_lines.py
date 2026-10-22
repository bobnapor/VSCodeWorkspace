"""DraftKings player prop lines from ESPN's public odds API: this week's lines and a cached history for backtests.

ESPN's core API publishes its partner sportsbook's player props for every game (`.../odds/{provider}/propBets`): the
line now ("current") and the line when it opened ("open"), but not the prices.  The book has been DraftKings
(provider 100) since late 2025 and was ESPN BET (provider 58) in 2024-2025; ESPN BET's feed also lists alternate lines
(see `_main_lines`).  Lines of finished games stay in the feed, which is what makes a backtest against real prop lines
possible.

Caveat for history: the book keeps moving a line during the game, so a stored "current" line is the closing line only
if it was last updated before kickoff (`pregame`).  The opening line is always a pre-game line.
"""

import logging
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import requests

from nflverse_data import CACHE_DIR

ESPN_ODDS_URL = 'https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{event}/competitions/{event}/odds'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/130.0.0.0 Safari/537.36', 'Accept': 'application/json'}
PROVIDERS = {100: 'DraftKings', 58: 'ESPN BET'}
FIRST_PROP_SEASON = 2024   # ESPN's prop feed has only a handful of lines per game before 2024


def snapshot_log_path():
    """NFL\\data\\prop_line_log.csv: the prop lines as each run saw them, with the rest of the daily run's files."""
    import nfl_sheet
    return os.path.join(nfl_sheet.nfl_folder('data'), 'prop_line_log.csv')

# market key: (ESPN prop type id, label, nflverse stat columns summed for the result)
MARKETS = {
    'pass_yds': ('8', 'Passing Yards', ('passing_yards',)),
    'pass_cmp': ('9', 'Pass Completions', ('completions',)),
    'pass_td': ('10', 'Passing TDs', ('passing_tds',)),
    'rush_att': ('11', 'Rush Attempts', ('carries',)),
    'rush_yds': ('12', 'Rushing Yards', ('rushing_yards',)),
    'rec_yds': ('13', 'Receiving Yards', ('receiving_yards',)),
    'rec': ('14', 'Receptions', ('receptions',)),
    'pass_int': ('15', 'Interceptions Thrown', ('passing_interceptions',)),
    'pass_att': ('16', 'Pass Attempts', ('attempts',)),
    'pass_rush_yds': ('18', 'Pass + Rush Yards', ('passing_yards', 'rushing_yards')),
    'rush_rec_yds': ('20', 'Rush + Rec Yards', ('rushing_yards', 'receiving_yards')),
}
TYPE_TO_MARKET = {type_id: market for market, (type_id, _, _) in MARKETS.items()}
LINE_COLUMNS = ['season', 'week', 'game_id', 'espn_event', 'provider', 'espn_athlete', 'market', 'line', 'open_line',
                'last_updated', 'kickoff', 'pregame']


def kickoff_utc(gameday, gametime):
    """Kickoff as a UTC timestamp from nflverse's gameday and gametime (US Eastern)."""
    return pd.Timestamp(f'{gameday} {gametime}').tz_localize('America/New_York').tz_convert('UTC')


def _get_json(url, params=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == max_retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def fetch_event_props(event, provider):
    """Every propBets entry ESPN has for one game and book (all pages); [] when the book has none."""
    items, page = [], 1
    while True:
        data = _get_json(ESPN_ODDS_URL.format(event=event) + f'/{provider}/propBets',
                         params={'lang': 'en', 'region': 'us', 'limit': 1000, 'page': page})
        if not data:
            return items
        items += data.get('items', [])
        if page >= data.get('pageCount', 1):
            return items
        page += 1


def _athlete_id(item):
    found = re.search(r'/athletes/(\d+)', (item.get('athlete') or {}).get('$ref', ''))
    return found.group(1) if found else None


def _target(item, which):
    return ((item.get(which) or {}).get('target') or {}).get('value')


def _main_lines(items):
    """One line per (ESPN athlete id, market): {(athlete, market): (line, open line, last updated)}.

    Every over/under line is listed twice (the over and the under, indistinguishable without prices).  ESPN BET's
    feed also lists alternate lines, once each, so the main line is the (line, open line) pair listed most often;
    a player whose most frequent count is shared by two different pairs is skipped as ambiguous.
    """
    counts, stamps = defaultdict(Counter), defaultdict(dict)
    for item in items:
        market = TYPE_TO_MARKET.get(str((item.get('type') or {}).get('id')))
        athlete, line, opened = _athlete_id(item), _target(item, 'current'), _target(item, 'open')
        if market is None or athlete is None or line is None:
            continue
        key, pair = (athlete, market), (float(line), None if opened is None else float(opened))
        counts[key][pair] += 1
        stamps[key][pair] = max(stamps[key].get(pair, ''), item.get('lastUpdated') or '')
    lines = {}
    for key, pairs in counts.items():
        ranked = pairs.most_common()
        (line, opened), top = ranked[0]
        if top < 2 or (len(ranked) > 1 and ranked[1][1] == top):
            continue
        lines[key] = (line, opened, stamps[key][(line, opened)])
    return lines


def event_lines(event, kickoff, providers=(100, 58)):
    """Main prop lines for one game from the first provider in `providers` that has any, as row dicts."""
    for provider in providers:
        items = fetch_event_props(event, provider)
        if items:
            break
    else:
        return []
    rows = []
    for (athlete, market), (line, opened, updated) in _main_lines(items).items():
        stamp = pd.Timestamp(updated) if updated else None
        if stamp is not None and stamp.tzinfo is None:
            stamp = stamp.tz_localize('UTC')
        rows.append({'espn_event': str(event), 'provider': provider, 'espn_athlete': athlete, 'market': market,
                     'line': line, 'open_line': opened, 'last_updated': updated,
                     'pregame': bool(stamp is not None and stamp <= kickoff)})
    return rows


def _game_rows(game):
    kickoff = kickoff_utc(game.gameday, game.gametime)
    rows = event_lines(int(game.espn), kickoff)
    for row in rows:
        row.update(season=int(game.season), week=int(game.week), game_id=game.game_id, kickoff=kickoff.isoformat())
    return rows


def _history_rows(game):
    """(rows, fetched) for one finished game; a failed download is skipped and retried on the next run."""
    try:
        return _game_rows(game), True
    except (requests.RequestException, ValueError) as e:
        logging.warning(f"Could not fetch prop lines for {game.game_id} (will retry next run): {e}")
        return [], False


def load_line_history(games, seasons, refresh_events=False, workers=6):
    """Prop lines of every finished game in `seasons`, cached per season in data_cache/props_espn_{season}.csv.

    Only games not fetched before are downloaded (about 2 requests a game).  `games` is nflverse's schedule
    (nflverse_data.load_games), which carries ESPN's event id for every game.  Returns LINE_COLUMNS.
    """
    frames = []
    for season in seasons:
        path = os.path.join(CACHE_DIR, f'props_espn_{season}.csv')
        done_path = os.path.join(CACHE_DIR, f'props_espn_{season}_events.csv')
        cached = pd.read_csv(path, dtype={'espn_event': str, 'espn_athlete': str}) if os.path.exists(path) else None
        done = set() if refresh_events or not os.path.exists(done_path) else set(pd.read_csv(done_path, dtype=str)['espn_event'])
        finished = games[(games['season'] == season) & games['result'].notna() & games['espn'].notna()]
        todo = [g for g in finished.itertuples() if str(int(g.espn)) not in done]
        if todo:
            logging.info(f"Downloading prop lines for {len(todo)} finished {season} games from ESPN")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fetched = list(pool.map(_history_rows, todo))
            new = pd.DataFrame([row for rows, _ in fetched for row in rows], columns=LINE_COLUMNS)
            if cached is not None and refresh_events:
                cached = cached[~cached['espn_event'].isin(new['espn_event'])]
            cached = new if cached is None else pd.concat([cached, new], ignore_index=True)
            cached.to_csv(path, index=False)
            done |= {str(int(g.espn)) for g, (_, ok) in zip(todo, fetched) if ok}
            pd.DataFrame({'espn_event': sorted(done)}).to_csv(done_path, index=False)
        if cached is not None:
            frames.append(cached)
    if not frames:
        return pd.DataFrame(columns=LINE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def fetch_week_lines(week_games, now=None):
    """Current lines for the games of `week_games` (nflverse schedule rows) that have not kicked off.

    Returns (lines, skipped) where skipped lists the game_ids left out because they are under way or over (their
    lines would be in-game lines).
    """
    now = now or pd.Timestamp.now(tz='UTC')
    rows, skipped = [], []
    for game in week_games.itertuples():
        if pd.isna(game.espn):
            skipped.append(game.game_id)
            continue
        if kickoff_utc(game.gameday, game.gametime) <= now:
            skipped.append(game.game_id)
            continue
        try:
            rows += _game_rows(game)
        except (requests.RequestException, ValueError) as e:
            logging.warning(f"Could not fetch prop lines for {game.game_id}: {e}")
    return pd.DataFrame(rows, columns=LINE_COLUMNS), skipped


def attach_players(lines, players):
    """Adds gsis_id (nflverse's player_id), player name and position from nflverse's players table by ESPN id."""
    known = players.dropna(subset=['espn_id']).copy()
    known['espn_athlete'] = known['espn_id'].astype('int64').astype(str)
    known = known.drop_duplicates('espn_athlete')[['espn_athlete', 'gsis_id', 'display_name', 'position']]
    merged = lines.merge(known, on='espn_athlete', how='left')
    missing = merged['gsis_id'].isna()
    if missing.any():
        logging.info(f"{int(missing.sum())} prop lines are for players nflverse has no ESPN id for; they are skipped")
    return merged[~missing].rename(columns={'display_name': 'player_name'}).reset_index(drop=True)


def append_prop_snapshot(lines):
    """Append the lines as they stand now to NFL\\data\\prop_line_log.csv (timestamped); returns the rows added.

    The ESPN feed keeps only the opening and the latest line, so this log is what shows how a line moved between the
    time a bet was placed and kickoff.
    """
    if lines.empty:
        return 0
    frame = lines.assign(logged_at=datetime.now().isoformat(timespec='seconds'))
    path = snapshot_log_path()
    frame.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return len(frame)


def closing_lines(history):
    """{(game_id, gsis_id, market): closing line} from history rows whose line was last updated before kickoff."""
    pre = history[history['pregame'].astype(str).str.lower() == 'true']
    return {(r.game_id, r.gsis_id, r.market): float(r.line) for r in pre.itertuples()}


def _number(text):
    try:
        return float(str(text).replace('+', ''))
    except (TypeError, ValueError):
        return None


def _game_opener(game):
    """(game_id, opening spread_line, opening total_line) from ESPN's odds for one game; None values if missing.

    spread_line follows nflverse's sign (positive = home favored), the opposite of the home point spread ESPN shows.
    """
    try:
        data = _get_json(ESPN_ODDS_URL.format(event=int(game.espn)), params={'lang': 'en', 'region': 'us'}) or {}
    except (requests.RequestException, ValueError) as e:
        logging.warning(f"Could not fetch the opening game lines for {game.game_id}: {e}")
        return None
    items = {int((item.get('provider') or {}).get('id', 0)): item for item in data.get('items', [])}
    item = items.get(100) or items.get(58)
    if item is None:
        return (game.game_id, None, None)
    opened = item.get('open') or {}
    total = _number(((opened.get('total') or {}).get('american')))
    home = _number((((item.get('homeTeamOdds') or {}).get('open') or {}).get('pointSpread') or {}).get('american'))
    return (game.game_id, None if home is None else -home, total)


def load_game_openers(games, seasons, workers=6):
    """Opening spread and total of every game in `seasons` from ESPN, cached in data_cache/game_openers.csv.

    Returns columns game_id, open_spread_line (positive = home favored, as nflverse's spread_line), open_total_line.
    Games not cached yet (or cached without a line, e.g. not posted yet) are fetched; the opening line never changes.
    """
    path = os.path.join(CACHE_DIR, 'game_openers.csv')
    cached = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=['game_id', 'open_spread_line', 'open_total_line'])
    have = set(cached.dropna(subset=['open_total_line'])['game_id'])
    todo = [g for g in games[games['season'].isin(seasons) & games['espn'].notna()].itertuples() if g.game_id not in have]
    if todo:
        logging.info(f"Downloading opening game lines for {len(todo)} games from ESPN")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            found = [row for row in pool.map(_game_opener, todo) if row is not None]
        fresh = pd.DataFrame(found, columns=['game_id', 'open_spread_line', 'open_total_line'])
        cached = pd.concat([cached[~cached['game_id'].isin(fresh['game_id'])], fresh], ignore_index=True)
        cached.to_csv(path, index=False)
    return cached
