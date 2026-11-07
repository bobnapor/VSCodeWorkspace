"""College football spreads and totals: every game's lines from many sportsbooks since 2006, as cfbfastR publishes them
(cfb_line_odds, from CollegeFootballData's archive), cached in cfb/data_cache.

The file has one row per side per book: a spread row for each team (that team's line and price), an over and an under
row for the total, and the moneylines.  Teams are named by abbreviations that do not match ESPN's, and a book's two
rows come in no fixed order, so each season every abbreviation is matched to an ESPN team id by voting
(abbreviation_teams): in games where ESPN's own spread (from the play-by-play) is CLEAR_SPREAD points or more, the row
with ESPN's sign is the home team's.  An abbreviation that points to one team in at least MIN_ABBR_SHARE of its votes
is kept (about 98% of them reach 95%), and a book's spread is the line of the row whose abbreviation is the home team.
A row whose abbreviation is not matched falls back to ESPN's sign when ESPN's spread is clear, and is dropped otherwise.

closing_lines() reduces the books to one line per game: the median across sportsbooks, leaving out the aggregators and
projection sites (consensus, teamrankings, numberfire), which is steadier than any single book.  Pinnacle (2010-2019),
the sharpest book in the archive, is also kept on its own, and the opening line where a book reported it (DraftKings,
Bovada and ESPN Bet from 2023).  Lines are as the books stood at kickoff, so they are closing lines.
"""

import numpy as np
import pandas as pd

import cfb_data

ODDS_URL = 'https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/betting/csv/cfb_line_odds.csv.gz'
FIRST_LINES_SEASON = 2006
AGGREGATORS = ('consensus', 'teamrankings', 'numberfire')
SHARP_BOOK = 'PINNACLE'
OPENING_BOOKS = ('DraftKings', 'Draft Kings', 'Bovada', 'ESPN Bet')
CLEAR_SPREAD = 3.0              # ESPN's spread at least this far from 0 says which row is the home team's
MIN_ABBR_SHARE = 0.8
SPREAD_LIMIT, TOTAL_RANGE = 70, (25, 110)
LINE_COLUMNS = ['game_id', 'season', 'close_home_spread', 'close_total', 'books_spread', 'books_total',
                'pinnacle_home_spread', 'pinnacle_total', 'home_spread_odds', 'over_odds', 'open_home_spread',
                'open_total', 'espn_home_spread', 'espn_total']


def load_odds(refresh=False):
    """Every book's line rows (spread, total), one row per side per book."""
    path = cfb_data.cached_download(ODDS_URL, 'cfb_line_odds.csv.gz', refresh)
    odds = pd.read_csv(path, low_memory=False, usecols=['game_id', 'season', 'market_type', 'abbr', 'lines', 'odds',
                                                        'opening_lines', 'opening_odds', 'book'])
    odds = odds[odds['market_type'].isin(('spread', 'total')) & odds['game_id'].notna() & odds['lines'].notna()]
    odds['game_id'] = odds['game_id'].astype('int64')
    odds['season'] = odds['season'].astype(int)
    odds['abbr'] = odds['abbr'].astype(str).str.strip()
    odds['book'] = odds['book'].astype(str).str.strip()
    return odds[~odds['book'].str.lower().isin(AGGREGATORS)]


def abbreviation_teams(spread_rows):
    """{(season, abbreviation): ESPN team id}, voted from the games where ESPN's spread is clear.  `spread_rows` are the
    books' spread rows with the game's espn_home_spread, home_id and away_id."""
    clear = spread_rows[spread_rows['espn_home_spread'].abs() >= CLEAR_SPREAD]
    home_side = np.sign(clear['lines']) == np.sign(clear['espn_home_spread'])
    votes = pd.DataFrame({'season': clear['season'], 'abbr': clear['abbr'],
                          'team_id': np.where(home_side, clear['home_id'], clear['away_id'])})
    counts = votes.groupby(['season', 'abbr', 'team_id']).size().rename('n').reset_index()
    counts['share'] = counts['n'] / counts.groupby(['season', 'abbr'])['n'].transform('sum')
    top = counts.sort_values('n', ascending=False).drop_duplicates(['season', 'abbr'])
    top = top[top['share'] >= MIN_ABBR_SHARE]
    return {(s, a): t for s, a, t in zip(top['season'], top['abbr'], top['team_id'])}


def _book_spreads(odds, espn):
    """One row per (game, book): the home spread, its price and opening line."""
    sp = odds[odds['market_type'] == 'spread']
    sp = sp.merge(espn[['game_id', 'espn_home_spread', 'home_id', 'away_id']], on='game_id', how='inner')
    teams = abbreviation_teams(sp)
    team = pd.Series([teams.get(key) for key in zip(sp['season'], sp['abbr'])], index=sp.index, dtype=float)
    by_name = team.eq(sp['home_id'])
    named = team.eq(sp['home_id']) | team.eq(sp['away_id'])
    by_espn = ~named & (sp['espn_home_spread'].abs() >= CLEAR_SPREAD) \
        & (np.sign(sp['lines']) == np.sign(sp['espn_home_spread']))
    home = sp[by_name | by_espn]
    home = home[home['lines'].abs() <= SPREAD_LIMIT].drop_duplicates(['game_id', 'book'])
    return home.rename(columns={'lines': 'home_spread', 'odds': 'home_spread_odds', 'opening_lines': 'open_home_spread'})[
        ['game_id', 'season', 'book', 'home_spread', 'home_spread_odds', 'open_home_spread']]


def _book_totals(odds):
    """One row per (game, book): the total, the over's price and the opening total."""
    over = odds[(odds['market_type'] == 'total') & (odds['abbr'].astype(str).str.lower() == 'over')]
    over = over[over['lines'].between(*TOTAL_RANGE)].drop_duplicates(['game_id', 'book'])
    return over.rename(columns={'lines': 'total', 'odds': 'over_odds', 'opening_lines': 'open_total'})[
        ['game_id', 'season', 'book', 'total', 'over_odds', 'open_total']]


def closing_lines(espn_games, refresh=False):
    """One row per game with a line: the median home spread and total across sportsbooks, how many books each, the
    Pinnacle lines, the median price of the home spread and the over, the median opening lines, and ESPN's own lines.
    `espn_games` has game_id, home_id, away_id, espn_home_spread and espn_total (cfb_data's reduced play-by-play games)."""
    odds = load_odds(refresh)
    odds = odds[odds['season'].isin(espn_games['season'].unique())]
    spreads = _book_spreads(odds, espn_games)
    totals = _book_totals(odds)
    sp = spreads.groupby('game_id').agg(season=('season', 'first'), close_home_spread=('home_spread', 'median'),
                                        books_spread=('home_spread', 'size'),
                                        home_spread_odds=('home_spread_odds', 'median'),
                                        open_home_spread=('open_home_spread', 'median'))
    tt = totals.groupby('game_id').agg(close_total=('total', 'median'), books_total=('total', 'size'),
                                       over_odds=('over_odds', 'median'), open_total=('open_total', 'median'))
    pin_sp = spreads[spreads['book'] == SHARP_BOOK].set_index('game_id')['home_spread'].rename('pinnacle_home_spread')
    pin_tt = totals[totals['book'] == SHARP_BOOK].set_index('game_id')['total'].rename('pinnacle_total')
    lines = sp.join(tt, how='outer').join(pin_sp).join(pin_tt).reset_index()
    seasons = odds.drop_duplicates('game_id').set_index('game_id')['season']
    lines['season'] = lines['season'].fillna(lines['game_id'].map(seasons))
    lines = lines.merge(espn_games[['game_id', 'espn_home_spread', 'espn_total']], on='game_id', how='left')
    for column in ('close_home_spread', 'open_home_spread', 'pinnacle_home_spread'):
        lines[column] = (lines[column] * 2).round() / 2      # medians of an even count land on quarter points
    for column in ('close_total', 'open_total', 'pinnacle_total'):
        lines[column] = (lines[column] * 2).round() / 2
    return lines.reindex(columns=LINE_COLUMNS)


def coverage(lines):
    """Games with a closing spread and total, and how often ESPN's line equals the books' median, by season."""
    g = lines.groupby('season')
    return pd.DataFrame({
        'games': g.size(), 'spread': g['close_home_spread'].count(), 'total': g['close_total'].count(),
        'books': g['books_spread'].median(), 'pinnacle': g['pinnacle_home_spread'].count(),
        'opening': g['open_home_spread'].count(),
        'espn_same_spread': g.apply(lambda x: np.mean(np.abs(x['close_home_spread'] - x['espn_home_spread']) <= 0.5)),
    })
