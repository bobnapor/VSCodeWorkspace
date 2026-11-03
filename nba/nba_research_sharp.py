"""How sharps beat NBA lines, tested: pre-registered hypotheses (written November 2, 2026, before any of their results
had been looked at), each with its rule fixed in advance from a mechanism, not tuned, and judged at the prices
actually offered.

The models of nba_research_validate.py and nba_research_props.py carry real information but do not beat the vig.
Professional NBA bettors mostly do not beat closing lines with public statistics either; their edges come from timing
(betting openers before the market matures), market structure (line shopping, stale books, prices that disagree
between related markets) and niche information (referees, injury news, where the money goes).  The ones free data can
test, with their fixed rules:

  H1  Outlier books (line shopping).  A book whose closing spread is 1+ point better for a side than the median of the
      books (totals: 1.5+): bet that side at that book's line and price.  ESPN's archive (several books a game,
      2013-14 to 2023-24).  H1b: DraftKings against the median of the other books on Action Network (2023-24 on).
  H2  Spread against moneyline at the same book.  The moneyline's no-vig chance that the home team wins against the
      chance its spread implies (a logistic fit of wins on closing spreads, fitted on the early period only); when
      they differ by 3+ points, bet the moneyline the spread makes cheap (H2a) and the spread the moneyline makes
      cheap (H2b), each at that book's price.
  H3  Favorite-longshot bias.  Every favorite at -400 or shorter on the closing consensus moneyline (H3a, the side
      expected to pay) and every underdog at +300 or longer (H3b, for contrast).
  H4  Early-season overs.  Totals are set from last season's scoring while scoring kept rising: the over in the first
      21 days of each regular season, at the closing total.
  H5  Referees.  A referee's tendency is the average over margin (points minus the closing total) of his games in the
      two seasons before, shrunk by 50 games; a crew's is its referees' mean.  Over when the crew's is +1 or more,
      under at -1 or less, at the closing total.
  H6  Beat the opener.  Model B without availability (only what is known the night before: ratings, rest, travel),
      the side it favors by 0.5+ at the opening spread, and the over or under it favors by 1+ at the opening total
      (-110: the archive has no opening prices).  Also: does it predict the line's move?
  H7  Follow the money (Action Network's splits, 2023-24 on), at the closing consensus line: (a) reverse line movement,
      the line moving 0.5+ toward the side with 40% of the tickets or fewer; (b) money against tickets, a side with 15+
      points more of the money than of the tickets; (c) reverse line movement on totals.
  H8  Overreaction.  Fade a team that covered its last game by 15+; back one that missed it by 15+.

Periods: EARLY (2013-14 to 2018-19) and LATE (2019-20 to 2025-26), or what each hypothesis's data allows (shown).  A
rule passes only if it returns money at actual prices in both periods and in the later one with z >= 2 against zero
(about a dozen rules: one fluke at z >= 2 is not unlikely, so both periods must agree).

    python nba_research_sharp.py

Results (November 2, 2026): none passes.  Prices of -100 to +100 are placeholders, not prices (5% of the early
archive's moneylines were 0, Bovada's and Wynn's above all) and were dropped before the final run; with them, H2a had
looked like +11.5% a bet in the early period.
  H1  a book 1+ off the median won 54-56% at its spread and 57-58% at its total, but at its own price -0.9% to +2.0%
      a bet: books charge for their outlying numbers, so at the close there is no free lunch.  H1b: DraftKings is
      rarely 1+ off the other books at the close (96 games in two seasons, lost); the best of the other books beats
      DraftKings' spread by about 0.4 points on average, the value of shopping.
  H2  -2.4% to -7.4% a bet in three of four cells; the moneylines and spreads agree well enough.
  H3  heavy favorites -3.3% / -4.4%, longshots -4.7% / -6.4%: no favorite-longshot edge after the vig.
  H4  early-season overs -4.7% / -7.3%: the totals do not lag.
  H5  referee crews -1.4% / -1.8%.
  H6  the model at the opening spread +1.1% (1,419 bets) / +1.8% (2,117), its lean moving with the line (r = +0.05):
      the right direction in both periods, as betting early should be, but far from significant (z 0.4 / 0.9) and at an
      assumed -110 (the archive has no opening prices).  Totals at the opener -4.9% / +3.3%.
  H7  reverse line movement -2.6% / +2.0% (spreads), -2.2% / +0.4% (totals); money 15+ points ahead of the tickets
      +18.3% (33 bets) / +13.0% (44 bets), about 60% won: the right direction twice but too rare to tell (z 1.1 / 0.9).
  H8  fading blowout covers -5.3% / -2.9%.
The two that point the right way in both periods (H6, H7b) are worth tracking forward with real prices.
"""

import math
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import nba_data
import nba_lab
import nba_lines
import nba_market
import nba_model

EARLY, LATE = (2014, 2019), (2020, 2026)
SPREAD_EDGE, TOTAL_EDGE = 1.0, 1.5
ML_GAP = 0.03
FAVORITE, LONGSHOT = -400, 300
EARLY_DAYS = 21
REF_SHRINK, REF_EDGE = 50, 1.0
OPEN_SPREAD_LEAN, OPEN_TOTAL_LEAN = 0.5, 1.0
TICKETS_MAX, MOVE_MIN, MONEY_GAP = 40, 0.5, 15
BLOWOUT = 15
AVAILABILITY = ('missing', 'returning', 'playing', 'missing_minutes')


def implied(odds):
    odds = np.asarray(odds, dtype=float)
    odds = np.where(np.abs(odds) < 100, np.nan, odds)
    return np.where(odds < 0, -odds / (-odds + 100), 100 / (np.abs(odds) + 100))


def profit(won, push, odds):
    """Units won per unit staked at American `odds` (-110 where there is no price)."""
    odds = np.asarray(pd.to_numeric(pd.Series(odds), errors='coerce').fillna(-110), dtype=float)
    odds = np.where(np.abs(odds) < 100, -110, odds)
    win = np.where(odds < 0, 100 / np.abs(odds), odds / 100)
    return np.where(push, 0.0, np.where(won, win, -1.0))


def bets(period, won, push, odds):
    return pd.DataFrame({'period': period, 'won': np.asarray(won, bool), 'push': np.asarray(push, bool),
                         'profit': profit(np.asarray(won, bool), np.asarray(push, bool), odds)})


def period_of(season, early=EARLY, late=LATE):
    season = np.asarray(season)
    return np.where((season >= early[0]) & (season <= early[1]), 'early',
                    np.where((season >= late[0]) & (season <= late[1]), 'late', None))


def report(name, frame, results):
    """One line per period: bets, win rate (pushes out), return a bet at the prices and its z; adds to `results`."""
    line = [f'{name}:']
    passed, rois = True, {}
    for period in ('early', 'late'):
        g = frame[frame['period'] == period]
        n = len(g)
        if not n:
            line.append(f'{period} -')
            passed = False
            continue
        decided = g[~g['push']]
        roi, sd = g['profit'].mean(), g['profit'].std(ddof=1) if n > 1 else float('nan')
        z = roi / (sd / math.sqrt(n)) if sd and sd > 0 else float('nan')
        rois[period] = (roi, z)
        line.append(f'{period} {n} bets, won {decided["won"].mean():.1%}, {roi:+.1%} a bet (z {z:+.2f})')
        if roi <= 0:
            passed = False
    if 'late' in rois and not (rois['late'][1] >= 2):
        passed = False
    print('  ' + ' | '.join(line) + ('  PASSES' if passed else ''), flush=True)
    results.append((name, passed))


# ---------------------------------------------------------------- the data

def finals():
    table = nba_lab.default_table()
    return table[table['margin'].notna()].copy()


def espn_books(seasons):
    """ESPN's archive, one row per game and sportsbook (no aggregators, openers or in-game lines), numbers as numbers."""
    odds = pd.concat([nba_lines.read_odds(s) for s in seasons], ignore_index=True)
    odds = odds[odds['provider_id'].notna() & ~odds['provider'].astype(str).str.contains('live', case=False)].copy()
    odds['provider_id'] = pd.to_numeric(odds['provider_id'], errors='coerce')
    odds = odds[~odds['provider_id'].isin(list(nba_lines.AGGREGATORS) + [nba_lines.OPENING_PROVIDER])]
    for column in ('home_spread', 'total', 'home_spread_odds', 'away_spread_odds', 'over_odds', 'under_odds',
                   'home_moneyline', 'away_moneyline'):
        odds[column] = pd.to_numeric(odds[column], errors='coerce')
    for column in ('home_spread_odds', 'away_spread_odds', 'over_odds', 'under_odds', 'home_moneyline',
                   'away_moneyline'):
        odds.loc[odds[column].abs() < 100, column] = np.nan        # no American price lies between -100 and +100
    odds.loc[odds['home_spread'].abs() > nba_lines.SPREAD_LIMIT, 'home_spread'] = np.nan
    odds.loc[~odds['total'].between(*nba_lines.TOTAL_RANGE), 'total'] = np.nan
    return odds


def spread_bets(rows, side_home, line, odds_home, odds_away, period):
    """Bets on the home (side_home True) or away side at `line` (the home spread)."""
    cover = rows['margin'].to_numpy() + np.asarray(line, float)
    sign = np.where(side_home, 1, -1)
    return bets(period, sign * cover > 0, cover == 0, np.where(side_home, odds_home, odds_away))


def total_bets(rows, over, line, odds_over, odds_under, period):
    diff = rows['points'].to_numpy() - np.asarray(line, float)
    sign = np.where(over, 1, -1)
    return bets(period, sign * diff > 0, diff == 0, np.where(over, odds_over, odds_under))


# ---------------------------------------------------------------- the hypotheses

def h1_outlier_books(games, results):
    print('\nH1  Outlier books (line shopping), ESPN archive, 2013-14 to 2023-24', flush=True)
    odds = espn_books(range(2014, 2025))
    odds = odds.merge(games[['game_id', 'margin', 'points']], on='game_id')
    counts = odds.groupby('game_id')['provider_id'].transform('nunique')
    odds = odds[counts >= 4]
    odds['med_spread'] = odds.groupby('game_id')['home_spread'].transform('median')
    odds['med_total'] = odds.groupby('game_id')['total'].transform('median')
    dev = odds['home_spread'] - odds['med_spread']
    pick = odds[dev.abs() >= SPREAD_EDGE]
    home = (pick['home_spread'] - pick['med_spread']).to_numpy() > 0      # more points for the home side
    report('H1  spreads, the book 1+ better than the median', spread_bets(
        pick, home, pick['home_spread'], pick['home_spread_odds'], pick['away_spread_odds'],
        period_of(pick['season'])), results)
    tdev = odds['total'] - odds['med_total']
    pick = odds[tdev.abs() >= TOTAL_EDGE]
    over = (pick['total'] - pick['med_total']).to_numpy() < 0              # a low total: the over
    report('H1  totals, the book 1.5+ off the median', total_bets(
        pick, over, pick['total'], pick['over_odds'], pick['under_odds'], period_of(pick['season'])), results)


def h1b_draftkings(games, results):
    print('\nH1b DraftKings against the other books (Action Network; early 2023-24 and 2024-25, late 2025-26)',
          flush=True)
    an = nba_market.load_history(range(2024, 2027), progress=False)
    for column in ('home_spread', 'home_spread_odds', 'away_spread_odds', 'total', 'over_odds', 'under_odds'):
        an[column] = pd.to_numeric(an[column], errors='coerce')
    others = an[~an['book'].isin([15, 30, 67, 68, 1534, 1548])]
    market = others.groupby('game_id').agg(med_spread=('home_spread', 'median'), med_total=('total', 'median'),
                                           books=('book', 'nunique'))
    dk = an[an['book'] == 68].merge(market[market['books'] >= 4], on='game_id').merge(
        games[['game_id', 'margin', 'points']], on='game_id')
    period = np.where(dk['season'] <= 2025, 'early', np.where(dk['season'] == 2026, 'late', None))
    dev = dk['home_spread'] - dk['med_spread']
    keep = (dev.abs() >= SPREAD_EDGE).to_numpy()
    pick = dk[keep]
    report('H1b spreads, DraftKings 1+ better than the others', spread_bets(
        pick, (dev[keep] > 0).to_numpy(), pick['home_spread'], pick['home_spread_odds'], pick['away_spread_odds'],
        period[keep]), results)
    tdev = dk['total'] - dk['med_total']
    keep = (tdev.abs() >= TOTAL_EDGE).to_numpy()
    pick = dk[keep]
    report('H1b totals, DraftKings 1.5+ off the others', total_bets(
        pick, (tdev[keep] < 0).to_numpy(), pick['total'], pick['over_odds'], pick['under_odds'], period[keep]), results)
    # what shopping is worth on average: the best of the books' home spreads against DraftKings'
    shop = others.groupby('game_id')['home_spread'].agg(['max', 'min']).join(
        an[an['book'] == 68].set_index('game_id')['home_spread'].rename('dk'), how='inner')
    print(f'      (shopping: the best home spread beats DraftKings\' by {(shop["max"] - shop["dk"]).mean():.2f} points '
          f'on average, the best away spread by {(shop["dk"] - shop["min"]).mean():.2f})', flush=True)


def h2_spread_vs_moneyline(games, results):
    print('\nH2  Spread against moneyline at the same book, ESPN archive', flush=True)
    early = games[games['season'].between(*EARLY) & games['close_home_spread'].notna()]
    fit = LogisticRegression(C=1e6).fit(early[['close_home_spread']].to_numpy(), (early['margin'] > 0).to_numpy())
    odds = espn_books(range(2014, 2027))
    odds = odds[odds['home_moneyline'].notna() & odds['away_moneyline'].notna() & odds['home_spread'].notna()]
    odds = odds.merge(games[['game_id', 'margin']], on='game_id')
    p_home, p_away = implied(odds['home_moneyline']), implied(odds['away_moneyline'])
    p_ml = p_home / (p_home + p_away)
    p_spread = fit.predict_proba(odds[['home_spread']].to_numpy())[:, 1]
    gap = p_ml - p_spread
    keep = np.abs(gap) >= ML_GAP
    pick, gap = odds[keep], gap[keep]
    period = period_of(pick['season'])
    # gap > 0: the moneyline rates the home team higher than its spread does, so the away moneyline is cheap against
    # the spread market and the home spread cheap against the moneyline market
    away_ml = gap > 0
    won = np.where(away_ml, pick['margin'] < 0, pick['margin'] > 0)
    report('H2a the moneyline the spread makes cheap', bets(
        period, won, pick['margin'] == 0, np.where(away_ml, pick['away_moneyline'], pick['home_moneyline'])), results)
    report('H2b the spread the moneyline makes cheap', spread_bets(
        pick, gap > 0, pick['home_spread'], pick['home_spread_odds'], pick['away_spread_odds'], period), results)


def h3_favorite_longshot(games, results):
    print('\nH3  Favorite-longshot bias, consensus closing moneylines', flush=True)
    g = games[games['home_moneyline'].notna() & games['away_moneyline'].notna()]
    period = period_of(g['season'])
    for name, pick_home, pick_away in (
            ('H3a favorites at -400 or shorter', g['home_moneyline'] <= FAVORITE, g['away_moneyline'] <= FAVORITE),
            ('H3b underdogs at +300 or longer', g['home_moneyline'] >= LONGSHOT, g['away_moneyline'] >= LONGSHOT)):
        home, away = g[pick_home], g[pick_away]
        frame = pd.concat([
            bets(period[pick_home.to_numpy()], home['margin'] > 0, home['margin'] == 0, home['home_moneyline']),
            bets(period[pick_away.to_numpy()], away['margin'] < 0, away['margin'] == 0, away['away_moneyline'])])
        report(name, frame, results)


def h4_early_overs(games, results):
    print(f'\nH4  Overs in the first {EARLY_DAYS} days of a season, at the closing total', flush=True)
    g = games[games['close_total'].notna()].copy()
    first = g.groupby('season')['tipoff'].transform('min').dt.normalize()
    g = g[g['tipoff'] < first + pd.Timedelta(days=EARLY_DAYS)]
    report('H4  early-season overs', total_bets(g, np.ones(len(g), bool), g['close_total'], g['over_odds'],
                                                g['under_odds'], period_of(g['season'])), results)


def h5_referees(games, results):
    print('\nH5  Referees (their over margin in the two seasons before), at the closing total; early 2015-16 to '
          '2018-19', flush=True)
    refs = nba_data.load_officials(range(2014, 2027))
    g = games[games['close_total'].notna()][['game_id', 'season', 'points', 'close_total', 'over_odds',
                                             'under_odds']].copy()
    g['over_margin'] = g['points'] - g['close_total']
    crews = refs.merge(g[['game_id', 'season', 'over_margin']], on=['game_id', 'season'])
    tendencies = {}
    for season in range(2016, 2027):
        past = crews[crews['season'].between(season - 2, season - 1)]
        stats = past.groupby('official_full_name')['over_margin'].agg(['sum', 'count'])
        tendencies[season] = (stats['sum'] / (stats['count'] + REF_SHRINK)).to_dict()
    this = refs[refs['season'] >= 2016].copy()
    this['tendency'] = [tendencies[s].get(name, 0.0) for s, name in zip(this['season'], this['official_full_name'])]
    crew = this.groupby('game_id')['tendency'].mean().rename('crew')
    g = g.merge(crew, on='game_id')
    pick = g[g['crew'].abs() >= REF_EDGE]
    report('H5  over/under by the crew', total_bets(
        pick, (pick['crew'] > 0).to_numpy(), pick['close_total'], pick['over_odds'], pick['under_odds'],
        period_of(pick['season'], early=(2016, 2019))), results)
    print(f'      (crews at +-{REF_EDGE:g}: {len(pick)} of {len(g)} games; crew tendencies range '
          f'{g["crew"].min():+.2f} to {g["crew"].max():+.2f})', flush=True)


def h6_opener(table, results):
    print('\nH6  Beat the opener: Model B without availability, at the opening line (early 2015-16 and 2016-17, late '
          '2023-24 to 2025-26)', flush=True)
    spread_features = [f for f in nba_model.SPREAD_FEATURES if f.replace('d_', '') not in AVAILABILITY]
    total_features = [f for f in nba_model.TOTAL_FEATURES if f.replace('sum_', '') not in AVAILABILITY]
    s = nba_model.scored(table, 'cover', 1000, 2016, 2026, features=spread_features)
    s = s[s['margin'].notna() & s['open_home_spread'].notna() & s['close_home_spread'].notna()]
    period = period_of(s['season'], early=(2016, 2017), late=(2024, 2026))
    keep = (np.abs(s['pred']) >= OPEN_SPREAD_LEAN).to_numpy() & (period != None)   # noqa: E711
    pick = s[keep]
    report('H6  spreads at the opener', spread_bets(pick, (pick['pred'] > 0).to_numpy(), pick['open_home_spread'],
                                                    -110, -110, period[keep]), results)
    move = s['close_home_spread'] - s['open_home_spread']
    r = np.corrcoef(s['pred'], -move)[0, 1]
    print(f'      (the correction against the line\'s move toward the home side: r = {r:+.3f}, {len(s)} games)', flush=True)
    t = nba_model.scored(table, 'over', 10000, 2016, 2026, features=total_features)
    t = t[t['points'].notna() & t['open_total'].notna()]
    period = period_of(t['season'], early=(2016, 2017), late=(2024, 2026))
    keep = (np.abs(t['pred']) >= OPEN_TOTAL_LEAN).to_numpy() & (period != None)    # noqa: E711
    pick = t[keep]
    report('H6  totals at the opener', total_bets(pick, (pick['pred'] > 0).to_numpy(), pick['open_total'], -110, -110,
                                                  period[keep]), results)


def h7_money(games, results):
    print('\nH7  Follow the money (Action Network splits; early 2023-24 and 2024-25, late 2025-26), at the closing '
          'consensus line', flush=True)
    an = nba_market.load_history(range(2024, 2027), progress=False)
    for column in an.columns[3:]:
        an[column] = pd.to_numeric(an[column], errors='coerce')
    close = an[an['book'] == 15].set_index('game_id')
    opened = an[an['book'] == 30].set_index('game_id')[['home_spread', 'total']].add_prefix('open_')
    g = close.join(opened, how='inner').join(games.set_index('game_id')[['margin', 'points']], how='inner')
    g = g.reset_index()
    period = np.where(g['season'] <= 2025, 'early', np.where(g['season'] == 2026, 'late', None))
    move = g['home_spread'] - g['open_home_spread']                # negative: the line moved toward the home side
    home_rlm = (g['tickets_home'] <= TICKETS_MAX) & (move <= -MOVE_MIN)
    away_rlm = (g['tickets_home'] >= 100 - TICKETS_MAX) & (move >= MOVE_MIN)
    keep = (home_rlm | away_rlm).to_numpy()
    report('H7a reverse line movement, spreads', spread_bets(
        g[keep], home_rlm[keep].to_numpy(), g.loc[keep, 'home_spread'], g.loc[keep, 'home_spread_odds'],
        g.loc[keep, 'away_spread_odds'], period[keep]), results)
    gap = g['money_home'] - g['tickets_home']
    keep = (gap.abs() >= MONEY_GAP).to_numpy()
    report('H7b money against tickets, spreads', spread_bets(
        g[keep], (gap[keep] > 0).to_numpy(), g.loc[keep, 'home_spread'], g.loc[keep, 'home_spread_odds'],
        g.loc[keep, 'away_spread_odds'], period[keep]), results)
    tmove = g['total'] - g['open_total']
    over_rlm = (g['tickets_over'] <= TICKETS_MAX) & (tmove >= MOVE_MIN)
    under_rlm = (g['tickets_over'] >= 100 - TICKETS_MAX) & (tmove <= -MOVE_MIN)
    keep = (over_rlm | under_rlm).to_numpy()
    report('H7c reverse line movement, totals', total_bets(
        g[keep], over_rlm[keep].to_numpy(), g.loc[keep, 'total'], g.loc[keep, 'over_odds'], g.loc[keep, 'under_odds'],
        period[keep]), results)


def h8_overreaction(games, results):
    print(f'\nH8  Overreaction: fade a team after a {BLOWOUT}+ point cover, back it after a {BLOWOUT}+ point miss',
          flush=True)
    g = games[games['close_home_spread'].notna()].sort_values('start')
    sides = pd.concat([
        pd.DataFrame({'game_id': g['game_id'], 'season': g['season'], 'start': g['start'], 'team': g['home_id'],
                      'home': True, 'team_cover': g['cover']}),
        pd.DataFrame({'game_id': g['game_id'], 'season': g['season'], 'start': g['start'], 'team': g['away_id'],
                      'home': False, 'team_cover': -g['cover']})]).sort_values(['team', 'start'])
    sides['last'] = sides.groupby(['team', 'season'])['team_cover'].shift()
    sides = sides[sides['last'].abs() >= BLOWOUT].merge(
        g[['game_id', 'margin', 'close_home_spread', 'home_spread_odds', 'away_spread_odds']], on='game_id')
    back = (sides['last'] <= -BLOWOUT).to_numpy()
    on_home = np.where(back, sides['home'], ~sides['home'])
    report('H8  fade the blowout cover, back the blowout miss', spread_bets(
        sides, on_home, sides['close_home_spread'], sides['home_spread_odds'], sides['away_spread_odds'],
        period_of(sides['season'])), results)


def main():
    table = nba_lab.default_table()
    games = table[table['margin'].notna()].copy()
    results = []
    for test in (h1_outlier_books, h1b_draftkings, h2_spread_vs_moneyline, h3_favorite_longshot, h4_early_overs,
                 h5_referees, h7_money, h8_overreaction):
        test(games, results)
    h6_opener(table, results)
    passed = [name for name, ok in results if ok]
    print('\nPassed (profitable at actual prices in both periods, z >= 2 in the later): '
          + (', '.join(passed) if passed else 'none'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
