"""Can the underdog signal supply at least N bets every week, and how well do those bets hit?

Uses the same walk-forward corrections as nfl_research_signal.py.  Each week's games are ranked and the top N are
bet at flat -110.  Rankings compared: underdog-side correction only, any side, and underdog-only with a minimum
spread size.  Run it from this folder:  python nfl_research_weekly.py
"""

import numpy as np
import pandas as pd

import nfl_lab as lab
from nfl_research_signal import FEATURES, picks, roi

TOP_N = (3, 4, 5, 6)


def weekly_top(frame, n, dog_only=True, min_spread=0.0):
    """Top-n games per season-week by correction size (underdog side, optionally with a minimum spread)."""
    f = frame[(frame['game_type'] == 'REG') & (frame['spread_line'].abs() >= min_spread)] if 'game_type' in frame else frame
    if dog_only:
        f = f[f['dog']]
    f = f.sort_values('conf', ascending=False)
    return f.groupby(['season', 'week']).head(n)


def report(name, g, weeks):
    d, h = g[g['season'] <= lab.DISCOVERY_END], g[g['season'] > lab.DISCOVERY_END]
    recent = g[g['season'] >= 2018]
    se = np.sqrt(g['win'].mean() * (1 - g['win'].mean()) / len(g))
    print('%-34s n=%5d (%.1f/wk)  win %.3f +/-%.3f  ROI %+.1f%% | discovery %.3f holdout %.3f' % (
        name, len(g), len(g) / weeks, g['win'].mean(), 1.96 * se, roi(g['win'].mean()) * 100, d['win'].mean(), h['win'].mean()))
    return recent


def main():
    table = lab.default_table()
    frame = lab.walk_forward(table, lab.ridge_margin(FEATURES, alpha=10000.0, target='resid'))
    meta = table[['season', 'week', 'home_team', 'away_team', 'game_type']].drop_duplicates(['season', 'week', 'home_team', 'away_team'])
    frame = frame.merge(meta, on=['season', 'week', 'home_team', 'away_team'], how='left', suffixes=('', '_m')) if 'home_team' in frame else frame
    f = picks(frame)
    f = f[f['season'] >= 2006]
    weeks = f.groupby(['season', 'week']).ngroups
    print('walk-forward games', len(f), 'in', weeks, 'season-weeks | break-even at -110 is 52.38%\n')

    print('=== top N per week ===')
    for n in TOP_N:
        report('underdog side, top %d' % n, weekly_top(f, n), weeks)
        report('either side, top %d' % n, weekly_top(f, n, dog_only=False), weeks)
        for ms in (3.0, 4.0):
            report('underdog, spread >= %g, top %d' % (ms, n), weekly_top(f, n, min_spread=ms), weeks)
        print()

    print('=== what each rank slot earns (underdog side, by rank within the week) ===')
    dogs = f[f['dog']].copy()
    dogs['rank'] = dogs.groupby(['season', 'week'])['conf'].rank(ascending=False, method='first')
    for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 20)):
        g = dogs[dogs['rank'].between(lo, hi)]
        print('ranks %2d-%2d: n=%5d  win %.3f  mean correction %.2f' % (lo, hi, len(g), g['win'].mean(), g['conf'].mean()))

    print('\n=== how many weeks already have 4+ games at correction >= 0.5 ===')
    counts = dogs[dogs['conf'] >= 0.5].groupby(['season', 'week']).size().reindex(f.groupby(['season', 'week']).size().index, fill_value=0)
    print(counts.value_counts().sort_index().to_string())

    print('\n=== by season, top 4 underdog picks ===')
    t4 = weekly_top(f, 4)
    by = t4.groupby('season')['win'].agg(['mean', 'size'])
    print(by.round(3).T.to_string())
    print('seasons above 50%%: %d of %d; above 52.38%%: %d' % ((by['mean'] > 0.5).sum(), len(by), (by['mean'] > 0.5238).sum()))

    print('\n=== simulated weekly results with top 4 picks (win rate from the 2018+ holdout) ===')
    p = t4[t4['season'] >= 2018]['win'].mean()
    rng = np.random.default_rng(7)
    wins = rng.binomial(4, p, 100000)
    print('win rate %.3f: P(at least 3 of 4) = %.2f, P(2 of 4) = %.2f, P(0-1 of 4) = %.2f' % (
        p, (wins >= 3).mean(), (wins == 2).mean(), (wins <= 1).mean()))


if __name__ == '__main__':
    main()
