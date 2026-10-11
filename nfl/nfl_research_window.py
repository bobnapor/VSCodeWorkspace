"""Training-window and model-agreement tests for the weekly picks.

1. Does fitting on only the last few seasons help?  (correction model and margin model, several windows)
2. When the margin model (fair line from team stats) and the correction model point at the same side of the
   spread, do those games hit more often, and how many a week are there?
Everything is walk-forward.  Run it from this folder:  python nfl_research_window.py
"""

import numpy as np
import pandas as pd

import nfl_lab as lab
from nfl_research_signal import FEATURES, roi

WINDOWS = (3, 5, 8, 14)
ALPHAS = (3000.0, 10000.0, 30000.0)


def line(label, g, weeks):
    if g.empty:
        return
    p = g['win'].mean()
    d, h = g[g['season'] <= lab.DISCOVERY_END], g[g['season'] > lab.DISCOVERY_END]
    print('%-40s n=%5d (%.1f/wk)  win %.3f  ROI %+5.1f%% | discovery %.3f (n=%d) holdout %.3f (n=%d)' % (
        label, len(g), len(g) / weeks, p, roi(p) * 100, d['win'].mean(), len(d), h['win'].mean(), len(h)))


def add_sides(frame, col):
    """win / dog flags for betting the side that `col` (points toward home covering) favors."""
    f = frame[frame['resid'] != 0].copy()
    f['win'] = np.where(f[col] > 0, f['resid'] > 0, f['resid'] < 0)
    f['dog'] = (f[col] > 0) == (f['spread_line'] < 0)
    f['conf'] = f[col].abs()
    return f


def top_per_week(f, n):
    return f.sort_values('conf', ascending=False).groupby(['season', 'week']).head(n)


def main():
    table = lab.default_table()
    table = table[table['season'] >= 2002]
    weeks = table[(table['season'] >= 2006) & (table['game_type'] == 'REG') & table['spread_line'].notna() & table['result'].notna()].groupby(['season', 'week']).ngroups

    print('=== 1. training window, correction model (underdog-side picks; correction >= 0.5, and top 4 a week) ===')
    corr_frames = {}
    for w in WINDOWS:
        for a in ALPHAS:
            frame = lab.walk_forward(table, lab.ridge_margin(FEATURES, alpha=a, target='resid'), window=w)
            corr_frames[(w, a)] = frame
            f = add_sides(frame, 'pred')
            dogs = f[f['dog']]
            line('window %2d alpha %6g dog >= 0.5' % (w, a), dogs[dogs['conf'] >= 0.5], weeks)
            line('window %2d alpha %6g dog top 4' % (w, a), top_per_week(dogs, 4), weeks)
        print()

    print('=== 2. training window, margin model (stats only): MAE against the final result, and the closing spread ===')
    margin_frames = {}
    for w in WINDOWS:
        frame = lab.walk_forward(table, lab.ridge_margin(FEATURES, alpha=300.0, target='result'), window=w)
        margin_frames[w] = frame
        t = frame[frame['season'] >= 2012]
        print('window %2d: MAE model %.3f, closing spread %.3f' % (w, (t['result'] - t['pred']).abs().mean(), (t['result'] - t['spread_line']).abs().mean()))

    print('\n=== 3. agreement: margin model side (fair line vs spread) and correction model side the same ===')
    for w in (5, 14):
        corr = add_sides(corr_frames[(w, 10000.0)], 'pred')[['game_id', 'pred', 'win', 'dog', 'conf', 'season', 'week', 'spread_line', 'resid']].rename(columns={'pred': 'corr', 'conf': 'cconf'})
        m = margin_frames[w][['game_id', 'pred']].rename(columns={'pred': 'margin'})
        both = corr.merge(m, on='game_id')
        both['edge'] = both['margin'] - both['spread_line']   # >0: stats model likes home against the spread
        both['agree'] = np.sign(both['edge']) == np.sign(both['corr'])
        print('--- window %d: agree on %.1f%% of games' % (w, both['agree'].mean() * 100))
        for name, g in (('agree, any side', both[both['agree']]), ('agree, underdog', both[both['agree'] & both['dog']]),
                        ('disagree, underdog', both[~both['agree'] & both['dog']]), ('agree, favorite', both[both['agree'] & ~both['dog']])):
            line(name, g, weeks)
        ag = both[both['agree']].copy()
        ag['conf'] = ag['cconf']
        for n in (3, 4):
            line('agree, any side, top %d by correction' % n, top_per_week(ag, n), weeks)
            line('agree, underdog, top %d by correction' % n, top_per_week(ag[ag['dog']], n), weeks)
        zz = both[both['agree']].copy()
        zz['conf'] = zz['cconf'] / zz['cconf'].std() + zz['edge'].abs() / zz['edge'].abs().std()
        for n in (3, 4):
            line('agree, any side, top %d by combined score' % n, top_per_week(zz, n), weeks)
            line('agree, underdog, top %d by combined score' % n, top_per_week(zz[zz['dog']], n), weeks)
        for q in (0.5, 0.67, 0.8):
            cut = zz['conf'].quantile(q)
            line('agree, any side, score above %.0f pct' % (q * 100), zz[zz['conf'] >= cut], weeks)
            line('agree, underdog, score above %.0f pct' % (q * 100), zz[(zz['conf'] >= cut) & zz['dog']], weeks)


if __name__ == '__main__':
    main()
