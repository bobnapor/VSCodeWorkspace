"""The evidence behind the underdog signal (nfl_signal.py), in one run.

Every test is walk-forward: each season is predicted by a model fitted only on earlier seasons.  The sections
are meant to be read in order, and the last one says what the evidence does and does not support:

1. Information: does the correction explain anything about covers beyond the spread?
2. Who it picks: underdog-side versus favorite-side, against the no-model underdog baseline.
3. Confidence: does the hit rate rise with the size of the correction?
4. Rules and stability: thresholds, seasons, and a simulated season of flat -110 betting.
5. Honest selection: choosing settings each season from prior seasons only, plus a placebo test.
"""

import itertools
import math

import numpy as np
import pandas as pd

import nfl_lab as lab

ALPHAS = (3000.0, 10000.0, 30000.0)
FEATURES = lab.PBP_FEATURES + lab.BOX_FEATURES + lab.BASIC_CONTEXT


def picks(frame):
    f = frame[frame['resid'] != 0].copy()
    f['win'] = np.where(f['pred'] > 0, f['resid'] > 0, f['resid'] < 0)
    f['dog'] = (f['pred'] > 0) == (f['spread_line'] < 0)
    f['conf'] = f['pred'].abs()
    return f


def roi(rate):
    return rate * 100 / 110 - (1 - rate)


def main():
    table = lab.default_table()
    frames = {a: lab.walk_forward(table, lab.ridge_margin(FEATURES, alpha=a, target='resid')) for a in ALPHAS}
    main_frame = frames[10000.0]
    f = picks(main_frame)
    print('games', len(f), 'test seasons', int(f['season'].min()), '-', int(f['season'].max()), '| break-even at -110 is 52.38%')

    print('\n=== 1. information beyond the spread (slope 0 = nothing, 1 = fully calibrated) ===')
    for alpha, frame in frames.items():
        lab.scorecard(f'ridge on cover margin, alpha={alpha:g}', frame, is_margin=False)

    print('\n=== 2. who it picks ===')
    t = main_frame[main_frame['resid'] != 0]
    dog_cover = np.where(t['spread_line'] < 0, t['resid'] > 0, t['resid'] < 0)
    print('always bet the underdog (no model): %.3f over %d games' % (dog_cover.mean(), len(t)))
    for name, g in (('all picks', f), ('picks on the underdog', f[f['dog']]), ('picks on the favorite', f[~f['dog']])):
        d, h = g[g['season'] <= lab.DISCOVERY_END], g[g['season'] > lab.DISCOVERY_END]
        print('%-24s all %.3f (n=%d) | discovery %.3f (n=%d) | holdout %.3f (n=%d)' % (
            name, g['win'].mean(), len(g), d['win'].mean(), len(d), h['win'].mean(), len(h)))
    dogs = f[f['dog']]
    rng = np.random.default_rng(1)
    draws = [pd.Series(dog_cover).sample(len(dogs), random_state=int(s)).mean() for s in rng.integers(0, 10 ** 6, 2000)]
    print('random underdog bets of the same size: mean %.3f, 95th pct %.3f; model underdog picks %.3f (p = %.4f)' % (
        np.mean(draws), np.percentile(draws, 95), dogs['win'].mean(), float((np.array(draws) >= dogs['win'].mean()).mean())))

    print('\n=== 3. confidence (underdog picks, thirds of the correction size) ===')
    for q, g in dogs.groupby(pd.qcut(dogs['conf'], 3, labels=['low', 'middle', 'high'])):
        print('   %-7s n=%4d  correction %.2f-%.2f  covered %.3f' % (q, len(g), g['conf'].min(), g['conf'].max(), g['win'].mean()))

    print('\n=== 4. rules and stability ===')
    for x in (0.0, 0.3, 0.5, 0.7, 1.0):
        g = dogs[dogs['conf'] >= x]
        d, h = g[g['season'] <= lab.DISCOVERY_END], g[g['season'] > lab.DISCOVERY_END]
        print('underdog picks, correction >= %.1f: discovery %.3f (n=%d) | holdout %.3f (n=%d) | %.0f picks/season' % (
            x, d['win'].mean(), len(d), h['win'].mean(), len(h), len(g) / g['season'].nunique()))
    rule = dogs[dogs['conf'] >= 0.5].sort_values(['season', 'week'])
    pnl = np.where(rule['win'], 100 / 110, -1.0)
    by_season = pd.Series(pnl, index=rule.index).groupby(rule['season']).sum()
    cum = np.cumsum(pnl)
    print('flat -110 betting on the >= 0.5 rule: %+.1f units over %d bets (ROI %+.1f%%), worst drawdown %.1f units, %d of %d seasons lost money' % (
        pnl.sum(), len(pnl), 100 * pnl.mean(), (np.maximum.accumulate(cum) - cum).max(), int((by_season < 0).sum()), len(by_season)))

    print('\n=== 5. honest selection: pick alpha, side and threshold each season from earlier seasons only ===')
    preds = {a: picks(fr) for a, fr in frames.items()}
    combos = list(itertools.product(ALPHAS, ('all', 'dog'), (0.0, 0.3, 0.5, 0.7, 1.0)))

    def chosen(alpha, side, x, seasons):
        g = preds[alpha]
        g = g[g['season'].isin(seasons) & (g['conf'] >= x)]
        return g[g['dog']] if side == 'dog' else g

    bets = won = 0
    for season in range(2012, 2027):
        best, best_score = None, -9
        for alpha, side, x in combos:
            g = chosen(alpha, side, x, list(range(2006, season)))
            if len(g) >= 150 and g['win'].mean() - math.sqrt(0.25 / len(g)) > best_score:
                best, best_score = (alpha, side, x), g['win'].mean() - math.sqrt(0.25 / len(g))
        g = chosen(*best, [season])
        bets, won = bets + len(g), won + int(g['win'].sum())
    rate = won / bets
    print('nested walk-forward 2012-2026: %d bets, %.1f%% won (+-%.1f%%), ROI %+.1f%%, z vs break-even %+.2f, z vs 50%% %+.2f' % (
        bets, 100 * rate, 100 * math.sqrt(0.25 / bets), 100 * roi(rate), (rate - 0.5238) / math.sqrt(0.25 / bets), (rate - 0.5) / math.sqrt(0.25 / bets)))
    placebo = []
    base = preds[10000.0]
    for _ in range(500):
        s = base.copy()
        s['pred'] = base.groupby('season')['pred'].transform(lambda v: rng.permutation(v.to_numpy()))
        s['win'] = np.where(s['pred'] > 0, s['resid'] > 0, s['resid'] < 0)
        s['dog'] = (s['pred'] > 0) == (s['spread_line'] < 0)
        s['conf'] = s['pred'].abs()
        placebo.append(s[s['dog'] & (s['conf'] >= 0.5)]['win'].mean())
    real = base[base['dog'] & (base['conf'] >= 0.5)]['win'].mean()
    print('placebo (corrections shuffled across games within each season), dog picks >= 0.5: mean %.3f, max %.3f vs real %.3f (p = %.4f)' % (
        np.mean(placebo), np.max(placebo), real, float((np.array(placebo) >= real).mean())))

    print('\nWhat this supports: a small, consistent tendency for the correction to find underpriced underdogs, strong enough')
    print('that chance alone is very unlikely to explain it.  What it does not settle: the underdog angle was found after')
    print('looking at several cuts of the same history, the estimated edge is uncertain (about 53-56%), and real lines')
    print('and limits at your sportsbook are worse than the closing consensus tested here.  Forward-test before trusting it.')


if __name__ == '__main__':
    main()
