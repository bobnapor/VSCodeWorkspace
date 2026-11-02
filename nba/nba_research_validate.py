"""NBA spreads and totals: is there a betting edge?  The pre-registered validation (written November 2, 2026, before
any NBA result had been looked at).

Data: every regular-season game from 2013-14 to 2025-26 with a closing line (the median across sportsbooks,
nba_lines.closing_lines), with the pre-game ratings, context and availability of nba_lab.  Win rates leave out pushes;
break-even at -110 is 52.38%.

Protocol, fixed before the first run:
  1. Discovery seasons 2016-2020 (2015-16 to 2019-20), each scored by models fitted only on earlier seasons from 2014
     on.  The untouched test seasons 2021-2026 (2020-21 to 2025-26) are not scored until step 3.
  2. On discovery only: Model B's ridge alpha for each market (spreads: the cover margin; totals: the over margin) is
     the one of ALPHAS whose corrections correlate best with the outcome.  Then the rule families below are scored at
     each threshold of THRESHOLDS, and one rule per market is chosen: the one beating break-even by the largest z-score
     with at least MIN_PICKS_PER_SEASON picks a season.
       Spreads  S1: the side the correction favors, |correction| >= t      S2: underdogs it favors by >= t
                S3: favorites it favors by >= t
       Totals   T1: the side the correction favors, |correction| >= t      T2: unders by >= t      T3: overs by >= t
  3. Test (python nba_research_validate.py test, run once the choices are written into CHOSEN): the chosen rule on
     2021-2026, each season by models fitted on all earlier seasons.  A rule is proposed for betting only if on the test
     seasons it wins more than break-even (52.38%) and its correction's correlation with the outcome is positive with
     t >= 2.  The owner decides.

Model A (the stats-only margin and total projections) is reported against the closing lines for reference.

    python nba_research_validate.py            discovery (steps 1-2)
    python nba_research_validate.py test       the untouched test of CHOSEN (step 3)

Results (November 2, 2026).  Model A misses the margin by 9.86 points (closing spread 9.61) and the total by 15.25
(closing total 14.20).  Discovery: the spread correction correlates +0.060 with the cover margin (t 4.7, positive in all
five seasons; alpha 1000) and the total correction +0.048 with the over margin (t 3.7; alpha 10000); the chosen rules
went 53.5% (S1 at 0.5, 3,859 picks) and 54.0% (T3 at 1, 341 picks).  Untouched test 2021-2026: the corrections kept
their information (spreads r +0.049, t 4.2; totals r +0.053, t 4.5), but the rules went 52.4% (2,392-2,174) and 52.5%
(1,033-935): over the letter of the bar, worth about nothing at -110.  Not proposed for betting; the corrections are a
real but small signal, about what supports 52-54% at any volume.
"""

import math
import sys

import numpy as np
import pandas as pd

import nba_lab
import nba_model

DISCOVERY = (2016, 2020)
TEST = (2021, 2026)
ALPHAS = (300, 1000, 3000, 10000, 30000)
THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
MIN_PICKS_PER_SEASON = 60
BREAK_EVEN = 110 / 210
# filled in after discovery, before the test: {'spreads': (alpha, family, threshold), 'totals': (...)}
# Frozen November 2, 2026 from the discovery run (spreads S1 at 0.5: 53.5% of 3,859; totals T3 at 1: 54.0% of 341)
CHOSEN = {'spreads': (1000, 'S1', 0.5), 'totals': (10000, 'T3', 1.0)}


def record(won):
    """(wins, losses, win rate, z against break-even) of a boolean series of graded picks (pushes left out)."""
    wins, n = int(won.sum()), len(won)
    rate = wins / n if n else float('nan')
    z = (rate - BREAK_EVEN) / math.sqrt(BREAK_EVEN * (1 - BREAK_EVEN) / n) if n else float('nan')
    return wins, n - wins, rate, z


def correlation(pred, outcome):
    """(r, t) of the correction with the outcome."""
    r = float(np.corrcoef(pred, outcome)[0, 1])
    return r, r * math.sqrt((len(pred) - 2) / max(1e-12, 1 - r * r))


def picks(scored, market, family, threshold):
    """The rule's picks as a boolean series: did the pick win (pushes dropped)."""
    if market == 'spreads':
        lean = scored['pred']                                   # toward the home team covering
        outcome = scored['cover']
        dog_home = scored['close_home_spread'] > 0
        toward_dog = np.where(dog_home, lean, -lean)
        chosen = {'S1': lean.abs() >= max(threshold, 1e-9), 'S2': toward_dog >= max(threshold, 1e-9),
                  'S3': -toward_dog >= max(threshold, 1e-9)}[family]
        # a pick'em has no underdog: only S1 takes it
        if family != 'S1':
            chosen &= scored['close_home_spread'] != 0
    else:
        lean = scored['pred']                                   # toward the over
        outcome = scored['over']
        chosen = {'T1': lean.abs() >= max(threshold, 1e-9), 'T2': -lean >= max(threshold, 1e-9),
                  'T3': lean >= max(threshold, 1e-9)}[family]
    taken = scored[chosen & (outcome != 0) & outcome.notna()]
    side = np.sign(taken['pred'])
    return pd.Series(np.sign(taken[outcome.name]) == side, index=taken.index), taken


def model_a(table):
    print('\n== Model A (stats only) against the closing lines, discovery seasons ==')
    for kind, line, truth in (('margin', 'close_home_spread', 'margin'), ('points', 'close_total', 'points')):
        s = nba_model.scored(table, kind, 3000, *DISCOVERY, train_from=nba_lab.FIRST_SEASON)
        s = s[s[truth].notna() & s[line].notna()]
        market = -s[line] if kind == 'margin' else s[line]
        print(f'  {kind}: model miss {np.abs(s["pred"] - s[truth]).mean():.2f} points, closing line '
              f'{np.abs(market - s[truth]).mean():.2f} ({len(s)} games)')


def discovery(table):
    model_a(table)
    families = {'spreads': ('S1', 'S2', 'S3'), 'totals': ('T1', 'T2', 'T3')}
    seasons = DISCOVERY[1] - DISCOVERY[0] + 1
    for market, kind in (('spreads', 'cover'), ('totals', 'over')):
        print(f'\n== {market}: Model B on the discovery seasons {DISCOVERY[0]}-{DISCOVERY[1]} ==')
        best_alpha, best_r, runs = None, -1, {}
        for alpha in ALPHAS:
            s = nba_model.scored(table, kind, alpha, *DISCOVERY)
            s = s[s[kind].notna()]
            r, t = correlation(s['pred'], s[kind])
            by_season = [correlation(g['pred'], g[kind])[0] for _, g in s.groupby('season')]
            print(f'  alpha {alpha:>6}: r = {r:+.4f} (t {t:+.2f}); by season ' + ' '.join(f'{x:+.3f}' for x in by_season)
                  + f'; corrections sd {s["pred"].std():.2f}')
            runs[alpha] = s
            if r > best_r:
                best_alpha, best_r = alpha, r
        s = runs[best_alpha]
        print(f'  chosen alpha {best_alpha}')
        best = None
        for family in families[market]:
            line = []
            for threshold in THRESHOLDS:
                won, _ = picks(s, market, family, threshold)
                wins, losses, rate, z = record(won)
                line.append(f't{threshold:g}: {wins}-{losses} {rate:.1%}')
                if wins + losses >= MIN_PICKS_PER_SEASON * seasons and (best is None or z > best[0]):
                    best = (z, family, threshold, wins, losses, rate)
            print(f'  {family}: ' + ' | '.join(line))
        if best:
            z, family, threshold, wins, losses, rate = best
            print(f'  -> best: {family} at {threshold:g} ({wins}-{losses}, {rate:.1%}, z {z:+.2f} against break-even)')
            print(f'     to test it, set CHOSEN[{market!r}] = ({best_alpha}, {family!r}, {threshold})')


def test(table):
    if not CHOSEN:
        print('CHOSEN is empty: run the discovery first and write its choices into CHOSEN.')
        return 2
    for market, (alpha, family, threshold) in CHOSEN.items():
        kind = 'cover' if market == 'spreads' else 'over'
        s = nba_model.scored(table, kind, alpha, *TEST)
        s = s[s[kind].notna()]
        r, t = correlation(s['pred'], s[kind])
        won, taken = picks(s, market, family, threshold)
        wins, losses, rate, z = record(won)
        per_season = (wins + losses) / (TEST[1] - TEST[0] + 1)
        print(f'\n== {market}: {family} at {threshold:g}, alpha {alpha}, untouched test {TEST[0]}-{TEST[1]} ==')
        print(f'  correction vs outcome: r = {r:+.4f} (t {t:+.2f})')
        print(f'  picks: {wins}-{losses}, {rate:.1%} (z {z:+.2f} against break-even), about {per_season:.0f} a season')
        for season, g in won.groupby(taken['season']):
            sw, sl, srate, _ = record(g)
            print(f'    {season}: {sw}-{sl} {srate:.1%}')
        passed = rate > BREAK_EVEN and r > 0 and t >= 2
        print(f'  {"PASSES" if passed else "does not pass"} the pre-registered bar (beats 52.38% and t >= 2)')
    return 0


def main(argv):
    # walk_forward scores only the seasons asked for and trains on earlier ones, so 2026-27 (live) never enters
    table = nba_lab.default_table()
    if argv and argv[0] == 'test':
        return test(table)
    discovery(table)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
