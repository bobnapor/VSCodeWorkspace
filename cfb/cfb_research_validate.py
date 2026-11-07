"""College football spreads and totals: is there a betting edge?  The pre-registered validation (written November 7,
2026, before any college football result had been looked at).

Data: every FBS-vs-FBS game from 2006 to 2025 with a closing line (the median across sportsbooks, cfb_lines), with the
pre-game ratings and context of cfb_lab.  Regular season only: bowls are left out because players sit out bowls for
the draft and transfers, which the data does not record.  Win rates leave out pushes; break-even at -110 is 52.38%.

Protocol, fixed before the first run:
  1. Discovery seasons 2010-2019, each scored by models fitted only on earlier seasons from 2006 on.  The untouched test
     seasons 2021-2025 are not scored until step 3.  2020 (the COVID season: shortened, uneven schedules, empty
     stadiums) is used only for training the test's models, never scored.
  2. On discovery only: Model B's ridge alpha for each market (spreads: the cover margin; totals: the over margin) is
     the one of ALPHAS whose corrections correlate best with the outcome.  Then the rule families below are scored at
     each threshold of THRESHOLDS, and one rule per market is chosen: the one beating break-even by the largest z-score
     with at least MIN_PICKS_PER_SEASON picks a season (clarified after the discovery run, before the test: a market
     where no such rule beats break-even gets no rule).
       Spreads  S1: the side the correction favors, |correction| >= t      S2: underdogs it favors by >= t
                S3: favorites it favors by >= t
       Totals   T1: the side the correction favors, |correction| >= t      T2: unders by >= t      T3: overs by >= t
  3. Test (python cfb_research_validate.py test, run once the choices are written into CHOSEN): the chosen rule on
     2021-2025, each season by models fitted on all earlier seasons from 2006.  A rule passes only if on the test seasons
       (a) its picks beat break-even by z >= 2 (at -110),
       (b) its correction correlates with the outcome with t >= 2, and
       (c) it beats break-even in at least 4 of the 5 test seasons.
     This is stricter than the NBA suite's bar (beat 52.38% with t >= 2), which let through rules worth about nothing at
     -110.  Its return at the books' median price for the picked side is reported beside it.  The owner decides.
  4. The 2026 season (being played; its lines are ESPN's, from the play-by-play) is a second, forward test of whatever
     passes, scored once the season is complete.

Model A (the stats-only margin and total projections) is reported against the closing lines for reference.

    python cfb_research_validate.py            discovery (steps 1-2)
    python cfb_research_validate.py test       the untouched test of CHOSEN (step 3)
    python cfb_research_validate.py forward    CHOSEN on the 2026 games played so far (step 4; lines from
                                               CollegeFootballData, the median of DraftKings and Bovada)

Results.  Discovery (November 7, 2026, run once): Model A misses the margin by 13.09 points (closing spread 12.33) and the
total by 13.70 (closing total 13.15).  The spread correction correlates +0.068 with the cover margin (t 5.6, alpha 3000,
positive in all ten seasons, +0.006 to +0.110); the best spread rule with 60+ picks a season is S1 at 2 (433-350,
55.3%, z +1.64), and it was frozen into CHOSEN before the test was run.  The total correction is weak (+0.025, t 2.1,
alpha 300, negative in three seasons) and no totals rule with 60+ picks a season beats break-even (best T1 at 3: 51.9%),
so no totals rule is chosen and the totals' test seasons are left unspent.

Test (November 7, 2026, run once, after CHOSEN was committed): on 2021-2025 the spread correction kept its information
(r +0.059, t 3.5), and S1 at 2 went 191-150, 56.0% (z +1.34), +6.9% a bet at the books' median price, about 68 picks a
season: 2021 36-36, 2022 39-26, 2023 38-22, 2024 43-35, 2025 35-31 (above break-even in 4 of 5).  It does not pass:
the win rate held from discovery (55.3%) but 341 picks are too few for z >= 2.  Not proposed for betting; the rule is
the one to track forward (step 4), unchanged.
"""

import math
import sys

import numpy as np
import pandas as pd

import cfb_lab
import cfb_model

DISCOVERY = (2010, 2019)
TEST = (2021, 2025)
FORWARD = 2026
ALPHAS = (300, 1000, 3000, 10000, 30000)
THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
MIN_PICKS_PER_SEASON = 60
MIN_TEST_SEASONS = 4
BREAK_EVEN = 110 / 210
# filled in after discovery, before the test: {'spreads': (alpha, family, threshold), 'totals': (...)}
# Frozen November 7, 2026 from the discovery run (spreads S1 at 2: 55.3% of 783; no totals rule beat break-even)
CHOSEN = {'spreads': (3000, 'S1', 2.0)}


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


def payout(price):
    """Profit on a 1-unit winning bet at American odds (-110 when unknown or implausible)."""
    price = np.where(np.isnan(price) | (np.abs(price) < 100), -110.0, price)
    return np.where(price < 0, 100 / -price, price / 100)


def picks(scored, market, family, threshold):
    """(did each pick win, the picked rows) for one rule; pushes and games without a result dropped."""
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


def roi_at_price(won, taken, market):
    """Return per unit at the median price of the picked side (the home spread's or the over's price; the other side's
    is taken as the mirror, -110 when unknown)."""
    if market == 'spreads':
        home = np.sign(taken['pred']) > 0
        price = taken['home_spread_odds'].astype(float).values
    else:
        home = np.sign(taken['pred']) > 0
        price = taken['over_odds'].astype(float).values
    # the other side of a -115 is about -105: mirror around -110
    mirrored = np.where(np.isnan(price), np.nan, np.where(price < 0, -220 - price, -220 + price))
    side_price = np.where(home, price, mirrored)
    profit = np.where(won.values, payout(side_price), -1.0)
    return float(profit.mean()) if len(profit) else float('nan')


def regular(table):
    return table['regular'] & table['season'].between(cfb_model.FIRST_LINES_SEASON, TEST[1])


def model_a(table):
    print('\n== Model A (stats only) against the closing lines, discovery seasons ==')
    rows = regular(table)
    for kind, line, truth in (('margin', 'close_home_spread', 'margin'), ('points', 'close_total', 'points')):
        s = cfb_model.scored(table, kind, 3000, *DISCOVERY, train_from=cfb_lab.FIRST_SEASON, rows=rows)
        s = s[s[truth].notna() & s[line].notna()]
        market = -s[line] if kind == 'margin' else s[line]
        print(f'  {kind}: model miss {np.abs(s["pred"] - s[truth]).mean():.2f} points, closing line '
              f'{np.abs(market - s[truth]).mean():.2f} ({len(s)} games)')


def discovery(table):
    model_a(table)
    rows = regular(table) & (table['season'] != 2020)
    families = {'spreads': ('S1', 'S2', 'S3'), 'totals': ('T1', 'T2', 'T3')}
    seasons = DISCOVERY[1] - DISCOVERY[0] + 1
    for market, kind in (('spreads', 'cover'), ('totals', 'over')):
        print(f'\n== {market}: Model B on the discovery seasons {DISCOVERY[0]}-{DISCOVERY[1]} ==')
        best_alpha, best_r, runs = None, -1, {}
        for alpha in ALPHAS:
            s = cfb_model.scored(table, kind, alpha, *DISCOVERY, rows=rows)
            s = s[s[kind].notna()]
            r, t = correlation(s['pred'], s[kind])
            by_season = [correlation(g['pred'], g[kind])[0] for _, g in s.groupby('season')]
            print(f'  alpha {alpha:>6}: r = {r:+.4f} (t {t:+.2f}); by season ' + ' '.join(f'{x:+.3f}' for x in by_season)
                  + f'; corrections sd {s["pred"].std():.2f}')
            runs[alpha] = s
            if r > best_r:
                best_alpha, best_r = alpha, r
        s = runs[best_alpha]
        print(f'  chosen alpha {best_alpha} ({len(s)} games)')
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
    # the test's models train on every earlier season, 2020 included; only 2021-2025 are scored
    rows = regular(table)
    for market, (alpha, family, threshold) in CHOSEN.items():
        kind = 'cover' if market == 'spreads' else 'over'
        s = cfb_model.scored(table, kind, alpha, *TEST, rows=rows)
        s = s[s[kind].notna()]
        r, t = correlation(s['pred'], s[kind])
        won, taken = picks(s, market, family, threshold)
        wins, losses, rate, z = record(won)
        per_season = (wins + losses) / (TEST[1] - TEST[0] + 1)
        print(f'\n== {market}: {family} at {threshold:g}, alpha {alpha}, untouched test {TEST[0]}-{TEST[1]} ==')
        print(f'  correction vs outcome: r = {r:+.4f} (t {t:+.2f})')
        print(f'  picks: {wins}-{losses}, {rate:.1%} (z {z:+.2f} against break-even), about {per_season:.0f} a season; '
              f'return at the median price {roi_at_price(won, taken, market):+.1%} a bet')
        above = 0
        for season, g in won.groupby(taken['season']):
            sw, sl, srate, _ = record(g)
            above += srate > BREAK_EVEN
            print(f'    {season}: {sw}-{sl} {srate:.1%}')
        passed = z >= 2 and t >= 2 and above >= MIN_TEST_SEASONS
        print(f'  {"PASSES" if passed else "does not pass"} the pre-registered bar (z >= 2 against break-even, t >= 2, '
              f'above break-even in {MIN_TEST_SEASONS}+ of 5 seasons: {above})')
    return 0


def forward(table):
    """Step 4: CHOSEN on the FORWARD season's finished regular-season games, by models fitted on 2006 to the season
    before, as in the test."""
    rows = table['regular'] & table['season'].between(cfb_model.FIRST_LINES_SEASON, FORWARD)
    for market, (alpha, family, threshold) in CHOSEN.items():
        kind = 'cover' if market == 'spreads' else 'over'
        s = cfb_model.scored(table, kind, alpha, FORWARD, FORWARD, rows=rows)
        s = s[s[kind].notna()]
        r, t = correlation(s['pred'], s[kind])
        won, taken = picks(s, market, family, threshold)
        wins, losses, rate, z = record(won)
        print(f'\n== {market}: {family} at {threshold:g}, alpha {alpha}, forward {FORWARD} '
              f'({len(s)} games played with a line) ==')
        print(f'  correction vs outcome: r = {r:+.4f} (t {t:+.2f})')
        print(f'  picks: {wins}-{losses}, {rate:.1%} (z {z:+.2f} against break-even)')
        for week, g in won.groupby(taken['week']):
            ww, wl, _, _ = record(g)
            print(f'    week {int(week)}: {ww}-{wl}')
    return 0


def main(argv):
    table = cfb_lab.default_table()
    if argv and argv[0] == 'test':
        return test(table)
    if argv and argv[0] == 'forward':
        return forward(table)
    discovery(table)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
