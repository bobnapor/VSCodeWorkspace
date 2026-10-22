"""Is there an edge on favorites?  A pre-registered test of the scoring-environment hypothesis and two classic angles.

The production rule picks only underdogs: the favorites the correction model favors cover about 51% in every era, which
fits books shading lines toward favorites.  This tests whether some favorites are underpriced for a reason the
correction model cannot see.  Nothing here changes the underdog rule.

Pre-registered before any outcome below was computed (October 20, 2026).  The Rule A threshold was set from the
distribution of x1 alone (x1 >= 0.5 holds in about 15% of games, 2.2 a week), never from results.

Frame: regular-season games, weeks 5-16, with a closing spread (not a pick'em) and a closing total.  The favorite is the
side laying points at the close; its cover margin is its final margin minus the spread (a push counts as 0 in
correlations and is dropped from hit rates).  Eras: 1999-2005, 2006-2017, 2018-2026.  c_fav is the production
correction (nfl_signal.scored_history: walk-forward from 2006, 1999-2005 scored by a 2006-2019 fit) signed toward the
favorite.  The underdog rule picks exactly the games with c_fav < 0, so a favorite rule limited to c_fav > 0 can never
touch an underdog pick.

H1 (primary), scoring-environment scaling.  Spreads come from additive power ratings ("A is 6 points better than B"),
but margins scale with how many points a game produces: a team 6 points better in a 44-point game should be about 7
points better in a 51-point game.  If spreads under-scale, the favorite's cover margin rises with
x1 = |spread| x (total / league total - 1), the league total being the season-to-date mean closing total (weeks up
to the game's).  x1 is in points (the extra margin a fully proportional market would have priced) and has no fitted
parameter, so every season is out of sample for it.  Earlier research used the raw total only as a separate linear
term or as underdog-only buckets, which cannot express this interaction.
H2, divisional compression.  Division rivals meet twice a year and know each other, which compresses margins:
x2 = -div_game.
H3, favorite-longshot.  The public over-bets big favorites: x3 = -|spread|.

Test: Pearson correlation of each x with the favorite's cover margin, and the slope's robust (HC0) t-statistic.
Pass: correlation > 0 in every era, partial correlation > 0 in every era (controlling for c_fav and |spread|; for H3
c_fav only), and pooled one-sided p < 0.05/3 (t >= 2.13, Bonferroni over the three hypotheses).

Rule A (primary): pick the favorite when the week is 5-16, x1 >= 0.5 and c_fav > 0.
Rule B (secondary), walk-forward stacking: each season, ordinary least squares of the favorite's cover margin on c_fav and
x1 (with intercept) over up to 14 previous seasons from 2006 (at least 3, so it is scored from 2009; 1999-2005 use the
2006-2019 fit); pick the favorite when the prediction is > 0 and c_fav > 0.
Adopt a favorite rule in production only if H1 passes and Rule A hits >= 53.5% pooled over the three eras and >= 50% in
every era (pushes excluded).  H2 and H3 can only become candidate filters for forward tracking, not production changes.

Reference lines (no decisions): every favorite; the production model's favorite side (c_fav > 0); every favorite with
x1 >= 0.5 regardless of the model; hit rates by quintile of x1; H1 over all weeks and the playoffs (outside the
betting window); ROI at the real closing spread odds (2006+).

Round 2, pre-registered after round 1 failed and before any of its outcomes were computed:

P1, the selection process on favorites.  Can a disciplined search find a pocket of favorites that keeps covering?  26
conditions fixed in advance (spread size, home/road, divisional, week block, dome, primetime, total relative to the
league, rest edge, each team's last result, key and half-point lines) x 3 thresholds on c_fav (> 0, >= 0.25, >= 0.5) =
78 rules, the same size as the underdog scan in nfl_research_validate.py.  Each season from 2011, the best rule on 2006
through the season before (at least 60 decided games; chosen by raw rate and, separately, by the 90% Wilson lower bound)
is bet in that season; 1999-2005 are bet with the best rule on 2006-2019.  The pooled result estimates what "scan and
keep the best" is worth on favorites.  Adopt a favorite rule only if the process hits >= 53.5% both forward (2011-2026)
and backward (1999-2005) under both selectors.
T1, favorite teaser legs and the total.  A 6-point teaser costs the same whatever the game, but margins spread out as
scoring rises, so a teased leg is worth more in a low-total game.  The published refinement of the Wong rule plays legs
only when the total is under 49.  Pass: favorite legs (-7.5/-8.5 teased to -1.5/-2.5, weeks 5-16) with a total under 49
win >= 72.4% in every era and >= 73.9% pooled, and more often than favorite legs with a total of 49 or more.  Mechanism
check (no decision): for every favorite at every line, how often it covers the spread less 6 points, against the total
(with the spread as a control).  Underdog legs are shown for reference only; nothing on the underdog side changes.

Run it from this folder:  python nfl_research_favorites.py          (round 1: H1-H3, Rules A and B)
                          python nfl_research_favorites.py scan     (round 2: P1, the selection process)
                          python nfl_research_favorites.py teasers  (round 2: T1, favorite teaser legs and the total)
                          python nfl_research_favorites.py night    (after P1: robustness of the rule it chose, no decisions)

Results (October 20, 2026; nothing was tuned after the pre-registered runs):

Round 1 failed.  H1: correlation with the favorite's cover margin -0.009 / -0.014 / +0.024 by era (pooled -0.002, t=-0.16,
n=4,769), so the market already scales spreads with the total.  H2 (+0.035 / -0.004 / +0.016) and H3 (-0.018 / -0.007 /
-0.024) failed too.  Rule A hit 50.0% / 51.2% / 50.4%, Rule B 51.5% / 53.0% / 50.6%; favorites cover 44-53% in every
quintile of x1 in every era.

Round 2.  T1 failed: favorite legs with a total under 49 won 78.0% / 81.6% / 71.4% (the last below the 72.4% break-even)
against 66.7% / 57.1% / 69.2% with 49 or more, and across all favorites the teased cover rate does not fall as the total
rises (-0.010 / +0.031 / +0.025 per 10 points), so the variance story does not show up either.  P1 passed: the process
hit 56.6% (133-102, raw-rate selector) and 55.3% (157-127, lower bound) over 2011-2026, and 13-4 on 1999-2005, where both
selectors chose "primetime favorite, c_fav >= 0.5" on 2006-2019.  The raw-rate selector chose that rule every season from
2015 and the lower-bound selector from 2017 (mostly road favorites before that).  Adopted as nfl_signal.is_favorite_pick
(night = kickoff 7 pm Eastern or later).

After P1 (diagnostics, no decisions; `night`): 1999 has no kickoff times, so its Monday games count as the night games
from here on, which added 0-3: the rule went 13-7 (65.0%) on 1999-2005, 81-55 (59.6%) on 2006-2017 and 56-44 (56.0%) on
2018-2026, about 11 picks a season; 77-59 (56.6%, +9.4% at the real closing odds) in the seasons after the process first
chose it and 90-66 (57.7%) with 1999-2005 added (p about 0.09 against the 52.4% break-even).  It holds on Thursday,
Sunday and Monday nights and for home and road favorites.  In day games the favorites the correction favors cover 48-53%
at every size of lean in every era.  The correction is more informative at night in every era: the c_fav x night
interaction is positive for the cover margin (+2.75 / +0.40 / +1.28 points per point) and for covering (+0.134 / +0.018 /
+0.070; pooled t=1.74), and with |c_fav| >= 0.5 the model's side covered 68.6% / 57.7% / 58.3% at night against 52.1% /
51.8% / 51.4% by day.  Weeks 1-4, where the model is not used for picks, did not support it (42-44); weeks 17-18 went
6-5.  Reading: standalone national-TV games draw the most recreational money, so their lines stray furthest from the
play-by-play fundamentals the correction measures; that is enough to clear the usual favorite tax at night, not by day.
Expect about 55%: the rule came out of a scan, and its evidence is thinner than the underdog rule's.
"""

import math

import numpy as np
import pandas as pd

import nfl_lab as lab
import nfl_signal

WEEKS = nfl_signal.PICK_WEEKS
ERAS = (('1999-2005', 1999, 2005), ('2006-2017', 2006, 2017), ('2018-2026', 2018, 2026))
BREAK_EVEN = 0.5238
SCALING_THRESHOLD = 0.5      # Rule A: x1 >= 0.5 points
PASS_T = 2.13                # one-sided p < 0.05/3
STACK_WINDOW, STACK_MIN_SEASONS, STACK_FIRST = 14, 3, 2006
BACKWARD_TRAIN = nfl_signal.BACKWARD_TRAIN


# ---------------------------------------------------------------- the favorite frame

def league_total(table):
    """Season-to-date mean closing total of regular-season games, keyed by (season, week); playoffs use the full season."""
    reg = table[(table['game_type'] == 'REG') & table['total_line'].notna()]
    by_week = reg.groupby(['season', 'week'])['total_line'].agg(['sum', 'count']).groupby(level=0).cumsum()
    to_date = (by_week['sum'] / by_week['count']).to_dict()
    season_mean = reg.groupby('season')['total_line'].mean().to_dict()
    return [to_date.get((s, w), season_mean.get(s)) if g == 'REG' else season_mean.get(s)
            for s, w, g in zip(table['season'], table['week'], table['game_type'])]


def add_favorite_columns(frame, table):
    """Favorite view of a game frame: cover margin, the three predictors and the favorite's closing spread odds."""
    f = frame[frame['spread_line'].notna() & (frame['spread_line'] != 0) & frame['total_line'].notna() & frame['result'].notna()].copy()
    totals = pd.Series(league_total(table), index=table['game_id'])
    f['league_total'] = totals.reindex(f['game_id']).to_numpy()
    fav_home = f['spread_line'] > 0
    f['fav_home'] = fav_home.astype(int)
    f['abs_s'] = f['spread_line'].abs()
    f['fav_cover'] = np.where(fav_home, f['resid'], -f['resid'])
    f['x1'] = f['abs_s'] * (f['total_line'] / f['league_total'] - 1)
    f['x2'] = -f['div_game'].fillna(0)
    f['x3'] = -f['abs_s']
    f['fav_odds'] = pd.to_numeric(np.where(fav_home, f['home_spread_odds'], f['away_spread_odds']), errors='coerce')
    if 'pred' in f:
        f['c_fav'] = np.where(fav_home, f['pred'], -f['pred'])
    return f


def favorite_frame(table, season=lab.LAST_SEASON):
    """Every scored regular-season game (out-of-sample correction) in the favorite view, weeks 5-16."""
    f = add_favorite_columns(nfl_signal.scored_history(table, season), table)
    return f[f['week'].between(*WEEKS) & (f['game_type'] == 'REG')]


def eras(f):
    return [(name, f[f['season'].between(lo, hi)]) for name, lo, hi in ERAS]


# ---------------------------------------------------------------- statistics

def slope_t(x, y):
    """OLS slope of y on x with a robust (HC0) t-statistic."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    slope = (xc * yc).sum() / (xc ** 2).sum()
    e = yc - slope * xc
    se = math.sqrt(((xc ** 2) * (e ** 2)).sum()) / (xc ** 2).sum()
    return slope, slope / se


def residualize(v, controls):
    design = np.column_stack([np.ones(len(v))] + [np.asarray(c, float) for c in controls])
    coef, *_ = np.linalg.lstsq(design, np.asarray(v, float), rcond=None)
    return np.asarray(v, float) - design @ coef


def partial_corr(x, y, controls):
    return float(np.corrcoef(residualize(x, controls), residualize(y, controls))[0, 1])


def record(g):
    """(wins, losses, pushes, hit rate) of betting the favorite in every game of g."""
    w, l = int((g['fav_cover'] > 0).sum()), int((g['fav_cover'] < 0).sum())
    return w, l, len(g) - w - l, (w / (w + l) if w + l else float('nan'))


def roi_at_odds(g):
    """Return per bet at the real closing spread odds (pushes dropped), over the games that have odds."""
    d = g[(g['fav_cover'] != 0) & g['fav_odds'].notna()]
    if d.empty:
        return float('nan'), 0
    odds = d['fav_odds'].to_numpy(float)
    profit = np.where(d['fav_cover'] > 0, np.where(odds < 0, 100.0 / -odds, odds / 100.0), -1.0)
    return float(profit.mean()), len(d)


def rate_line(label, g):
    w, l, p, rate = record(g)
    n = w + l
    if n == 0:
        print('    %-50s no games' % label)
        return rate, n
    half = 1.96 * math.sqrt(rate * (1 - rate) / n)
    by = g[g['fav_cover'] != 0].assign(win=lambda d: d['fav_cover'] > 0).groupby('season')['win'].mean()
    roi, priced = roi_at_odds(g)
    odds_part = '  ROI at real odds %+.1f%% (n=%d)' % (100 * roi, priced) if priced else ''
    print('    %-50s %3d-%3d-%d  %.3f [%.3f, %.3f]  %4.1f/season  seasons > 52.4%%: %2d/%-2d  ROI@-110 %+.1f%%%s' % (
        label, w, l, p, rate, rate - half, rate + half, n / g['season'].nunique(), (by > BREAK_EVEN).sum(), len(by),
        100 * (rate * 100 / 110 - (1 - rate)), odds_part))
    return rate, n


# ---------------------------------------------------------------- the pre-registered tests

def information_tests(f):
    """H1-H3: correlation with the favorite's cover margin by era, partial correlation, and the pooled slope t."""
    tests = (('H1 scaling: x1 = |spread| x (total/league - 1)', 'x1', ['c_fav', 'abs_s']),
             ('H2 divisional compression: -div_game', 'x2', ['c_fav', 'abs_s']),
             ('H3 favorite-longshot: -|spread|', 'x3', ['c_fav']))
    print('\n=== information tests: correlation with the favorite cover margin, weeks %d-%d (pass: every era > 0, partial > 0, pooled t >= %.2f) ===' % (*WEEKS, PASS_T))
    verdicts = {}
    for label, col, controls in tests:
        print(label)
        ok = True
        for name, g in eras(f):
            r = float(np.corrcoef(g[col], g['fav_cover'])[0, 1])
            pr = partial_corr(g[col], g['fav_cover'], [g[c] for c in controls])
            slope, t = slope_t(g[col], g['fav_cover'])
            ok &= r > 0 and pr > 0
            print('    %-10s r=%+.3f  partial r=%+.3f  slope %+.3f pts per unit  t=%+.2f  n=%d' % (name, r, pr, slope, t, len(g)))
        slope, t = slope_t(f[col], f['fav_cover'])
        r = float(np.corrcoef(f[col], f['fav_cover'])[0, 1])
        passed = ok and t >= PASS_T
        print('    %-10s r=%+.3f  slope %+.3f  t=%+.2f  n=%d  -> %s' % ('pooled', r, slope, t, len(f), 'PASS' if passed else 'fail'))
        verdicts[col] = passed
    return verdicts


def stacked_predictions(f):
    """Rule B: walk-forward OLS of the favorite cover margin on [1, c_fav, x1]; 1999-2005 use the 2006-2019 fit."""
    out = []

    def fit_predict(train, test):
        design = np.column_stack([np.ones(len(train)), train['c_fav'], train['x1']])
        coef, *_ = np.linalg.lstsq(design, train['fav_cover'].to_numpy(float), rcond=None)
        return np.column_stack([np.ones(len(test)), test['c_fav'], test['x1']]) @ coef, coef

    train = f[f['season'].between(*BACKWARD_TRAIN)]
    old = f[f['season'] < STACK_FIRST]
    pred, coef = fit_predict(train, old)
    out.append(old.assign(stack=pred))
    print('    Rule B fit on %d-%d: intercept %+.2f, c_fav %+.2f, x1 %+.2f (points of cover margin)' % (*BACKWARD_TRAIN, *coef))
    for season in sorted(f.loc[f['season'] >= STACK_FIRST, 'season'].unique()):
        past = f[(f['season'] < season) & (f['season'] >= max(STACK_FIRST, season - STACK_WINDOW))]
        if past['season'].nunique() < STACK_MIN_SEASONS:
            continue
        test = f[f['season'] == season]
        pred, coef = fit_predict(past, test)
        out.append(test.assign(stack=pred))
    print('    Rule B last fit (through %d): intercept %+.2f, c_fav %+.2f, x1 %+.2f' % (season - 1, *coef))
    return pd.concat(out)


def rules(f):
    print('\n=== favorite picks, weeks %d-%d (break-even 52.4%%; pushes excluded) ===' % WEEKS)
    rule_a = f[(f['x1'] >= SCALING_THRESHOLD) & (f['c_fav'] > 0)]
    stacked = stacked_predictions(f)
    rule_b = stacked[(stacked['stack'] > 0) & (stacked['c_fav'] > 0)]
    results = {}
    for name, g in eras(f):
        print(name)
        rate_line('reference: every favorite', g)
        rate_line('reference: production model favorite side (c_fav > 0)', g[g['c_fav'] > 0])
        rate_line('reference: x1 >= %.1f, no model' % SCALING_THRESHOLD, g[g['x1'] >= SCALING_THRESHOLD])
        results[name] = rate_line('RULE A: x1 >= %.1f and c_fav > 0' % SCALING_THRESHOLD, rule_a[rule_a['season'].isin(g['season'].unique())])
        b = rule_b[rule_b['season'].isin(g['season'].unique())]
        rate_line('Rule B: stacked prediction > 0 and c_fav > 0', b)
    print('pooled')
    pooled = rate_line('RULE A, all three eras', rule_a)
    rate_line('Rule B, all scored seasons', rule_b)
    by = rule_a[rule_a['fav_cover'] != 0].assign(win=lambda d: d['fav_cover'] > 0).groupby('season')['win'].agg(['sum', 'size'])
    print('    Rule A by season: ' + '  '.join('%d: %d-%d' % (s, w, n - w) for s, (w, n) in by.iterrows()))
    return pooled, results


def diagnostics(f, table):
    print('\n=== diagnostics (no decisions) ===')
    cuts = np.quantile(f['x1'], [0.2, 0.4, 0.6, 0.8])
    print('favorite cover rate by quintile of x1 (cut points %s):' % ', '.join('%.2f' % c for c in cuts))
    q = np.searchsorted(cuts, f['x1'], side='right')
    for name, g in eras(f.assign(q=q)):
        cells = []
        for k in range(5):
            w, l, _, rate = record(g[g['q'] == k])
            cells.append('Q%d %.3f (n=%d)' % (k + 1, rate, w + l))
        print('    %-10s %s' % (name, ' | '.join(cells)))
    print('favorite cover rate, x1 >= %.1f, by the model lean (c_fav):' % SCALING_THRESHOLD)
    for name, g in eras(f[f['x1'] >= SCALING_THRESHOLD]):
        cells = []
        for label, mask in (('c_fav <= 0', g['c_fav'] <= 0), ('c_fav > 0', g['c_fav'] > 0)):
            w, l, _, rate = record(g[mask])
            cells.append('%s %.3f (n=%d)' % (label, rate, w + l))
        print('    %-10s %s' % (name, ' | '.join(cells)))
    print('H1 outside the betting window (no model needed; every regular-season week and the playoffs):')
    every = add_favorite_columns(table[table['game_type'].isin(['REG', 'WC', 'DIV', 'CON', 'SB'])], table)
    for label, g in (('weeks 1-4', every[(every['game_type'] == 'REG') & (every['week'] <= 4)]),
                     ('weeks 17-18', every[(every['game_type'] == 'REG') & (every['week'] >= 17)]),
                     ('playoffs', every[every['game_type'] != 'REG']),
                     ('all regular-season weeks', every[every['game_type'] == 'REG'])):
        slope, t = slope_t(g['x1'], g['fav_cover'])
        w, l, _, rate = record(g[g['x1'] >= SCALING_THRESHOLD])
        print('    %-26s r=%+.3f  slope %+.3f  t=%+.2f  n=%d | favorites with x1 >= %.1f cover %.3f (n=%d)' % (
            label, np.corrcoef(g['x1'], g['fav_cover'])[0, 1], slope, t, len(g), SCALING_THRESHOLD, rate, w + l))


def main():
    table = lab.default_table()
    f = favorite_frame(table)
    print('favorite frame: %d games, weeks %d-%d, seasons %d-%d' % (len(f), *WEEKS, f['season'].min(), f['season'].max()))
    verdicts = information_tests(f)
    (pooled_rate, _), per_era = rules(f)
    diagnostics(f, table)
    era_ok = all(rate >= 0.50 for rate, _ in per_era.values())
    adopt = verdicts['x1'] and pooled_rate >= 0.535 and era_ok
    print('\n=== verdict (pre-registered) ===')
    print('H1 information test: %s; Rule A pooled %.3f (needs >= 0.535); every era >= 0.50: %s -> %s' % (
        'pass' if verdicts['x1'] else 'fail', pooled_rate, era_ok, 'ADOPT Rule A' if adopt else 'do not adopt'))
    print('H2 (divisional) %s, H3 (favorite-longshot) %s: candidate filters for forward tracking only' % (
        'pass' if verdicts['x2'] else 'fail', 'pass' if verdicts['x3'] else 'fail'))


# ---------------------------------------------------------------- round 2: P1, the selection process on favorites

SCAN_THRESHOLDS = (0.0, 0.25, 0.5)
SCAN_MIN_N = 60
SCAN_FIRST_TEST = 2011
WILSON_Z = 1.2816   # 90% one-sided lower bound, as in the underdog scan


def add_last_results(f, table):
    """Each side's final margin in its previous game of the same season (NaN before its first game)."""
    played = table[table['result'].notna()]
    long = pd.concat([
        pd.DataFrame({'game_id': played['game_id'], 'season': played['season'], 'gameday': played['gameday'],
                      'team': played['home_team'], 'margin': played['result']}),
        pd.DataFrame({'game_id': played['game_id'], 'season': played['season'], 'gameday': played['gameday'],
                      'team': played['away_team'], 'margin': -played['result']}),
    ]).sort_values(['team', 'gameday'])
    long['prev'] = long.groupby(['team', 'season'])['margin'].shift(1)
    prev = long.set_index(['game_id', 'team'])['prev']
    fav_team = np.where(f['fav_home'] == 1, f['home_team'], f['away_team'])
    dog_team = np.where(f['fav_home'] == 1, f['away_team'], f['home_team'])
    f = f.copy()
    f['fav_prev'] = prev.reindex(pd.MultiIndex.from_arrays([f['game_id'], fav_team])).to_numpy()
    f['dog_prev'] = prev.reindex(pd.MultiIndex.from_arrays([f['game_id'], dog_team])).to_numpy()
    return f


def favorite_conditions(f):
    """The 26 conditions of the favorite scan, fixed before any result was seen."""
    relative = f['total_line'] / f['league_total']
    rest = np.where(f['fav_home'] == 1, f['rest_diff'], -f['rest_diff'])
    return {
        'all favorites': pd.Series(True, index=f.index),
        'lays 1-3': f['abs_s'] <= 3, 'lays 3.5-6.5': f['abs_s'].between(3.5, 6.5),
        'lays 7-9.5': f['abs_s'].between(7, 9.5), 'lays 10+': f['abs_s'] >= 10,
        'home favorite': f['fav_home'] == 1, 'road favorite': f['fav_home'] == 0,
        'divisional game': f['div_game'] == 1, 'non-divisional game': f['div_game'] == 0,
        'weeks 5-8': f['week'].between(5, 8), 'weeks 9-12': f['week'].between(9, 12), 'weeks 13-16': f['week'].between(13, 16),
        'dome': f['dome'] == 1, 'outdoors': f['dome'] == 0,
        'primetime': f['primetime'] == 1, 'not primetime': f['primetime'] == 0,
        'total < 0.95 x league': relative < 0.95, 'total 0.95-1.05 x league': relative.between(0.95, 1.05),
        'total > 1.05 x league': relative > 1.05,
        'favorite has the rest edge': pd.Series(rest > 0, index=f.index), 'underdog has the rest edge': pd.Series(rest < 0, index=f.index),
        'favorite lost its last game': f['fav_prev'] < 0, 'favorite won its last game': f['fav_prev'] > 0,
        'underdog won its last game': f['dog_prev'] > 0,
        'key line 3 or 7': f['abs_s'].isin([3.0, 7.0]), 'half-point line': f['abs_s'] % 1 == 0.5,
    }


def wilson_lower(p, n, z=WILSON_Z):
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def scan_rule_mask(conds, f, rule):
    condition, threshold = rule
    return conds[condition] & (f['c_fav'] > 0) & (f['c_fav'] >= threshold)


def best_rules(f, conds, rules, past):
    """Best rule on the `past` games by raw rate and by the Wilson lower bound (pushes excluded, at least SCAN_MIN_N games)."""
    decided = f['fav_cover'] != 0
    best = {'raw rate': (-1.0, None), 'lower bound': (-1.0, None)}
    for rule in rules:
        g = f[scan_rule_mask(conds, f, rule) & past & decided]
        if len(g) < SCAN_MIN_N:
            continue
        p = float((g['fav_cover'] > 0).mean())
        for name, score in (('raw rate', p), ('lower bound', wilson_lower(p, len(g)))):
            if score > best[name][0]:
                best[name] = (score, rule)
    return {name: rule for name, (_, rule) in best.items()}


def selection_process(f):
    """P1: each season, bet the favorite rule that looked best on earlier seasons; 1999-2005 use the best on 2006-2019."""
    conds = favorite_conditions(f)
    rules = [(c, t) for c in conds for t in SCAN_THRESHOLDS]
    picked = {'raw rate': [], 'lower bound': []}
    chosen = []
    for season in range(SCAN_FIRST_TEST, int(f['season'].max()) + 1):
        best = best_rules(f, conds, rules, f['season'].between(STACK_FIRST, season - 1))
        for name, rule in best.items():
            picked[name].append(f[scan_rule_mask(conds, f, rule) & (f['season'] == season)])
        chosen.append((season, best['raw rate'], best['lower bound']))
    backward = best_rules(f, conds, rules, f['season'].between(*BACKWARD_TRAIN))
    old = f['season'] < STACK_FIRST
    print('\n=== P1: selection process on favorites, %d rules (26 conditions x thresholds %s on c_fav), weeks %d-%d ===' % (
        len(rules), '/'.join('%g' % t for t in SCAN_THRESHOLDS), *WEEKS))
    results = {}
    for name in picked:
        forward = pd.concat(picked[name])
        back = f[scan_rule_mask(conds, f, backward[name]) & old]
        print('chosen by %s' % name)
        results[(name, 'forward')] = rate_line('forward: %d-%d, best rule on earlier seasons' % (SCAN_FIRST_TEST, f['season'].max()), forward)[0]
        results[(name, 'backward')] = rate_line('backward: 1999-2005, best rule on %d-%d (%s, c_fav >= %g)' % (
            *BACKWARD_TRAIN, backward[name][0], backward[name][1]), back)[0]
    print('rules chosen each season (raw rate | lower bound):')
    for season, raw, low in chosen:
        print('    %d: %s, c_fav >= %g | %s, c_fav >= %g' % (season, raw[0], raw[1], low[0], low[1]))
    latest = best_rules(f, conds, rules, f['season'].between(STACK_FIRST, f['season'].max() - 1))
    print('what the process would bet now (chosen on %d-%d), shown on every era for reference:' % (STACK_FIRST, f['season'].max() - 1))
    for name, rule in latest.items():
        for era, g in eras(f.assign(_m=scan_rule_mask(conds, f, rule))):
            rate_line('%s: %s, c_fav >= %g, %s' % (name, rule[0], rule[1], era), g[g['_m']])
    print('\nall 26 conditions with c_fav > 0, by era (for transparency; this is the menu the process chose from):')
    for condition in conds:
        cells = []
        for era, g in eras(f.assign(_m=scan_rule_mask(conds, f, (condition, 0.0)))):
            w, l, _, rate = record(g[g['_m']])
            cells.append('%s %.3f (n=%4d)' % (era, rate, w + l))
        print('    %-30s %s' % (condition, ' | '.join(cells)))
    passed = all(results[(name, part)] >= 0.535 for name in picked for part in ('forward', 'backward'))
    print('\n=== verdict (pre-registered): process >= 53.5%% forward and backward under both selectors -> %s ===' % (
        'ADOPT the process rule' if passed else 'do not adopt'))
    return passed


# ---------------------------------------------------------------- round 2: T1, favorite teaser legs and the total

TEASER_TOTAL_CUTOFF = 49.0
TEASER_POINTS = nfl_signal.TEASER_POINTS


def teaser_frame(table):
    g = table[(table['game_type'] == 'REG') & table['result'].notna() & table['spread_line'].notna() & (table['spread_line'] != 0)
              & table['total_line'].notna() & table['week'].between(*WEEKS)].copy()
    g['abs_s'] = g['spread_line'].abs()
    g['fav_margin'] = np.where(g['spread_line'] > 0, g['result'], -g['result'])
    g['fav_leg'] = g['abs_s'].isin(nfl_signal.TEASER_FAV_LINES)
    g['dog_leg'] = g['abs_s'].isin(nfl_signal.TEASER_DOG_LINES)
    g['leg_win'] = np.where(g['fav_leg'], g['fav_margin'] > g['abs_s'] - TEASER_POINTS, g['fav_margin'] < g['abs_s'] + TEASER_POINTS)
    g['low_total'] = g['total_line'] < TEASER_TOTAL_CUTOFF
    return g


def leg_cell(g):
    return '%.3f (n=%3d)' % (g['leg_win'].mean(), len(g)) if len(g) else '  -   (n=  0)'


def teaser_tests(table):
    """T1: do favorite teaser legs win more when the total is under 49?  Underdog legs shown for reference only."""
    g = teaser_frame(table)
    print('\n=== T1: Wong teaser legs by total (weeks %d-%d; a leg needs 72.4%% at -110 and 73.9%% at -120 in a 2-team teaser) ===' % WEEKS)
    era_rates = []
    for side, mask in (('favorite legs (-7.5/-8.5 to -1.5/-2.5)', g['fav_leg']), ('underdog legs, reference only', g['dog_leg'])):
        print(side)
        for name, lo, hi in ERAS + (('pooled', 1999, 2026),):
            e = g[mask & g['season'].between(lo, hi)]
            print('    %-10s all %s | total < %g: %s | total >= %g: %s' % (
                name, leg_cell(e), TEASER_TOTAL_CUTOFF, leg_cell(e[e['low_total']]), TEASER_TOTAL_CUTOFF, leg_cell(e[~e['low_total']])))
            if side.startswith('favorite'):
                era_rates.append((name, e[e['low_total']]['leg_win'].mean(), e[~e['low_total']]['leg_win'].mean()))
    print('mechanism, every favorite at every line: covers the spread less 6 points, linear slope per 10 points of total (spread held fixed)')
    every = g.copy()
    every['teased'] = every['fav_margin'] - (every['abs_s'] - TEASER_POINTS)
    every = every[every['teased'] != 0]
    for name, lo, hi in ERAS:
        e = every[every['season'].between(lo, hi)]
        win = (e['teased'] > 0).astype(float)
        x = residualize(e['total_line'], [e['abs_s']])
        slope, t = slope_t(x, residualize(win, [e['abs_s']]))
        print('    %-10s %+.3f per 10 points of total (t=%+.2f), n=%d, overall %.3f' % (name, 10 * slope, t, len(e), win.mean()))
    low_by_era = {name: low for name, low, _ in era_rates}
    pooled_low, pooled_high = low_by_era['pooled'], era_rates[-1][2]
    passed = (all(low_by_era[name] >= 0.724 for name, _, _ in ERAS) and pooled_low >= 0.739 and pooled_low > pooled_high)
    print('\n=== verdict (pre-registered): favorite legs with total < %g: %s; pooled %.3f vs %.3f with total >= %g -> %s ===' % (
        TEASER_TOTAL_CUTOFF, ', '.join('%s %.3f' % (name, low_by_era[name]) for name, _, _ in ERAS), pooled_low, pooled_high,
        TEASER_TOTAL_CUTOFF, 'PASS' if passed else 'fail'))
    return passed


# ---------------------------------------------------------------- after P1: robustness of the chosen rule (no decisions)

def ols_t(columns, y):
    """OLS coefficients of y on [1, *columns] with robust (HC0) t-statistics."""
    x = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in columns])
    y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    e = y - x @ coef
    bread = np.linalg.inv(x.T @ x)
    cov = bread @ (x.T * e ** 2) @ x @ bread
    return coef, coef / np.sqrt(np.diag(cov))


def add_night(f):
    """Night game (kickoff 7 pm Eastern or later); 1999 has no kickoff times, so its Monday games count as the night games."""
    return f.assign(night=[int(nfl_signal.is_night_game(g, d)) for g, d in zip(f['gametime'], f['weekday'])])


def night_game_checks(table):
    """Is the rule P1 chose (night-game favorite, c_fav >= 0.5) more than a lucky pocket?  Diagnostics only."""
    every = add_night(add_favorite_columns(nfl_signal.scored_history(table, lab.LAST_SEASON), table))
    every = every[every['game_type'] == 'REG']
    f = every[every['week'].between(*WEEKS)]
    lean = nfl_signal.FAVORITE_MIN_LEAN
    rule = lambda g: g[(g['night'] == 1) & (g['c_fav'] >= lean)]
    print('=== the night-game favorite rule (c_fav >= %g), weeks %d-%d ===' % (lean, *WEEKS))
    for name, g in eras(f):
        rate_line('%s, night flag as scanned (no 1999 kickoffs)' % name, g[(g['primetime'] == 1) & (g['c_fav'] >= lean)])
        rate_line('%s, night flag with 1999 Monday games' % name, rule(g))
    rate_line('seasons after the process first chose it (2015 on)', rule(f[f['season'] >= 2015]))
    rate_line('those seasons plus 1999-2005', rule(f[(f['season'] >= 2015) | (f['season'] <= 2005)]))
    print('\nby slot and side (pushes excluded):')
    for name, g in eras(rule(f)):
        cells = ['%s %d-%d' % (day, *record(g[g['weekday'] == day])[:2]) for day in ('Thursday', 'Sunday', 'Monday', 'Saturday')]
        cells += ['home favorite %d-%d' % record(g[g['fav_home'] == 1])[:2], 'road favorite %d-%d' % record(g[g['fav_home'] == 0])[:2]]
        print('    %-10s %s' % (name, ' | '.join(cells)))
    print('\nfavorites the correction favors, by size of lean, night and day:')
    for name, g in eras(f):
        for label, part in (('night', g[g['night'] == 1]), ('day', g[g['night'] == 0])):
            cells = ['>= %g: %.3f (n=%d)' % (k, record(part[(part['c_fav'] > 0) & (part['c_fav'] >= k)])[3],
                                            sum(record(part[(part['c_fav'] > 0) & (part['c_fav'] >= k)])[:2])) for k in (0, 0.25, 0.5, 0.75, 1.0)]
            print('    %-10s %-5s %s' % (name, label, ' | '.join(cells)))
    print('\nis the correction more informative at night?  favorite cover ~ c_fav + night + c_fav x night (robust t):')
    for name, g in eras(f) + [('pooled', f)]:
        decided = g[g['fav_cover'] != 0]
        (_, b1, b2, b3), (_, t1, t2, t3) = ols_t([g['c_fav'], g['night'], g['c_fav'] * g['night']], g['fav_cover'])
        (_, c1, _, c3), (_, s1, _, s3) = ols_t([decided['c_fav'], decided['night'], decided['c_fav'] * decided['night']],
                                               (decided['fav_cover'] > 0).astype(float))
        print('    %-10s margin: c_fav %+.2f (t=%+.2f), x night %+.2f (t=%+.2f) | covered: c_fav %+.3f (t=%+.2f), x night %+.3f (t=%+.2f) | night n=%d' % (
            name, b1, t1, b3, t3, c1, s1, c3, s3, int(g['night'].sum())))
    print("\nthe model's side (either side) when |c_fav| >= %g, night and day:" % lean)
    for name, g in eras(f):
        g = g[(g['fav_cover'] != 0) & (g['c_fav'].abs() >= lean)]
        hit = np.where(g['c_fav'] > 0, g['fav_cover'] > 0, g['fav_cover'] < 0)
        cells = []
        for label, mask in (('night', g['night'] == 1), ('day', g['night'] == 0)):
            fav, dog = mask & (g['c_fav'] > 0), mask & (g['c_fav'] < 0)
            cells.append('%s %.3f (n=%d; favorite side %.3f, underdog side %.3f)' % (label, hit[mask].mean(), mask.sum(), hit[fav].mean(), hit[dog].mean()))
        print('    %-10s %s' % (name, ' | '.join(cells)))
    print('\noutside the pick weeks (the model is not used for picks there):')
    for label, g in (('weeks 1-4', every[every['week'] <= 4]), ('weeks 17-18', every[every['week'] >= 17])):
        rate_line('%s, the rule' % label, rule(g))
        rate_line('%s, day favorites with c_fav >= %g' % (label, lean), g[(g['night'] == 0) & (g['c_fav'] >= lean)])


if __name__ == '__main__':
    import sys

    if 'scan' in sys.argv[1:]:
        selection_process(add_last_results(favorite_frame(lab.default_table()), lab.default_table()))
    elif 'teasers' in sys.argv[1:]:
        teaser_tests(lab.default_table())
    elif 'night' in sys.argv[1:]:
        night_game_checks(lab.default_table())
    else:
        main()
