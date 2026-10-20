"""Predict the final margin, attach a confidence score, compare with the spread: is there an edge?

The owner's original method: average 10,000 bootstrap regressions into a projected margin, compare it with the spread,
bet the side the projection favors and trust bigger gaps more.  This tests that method and two other margin models,
each with confidence scores, out of sample on every season since 1999 (2006 on walk-forward; 1999-2005 scored by fits
on 2006-2019, as nfl_signal.scored_history does).  Pre-registered before any result below was computed (October 20,
2026); no current logic changes either way.

Margin models (each predicts the home team's final margin before kickoff):
 A  the production stats model (nfl_model: ridge alpha 300 on points, yards, first downs, turnovers and plays for and
    against both teams, Elo, rest, neutral site and quarterback features), refitted every week on the 12 seasons before
    plus the season's earlier weeks, as nfl_prod_projections.py does but on the research table, which starts in 1999.
    Its 10,000 "simulations" are bootstrap refits; 200 per week here (the average settles long before that).
 K  a state-space (Kalman filter) rating model: each team's strength drifts a little every week and regresses toward
    average each offseason, with home field estimated alongside; every game moves the two teams by how surprising the
    result was, weighted by how uncertain their ratings were.  It gives a predicted margin and a variance (rating
    uncertainty plus the game's own randomness), the natural confidence score.  Game noise, weekly drift, offseason
    carry and offseason change are chosen from a fixed grid by predictive likelihood on 2006-2017.
 F  every pre-game rating (37 stats, offense and defense) plus Elo, rest and neutral site, ridge alpha 1000 on the final
    margin, fitted on the 14 seasons before (the correction model's features and window, aimed at the margin).

Confidence scores (the gap is projected margin minus the closing spread, positive = the home side):
 C1  the gap itself (the original method: the bigger the gap, the surer the bet).
 C2  the bootstrap z (model A): the gap over the spread of its bootstrap projections.
 C3  the cover probability if the model is taken at its word: normal around its projection, with the size of its own
     out-of-sample errors in earlier seasons (K: its predictive variance).
 C4  the honest cover probability: Phi(beta x gap / sigma), with beta the slope of the cover margin on the gap and
     sigma the spread of the cover margin, both from earlier seasons' out-of-sample predictions (from 2006, at least
     3 seasons; 1999-2005 use 2006-2019).  It pulls the projection toward the spread by what its gaps have been worth.

Tests, weeks 5-16, eras 1999-2005 / 2006-2017 / 2018-2026, pushes excluded from hit rates:
 1. Accuracy: mean absolute error of the projection against the closing spread's.
 2. Information: the correlation of the gap with the cover margin (result minus spread) must be > 0 in every era.
 3. Is the confidence real?  Hit rate of the gap's side by C1 (and C2) bucket; calibration of C3 and C4 (predicted
    against actual cover rate in bins).
 4. A margin-model rule is worth adding only if test 2 passes, bets with C4 >= 55% hit >= 55% pooled (at least 30
    bets) and >= 52.4% in every era with bets, and the gap's partial correlation with the cover margin given the
    production correction is > 0 in every era.  The production correction (nfl_signal) read as a margin model
    (spread + correction) is shown alongside for reference, not tested.

Run it from this folder:  python nfl_research_margin.py

Results (October 20, 2026; eras 1999-2005 / 2006-2017 / 2018-2026, weeks 5-16; one bug fix after the pre-registration
snapshot, in how test 2 was printed):

A, the original method: mean absolute error 10.13 / 10.79 / 10.06 against the closing spread's 10.01 / 10.59 / 9.83.
   The 200-run bootstrap average differed from a single fit by 0.08 points on average, and the runs spread about 0.95
   points (5-95%: 0.69-1.57) while games missed the projection by 13.4 points (SD): averaging simulations makes the
   projection precise, not accurate, and their spread measures the model's wobble, not the game's.  Gap information
   +0.048 / +0.011 / +0.017 (given the correction +0.038 / -0.002 / +0.015).  The gap's side covered 49.2% (gap under 1
   point), 50.4% (1-2), 51.9% (2-3), 51.8% (3-5), 52.4% (5-7) and 57.9% (7+, 57 games in 28 seasons); by bootstrap z,
   49-53% in every bucket.  Taken at its word it is overconfident: games it called 60-65% covered 51.2%.  Honest, it
   calls 95% of games 50-52%, and its 37 bets at 55% or more went 18-19.  Fail.
K, Kalman (game noise 13, weekly drift 0.5, offseason carry 0.65, offseason change 4): error 10.09 / 10.75 / 10.16;
   gap information +0.079 / +0.013 / +0.004 (given the correction +0.079 / +0.013 / +0.011).  Its stated uncertainty
   barely moves, 13.72-14.11 points (SD of the margin) for 90% of games (the rating part alone about 4.8), because the
   game's own randomness dominates; games it called 60-65% covered 52.1%.  Honest bets at 55% or more: 4-3; at the
   break-even or more: 76-70 (52.1%).  Fail.
F, every rating aimed at the margin: error 10.06 / 10.76 / 10.06, the closest of the three; gap information +0.091 /
   +0.016 / +0.030 but +0.061 / -0.011 / +0.012 given the correction.  Its gap's side covered 52-55% in every bucket
   under 7 points and 57.3% at 7+ (75 games; 46.9% in 2006-2017); games it called 60-65% covered 51.1%; honest bets at
   55% or more 19-21.  Fail.
Reference, the production correction read as a margin (spread + correction): as accurate as the spread (10.01 / 10.58
   / 9.83), with the only steady information (+0.083 / +0.055 / +0.063), and honest probabilities roughly calibrated
   (stated 54.8% covered 52.5%, stated 56.8% covered 56.0%).

Reading: a margin model built from team stats misses by more than the closing line in every era, so when the two
disagree the line is usually right.  The disagreement carries a little information (mostly in 1999-2005, near zero since
2018) and almost none beyond the correction model, and no honest confidence score finds 55% bets: the game's randomness
(about 13.5 points) swamps any model's uncertainty about its own number, so confidence can hardly vary from game to game.
The useful form of "projected margin plus confidence" is already in production: the spread plus the correction (the
sheet's Fair Line), with Hist. Win % as its calibrated confidence.
"""

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

import nfl_lab as lab
import nfl_model as nm
import nfl_signal

WEEKS = nfl_signal.PICK_WEEKS
ERAS = (('1999-2005', 1999, 2005), ('2006-2017', 2006, 2017), ('2018-2026', 2018, 2026))
BACKWARD_TRAIN = nfl_signal.BACKWARD_TRAIN
A_COLUMNS = nm.feature_columns(nm.PRODUCTION_METRICS, use_qb=True, use_elo=True)
BOOT_RUNS = 200
F_COLUMNS = lab.PBP_FEATURES + lab.BOX_FEATURES + lab.BASIC_CONTEXT
F_ALPHA = 1000.0
KALMAN_GRID = {'game_sd': (13.0, 13.5, 14.0), 'week_sd': (0.0, 0.5, 1.0), 'carry': (0.5, 0.65, 0.8), 'offseason_sd': (2.0, 3.0, 4.0)}
KALMAN_FIXED = {'start_sd': 6.0, 'hfa': 2.5, 'hfa_sd': 1.0, 'hfa_offseason_sd': 0.5}
KALMAN_FIT = (2006, 2017)
STACK_FIRST, STACK_MIN_SEASONS = 2006, 3
RULE_P = 0.55
BREAK_EVEN = 0.5238
GAP_BUCKETS = (0, 1, 2, 3, 5, 7, 99)
Z_BUCKETS = (0, 1, 2, 3, 5, 99)
P_BINS = (0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 1.01)


def final_week(season):
    return 18 if season >= 2021 else 17


def eval_games(table):
    """The games every model is scored on: played regular-season games of weeks 5-16 with a closing spread."""
    return table[(table['game_type'] == 'REG') & table['week'].between(*WEEKS) & table['result'].notna() & table['spread_line'].notna()]


# ---------------------------------------------------------------- model A: the production stats model, refitted weekly

def project_a(train, games, seed):
    draws = nm.bootstrap_projection(train, games, A_COLUMNS, runs=BOOT_RUNS, seed=seed)
    single = nm.predict(nm.fit(train, A_COLUMNS), games)
    return games.assign(mu=draws.mean(axis=0), boot_sd=draws.std(axis=0), single=single)


def model_a(table):
    played = table[table['result'].notna()]
    pool = played[~((played['game_type'] == 'REG') & (played['week'] == played['season'].map(final_week)))]   # starters rest
    test = eval_games(table)
    parts = [project_a(pool[pool['season'].between(*BACKWARD_TRAIN)], test[test['season'] < STACK_FIRST], seed=1)]
    for (season, week), games in test[test['season'] >= STACK_FIRST].groupby(['season', 'week']):
        window = pool[(pool['season'] >= season - nm.TRAIN_SEASONS)
                      & ((pool['season'] < season) | ((pool['season'] == season) & (pool['week'] < week)))]
        parts.append(project_a(window, games, seed=int(season) * 100 + int(week)))
    return pd.concat(parts)


# ---------------------------------------------------------------- model K: Kalman-filter team strengths

def kalman(table, game_sd, week_sd, carry, offseason_sd, start_sd, hfa, hfa_sd, hfa_offseason_sd):
    """Pre-game predicted home margin and its variance for every played game, processed in date order."""
    games = table[table['result'].notna()].sort_values(['gameday', 'game_id'])
    teams = sorted(set(games['home_team']) | set(games['away_team']))
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    mean = np.zeros(n + 1)
    mean[n] = hfa
    cov = np.eye(n + 1) * start_sd ** 2
    cov[n, n] = hfa_sd ** 2
    eye = np.eye(n)
    mus, variances = np.empty(len(games)), np.empty(len(games))
    season = week = None
    rows = games[['season', 'week', 'home_team', 'away_team', 'neutral', 'result']].itertuples(index=False)
    for i, g in enumerate(rows):
        if g.season != season:
            if season is not None:   # offseason: strengths regress toward average and become less certain
                mean[:n] = carry * (mean[:n] - mean[:n].mean())
                cov[:n, :n] = carry ** 2 * cov[:n, :n] + offseason_sd ** 2 * eye
                cov[:n, n] *= carry
                cov[n, :n] *= carry
                cov[n, n] += hfa_offseason_sd ** 2
            season, week = g.season, None
        if g.week != week:
            if week is not None:
                cov[:n, :n] += week_sd ** 2 * eye
            week = g.week
        h, a, site = index[g.home_team], index[g.away_team], 0.0 if g.neutral else 1.0
        mu = mean[h] - mean[a] + site * mean[n]
        px = cov[:, h] - cov[:, a] + site * cov[:, n]
        variance = px[h] - px[a] + site * px[n] + game_sd ** 2
        mus[i], variances[i] = mu, variance
        gain = px / variance
        mean += gain * (g.result - mu)
        cov -= np.outer(gain, px)
    return pd.DataFrame({'game_id': games['game_id'].to_numpy(), 'season': games['season'].to_numpy(), 'mu': mus,
                         'variance': variances, 'result': games['result'].to_numpy()})


def fit_kalman(table):
    """The grid setting with the best predictive log likelihood on KALMAN_FIT seasons, and the filter run with it."""
    best = None
    for values in itertools.product(*KALMAN_GRID.values()):
        params = {**dict(zip(KALMAN_GRID, values)), **KALMAN_FIXED}
        k = kalman(table, **params)
        fit = k[k['season'].between(*KALMAN_FIT)]
        loglik = float((-0.5 * np.log(2 * np.pi * fit['variance']) - (fit['result'] - fit['mu']) ** 2 / (2 * fit['variance'])).sum())
        if best is None or loglik > best[0]:
            best = (loglik, params, k)
    return best


def model_k(table):
    loglik, params, k = fit_kalman(table)
    print('Kalman settings chosen on %d-%d (log likelihood %.1f): %s' % (*KALMAN_FIT, loglik, ', '.join(
        '%s %g' % (key, params[key]) for key in KALMAN_GRID)))
    test = eval_games(table)
    return test.merge(k[['game_id', 'mu', 'variance']], on='game_id'), params


# ---------------------------------------------------------------- model F: every rating, aimed at the margin

def model_f(table):
    fit = lab.ridge_margin(F_COLUMNS, alpha=F_ALPHA, target='result')
    forward = lab.walk_forward(table, fit)
    played = table[table['result'].notna() & (table['game_type'] == 'REG')]
    train = played[played['season'].between(*BACKWARD_TRAIN)]
    backward = [lab.test_games(table, s).assign(pred=lambda t: np.asarray(fit(train, t), float)) for s in range(1999, STACK_FIRST)]
    scored = pd.concat(backward + [forward], ignore_index=True).rename(columns={'pred': 'mu'})
    return scored[scored['week'].between(*WEEKS)]


# ---------------------------------------------------------------- confidence and scoring

def add_confidence(f, variance=None):
    """Gap, cover margin and the C3 (taken at its word) and C4 (honest) cover probabilities, season by season."""
    f = f.copy()
    f['gap'] = f['mu'] - f['spread_line']
    f['cover'] = f['result'] - f['spread_line']
    for column in ('p_naive', 'p_honest', 'beta'):
        f[column] = np.nan
    for season in sorted(f['season'].unique()):
        if season < STACK_FIRST:
            history = f[f['season'].between(*BACKWARD_TRAIN)]
        else:
            history = f[f['season'].between(STACK_FIRST, season - 1)]
            if history['season'].nunique() < STACK_MIN_SEASONS:
                continue
        beta = np.polyfit(history['gap'], history['cover'], 1)[0]
        cover_sd, model_sd = history['cover'].std(), (history['result'] - history['mu']).std()
        rows = f['season'] == season
        scale = np.sqrt(f.loc[rows, variance]) if variance else model_sd
        f.loc[rows, 'beta'] = beta
        f.loc[rows, 'p_naive'] = norm.cdf(f.loc[rows, 'gap'] / scale)
        f.loc[rows, 'p_honest'] = norm.cdf(beta * f.loc[rows, 'gap'] / cover_sd)
    return f


def side_record(g, home_side):
    """(wins, losses, hit rate) of betting home where home_side is True, else away; pushes excluded."""
    d = g[g['cover'] != 0]
    side = home_side.loc[d.index]
    win = np.where(side, d['cover'] > 0, d['cover'] < 0)
    n = len(win)
    return int(win.sum()), n - int(win.sum()), (win.mean() if n else float('nan'))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 50 else float('nan')


def partial_corr(x, y, control):
    x, y, c = (np.asarray(v, float) for v in (x, y, control))
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(c))
    design = np.column_stack([np.ones(ok.sum()), c[ok]])
    rx = x[ok] - design @ np.linalg.lstsq(design, x[ok], rcond=None)[0]
    ry = y[ok] - design @ np.linalg.lstsq(design, y[ok], rcond=None)[0]
    return corr(rx, ry)


def eras(f):
    return [(name, f[f['season'].between(lo, hi)]) for name, lo, hi in ERAS]


def bucket_lines(f, values, edges, label):
    """Hit rate of the gap's side by bucket of `values` (absolute), each era and pooled."""
    print('    hit rate of the side the projection favors, by %s:' % label)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values.abs() >= lo) & (values.abs() < hi)
        cells = []
        for name, g in eras(f) + [('all', f)]:
            w, l, rate = side_record(g[mask.loc[g.index]], g['gap'] > 0)
            cells.append('%s %s' % (name, '  -   ' if not w + l else '%.3f (%4d)' % (rate, w + l)))
        print('      %-14s %s' % ('%g-%s' % (lo, '' if hi >= 99 else '%g' % hi), ' | '.join(cells)))


def calibration(f, column, label):
    """Predicted against actual cover rate of the side a probability favors, in bins of its stated confidence."""
    print('    calibration of %s (stated chance the favored side covers vs how often it did, all eras):' % label)
    d = f[f[column].notna()]
    stated = np.maximum(d[column], 1 - d[column])
    for lo, hi in zip(P_BINS[:-1], P_BINS[1:]):
        g = d[(stated >= lo) & (stated < hi)]
        w, l, rate = side_record(g, g[column] > 0.5)
        if w + l:
            print('      stated %.2f-%.2f: said %.3f, covered %.3f (%d games)' % (lo, min(hi, 1.0), stated.loc[g.index].mean(), rate, w + l))


def evaluate(name, f, scored):
    """The four pre-registered tests for one margin model; returns whether test 4 passed."""
    f = f.merge(scored[['game_id', 'pred']], on='game_id', how='left')
    print('\n' + '=' * 26 + ' ' + name + ' ' + '=' * 26)
    print('  1. mean absolute error, model vs closing spread:  ' + ' | '.join(
        '%s %.2f vs %.2f' % (era, (g['mu'] - g['result']).abs().mean(), (g['spread_line'] - g['result']).abs().mean()) for era, g in eras(f)))
    infos = [corr(g['gap'], g['cover']) for _, g in eras(f)]
    partials = [partial_corr(g['gap'], g['cover'], g['pred']) for _, g in eras(f)]
    print('  2. correlation of the gap with the cover margin:  ' + ' | '.join('%s %+.3f' % (era, v) for (era, _, _), v in zip(ERAS, infos))
          + '   (given the production correction: ' + ' / '.join('%+.3f' % v for v in partials) + ')')
    print('  3. is the confidence real?')
    bucket_lines(f, f['gap'], GAP_BUCKETS, 'C1, the gap in points')
    if 'boot_sd' in f:
        bucket_lines(f, f['gap'] / f['boot_sd'], Z_BUCKETS, 'C2, the bootstrap z (gap / spread of the bootstrap runs)')
    calibration(f, 'p_naive', 'C3, taken at its word')
    calibration(f, 'p_honest', 'C4, honest')
    rule = f[f['p_honest'].notna() & ((f['p_honest'] >= RULE_P) | (f['p_honest'] <= 1 - RULE_P))]
    cells, rates, total = [], [], [0, 0]
    for era, g in eras(rule):
        w, l, rate = side_record(g, g['p_honest'] > 0.5)
        total = [total[0] + w, total[1] + l]
        if w + l:
            rates.append(rate)
        cells.append('%s %s' % (era, '%d-%d %.3f' % (w, l, rate) if w + l else 'no bets'))
    pooled = total[0] / sum(total) if sum(total) else float('nan')
    print('  4. bets with C4 >= %.0f%%: %s | pooled %d-%d %s' % (100 * RULE_P, ' | '.join(cells), *total,
                                                              '%.3f' % pooled if sum(total) else ''))
    near = f[f['p_honest'].notna() & ((f['p_honest'] >= BREAK_EVEN) | (f['p_honest'] <= 1 - BREAK_EVEN))]
    w, l, rate = side_record(near, near['p_honest'] > 0.5)
    print('     (bets with C4 >= 52.4%%, the break-even: %d-%d %.3f; honest beta by season ranged %+.2f to %+.2f)' % (
        w, l, rate, f['beta'].min(), f['beta'].max()))
    passed = (all(v > 0 for v in infos) and sum(total) >= 30 and pooled >= RULE_P and all(r >= BREAK_EVEN for r in rates)
              and all(v > 0 for v in partials))
    print('  verdict (pre-registered test 4): %s' % ('PASS' if passed else 'fail'))
    return f, passed


def main():
    table = lab.default_table()
    scored = nfl_signal.scored_history(table, lab.LAST_SEASON)

    a = add_confidence(model_a(table))
    a_eval, a_pass = evaluate('A: the production stats model (the original method)', a, scored)
    print('    the 200-run bootstrap average vs a single fit: mean difference %.3f points; spread of the bootstrap runs: median %.2f '
          'points (5-95%%: %.2f-%.2f); the game itself: errors of %.1f points (SD)' % (
              (a['mu'] - a['single']).abs().mean(), a['boot_sd'].median(), a['boot_sd'].quantile(0.05), a['boot_sd'].quantile(0.95),
              (a['result'] - a['mu']).std()))

    k_frame, k_params = model_k(table)
    k = add_confidence(k_frame, variance='variance')
    _, k_pass = evaluate('K: Kalman-filter team strengths (margin with its own uncertainty)', k, scored)
    sd = np.sqrt(k['variance'])
    print('    its stated uncertainty (SD of the margin): median %.2f points, 5-95%% %.2f-%.2f; the rating part alone: median %.2f' % (
        sd.median(), sd.quantile(0.05), sd.quantile(0.95), np.sqrt(np.maximum(k['variance'] - k_params['game_sd'] ** 2, 0)).median()))

    f = add_confidence(model_f(table))
    _, f_pass = evaluate('F: every rating, aimed at the margin', f, scored)

    b = eval_games(table).merge(scored[['game_id', 'pred']], on='game_id')
    b = add_confidence(b.assign(mu=b['spread_line'] + b['pred']))
    evaluate('reference: the production correction read as a margin (spread + correction)', b.drop(columns=['pred']), scored)

    print('\n=== verdict (pre-registered): A %s, K %s, F %s ===' % tuple('PASS' if p else 'fail' for p in (a_pass, k_pass, f_pass)))


if __name__ == '__main__':
    main()
