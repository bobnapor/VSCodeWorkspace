"""Honest validation of the pick rule: untouched seasons, the selection process itself, and a quarterback-change filter.

Pre-registered before running (nothing below was tuned after seeing these results):

1. Untouched-data test.  The rule is frozen as in nfl_signal.is_pick (underdog, line exactly 3/6/7/10/14, correction toward
   the underdog >= 0.25, weeks 5-16).  The production model (same features and alpha) is fitted on the 14 seasons
   2006-2019 and scores 1999-2005, seasons that no research step ever scored or selected on.  Read: a hit rate of 57% or
   more supports an edge near 60%; 53% or less says the 2006-2026 backtest was mostly selection.
2. Selection-process test.  The scan that found the key-number rule is re-run walk-forward: each season, the best of
   about 80 candidate rules (26 conditions x 3 thresholds, underdogs only) is chosen using earlier seasons only and then
   bet in that season.  The pooled result is an unbiased estimate of what "scan and keep the best" is worth.
3. Quarterback-change filter.  The model rates teams, not quarterbacks, so when a starting quarterback changes the
   market reprices at once while the team ratings go stale, which makes the model like the team the market just
   downgraded (or dislike one whose starter just returned).  Filter: skip the game if either team's starter differs
   from the one who started its previous game.  Judged on all three periods, including the untouched seasons.

Run it from this folder:  python nfl_research_validate.py            (sections 1-3, information by era, robust rule)
                          python nfl_research_validate.py models     (production vs 4-stat model, pre-registered)
                          python nfl_research_validate.py threshold  (both-sides threshold rule set from earlier seasons)
                          python nfl_research_validate.py teasers    (the published Wong teaser rule, no model)

Results (October 2026): the frozen key-number rule hit 51.5% on the untouched 1999-2005 seasons (53-50) against 65.8% on
2006-2026, so it was a selection artifact.  The quarterback-change filter helped on the untouched seasons and hurt on the
others, so it was not adopted.  The correction does carry stable information (correlation with the cover margin 0.06-0.08 in
all three eras, t of about 2.5 each), and every underdog it favors covered 53.7% / 55.4% / 55.0% across the eras.  A signal
that size supports about 55-56% at any volume; 60% needs a correlation of about 0.11 (1 pick a week) to 0.14 (3 a week).
The 4-stat model (drives, special-teams EPA, rushing success, interception rate) passed its pre-registered test (correlation
0.10 vs 0.08 on 1999-2005 and 0.11 vs 0.06 on 2018-2026), but its production-like picks also land at about 54-55%.
Wong teaser legs won 74-76% in every era, weeks 5-16.
"""

import numpy as np
import pandas as pd

import nfl_lab as lab
import nfl_signal
from nfl_sheet import KEY_LINES

BREAK_EVEN = 0.5238
WEEKS = (5, 16)
BACKWARD_TRAIN = (2006, 2019)
BACKWARD_TEST = range(1999, 2006)


def add_qb_change(table):
    """Whether each team's starting QB differs from the one who started its previous game (False when unknown)."""
    long = pd.concat([
        table[['game_id', 'gameday', 'home_team', 'home_qb_id']].rename(columns={'home_team': 'team', 'home_qb_id': 'qb'}).assign(side='home'),
        table[['game_id', 'gameday', 'away_team', 'away_qb_id']].rename(columns={'away_team': 'team', 'away_qb_id': 'qb'}).assign(side='away'),
    ]).sort_values(['team', 'gameday'])
    long['prev_qb'] = long.groupby('team')['qb'].shift(1)
    long['changed'] = long['qb'].notna() & long['prev_qb'].notna() & (long['qb'] != long['prev_qb'])
    wide = long.pivot_table(index='game_id', columns='side', values='changed', aggfunc='first')
    t = table.copy()
    t['home_qb_changed'] = t['game_id'].map(wide['home']).fillna(False).astype(bool)
    t['away_qb_changed'] = t['game_id'].map(wide['away']).fillna(False).astype(bool)
    t['qb_changed_any'] = t['home_qb_changed'] | t['away_qb_changed']
    return t


def annotate(frame):
    """Underdog view of a scored frame: correction toward the underdog, whether it covered, line facts.  Pushes dropped."""
    f = frame[(frame['resid'] != 0) & (frame['spread_line'] != 0) & frame['week'].between(*WEEKS) & (frame['game_type'] == 'REG')].copy()
    dog_home = f['spread_line'] < 0
    sign = np.where(dog_home, 1.0, -1.0)
    f['dog_is_home'] = dog_home.astype(int)
    f['toward_dog'] = f['pred'] * sign
    f['win'] = f['resid'] * sign > 0
    f['line'] = f['spread_line'].abs()
    f['key'] = f['line'].isin(KEY_LINES)
    f['whole'] = f['line'] % 1 == 0
    f['dog_odds'] = np.where(dog_home, f['home_spread_odds'], f['away_spread_odds'])
    return f


def summary(label, g, seasons):
    n = len(g)
    if n == 0:
        print('%-58s no picks' % label)
        return
    p = g['win'].mean()
    lo, hi = p - 1.96 * np.sqrt(p * (1 - p) / n), p + 1.96 * np.sqrt(p * (1 - p) / n)
    by = g.groupby('season')['win'].mean()
    print('%-58s n=%4d (%4.1f/season) win %.3f  [%.3f, %.3f]  seasons above 52.4%%: %d/%d  ROI@-110 %+.1f%%' % (
        label, n, n / len(seasons), p, lo, hi, (by > BREAK_EVEN).sum(), len(by), 100 * (p * 100 / 110 - (1 - p))))


def rule_frozen(f):
    return f[f['key'] & (f['toward_dog'] > 0) & (f['toward_dog'] >= nfl_signal.SIGNAL_THRESHOLD)]


def top_n(f, n):
    d = f[f['toward_dog'] > 0]
    return d.sort_values('toward_dog', ascending=False).groupby(['season', 'week']).head(n)


def backward_frame(table, fit=None):
    fit = fit or lab.ridge_margin(nfl_signal.SIGNAL_FEATURES, alpha=nfl_signal.SIGNAL_ALPHA, target='resid')
    played = table[table['result'].notna() & (table['game_type'] == 'REG')]
    train = played[played['season'].between(*BACKWARD_TRAIN)]
    out = []
    for season in BACKWARD_TEST:
        test = lab.test_games(table, season)
        if test.empty:
            continue
        scored = test.copy()
        scored['pred'] = np.asarray(fit(train, test), dtype=float)
        out.append(scored)
    return pd.concat(out, ignore_index=True)


def report_period(name, f):
    seasons = sorted(f['season'].unique())
    print('\n--- %s (%d-%d, weeks %d-%d) ---' % (name, seasons[0], seasons[-1], *WEEKS))
    summary('no model: every underdog', f, seasons)
    summary('no model: every underdog on a key line', f[f['key']], seasons)
    summary('FROZEN RULE: key-line underdog, correction >= 0.25', rule_frozen(f), seasons)
    summary('key-line underdog, any positive correction', f[f['key'] & (f['toward_dog'] > 0)], seasons)
    summary('any line, underdog correction >= 0.5', f[f['toward_dog'] >= 0.5], seasons)
    summary('top 3 underdogs a week by correction', top_n(f, 3), seasons)
    summary('half-point lines, underdog correction >= 0.25', f[~f['whole'] & (f['toward_dog'] >= 0.25)], seasons)
    summary('other whole lines, underdog correction >= 0.25', f[f['whole'] & ~f['key'] & (f['toward_dog'] >= 0.25)], seasons)
    rule = rule_frozen(f)
    summary('FROZEN RULE + QB filter (neither starter changed)', rule[~rule['qb_changed_any']], seasons)
    summary('FROZEN RULE, games the QB filter removes', rule[rule['qb_changed_any']], seasons)
    any_dog = f[f['toward_dog'] >= 0.25]
    summary('all underdogs >= 0.25, starter changed (either team)', any_dog[any_dog['qb_changed_any']], seasons)
    summary('all underdogs >= 0.25, underdog starter changed', any_dog[np.where(any_dog['dog_is_home'] == 1, any_dog['home_qb_changed'], any_dog['away_qb_changed'])], seasons)
    summary('all underdogs >= 0.25, favorite starter changed', any_dog[np.where(any_dog['dog_is_home'] == 1, any_dog['away_qb_changed'], any_dog['home_qb_changed'])], seasons)
    by = rule.groupby('season')['win'].agg(['sum', 'size'])
    print('frozen rule by season: ' + '  '.join('%d: %d-%d' % (s, w, n - w) for s, (w, n) in by.iterrows()))


# ---------------------------------------------------------------- selection-process test

def conditions(f):
    return {
        'all lines': pd.Series(True, index=f.index),
        'key line 3/6/7/10/14': f['key'], 'whole-number line': f['whole'], 'half-point line': ~f['whole'],
        'dog gets 3 or less': f['line'] <= 3, 'dog gets 3.5-6': f['line'].between(3.5, 6), 'dog gets 6.5-9.5': f['line'].between(6.5, 9.5),
        'dog gets 10+': f['line'] >= 10, 'dog gets 4+': f['line'] >= 4, 'dog gets 7+': f['line'] >= 7,
        'dog is home': f['dog_is_home'] == 1, 'dog is away': f['dog_is_home'] == 0,
        'weeks 5-8': f['week'].between(5, 8), 'weeks 9-12': f['week'].between(9, 12), 'weeks 13-16': f['week'].between(13, 16),
        'divisional game': f['div_game'] == 1, 'dome': f['dome'] == 1, 'outdoors': f['dome'] == 0,
        'primetime': f['primetime'] == 1, 'thursday': f['thursday'] == 1, 'monday': f['monday'] == 1,
        'total under 42': f['total_line'] < 42, 'total 42-47': f['total_line'].between(42, 47), 'total over 47': f['total_line'] > 47,
        'cold or windy': (f['cold'] == 1) | (f['wind_high'] == 1), 'QB out/flagged': f['qb_out_any'].fillna(0) > 0,
    }


def nested_selection(f, first_test=2011, thresholds=(0.0, 0.25, 0.5), min_n=60):
    """Each season, pick the best rule on earlier seasons only, then bet it.  Two selectors: raw rate, and Wilson lower bound."""
    conds = conditions(f)
    rules = [(c, t) for c in conds for t in thresholds]
    picked = {'raw rate': [], 'lower bound': []}
    chosen_log = []
    for season in range(first_test, int(f['season'].max()) + 1):
        past = f['season'] < season
        best = {'raw rate': (-1, None), 'lower bound': (-1, None)}
        for c, t in rules:
            mask = conds[c] & (f['toward_dog'] > 0) & (f['toward_dog'] >= t)
            g = f[mask & past]
            n = len(g)
            if n < min_n:
                continue
            p = g['win'].mean()
            z = 1.2816
            lb = (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
            for name, score in (('raw rate', p), ('lower bound', lb)):
                if score > best[name][0]:
                    best[name] = (score, (c, t))
        for name, (_, rule) in best.items():
            c, t = rule
            mask = conds[c] & (f['toward_dog'] > 0) & (f['toward_dog'] >= t) & (f['season'] == season)
            picked[name].append(f[mask])
        chosen_log.append((season, best['raw rate'][1], best['lower bound'][1]))
    seasons = list(range(first_test, int(f['season'].max()) + 1))
    print('\n=== selection-process test: best of %d rules chosen on earlier seasons, bet in the next (%d-%d) ===' % (len(rules), first_test, seasons[-1]))
    for name, frames in picked.items():
        summary('process, chosen by %s' % name, pd.concat(frames), seasons)
    in_window = f[f['season'] >= first_test]
    summary('hindsight benchmark: frozen rule over the same seasons', rule_frozen(in_window), seasons)
    print('rules chosen each season (raw rate | lower bound):')
    for season, raw, lb in chosen_log:
        print('  %d: %s >= %.2f | %s >= %.2f' % (season, raw[0], raw[1], lb[0], lb[1]))


def shrinkage(f):
    """Empirical-Bayes shrinkage of the key-line rule toward the other scanned conditions (rough: conditions overlap)."""
    conds = conditions(f)
    rows = []
    for c, mask in conds.items():
        g = f[mask & (f['toward_dog'] >= 0.25)]
        if len(g) >= 30:
            rows.append((c, g['win'].mean(), len(g)))
    rates = np.array([r[1] for r in rows])
    ns = np.array([r[2] for r in rows])
    noise = (rates * (1 - rates) / ns).mean()
    tau2 = max(rates.var() - noise, 0.0)
    mean = np.average(rates, weights=ns)
    key = next(r for r in rows if r[0] == 'key line 3/6/7/10/14')
    se2 = key[1] * (1 - key[1]) / key[2]
    shrunk = mean + tau2 / (tau2 + se2) * (key[1] - mean)
    print('\n=== empirical-Bayes shrinkage across %d scanned conditions (underdogs, correction >= 0.25, 2006-2026) ===' % len(rows))
    print('condition rates: mean %.3f, spread of true effects (tau) %.3f vs sampling noise %.3f' % (mean, np.sqrt(tau2), np.sqrt(noise)))
    print('key-line rule: observed %.3f (n=%d) -> shrunk estimate %.3f' % (key[1], key[2], shrunk))


def information_by_era(eras):
    """Correlation of the correction with the cover margin, and the regression slope with its t-statistic, per era."""
    print('\n=== information in the correction (all games, weeks 5-16): correlation with the cover margin ===')
    for name, f in eras:
        slope, t = lab.information_test(f, is_margin=False)
        print('%-22s corr %.3f   slope %+.2f  t=%+.1f   n=%d' % (name, np.corrcoef(f['pred'], f['resid'])[0, 1], slope, t, len(f)))


def side_size_table(eras):
    """Hit rate of the correction's side by size of correction, underdog side versus favorite side, per era."""
    print('\n=== hit rate of the side the correction favors, by size (weeks 5-16) ===')
    for name, f in eras:
        f = f.assign(pick_win=np.where(f['toward_dog'] > 0, f['win'], ~f['win']), size=f['toward_dog'].abs())
        print(name)
        for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 99)):
            b = f[(f['size'] >= lo) & (f['size'] < hi)]
            dog, fav = b[b['toward_dog'] > 0], b[b['toward_dog'] < 0]
            print('  |correction| %.2f-%-5.2f underdog side %.3f (n=%4d) | favorite side %.3f (n=%4d)' % (
                lo, hi, dog['pick_win'].mean(), len(dog), fav['pick_win'].mean(), len(fav)))


def robust_rule(eras):
    """The rule that holds in every era: bet each underdog the correction favors (any positive correction)."""
    print('\n=== every underdog the correction favors (any positive correction), weeks 5-16 ===')
    for name, f in eras:
        g = f[f['toward_dog'] > 0]
        seasons = sorted(f['season'].unique())
        summary(name, g, seasons)
        priced = g[g['dog_odds'].notna()]
        if len(priced):
            odds = pd.to_numeric(priced['dog_odds'])
            profit = np.where(priced['win'], np.where(odds < 0, 100.0 / -odds, odds / 100.0), -1.0)
            print('%58s ROI at the real closing odds %+.1f%% (n=%d)' % ('', 100 * profit.mean(), len(priced)))
        by = g.groupby('season')['win'].agg(['sum', 'size'])
        print('   by season: ' + '  '.join('%d: %d-%d' % (s, w, n - w) for s, (w, n) in by.iterrows()))


def required_signal(games_per_season=178):
    """How informative a signal must be for its picks to hit 60%, in a simple bivariate-normal picture.

    A signal s correlated rho with the cover margin; bet the side it favors on the games where |s| is largest.
    """
    from scipy.stats import norm

    def hit_rate(rho, picks):
        q = picks / games_per_season
        t = norm.ppf(1 - q / 2)
        s = np.linspace(t, 8, 4000)
        w = norm.pdf(s)
        return float((norm.cdf(rho * s / np.sqrt(1 - rho ** 2)) * w).sum() / w.sum())

    print('\n=== how strong a signal 60%% needs (bivariate-normal approximation, %d games a season in weeks 5-16) ===' % games_per_season)
    for picks in (12, 24, 36):
        rhos = np.linspace(0.01, 0.4, 400)
        need = next(r for r in rhos if hit_rate(r, picks) >= 0.60)
        print('  %2d picks a season (%.1f a week): 60%% needs correlation %.2f with the cover margin; at 0.07 (this model) the picks hit %.1f%%' % (
            picks, picks / 12, need, 100 * hit_rate(0.07, picks)))


def compare_models(table, eras_full):
    """Pre-registered model comparison on information (correlation with the cover margin), all three eras.

    Candidates: the production model; the 4-stat model forward-selected on 2006-2017 only (nfl_research_features.py,
    alpha 1000), so 1999-2005 and 2018-2026 are out of sample for its selection; and their average.  Adopt a challenger
    only if it beats production in both of those eras.
    """
    from nfl_research_features import ALL, SLIM, pair_features

    slim4 = pair_features([ALL[n] for n in SLIM[4]])
    fits = {
        'production (37 stat pairs + context, a=10000)': lab.ridge_margin(nfl_signal.SIGNAL_FEATURES, alpha=nfl_signal.SIGNAL_ALPHA, target='resid'),
        '4-stat (drives, ST EPA, rush success, INT rate; a=1000)': lab.ridge_margin(slim4, alpha=1000.0, target='resid'),
    }
    scored = {}
    for name, fit in fits.items():
        fwd = annotate(lab.walk_forward(table, fit))
        bwd = annotate(backward_frame(table, fit))
        scored[name] = {'untouched 1999-2005': bwd, '2006-2017': fwd[fwd['season'] <= lab.DISCOVERY_END],
                        '2018-2026': fwd[fwd['season'] > lab.DISCOVERY_END]}
    names = list(fits)
    blend = {}
    for era in scored[names[0]]:
        a, b = scored[names[0]][era], scored[names[1]][era]
        merged = a.copy()
        pb = b.set_index('game_id')['pred'].reindex(a['game_id']).to_numpy()
        merged['pred'] = (a['pred'].to_numpy() / a['pred'].std() + pb / np.nanstd(pb)) / 2 * a['pred'].std()
        merged['toward_dog'] = merged['pred'] * np.where(merged['dog_is_home'] == 1, 1.0, -1.0)
        blend[era] = merged
    scored['average of the two'] = blend
    print('\n=== model comparison: correlation with the cover margin (weeks 5-16) and every underdog the model favors ===')
    for name, eras in scored.items():
        cells = []
        for era, f in eras.items():
            dogs = f[f['toward_dog'] > 0]
            cells.append('%s r=%.3f dogs %.3f (n=%d)' % (era.split()[-1], np.corrcoef(f['pred'], f['resid'])[0, 1], dogs['win'].mean(), len(dogs)))
        print('%-56s %s' % (name, ' | '.join(cells)))
    print('\n=== binary view: correlation with covering (yes/no), and hit rate of the model side for the top-k games a season by |correction| ===')
    for name, eras in scored.items():
        print(name)
        for era, f in eras.items():
            f = f.assign(pick_win=np.where(f['toward_dog'] > 0, f['win'], ~f['win']), size=f['toward_dog'].abs(),
                         home_cover=(f['resid'] > 0).astype(float))
            r_bin = np.corrcoef(f['pred'], f['home_cover'])[0, 1]
            cells = []
            for k in (12, 24, 36, 60):
                top = f.sort_values('size', ascending=False).groupby('season').head(k)
                cells.append('top%d %.3f' % (k, top['pick_win'].mean()))
            dog, fav = f[f['toward_dog'] > 0], f[f['toward_dog'] < 0]
            print('   %-20s binary r=%.3f | both sides %s | all dog-side %.3f (n=%d) fav-side %.3f (n=%d)' % (
                era, r_bin, ' '.join(cells), dog['pick_win'].mean(), len(dog), fav['pick_win'].mean(), len(fav)))
    return scored


def wong_teasers(table):
    """The published (2001) Wong teaser rule: 6-point teaser legs through 3 and 7.  No model, nothing tuned here.

    Legs: underdogs of +1.5/+2.5 teased to +7.5/+8.5, favorites of -7.5/-8.5 teased to -1.5/-2.5.  A 2-team teaser
    needs each leg to win 72.4% at -110 or 73.9% at -120.
    """
    g = table[(table['game_type'] == 'REG') & table['result'].notna() & table['spread_line'].notna()].copy()
    g['line'] = g['spread_line'].abs()
    g['fav_margin'] = np.where(g['spread_line'] > 0, g['result'], -g['result'])
    fav_leg = g['line'].isin([7.5, 8.5])
    dog_leg = g['line'].isin([1.5, 2.5])
    g['leg'] = np.where(fav_leg, 'favorite', np.where(dog_leg, 'underdog', ''))
    g['leg_win'] = np.where(fav_leg, g['fav_margin'] > g['line'] - 6, g['fav_margin'] < g['line'] + 6)
    legs = g[(g['leg'] != '') & g['week'].between(*WEEKS)]
    print('\n=== Wong teaser legs, weeks 5-16 (break-even per leg 72.4%% at -110, 73.9%% at -120) ===')
    for era, mask in (('1999-2005', legs['season'] <= 2005), ('2006-2017', legs['season'].between(2006, 2017)), ('2018-2026', legs['season'] >= 2018)):
        l = legs[mask]
        parts = ' | '.join('%s legs %.3f (n=%d)' % (k, v['leg_win'].mean(), len(v)) for k, v in l.groupby('leg'))
        print('%-10s all legs %.3f (n=%d, %.1f/season) | %s | 2-team teaser wins about %.3f' % (
            era, l['leg_win'].mean(), len(l), len(l) / l['season'].nunique(), parts, l['leg_win'].mean() ** 2))


def main():
    table = add_qb_change(lab.default_table())
    forward = annotate(lab.walk_forward(table, lab.ridge_margin(nfl_signal.SIGNAL_FEATURES, alpha=nfl_signal.SIGNAL_ALPHA, target='resid')))
    backward = annotate(backward_frame(table))
    print('=== 1. untouched seasons: model fitted on %d-%d, scoring %d-%d ===' % (*BACKWARD_TRAIN, BACKWARD_TEST[0], BACKWARD_TEST[-1]))
    report_period('UNTOUCHED', backward)
    report_period('discovery', forward[forward['season'] <= lab.DISCOVERY_END])
    report_period('holdout', forward[forward['season'] > lab.DISCOVERY_END])
    nested_selection(forward)
    shrinkage(forward)
    eras = (('untouched 1999-2005', backward), ('2006-2017', forward[forward['season'] <= lab.DISCOVERY_END]),
            ('2018-2026', forward[forward['season'] > lab.DISCOVERY_END]))
    information_by_era(eras)
    side_size_table(eras)
    robust_rule(eras)
    required_signal()


def threshold_rule(table):
    """Production-like rule: bet the side the correction favors when |correction| >= T, with T set from earlier seasons only.

    T is the |correction| quantile of the walk-forward predictions from seasons before the test season (weeks 5-16) that
    would have produced the target number of picks per season.  The untouched seasons use the 2006-2019 distribution.
    """
    from nfl_research_features import ALL, SLIM, pair_features

    slim4 = pair_features([ALL[n] for n in SLIM[4]])
    fits = {'production model': lab.ridge_margin(nfl_signal.SIGNAL_FEATURES, alpha=nfl_signal.SIGNAL_ALPHA, target='resid'),
            '4-stat model': lab.ridge_margin(slim4, alpha=1000.0, target='resid')}
    print('\n=== threshold rule, both sides: |correction| >= T, T from earlier seasons only (weeks 5-16) ===')
    for name, fit in fits.items():
        fwd = annotate(lab.walk_forward(table, fit))
        bwd = annotate(backward_frame(table, fit))
        fwd['size'], bwd['size'] = fwd['toward_dog'].abs(), bwd['toward_dog'].abs()
        for target in (12, 24, 36):
            out = []
            ref = fwd[fwd['season'].between(*BACKWARD_TRAIN)]
            share = target / ref.groupby('season').size().mean()
            t_back = ref['size'].quantile(1 - share)
            out.append(('untouched 1999-2005', bwd[bwd['size'] >= t_back]))
            picks = []
            for season in sorted(fwd['season'].unique()):
                past = fwd[(fwd['season'] < season)]
                if past['season'].nunique() < 4:
                    continue
                share = target / past.groupby('season').size().mean()
                t = past['size'].quantile(1 - share)
                picks.append(fwd[(fwd['season'] == season) & (fwd['size'] >= t)])
            picks = pd.concat(picks)
            out += [('2010-2017', picks[picks['season'] <= lab.DISCOVERY_END]), ('2018-2026', picks[picks['season'] > lab.DISCOVERY_END])]
            cells = []
            for era, g in out:
                g = g.assign(pick_win=np.where(g['toward_dog'] > 0, g['win'], ~g['win']))
                cells.append('%s %.3f (n=%d, %.1f/season)' % (era, g['pick_win'].mean(), len(g), len(g) / g['season'].nunique()))
            print('%-16s target %2d/season: %s' % (name, target, ' | '.join(cells)))


def coefficients(table):
    """Standardized ridge coefficients of the 4-stat model fitted on 2012-2025, for interpretation."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from nfl_research_features import ALL, SLIM, pair_features

    cols = pair_features([ALL[n] for n in SLIM[4]])
    train = table[table['result'].notna() & (table['game_type'] == 'REG') & table['season'].between(2012, 2025)]
    x = StandardScaler().fit_transform(train[cols].fillna(train[cols].median()))
    model = Ridge(alpha=1000.0).fit(x, train['resid'])
    print('\n=== 4-stat model, points of cover margin per 1 SD of each home-minus-away rating (fit 2012-2025) ===')
    for c, b in sorted(zip(cols, model.coef_), key=lambda z: -abs(z[1])):
        print('  %-24s %+.2f' % (c, b))


if __name__ == '__main__':
    import sys

    if 'models' in sys.argv[1:]:
        compare_models(add_qb_change(lab.default_table()), None)
    elif 'threshold' in sys.argv[1:]:
        t = add_qb_change(lab.default_table())
        threshold_rule(t)
        coefficients(t)
    elif 'teasers' in sys.argv[1:]:
        wong_teasers(lab.default_table())
    else:
        main()
