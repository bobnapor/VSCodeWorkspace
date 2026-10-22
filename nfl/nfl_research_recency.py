"""Do the models do better when a team's most recent 3-4 games count for more?

Every rating in the system is already recency-weighted: each earlier game counts 0.90 times the next (a 6.6-game
half-life), so at mid-season a team's last four games carry about 30% of its rating (42% of the weight its games get;
the rest is the pull toward average).  Earlier research tried faster memory only for the correction model (decay 0.80,
half-life 3 games, in nfl_research_data.py's rating-memory grid; flat).  This tests sharper recency in both kinds of
model.  Pre-registered before any result below was computed (October 20, 2026); no current logic changes either way.

Ratings are rebuilt from the same data three times: production (decay 0.90, 3 prior pseudo-games), decay 0.75
(half-life 2.4 games; the last four games carry about 54% of a mid-season rating) and decay 0.60 (half-life 1.4 games;
about 67%).  The prior pseudo-games shrink with the memory (3 x 0.10 / (1 - decay): 1.2 and 0.75) so the pull toward
average keeps the same share of a full-memory rating; everything else is as production.  All three are rebuilt fresh so
they see the same games.

R1, the correction model on recent ratings: the production correction (same features, alpha and training windows;
    2006 on walk-forward, 1999-2005 from a 2006-2019 fit) on each set of ratings; its correlation with the cover margin,
    weeks 5-16, by era.
R2, recent form on top of the current ratings: the production features plus the same 74 ratings at decay 0.60, in one
    ridge (alpha 10000), so the fit can weigh recent form against the season-long ratings.
R3, the projection model (Model A: its production features, ridge alpha 300 on the final margin, refitted every week on
    the 12 seasons before plus the season's earlier weeks; 1999-2005 from a 2006-2019 fit) on each set of ratings: mean
    absolute error, and the correlation of its gap to the spread with the cover margin.
Pass: a variant beats production in every era (R1 and R2: a higher correlation with the cover margin; R3: a lower mean
absolute error).  The production pick rule's record on each correction is shown alongside, not judged.

Run it from this folder:  python nfl_research_recency.py

Results (October 20, 2026; eras 1999-2005 / 2006-2017 / 2018-2026):
R1: the more weight on the last few games, the less the correction knows: production +0.083 / +0.055 / +0.064, decay
    0.75 +0.074 / +0.052 / +0.050, decay 0.60 +0.071 / +0.042 / +0.042.  The underdog picks went from 53.7% / 55.4% /
    55.1% to 53.4% / 54.9% / 53.7% and 54.1% / 54.2% / 51.7%.  Fail.
R2: recent form on top of the current ratings, +0.084 / +0.051 / +0.061, is no better than the current ratings alone
    (underdog picks 54.3% / 53.7% / 53.9%).  Fail.
R3: the projection model's error barely moves (production 10.134 / 10.775 / 10.038; decay 0.75 10.126 / 10.792 /
    10.053; decay 0.60 10.127 / 10.804 / 10.058; the closing spread's 10.01 / 10.59 / 9.80), and its gap to the spread
    carries less information with more recency (+0.047 / +0.003 / +0.018 at production, +0.050 / -0.005 / +0.004 at
    decay 0.60).  Fail.
Reading: a team's last three or four games are mostly noise on top of what its longer record says, and the market
already leans on them (if anything too much).  The correction's edge comes from the longer memory, which captures
stable quality the market underweights.  The Kalman model in nfl_research_margin.py agrees: by predictive likelihood
it chose the slower of its weekly drifts (0.5 points a week over 1.0).
"""

import numpy as np
import pandas as pd

import nfl_lab as lab
import nfl_model as nm
import nfl_signal

WEEKS = nfl_signal.PICK_WEEKS
ERAS = (('1999-2005', 1999, 2005), ('2006-2017', 2006, 2017), ('2018-2026', 2018, 2026))
BACKWARD_TRAIN = nfl_signal.BACKWARD_TRAIN
SETTINGS = {
    'production (decay 0.90)': {'decay': 0.90, 'prior_games': 3.0},
    'decay 0.75': {'decay': 0.75, 'prior_games': 1.2},
    'decay 0.60': {'decay': 0.60, 'prior_games': 0.75},
}
PRODUCTION, FORM = 'production (decay 0.90)', 'decay 0.60'
RATING_COLUMNS = lab.PBP_FEATURES + lab.BOX_FEATURES
A_COLUMNS = nm.feature_columns(nm.PRODUCTION_METRICS, use_qb=True, use_elo=True)


def final_week(season):
    return 18 if season >= 2021 else 17


def eras(f):
    return [(name, f[f['season'].between(lo, hi)]) for name, lo, hi in ERAS]


def scored(table, columns, alpha=nfl_signal.SIGNAL_ALPHA):
    """Out-of-sample correction for every regular-season game, weeks 5-16: walk-forward from 2006, 1999-2005 from 2006-2019."""
    fit = lab.ridge_margin(columns, alpha=alpha, target='resid')
    forward = lab.walk_forward(table, fit)
    played = table[table['result'].notna() & (table['game_type'] == 'REG')]
    train = played[played['season'].between(*BACKWARD_TRAIN)]
    backward = [lab.test_games(table, s).assign(pred=lambda t: np.asarray(fit(train, t), float)) for s in range(1999, 2006)]
    f = pd.concat(backward + [forward], ignore_index=True)
    return f[f['week'].between(*WEEKS)]


def information(f, column='pred', target='resid'):
    return [float(np.corrcoef(g[column], g[target])[0, 1]) for _, g in eras(f)]


def underdog_record(f):
    """The production pick rule on these corrections (every underdog the correction favors), by era."""
    d = f[(f['resid'] != 0) & (f['spread_line'] != 0)]
    toward_dog = np.where(d['spread_line'] < 0, d['pred'], -d['pred'])
    picks = d[toward_dog > 0]
    covered = pd.Series(np.where(picks['spread_line'] < 0, picks['resid'] > 0, picks['resid'] < 0), index=picks.index)
    return ['%.3f (%d)' % (covered[g.index].mean(), len(g)) for _, g in eras(picks)]


def projection(table):
    """Model A's out-of-sample projected margins, refitted every week (single fits; the bootstrap average matches them)."""
    played = table[table['result'].notna()]
    pool = played[~((played['game_type'] == 'REG') & (played['week'] == played['season'].map(final_week)))]
    test = played[(played['game_type'] == 'REG') & played['week'].between(*WEEKS) & played['spread_line'].notna()]
    old = test[test['season'] < 2006]
    parts = [old.assign(mu=nm.predict(nm.fit(pool[pool['season'].between(*BACKWARD_TRAIN)], A_COLUMNS), old))]
    for (season, week), games in test[test['season'] >= 2006].groupby(['season', 'week']):
        window = pool[(pool['season'] >= season - nm.TRAIN_SEASONS)
                      & ((pool['season'] < season) | ((pool['season'] == season) & (pool['week'] < week)))]
        parts.append(games.assign(mu=nm.predict(nm.fit(window, A_COLUMNS), games)))
    f = pd.concat(parts)
    return f.assign(gap=f['mu'] - f['spread_line'], resid=f['result'] - f['spread_line'])


def verdict(label, values, base, better):
    passed = all(better(v, b) for v, b in zip(values, base))
    return '%s: %s' % (label, 'PASS (beats production in every era)' if passed else 'fail')


def main():
    tables = {}
    for name, params in SETTINGS.items():
        tables[name] = lab.build_table(lab.ALL_PAIRS, params)
        print('built ratings: %s %s' % (name, params))

    print('\n=== R1: the correction model on each set of ratings, weeks %d-%d (correlation with the cover margin; '
          'the pick rule\'s underdog record beside it) ===' % WEEKS)
    r1 = {}
    for name, table in tables.items():
        f = scored(table, nfl_signal.SIGNAL_FEATURES)
        r1[name] = information(f)
        print('  %-24s r %s | underdog picks %s' % (name, ' / '.join('%+.3f' % v for v in r1[name]), ' / '.join(underdog_record(f))))

    print('\n=== R2: the current ratings plus recent form (the 74 ratings at %s) in one ridge ===' % FORM)
    form = tables[FORM][['game_id'] + RATING_COLUMNS].rename(columns={c: c + '__form' for c in RATING_COLUMNS})
    combined = tables[PRODUCTION].merge(form, on='game_id', how='left')
    f2 = scored(combined, nfl_signal.SIGNAL_FEATURES + [c + '__form' for c in RATING_COLUMNS])
    r2 = information(f2)
    print('  %-24s r %s | underdog picks %s' % ('current + recent form', ' / '.join('%+.3f' % v for v in r2), ' / '.join(underdog_record(f2))))
    print('  %-24s r %s' % ('current alone', ' / '.join('%+.3f' % v for v in r1[PRODUCTION])))

    print('\n=== R3: the projection model (Model A) on each set of ratings, weeks %d-%d ===' % WEEKS)
    r3 = {}
    for name, table in tables.items():
        f = projection(table)
        r3[name] = [float((g['mu'] - g['result']).abs().mean()) for _, g in eras(f)]
        market = [float((g['spread_line'] - g['result']).abs().mean()) for _, g in eras(f)]
        print('  %-24s mean absolute error %s (closing spread %s) | gap information %s' % (
            name, ' / '.join('%.3f' % v for v in r3[name]), ' / '.join('%.2f' % v for v in market),
            ' / '.join('%+.3f' % v for v in information(f, 'gap', 'resid'))))

    print('\n=== verdicts (pre-registered; eras 1999-2005 / 2006-2017 / 2018-2026) ===')
    for name in SETTINGS:
        if name == PRODUCTION:
            continue
        print('  ' + verdict('R1 correction on %s ratings' % name, r1[name], r1[PRODUCTION], lambda v, b: v > b))
        print('  ' + verdict('R3 projection on %s ratings' % name, r3[name], r3[PRODUCTION], lambda v, b: v < b))
    print('  ' + verdict('R2 current ratings + recent form', r2, r1[PRODUCTION], lambda v, b: v > b))


if __name__ == '__main__':
    main()
