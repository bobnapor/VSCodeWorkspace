"""Which team stats carry information the spread does not, and how few are enough?

Everything is walk-forward (each season predicted by a model fitted on the 14 seasons before it), scored on regular-season
weeks 5-16 only, and split into a discovery period (2006-2017, where stats are chosen) and a holdout (2018 on, where the
choice is confirmed).  The target is the cover margin (result minus closing spread), so a stat only counts if it knows
something the market does not already price.  Run it from this folder:  python nfl_research_features.py [stage]
"""

import sys

import numpy as np
import pandas as pd

import nfl_lab as lab

WEEKS = (5, 16)
ALPHAS = (1000.0, 3000.0, 10000.0, 30000.0)
CONTEXT = lab.BASIC_CONTEXT


def pair_features(pairs):
    return lab.off_def_columns(pairs)


BOX = {p[0]: p for p in lab.BOX_PAIRS}
PBP = {p[0]: p for p in lab.PBP_PAIRS}
ALL = {**BOX, **PBP}
# box-score stats exist as 'yds' (net yards) and pbp stats as 'o_*'; first_downs/plays also exist in both families
GUT = [BOX['yds'], BOX['fd'], BOX['to']]


def fit_frame(table, columns, alpha):
    return lab.walk_forward(table, lab.ridge_margin(columns, alpha=alpha, target='resid'))


def in_weeks(frame):
    return frame[frame['week'].between(*WEEKS) & (frame['game_type'] == 'REG')]


def summarize(frame):
    """Information test and pick rates on weeks 5-16, split into discovery and holdout."""
    f = in_weeks(frame)
    out = {}
    for name, g in (('disc', f[f['season'] <= lab.DISCOVERY_END]), ('hold', f[f['season'] > lab.DISCOVERY_END])):
        slope, t = lab.information_test(g, is_margin=False)
        d = g[g['resid'] != 0].copy()
        d['win'] = np.where(d['pred'] > 0, d['resid'] > 0, d['resid'] < 0)
        d['dog'] = (d['pred'] > 0) == (d['spread_line'] < 0)
        d['conf'] = d['pred'].abs()
        dogs = d[d['dog']]
        top3 = dogs.sort_values('conf', ascending=False).groupby(['season', 'week']).head(3)
        out[name] = dict(slope=slope, t=t, corr=np.corrcoef(g['pred'], g['resid'])[0, 1],
                         top3=top3['win'].mean(), n3=len(top3), ge5=dogs[dogs['conf'] >= 0.5]['win'].mean(), n5=int((dogs['conf'] >= 0.5).sum()))
    return out


def line(label, s):
    d, h = s['disc'], s['hold']
    print('%-44s disc: slope %+.2f t=%+.1f r=%+.3f top3 %.3f (n=%d) | hold: slope %+.2f t=%+.1f r=%+.3f top3 %.3f (n=%d) >=.5 %.3f (n=%d)' % (
        label, d['slope'], d['t'], d['corr'], d['top3'], d['n3'], h['slope'], h['t'], h['corr'], h['top3'], h['n3'], h['ge5'], h['n5']))


def stage_sets(table):
    print('=== stat sets (weeks 5-16), best alpha per set shown for each of %s ===' % (ALPHAS,))
    sets = {
        'gut: yards, first downs, turnovers': pair_features(GUT) + CONTEXT,
        'gut, no context': pair_features(GUT),
        'gut + points + plays': pair_features(GUT + [BOX['pts'], BOX['plays']]) + CONTEXT,
        'current box (fd yds plays to pts)': lab.BOX_FEATURES + CONTEXT,
        'current pbp (31 stats)': lab.PBP_FEATURES + CONTEXT,
        'current model (box + pbp)': lab.PBP_FEATURES + lab.BOX_FEATURES + lab.BASIC_CONTEXT,
        'epa + success only': pair_features([PBP['o_epa_pp'], PBP['o_success']]) + CONTEXT,
        'context only (elo, rest, neutral)': CONTEXT,
    }
    for label, columns in sets.items():
        for alpha in ALPHAS:
            line('%s a=%g' % (label, alpha), summarize(fit_frame(table, columns, alpha)))
        print()


def stage_singles(table):
    print('=== each stat alone (offense and defense side), alpha 3000, weeks 5-16, ranked by discovery t ===')
    rows = []
    for name, pair in ALL.items():
        s = summarize(fit_frame(table, pair_features([pair]), 3000.0))
        rows.append((s['disc']['t'], name, s))
    for _, name, s in sorted(rows, key=lambda r: -abs(r[0])):
        line(name, s)


def disc_corr(table, columns, alpha):
    """Correlation of the model's correction with the cover margin on discovery-period weeks 5-16 only."""
    frame = lab.walk_forward(table, lab.ridge_margin(columns, alpha=alpha, target='resid'), last_test=lab.DISCOVERY_END)
    f = in_weeks(frame)
    return float(np.corrcoef(f['pred'], f['resid'])[0, 1])


def stage_forward(table, steps=10, alpha=1000.0):
    """Greedy forward selection on the discovery period; the holdout is only looked at to report the path."""
    print('=== forward selection on 2006-2017 weeks 5-16 (alpha %g), holdout 2018+ reported at each step ===' % alpha)
    chosen, columns = [], []
    for step in range(1, steps + 1):
        best = max(((disc_corr(table, columns + pair_features([pair]), alpha), name) for name, pair in ALL.items() if name not in chosen))
        chosen.append(best[1])
        columns = columns + pair_features([ALL[best[1]]])
        line('step %d +%s (disc r %.3f)' % (step, best[1], best[0]), summarize(fit_frame(table, columns, alpha)))
    print('chosen:', chosen)


SLIM = {
    2: ['o_drives', 'o_st_epa_for'],
    4: ['o_drives', 'o_st_epa_for', 'o_success_rush', 'o_int_rate'],
    6: ['o_drives', 'o_st_epa_for', 'o_success_rush', 'o_int_rate', 'o_early_epa', 'o_explosive'],
}


def frontier(frame, label):
    """Hit rate of the top-k picks per week (weeks 5-16), underdogs only and either side, discovery vs holdout."""
    f = in_weeks(frame)
    d = f[f['resid'] != 0].copy()
    d['win'] = np.where(d['pred'] > 0, d['resid'] > 0, d['resid'] < 0)
    d['dog'] = (d['pred'] > 0) == (d['spread_line'] < 0)
    d['conf'] = d['pred'].abs()
    for side, g in (('dog', d[d['dog']]), ('any', d)):
        cells = []
        for k in (2, 3, 4, 5):
            t = g.sort_values('conf', ascending=False).groupby(['season', 'week']).head(k)
            a, b = t[t['season'] <= lab.DISCOVERY_END], t[t['season'] > lab.DISCOVERY_END]
            cells.append('top%d %.3f/%.3f (h n=%d)' % (k, a['win'].mean(), b['win'].mean(), len(b)))
        print('%-34s %-3s disc/hold: %s' % (label, side, ' | '.join(cells)))


def stage_frontier(table):
    print('=== pick frontier, weeks 5-16 (discovery / holdout hit rate of the top-k picks per week) ===')
    full = lab.PBP_FEATURES + lab.BOX_FEATURES + lab.BASIC_CONTEXT
    frontier(fit_frame(table, full, 10000.0), 'full model a=10000')
    for size, names in SLIM.items():
        columns = pair_features([ALL[n] for n in names])
        for alpha in (1000.0, 3000.0, 10000.0):
            frontier(fit_frame(table, columns, alpha), 'slim%d a=%g' % (size, alpha))
    # ensemble: the 4-stat model and the full model averaged after scaling each to unit spread
    a = fit_frame(table, pair_features([ALL[n] for n in SLIM[4]]), 3000.0)
    b = fit_frame(table, full, 10000.0)
    blend = a.copy()
    blend['pred'] = a['pred'] / a['pred'].std() + b['pred'] / b['pred'].std()
    blend['pred'] *= (a['pred'].std() + b['pred'].std()) / 2
    frontier(blend, 'blend slim4 + full')


def add_dog_columns(table, columns):
    """Re-express the game from the underdog's side: features become underdog-minus-favorite and y_dog > 0 means it covered."""
    t = table.copy()
    sign = np.where(t['spread_line'] < 0, 1.0, -1.0)   # +1 when the home team is the underdog
    for c in columns:
        t['dog__' + c] = sign * t[c]
    t['y_dog'] = sign * t['resid']
    t['dog_is_home'] = (sign > 0).astype(int)
    return t


def dog_frontier(frame, label):
    f = in_weeks(frame)
    d = f[f['resid'] != 0].copy()
    d['win'] = d['y_dog'] > 0
    cells = []
    for k in (2, 3, 4, 5):
        t = d.sort_values('pred', ascending=False).groupby(['season', 'week']).head(k)
        a, b = t[t['season'] <= lab.DISCOVERY_END], t[t['season'] > lab.DISCOVERY_END]
        cells.append('top%d %.3f/%.3f (h n=%d)' % (k, a['win'].mean(), b['win'].mean(), len(b)))
    print('%-40s disc/hold: %s' % (label, ' | '.join(cells)))


def stage_dog(table):
    print('=== underdog-frame model: ridge on the underdog cover margin, top-k predicted underdogs each week (weeks 5-16) ===')
    full = lab.PBP_FEATURES + lab.BOX_FEATURES + ['elo_diff', 'rest_diff']
    slim4 = pair_features([ALL[n] for n in SLIM[4]])
    signed_sets = {'full': full, 'slim4': slim4, 'none': []}
    extras = {'abs_spread': ['abs_spread'], 'abs_spread+total': ['abs_spread', 'total_line'],
              'market shape': ['abs_spread', 'total_line', 'on_key_number', 'half_point', 'neutral']}
    dog_table = add_dog_columns(table, ['elo_diff', 'rest_diff', 'ml_logit', 'juice_lean'] + full + slim4)
    for set_name, cols in signed_sets.items():
        for extra_name, extra in extras.items():
            for alpha in (3000.0, 10000.0, 30000.0):
                columns = ['dog__' + c for c in cols] + extra
                frame = lab.walk_forward(dog_table, lab.ridge_margin(columns, alpha=alpha, target='y_dog'))
                dog_frontier(frame, '%s + %s a=%g' % (set_name, extra_name, alpha))
        print()


def stage_consensus(table):
    print('=== consensus: average (and unanimity) of several differently built models, toward the underdog, weeks 5-16 ===')
    full = lab.PBP_FEATURES + lab.BOX_FEATURES + ['elo_diff', 'rest_diff']
    slim4 = pair_features([ALL[n] for n in SLIM[4]])
    dog_table = add_dog_columns(table, ['elo_diff', 'rest_diff'] + full + slim4)
    sign = 2.0 * dog_table['dog_is_home'] - 1.0
    shape = ['abs_spread', 'total_line', 'on_key_number', 'half_point', 'neutral']
    runs = {
        'full': (lab.ridge_margin(full, alpha=10000.0, target='resid'), False),
        'slim4': (lab.ridge_margin(slim4, alpha=3000.0, target='resid'), False),
        'dogfull': (lab.ridge_margin(['dog__' + c for c in full] + shape, alpha=10000.0, target='y_dog'), True),
        'dogslim': (lab.ridge_margin(['dog__' + c for c in slim4] + shape, alpha=10000.0, target='y_dog'), True),
    }
    scores = {}
    for name, (fn, already_dog) in runs.items():
        frame = lab.walk_forward(dog_table, fn)
        s = frame['pred'] if already_dog else frame['pred'] * (2.0 * frame['dog_is_home'] - 1.0)
        scores[name] = s / s.std()
        base = frame
    for combo in (('full',), ('full', 'dogfull'), ('full', 'slim4'), ('full', 'slim4', 'dogfull'), ('full', 'slim4', 'dogfull', 'dogslim')):
        frame = base.copy()
        frame['pred'] = sum(scores[c] for c in combo) / len(combo)
        dog_frontier(frame, 'mean of ' + '+'.join(combo))
        unanimous = frame[(pd.concat([scores[c] for c in combo], axis=1) > 0).all(axis=1)]
        dog_frontier(unanimous, '  unanimous ' + '+'.join(combo))


if __name__ == '__main__':
    table = lab.default_table()
    table = table[table['season'] >= 2002]
    for stage in (sys.argv[1:] or ['sets', 'singles']):
        {'sets': stage_sets, 'singles': stage_singles, 'forward': stage_forward, 'frontier': stage_frontier, 'dog': stage_dog,
         'consensus': stage_consensus}[stage](table)
