"""Selection rules: which games are 'worth picking', and what hit rate do the picks earn over a whole season?

A rule says "pick this game" whenever a score clears a cutoff, so the number of picks varies by week.  The cutoff is set as a
quantile of the model's own scores (outcomes are never used to set it), and results are pooled over weeks 5-16 of every
season, then split into discovery (2006-2017) and holdout (2018 on) and looked at season by season.
Run it from this folder:  python nfl_research_select.py
"""

import numpy as np
import pandas as pd

import nfl_lab as lab
from nfl_research_features import ALL, SLIM, add_dog_columns, pair_features
from nfl_sheet import KEY_LINES

BREAK_EVEN = 0.5238
QUANTILES = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95)


def score_frames(table):
    """Walk-forward scores toward the underdog (positive = the model thinks the underdog covers) for weeks 5-16."""
    full = lab.PBP_FEATURES + lab.BOX_FEATURES + ['elo_diff', 'rest_diff']
    slim4 = pair_features([ALL[n] for n in SLIM[4]])
    dog_table = add_dog_columns(table, ['elo_diff', 'rest_diff'] + full + slim4)
    shape = ['abs_spread', 'total_line', 'on_key_number', 'half_point', 'neutral']
    runs = {
        'full': (lab.ridge_margin(full, alpha=10000.0, target='resid'), False),
        'slim4': (lab.ridge_margin(slim4, alpha=3000.0, target='resid'), False),
        'dogfull': (lab.ridge_margin(['dog__' + c for c in full] + shape, alpha=10000.0, target='y_dog'), True),
    }
    scores, base = {}, None
    for name, (fn, already_dog) in runs.items():
        frame = lab.walk_forward(dog_table, fn)
        s = frame['pred'] if already_dog else frame['pred'] * (2.0 * frame['dog_is_home'] - 1.0)
        scores[name] = s
        base = frame
    base = base.drop(columns=['pred'])
    for name, s in scores.items():
        base['s_' + name] = s
    z = {name: scores[name] / scores[name].std() for name in scores}
    base['s_agree'] = (z['full'] + z['slim4']) / 2
    base['s_agree3'] = (z['full'] + z['slim4'] + z['dogfull']) / 3
    base['all_pos'] = (pd.concat([scores[n] for n in ('full', 'slim4')], axis=1) > 0).all(axis=1)
    base['win'] = base['y_dog'] > 0
    base = base[(base['resid'] != 0) & base['week'].between(5, 16) & (base['game_type'] == 'REG')]
    return base


def describe(label, picks, seasons):
    n = len(picks)
    if n == 0:
        print('%-46s no picks' % label)
        return
    rate = picks['win'].mean()
    se = np.sqrt(rate * (1 - rate) / n)
    d, h = picks[picks['season'] <= lab.DISCOVERY_END], picks[picks['season'] > lab.DISCOVERY_END]
    by = picks.groupby('season')['win'].agg(['mean', 'size']).reindex(seasons)
    print('%-46s %5.1f/season  all %.3f +-%.3f | disc %.3f (n=%d) hold %.3f (n=%d) | seasons >=52.4%%: %d/%d, >=60%%: %d/%d  worst %.2f best %.2f' % (
        label, n / len(seasons), rate, 1.96 * se, d['win'].mean() if len(d) else float('nan'), len(d),
        h['win'].mean() if len(h) else float('nan'), len(h), (by['mean'] >= BREAK_EVEN).sum(), by['mean'].notna().sum(),
        (by['mean'] >= 0.60).sum(), by['mean'].notna().sum(), by['mean'].min(), by['mean'].max()))


def condition_scan(base, picks_mask, label):
    """Hit rate of the picks inside and outside simple game conditions, discovery versus holdout."""
    p = base[picks_mask].copy()
    conditions = {
        'dog gets 3 or less': p['abs_spread'] <= 3, 'dog gets 3.5-6': p['abs_spread'].between(3.5, 6), 'dog gets 6.5-9.5': p['abs_spread'].between(6.5, 9.5),
        'dog gets 10+': p['abs_spread'] >= 10, 'dog gets 4+': p['abs_spread'] >= 4, 'dog gets 7+': p['abs_spread'] >= 7,
        'dog is home': p['dog_is_home'] == 1, 'dog is away': p['dog_is_home'] == 0,
        'weeks 5-8': p['week'].between(5, 8), 'weeks 9-12': p['week'].between(9, 12), 'weeks 13-16': p['week'].between(13, 16),
        'divisional game': p['div_game'] == 1, 'dome': p['dome'] == 1, 'outdoors': p['dome'] == 0,
        'primetime': p['primetime'] == 1, 'thursday': p['thursday'] == 1, 'monday': p['monday'] == 1,
        'on 3/7/6/10/14': p['on_key_number'] == 1, 'half-point line': p['half_point'] == 1,
        'total under 42': p['total_line'] < 42, 'total 42-47': p['total_line'].between(42, 47), 'total over 47': p['total_line'] > 47,
        'cold or windy': (p['cold'] == 1) | (p['wind_high'] == 1), 'neutral site': p['neutral'] == 1,
        'QB out/flagged': p['qb_out_any'].fillna(0) > 0,
    }
    print('\n=== conditions within %s (%d picks) ===' % (label, len(p)))
    rows = []
    for name, mask in conditions.items():
        mask = mask.fillna(False)
        d, h = p[mask & (p['season'] <= lab.DISCOVERY_END)], p[mask & (p['season'] > lab.DISCOVERY_END)]
        if len(d) < 20 or len(h) < 15:
            continue
        rows.append((min(d['win'].mean(), h['win'].mean()), name, d, h, p[~mask & (p['season'] <= lab.DISCOVERY_END)], p[~mask & (p['season'] > lab.DISCOVERY_END)]))
    for _, name, d, h, od, oh in sorted(rows, key=lambda r: -r[0]):
        print('%-20s in: disc %.3f (n=%3d) hold %.3f (n=%3d) | out: disc %.3f hold %.3f' % (
            name, d['win'].mean(), len(d), h['win'].mean(), len(h), od['win'].mean() if len(od) else float('nan'), oh['win'].mean() if len(oh) else float('nan')))
    print('conditions tested:', len(conditions), '(expect a few to look good by chance alone)')


def american_profit(odds):
    """Profit on a 1-unit winning bet at American odds."""
    odds = pd.to_numeric(odds, errors='coerce')
    return np.where(odds < 0, 100.0 / -odds, odds / 100.0)


def key_number_study(base):
    """Is the edge about the line sitting on a key number?  Baselines, real juice, each number, both sides."""
    b = base.copy()
    b['dog_odds'] = np.where(b['dog_is_home'] == 1, b['home_spread_odds'], b['away_spread_odds'])
    b['fav_odds'] = np.where(b['dog_is_home'] == 1, b['away_spread_odds'], b['home_spread_odds'])
    b['line'] = b['abs_spread']
    b['key'] = b['line'].isin([3.0, 6.0, 7.0, 10.0, 14.0])
    b['integer_line'] = (b['line'] % 1 == 0)
    print('\n=== key numbers: is the edge about where the line sits? (weeks 5-16, decided bets; pushes excluded) ===')
    print('-- baseline: bet EVERY underdog / favorite, no model --')
    for name, mask in (('line on 3/6/7/10/14', b['key']), ('other whole-number line', b['integer_line'] & ~b['key']), ('half-point line', ~b['integer_line'])):
        g = b[mask]
        print('%-26s n=%5d  underdog covers %.3f  (disc %.3f hold %.3f)' % (
            name, len(g), g['win'].mean(), g[g['season'] <= lab.DISCOVERY_END]['win'].mean(), g[g['season'] > lab.DISCOVERY_END]['win'].mean()))
    print('-- how often a game is a push, by line --')
    print(base.assign(l=base['abs_spread']).groupby('l').size().loc[[3.0, 7.0]].to_dict(), '(decided games with line 3 / 7, after removing pushes)')

    score = 's_full'
    for side_name, sign in (('underdog side', 1.0), ('favorite side', -1.0)):
        s = sign * b[score]
        print('\n-- model picks, %s (cutoff on %s toward that side) --' % (side_name, score))
        for cut in (0.0, 0.25, 0.5, 0.75):
            for label, mask in (('key line', b['key']), ('other whole', b['integer_line'] & ~b['key']), ('half-point', ~b['integer_line'])):
                g = b[mask & (s >= cut)]
                win = g['win'] if sign > 0 else ~g['win']
                odds = g['dog_odds'] if sign > 0 else g['fav_odds']
                profit = np.where(win, american_profit(odds), -1.0)
                has_odds = odds.notna().to_numpy()
                d, h = g['season'] <= lab.DISCOVERY_END, g['season'] > lab.DISCOVERY_END
                print('  corr>=%.2f %-12s n=%4d (%.1f/season) win %.3f | disc %.3f (n=%d) hold %.3f (n=%d) | ROI at real odds %+.1f%% (flat -110: %+.1f%%)' % (
                    cut, label, len(g), len(g) / 21, win.mean(), win[d].mean() if d.any() else float('nan'), d.sum(), win[h].mean() if h.any() else float('nan'), h.sum(),
                    100 * profit[has_odds].mean() if has_odds.any() else float('nan'), 100 * (win.mean() * 100 / 110 - (1 - win.mean()))))
    print('\n-- underdog picks on each key number (s_full >= 0) --')
    for line in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 13.0, 14.0):
        g = b[(b['line'] == line) & (b['s_full'] >= 0)]
        if len(g) >= 20:
            d, h = g[g['season'] <= lab.DISCOVERY_END], g[g['season'] > lab.DISCOVERY_END]
            print('  line %4.1f n=%3d win %.3f | disc %.3f (n=%d) hold %.3f (n=%d)' % (line, len(g), g['win'].mean(), d['win'].mean(), len(d), h['win'].mean(), len(h)))


def main():
    table = lab.default_table()
    table = table[table['season'] >= 2002]
    base = score_frames(table)
    key_number_study(base)


def condition_main():
    table = lab.default_table()
    table = table[table['season'] >= 2002]
    base = score_frames(table)
    seasons = sorted(base['season'].unique())
    print('games', len(base), 'seasons', seasons[0], '-', seasons[-1], '| break-even at -110 is 52.4%\n')

    print('=== cutoff = quantile of the score over all games (underdog side, weeks 5-16) ===')
    for score in ('s_full', 's_agree', 's_agree3', 's_dogfull'):
        for q in QUANTILES:
            cut = base[score].quantile(q)
            describe('%s >= q%.2f (%.2f)' % (score, q, cut), base[base[score] >= cut], seasons)
        print()
    print('--- both full and slim4 agree the underdog covers, then cutoff on the full score ---')
    agree = base[base['all_pos']]
    for q in (0.5, 0.6, 0.7, 0.8, 0.9):
        cut = agree['s_agree'].quantile(q)
        describe('agree-only, s_agree >= its q%.2f' % q, agree[agree['s_agree'] >= cut], seasons)


def production_rule_check():
    """The rule on the exact production correction (nfl_signal): underdog, line on a key number, correction at least X."""
    import nfl_signal

    table = lab.default_table()
    table = table[table['season'] >= 2002]
    frame = lab.walk_forward(table, lab.ridge_margin(nfl_signal.SIGNAL_FEATURES, alpha=nfl_signal.SIGNAL_ALPHA, target='resid'))
    f = frame[(frame['resid'] != 0) & frame['week'].between(5, 16)].copy()
    f['dog_is_home'] = (f['spread_line'] < 0).astype(int)
    f['toward_dog'] = f['pred'] * (2.0 * f['dog_is_home'] - 1.0)
    f['win'] = f['resid'] * (2.0 * f['dog_is_home'] - 1.0) > 0
    f['line'] = f['spread_line'].abs()
    f['dog_odds'] = np.where(f['dog_is_home'] == 1, f['home_spread_odds'], f['away_spread_odds'])
    seasons = sorted(f['season'].unique())

    def report(label, g):
        d, h = g[g['season'] <= lab.DISCOVERY_END], g[g['season'] > lab.DISCOVERY_END]
        profit = np.where(g['win'], american_profit(g['dog_odds']), -1.0)
        ok = g['dog_odds'].notna().to_numpy()
        rate = g['win'].mean()
        print('%-44s n=%4d (%4.1f/season, %.1f/wk) win %.3f +-%.3f | disc %.3f (n=%d) hold %.3f (n=%d) | ROI real odds %+.1f%%' % (
            label, len(g), len(g) / len(seasons), len(g) / len(seasons) / 12, rate, 1.96 * np.sqrt(rate * (1 - rate) / len(g)),
            d['win'].mean(), len(d), h['win'].mean(), len(h), 100 * profit[ok].mean()))

    print('\n=== production correction, weeks 5-16, underdog side ===')
    key = f['line'].isin([3.0, 6.0, 7.0, 10.0, 14.0])
    for name, lines in (('key lines 3/6/7/10/14', key), ('only 3 and 7', f['line'].isin([3.0, 7.0])), ('only 3', f['line'] == 3.0),
                        ('every whole-number line', f['line'] % 1 == 0), ('whole-number lines of 3 or more', (f['line'] % 1 == 0) & (f['line'] >= 3))):
        for cut in (0.0, 0.25, 0.5):
            report('%s, correction >= %.2f' % (name, cut), f[lines & (f['toward_dog'] > 0) & (f['toward_dog'] >= cut)])
        print()

    chosen = f[key & (f['toward_dog'] > 0) & (f['toward_dog'] >= 0.25)]
    by = chosen.groupby('season')['win'].agg(['sum', 'size'])
    print('--- rule: key line, underdog correction >= 0.25, season by season (wins-losses) ---')
    print('  '.join('%d: %d-%d' % (s, w, n - w) for s, (w, n) in by.iterrows()))
    rate = by['sum'] / by['size']
    print('seasons above 50%%: %d/%d, above 52.4%%: %d/%d, at or above 60%%: %d/%d' % (
        (rate > 0.5).sum(), len(by), (rate > BREAK_EVEN).sum(), len(by), (rate >= 0.6).sum(), len(by)))
    # how often a random draw of key-line underdogs (no model) does as well as the model's picks
    pool = f[key & (f['toward_dog'] > 0)]
    rng = np.random.default_rng(3)
    base_rate = f[key]['win'].mean()
    draws = [pool['win'].sample(len(chosen), random_state=int(s)).mean() for s in rng.integers(0, 10 ** 6, 2000)]
    print('random key-line underdogs of the same count: mean %.3f, best of 2000 %.3f; model picks %.3f (p = %.4f); all key-line dogs %.3f' % (
        np.mean(draws), np.max(draws), chosen['win'].mean(), float((np.array(draws) >= chosen['win'].mean()).mean()), base_rate))
    # weekly volume
    per_week = chosen.groupby(['season', 'week']).size().reindex(f.groupby(['season', 'week']).size().index, fill_value=0)
    print('picks per week-slate: %s' % per_week.value_counts().sort_index().to_dict())


def side_models(table):
    """Is the favorite side really edgeless, or only so for a model that was found on underdogs?

    Besides the pooled model, fits models on subsets of games (key-number lines only, whole-number lines only, and
    games with a big favorite) and evaluates both sides on key lines and on all lines.
    """
    full = lab.PBP_FEATURES + lab.BOX_FEATURES + ['elo_diff', 'rest_diff']
    shape = ['abs_spread', 'total_line', 'neutral']
    dog_table = add_dog_columns(table, ['elo_diff', 'rest_diff'] + full)
    dog_table['line'] = dog_table['abs_spread']
    dog_table['key'] = dog_table['line'].isin(KEY_LINES)
    dog_table['whole'] = dog_table['line'] % 1 == 0
    columns = ['dog__' + c for c in full] + shape
    subsets = {
        'all games': lambda t: np.ones(len(t), bool),
        'key lines only': lambda t: t['key'].to_numpy(),
        'whole-number lines only': lambda t: t['whole'].to_numpy(),
        'line >= 3 only': lambda t: (t['line'] >= 3).to_numpy(),
    }
    print('=== models trained on subsets of games; picks on each side, weeks 5-16 (win rate disc/hold, n) ===')
    for name, mask_fn in subsets.items():
        for alpha in (3000.0, 10000.0, 30000.0):
            inner = lab.ridge_margin(columns, alpha=alpha, target='y_dog')
            fn = lambda train, test, inner=inner, mask_fn=mask_fn: inner(train[mask_fn(train)], test)
            frame = lab.walk_forward(dog_table, fn)
            f = frame[(frame['resid'] != 0) & frame['week'].between(5, 16) & (frame['game_type'] == 'REG')].copy()
            f['win_dog'] = f['y_dog'] > 0
            cells = []
            for side, sign in (('dog', 1.0), ('fav', -1.0)):
                for where, wmask in (('key', f['key']), ('any', pd.Series(True, index=f.index))):
                    for cut in (0.5, 1.0):
                        g = f[wmask & (sign * f['pred'] >= cut)]
                        win = g['win_dog'] if sign > 0 else ~g['win_dog']
                        d, h = g['season'] <= lab.DISCOVERY_END, g['season'] > lab.DISCOVERY_END
                        cells.append('%s/%s>=%.1f %.3f/%.3f (n=%d,%d)' % (side, where, cut, win[d].mean() if d.any() else float('nan'),
                                                                       win[h].mean() if h.any() else float('nan'), d.sum(), h.sum()))
            print('%-24s a=%-6g %s' % (name, alpha, ' | '.join(cells)))
        print()


def favorites_main():
    table = lab.default_table()
    table = table[table['season'] >= 2002]
    side_models(table)


if __name__ == '__main__':
    import sys

    if 'production' in sys.argv[1:]:
        production_rule_check()
    elif 'favorites' in sys.argv[1:]:
        favorites_main()
    else:
        main()
