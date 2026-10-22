"""Which team statistics go with winning NFL games, which predict the winner before kickoff, and which does the market miss?

Four questions in order; the first three describe, only the fourth could hold a betting edge.

1. What wins games: each offense and defense stat measured in the game itself (the home team's value) against whether
   the home team won, every regular-season game 1999-2026.  Same-game stats explain a result but are not known before
   kickoff.  Points are left out (they are the result).
2. What repeats: how well a team's average in its odd-numbered games predicts its average in its even-numbered games
   (split-half reliability, team-seasons 1999-2025), and its season average the next season.  A stat that decides games
   but does not repeat cannot predict the next one.
3. What predicts the winner: the pre-game ratings the models use (recency-weighted, opponent-adjusted, built only from
   earlier games; nfl_lab) as home-minus-away differences against whether the home team won, weeks 5-16, with the
   closing spread and Elo as benchmarks; then a forward selection of the stats that predict winners together, chosen on
   2006-2017 and scored on 1999-2005 and 2018-2026.
4. What the market misses.  Pre-registered before any of it was computed (October 20, 2026); nothing in production
   changes either way:
   M1, information beyond the closing spread.  For each of the 74 ratings (37 stats, offense and defense), the
   correlation of its home-minus-away difference with the cover margin (result minus closing spread), weeks 5-16, in
   each era (1999-2005, 2006-2017, 2018-2026; air yards, CPOE and pass rate over expected exist from 2006, so they are
   judged on the later two).  An edge candidate needs the same sign in every era, a pooled robust |t| >= 3.4 (two-sided
   0.05 after Bonferroni over 74 ratings), and a partial correlation with the cover margin, controlling for the
   production correction, of that same sign in every era (it adds to what the picks already use).  The straight-up
   version (won minus the spread-implied win probability) is shown alongside as a check, not a criterion.
   M2, does the market overreact to stats that do not repeat?  Across the stats with a better direction, the Spearman
   correlation between a stat's split-half reliability and its direction-signed correlation with the cover margin
   (positive = being better at the stat helps a team cover).  Positive in all three eras would mean the market
   overweights noisy stats and underweights stable ones.
   M3, added after M2 passed and before it was computed: does that pattern add to the production correction?  S = the
   average direction-signed rating difference (home minus away, in standard deviations) over the most reliable third of
   the directed ratings measured in every era, minus the same over the least reliable third.  Pass: correlation with the
   cover margin > 0 in every era, partial correlation given the production correction > 0 in every era, and a pooled
   robust t >= 3.0 for S against the cover margin left after the production correction.  A pass would make S a
   candidate signal for forward tracking; no current logic changes either way.  Diagnostic beside it (no decision): M2
   within offense ratings only and within defense ratings only, since offense stats are the more stable ones.
   M4, added after M3 failed and before it was computed: does the market overreact to last week's turnover luck?
   Turnovers decide games (question 1) but barely repeat (question 2).  Feature: the home team's turnover margin
   (takeaways minus giveaways) in its previous game of the season minus the away team's.  Pass: correlation with the
   cover margin < 0 in every era (teams coming off a turnover windfall are overpriced), pooled robust t <= -3.0, and a
   partial correlation given the production correction < 0 in every era.

Run it from this folder:  python nfl_research_winning.py

Results (October 20, 2026; correlations by era are 1999-2005 / 2006-2017 / 2018-2026):

1. What wins games (same-game, 7,060 games): per-play efficiency leads, offense and defense alike, nearly unchanged
   across eras: EPA per play in competitive game states +0.55 (allowed -0.55), EPA per play +0.50 (-0.51), EPA per
   dropback +0.49 (-0.50), success rate in competitive game states +0.41 (-0.42), giveaways -0.39 and takeaways +0.38,
   early-down EPA +0.38 (-0.39), touchdowns per drive +0.37 (-0.40), pass success rate +0.36 (-0.37), yards per pass
   attempt +0.35 (-0.36).  Pass rate is -0.52, but that is game script (teams ahead run the ball), not a cause.  Volume
   matters less than efficiency: net yards +0.29 (-0.31), first downs +0.25 (-0.28).  Holding the rest fixed, the parts
   of a win in order of what each adds: interception rate, yards per pass attempt, game script, fumbles, third downs,
   special teams, sacks, penalties, yards per carry (McFadden R2 0.63); explosive plays add nothing beyond yards per play.
2. What repeats: offense far more than defense (split-half reliability: EPA per play 0.62 vs 0.39, EPA per dropback 0.57
   vs 0.28, yards per pass attempt 0.51 vs 0.27, success rate 0.65 vs 0.48, third-down rate 0.42 vs 0.17).  Turnovers
   barely repeat: fumbles lost 0.07, fumbles recovered 0.13, giveaways and takeaways 0.27, interception rates 0.24-0.26.
   Special teams (0.59) and pace (drives, 0.50) are among the most stable.
3. What predicts the winner (pre-game ratings, weeks 5-16): offense leads: points per game +0.31, EPA per play +0.30,
   EPA per dropback +0.29, success rate in competitive game states +0.29, early-down EPA +0.28.  The best defense ratings
   are points allowed -0.21 and EPA allowed in competitive game states -0.20.  Turnovers fall from +-0.39 in the game to
   -0.17 (giveaways) and +0.13 (takeaways) before it.  The closing spread (+0.38) beats every stat: the best stats-only
   model reached a log loss of 0.6207 and 0.6229 out of sample against the spread's 0.6189 and 0.6092, and the spread
   plus 8 stats did no better than the spread alone (0.6208 and 0.6094).
4. What the market misses.  M1: 4 of 74 pass: EPA per run (+0.083 / +0.055 / +0.045), rushing success rate (+0.065 /
   +0.071 / +0.053), early-down success rate (+0.058 / +0.069 / +0.047) and opponent drives per game (-0.054 / -0.052 /
   -0.058).  The market prices 1 SD of rushing efficiency at about 2.5 points when it is worth about 3.3, while passing,
   points, yards, first downs and turnovers are priced within about half a point of their worth.  But their partial
   correlations given the production correction are small (0.00 to 0.05): the correction already bets them; they are
   the blind spots nfl_research_features.py found.  M2 passes (rho +0.51 / +0.49 / +0.49, p < 0.001 each): the market
   underweights stats that repeat and overweights those that do not.  Within one side it is weak (offense +0.19 / +0.30
   / +0.32, defense +0.69 / -0.04 / +0.29), so most of it is the market leaning on defense, which repeats less, relative
   to offense.  M3 fails: the stability composite tracks the cover margin (+0.041 / +0.068 / +0.043) but adds nothing
   to the production correction (partial -0.021 / +0.042 / -0.003, pooled t +0.94).  M4 fails: last week's turnover
   margin says nothing about this week's cover (-0.017 / -0.004 / +0.007, t -0.34); the market discounts turnover luck.

Reading: efficiency per play and turnovers decide games; offensive efficiency predicts the next one, because turnovers
and most defensive numbers do not repeat.  The market knows all of that.  Its remaining errors (too little weight on
rushing efficiency and early-down success, too much on pace and on defense) are what the correction model already bets,
so this lens explains the current edge rather than adding a new one.  The four M1 stats are for forward watching only.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import nfl_lab as lab
import nfl_model as nm
import nfl_signal
from nfl_pbp_features import load_pbp_team_games
from nfl_reasons import feature_label, higher_is_better
from nflverse_data import load_games, load_team_weeks

WEEKS = nfl_signal.PICK_WEEKS
ERAS = (('1999-2005', 1999, 2005), ('2006-2017', 2006, 2017), ('2018-2026', 2018, 2026))
BOX_TWINS = {'fd', 'plays', 'epa_pp'}   # box-score stats that also have a play-by-play version
M1_T = 3.4                              # two-sided 0.05 after Bonferroni over 74 ratings
MIN_GAMES = 100
TOP = 12


# ---------------------------------------------------------------- data

def catalog():
    """One row per rated stat and side: rating feature, same-game column, plain words and the better direction."""
    box = {own for own, _ in lab.BOX_PAIRS}
    rows = []
    for own, allowed in lab.ALL_PAIRS:
        for side, column in (('off', own), ('def', allowed)):
            feature = f'{side}__{own}'
            label = feature_label(feature) + (' (box score)' if own in box and own in BOX_TWINS else '')
            rows.append({'feature': feature, 'column': column, 'side': 'offense' if side == 'off' else 'defense',
                         'label': label, 'better': higher_is_better(feature)})
    return pd.DataFrame(rows).set_index('feature')


def team_games():
    """One row per team per played regular-season game with what it produced (own) and allowed, as nfl_lab builds them."""
    games = nm.normalize_teams(load_games())
    games = games[games['season'] >= lab.FIRST_SEASON]
    last = int(games['season'].max())
    rows = nm.build_team_rows(games, load_team_weeks(lab.FIRST_SEASON, last))
    own = load_pbp_team_games(lab.FIRST_SEASON, last).drop(columns=['season', 'week'])
    allowed = own.rename(columns={'team': 'opp', **{c: 'd_' + c[2:] for c in own.columns if c.startswith('o_')}})
    rows = rows.merge(own, on=['game_id', 'team'], how='left').merge(allowed, on=['game_id', 'opp'], how='left')
    rows = rows[(rows['game_type'] == 'REG') & rows['pts'].notna() & rows['pts_allowed'].notna()].copy()
    rows['margin'] = rows['pts'] - rows['pts_allowed']
    rows['win'] = np.where(rows['margin'] > 0, 1.0, np.where(rows['margin'] < 0, 0.0, np.nan))
    return rows


def pregame_frame(table):
    """Scored regular-season games (out-of-sample production correction in 'pred'), weeks 5-16, ties dropped."""
    f = nfl_signal.scored_history(table, lab.LAST_SEASON)
    f = f[f['week'].between(*WEEKS) & (f['game_type'] == 'REG') & (f['result'] != 0)].copy()
    f['win'] = (f['result'] > 0).astype(float)
    f['p_market'] = np.nan
    for _, lo, hi in ERAS:
        era = f['season'].between(lo, hi)
        fit = LogisticRegression(C=1e6, max_iter=1000).fit(f.loc[era, ['spread_line']], f.loc[era, 'win'])
        f.loc[era, 'p_market'] = fit.predict_proba(f.loc[era, ['spread_line']])[:, 1]
    f['win_resid'] = f['win'] - f['p_market']
    return f


# ---------------------------------------------------------------- statistics

def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < MIN_GAMES or np.std(x[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def slope_t(x, y):
    """OLS slope of y on x with a robust (HC0) t-statistic."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    slope = (x * y).sum() / (x ** 2).sum()
    e = y - slope * x
    return slope, slope / (math.sqrt(((x ** 2) * (e ** 2)).sum()) / (x ** 2).sum())


def partial_corr(x, y, control):
    x, y, c = (np.asarray(v, float) for v in (x, y, control))
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(c))
    design = np.column_stack([np.ones(ok.sum()), c[ok]])
    rx = x[ok] - design @ np.linalg.lstsq(design, x[ok], rcond=None)[0]
    ry = y[ok] - design @ np.linalg.lstsq(design, y[ok], rcond=None)[0]
    return corr(rx, ry)


def era_frames(f):
    return [(name, f[f['season'].between(lo, hi)]) for name, lo, hi in ERAS]


def direction(better):
    return 'more is better' if better is True else 'less is better' if better is False else 'style / pace'


# ---------------------------------------------------------------- 1. what wins games

def same_game(rows, cat):
    home = rows[(rows['is_home'] == 1) & rows['win'].notna()]
    out = {}
    for feature, c in cat.iterrows():
        if feature.endswith('__pts'):
            continue
        out[feature] = {'r_win': corr(home[c['column']], home['win']), 'r_margin': corr(home[c['column']], home['margin']),
                        **{f'r_win {name}': corr(g[c['column']], g['win']) for name, g in era_frames(home)}}
    return pd.DataFrame(out).T


def same_game_components(rows):
    """Logistic regression of winning on same-game differences (team minus opponent) of non-overlapping components."""
    home = rows[(rows['is_home'] == 1) & rows['win'].notna()].copy()
    parts = {
        'yards per pass attempt': home['o_ypa'] - home['d_ypa'],
        'sack rate (taken minus made)': home['o_sack_rate'] - home['d_sack_rate'],
        'interception rate (thrown minus forced)': home['o_int_rate'] - home['d_int_rate'],
        'yards per carry': home['o_ypc'] - home['d_ypc'],
        'fumbles lost minus recovered': home['o_fumbles_lost'] - home['d_fumbles_lost'],
        'third-down conversion rate': home['o_third_rate'] - home['d_third_rate'],
        'explosive-play rate': home['o_explosive'] - home['d_explosive'],
        'penalties committed minus drawn': home['o_penalties'] - home['d_penalties'],
        'special-teams EPA (net)': home['o_st_epa_for'] - home['o_st_epa_against'],
        'pass rate (game script)': home['o_pass_rate'] - home['d_pass_rate'],
    }
    x = pd.DataFrame(parts).dropna()
    y = home.loc[x.index, 'win']
    z = (x - x.mean()) / x.std()
    base = log_loss(y, np.full(len(y), y.mean()))

    def fitted_loss(columns):
        model = LogisticRegression(C=1e6, max_iter=2000).fit(z[columns], y)
        return log_loss(y, model.predict_proba(z[columns])[:, 1]), model

    full_loss, model = fitted_loss(list(z.columns))
    print('\nwhat explains a win once the others are held fixed: logistic regression on same-game differences '
          '(home minus away, 1 SD each), %d games, McFadden R2 %.2f' % (len(y), 1 - full_loss / base))
    print('    %-42s %10s %22s' % ('component', 'log-odds/SD', 'R2 lost if dropped'))
    rows_out = []
    for i, column in enumerate(z.columns):
        loss, _ = fitted_loss([c for c in z.columns if c != column])
        rows_out.append((column, model.coef_[0][i], (loss - full_loss) / base))
    for column, coef, lost in sorted(rows_out, key=lambda r: -r[2]):
        print('    %-42s %+10.2f %22.3f' % (column, coef, lost))


# ---------------------------------------------------------------- 2. what repeats

def reliability(rows, cat):
    complete = rows[rows['season'] < lab.LAST_SEASON].sort_values(['team', 'season', 'gameday'])
    complete = complete.assign(n=complete.groupby(['team', 'season']).cumcount())
    key = ['team', 'season']
    out = {}
    for feature, c in cat.iterrows():
        column = c['column']
        odd = complete[complete['n'] % 2 == 0].groupby(key)[column].mean()
        even = complete[complete['n'] % 2 == 1].groupby(key)[column].mean()
        halves = pd.concat([odd, even], axis=1, keys=['odd', 'even']).dropna()
        means = complete.groupby(key)[column].mean().reset_index().sort_values(key)
        means['next'] = means.groupby('team')[column].shift(-1)
        means['next_season'] = means.groupby('team')['season'].shift(-1)
        consecutive = means[means['next_season'] == means['season'] + 1]
        out[feature] = {'split_half': corr(halves['odd'], halves['even']), 'next_season': corr(consecutive[column], consecutive['next'])}
    return pd.DataFrame(out).T


# ---------------------------------------------------------------- 3. what predicts the winner

def pregame(f, cat):
    out = {}
    for feature in cat.index:
        out[feature] = {'pre_r_win': corr(f[feature], f['win']), 'pre_r_margin': corr(f[feature], f['result']),
                        **{f'pre_r_win {name}': corr(g[feature], g['win']) for name, g in era_frames(f)}}
    return pd.DataFrame(out).T


def forward_selection(f, cat, steps=8):
    """Stats that predict winners together: forward selection on 2006-2017 by log loss, scored on the other two eras."""
    features = [c for c in cat.index if f.loc[f['season'] <= 2005, c].std() > 1e-9]   # stats measured in every era
    train = f[f['season'].between(2006, 2017)]
    tests = [('1999-2005', f[f['season'] <= 2005]), ('2018-2026', f[f['season'] >= 2018])]
    mean, sd = train[features + ['spread_line']].mean(), train[features + ['spread_line']].std()

    def z(frame, columns):
        return ((frame[columns] - mean[columns]) / sd[columns]).fillna(0.0)

    def losses(columns):
        model = LogisticRegression(C=1e4, max_iter=2000).fit(z(train, columns), train['win'])
        return [log_loss(frame['win'], model.predict_proba(z(frame, columns))[:, 1]) for _, frame in [('train', train)] + tests]

    base = [log_loss(frame['win'], np.full(len(frame), train['win'].mean())) for frame in [train] + [t for _, t in tests]]
    print('\nstats that predict winners together (forward selection on 2006-2017, weeks 5-16; log loss, lower is better)')
    print('    %-58s %9s %9s %9s' % ('', '2006-2017', '1999-2005', '2018-2026'))
    print('    %-58s %9.4f %9.4f %9.4f' % ('home-field rate only (no information)', *base))
    print('    %-58s %9.4f %9.4f %9.4f' % ('closing spread only (the market)', *losses(['spread_line'])))
    chosen = []
    for step in range(1, steps + 1):
        best = min(((losses(chosen + [c])[0], c) for c in features if c not in chosen))
        chosen.append(best[1])
        print('    %-58s %9.4f %9.4f %9.4f' % ('+ %d. %s (%s)' % (step, cat.loc[best[1], 'label'], cat.loc[best[1], 'side']), *losses(chosen)))
    print('    %-58s %9.4f %9.4f %9.4f' % ('closing spread + those %d stats' % steps, *losses(['spread_line'] + chosen)))


# ---------------------------------------------------------------- 4. what the market misses

def market(f, cat):
    out = {}
    for feature in cat.index:
        row = {}
        for name, g in era_frames(f):
            row[f'cover {name}'] = corr(g[feature], g['resid'])
            row[f'partial {name}'] = partial_corr(g[feature], g['resid'], g['pred'])
            row[f'win resid {name}'] = corr(g[feature], g['win_resid'])
        first = 2006 if f.loc[f['season'] <= 2005, feature].std() < 1e-9 else lab.FIRST_SEASON   # stats that start in 2006
        valid = f[(f['season'] >= first) & f[feature].notna()]
        row['cover pooled'] = corr(valid[feature], valid['resid'])
        row['cover t'] = slope_t(valid[feature], valid['resid'])[1]
        x = (valid[feature] - valid[feature].mean()) / valid[feature].std()
        row['worth'] = slope_t(x, valid['result'])[0]       # points of final margin per 1 SD of the rating difference
        row['priced'] = slope_t(x, valid['spread_line'])[0]  # points of closing spread per 1 SD
        out[feature] = row
    return pd.DataFrame(out).T


def m1_verdict(table):
    eras = [name for name, _, _ in ERAS]
    passed, near = [], []
    for feature, row in table.iterrows():
        covers = [row[f'cover {e}'] for e in eras if not np.isnan(row[f'cover {e}'])]
        partials = [row[f'partial {e}'] for e in eras if not np.isnan(row[f'partial {e}'])]
        sign = np.sign(row['cover pooled'])
        same = len(covers) >= 2 and all(np.sign(v) == sign for v in covers)
        adds = len(partials) == len(covers) and all(np.sign(v) == sign for v in partials)
        if same and adds and abs(row['cover t']) >= M1_T:
            passed.append(feature)
        elif same and abs(row['cover t']) >= 2.0:
            near.append(feature)
    return passed, near


def print_ranked(title, frame, cat, column, extra, n=TOP):
    print(f'\n{title}')
    for side in ('offense', 'defense'):
        rows = frame.join(cat)
        rows = rows[rows['side'] == side].dropna(subset=[column])
        rows = rows.reindex(rows[column].abs().sort_values(ascending=False).index).head(n)
        print(f'  {side.upper()}')
        for feature, row in rows.iterrows():
            print('    %-50s %+.3f   %-14s %s' % (row['label'], row[column], direction(row['better']), extra(feature, row)))


def main():
    cat = catalog()
    rows = team_games()
    table = lab.default_table()
    f = pregame_frame(table)
    print('data: %d team-games (%d games) for same-game stats; %d games, weeks %d-%d, for the pre-game ratings' % (
        len(rows), int((rows['is_home'] == 1).sum()), len(f), *WEEKS))

    print('\n' + '=' * 30 + ' 1. WHAT WINS GAMES (stats from the game itself) ' + '=' * 30)
    game = same_game(rows, cat)
    stable = reliability(rows, cat)
    print_ranked('correlation with the home team winning, every regular-season game 1999-2026 (home team\'s own numbers; '
                 'eras 1999-2005 / 2006-2017 / 2018-2026 at right)', game, cat, 'r_win',
                 lambda _, r: 'eras %+.2f / %+.2f / %+.2f   margin %+.2f' % (r['r_win 1999-2005'], r['r_win 2006-2017'], r['r_win 2018-2026'], r['r_margin']))
    same_game_components(rows)

    print('\n' + '=' * 30 + ' 2. WHAT REPEATS (stability) ' + '=' * 30)
    joined = game.join(stable, how='outer').join(cat)
    print('split-half reliability (odd vs even games of a team-season) and season-to-season, beside the same-game correlation with winning')
    print('    %-50s %-8s %8s %10s %12s' % ('stat', 'side', 'win r', 'split-half', 'next season'))
    for feature, row in joined.reindex(joined['r_win'].abs().sort_values(ascending=False).index).iterrows():
        if np.isnan(row.get('r_win', np.nan)):
            continue
        print('    %-50s %-8s %+8.3f %10.2f %12.2f' % (row['label'], row['side'], row['r_win'], row['split_half'], row['next_season']))

    print('\n' + '=' * 30 + ' 3. WHAT PREDICTS THE WINNER (pre-game ratings, weeks %d-%d) ' % WEEKS + '=' * 30)
    pre = pregame(f, cat)
    print('benchmarks, correlation with the home team winning: closing spread %+.3f, Elo %+.3f, production correction %+.3f' % (
        corr(f['spread_line'], f['win']), corr(f['elo_diff'], f['win']), corr(f['pred'], f['win'])))
    print_ranked('correlation of the home-minus-away rating with the home team winning (eras 1999-2005 / 2006-2017 / 2018-2026 '
                 'and split-half reliability at right)', pre, cat, 'pre_r_win',
                 lambda feature, r: 'eras %+.2f / %+.2f / %+.2f   reliability %.2f' % (
                     r['pre_r_win 1999-2005'], r['pre_r_win 2006-2017'], r['pre_r_win 2018-2026'], stable.loc[feature, 'split_half']))
    forward_selection(f, cat)

    print('\n' + '=' * 30 + ' 4. WHAT THE MARKET MISSES (pre-registered) ' + '=' * 30)
    mk = market(f, cat)
    full = mk.join(pre).join(stable).join(game[['r_win']]).join(cat)
    print('how the market prices the strongest predictors: points of final margin per 1 SD of the rating difference '
          '("worth") against points of closing spread ("priced"), weeks %d-%d, 1999-2026' % WEEKS)
    print('    %-50s %-8s %7s %7s %7s %8s' % ('stat', 'side', 'worth', 'priced', 'gap', 'gap t'))
    for feature, row in full.reindex(full['pre_r_win'].abs().sort_values(ascending=False).index).head(2 * TOP).iterrows():
        print('    %-50s %-8s %+7.2f %+7.2f %+7.2f %+8.2f' % (row['label'], row['side'], row['worth'], row['priced'],
                                                          row['worth'] - row['priced'], row['cover t']))
    print('\nM1: correlation with the cover margin by era, partial (given the production correction) by era, pooled t; '
          'and the straight-up win residual by era, every rating ranked by |pooled t|')
    print('    %-50s %-8s %-23s %-23s %7s   %s' % ('stat', 'side', 'cover r by era', 'partial r by era', 'pool t', 'win-residual r by era'))
    fmt = lambda v: '  n/a' if np.isnan(v) else '%+.3f' % v
    for feature, row in full.reindex(full['cover t'].abs().sort_values(ascending=False).index).iterrows():
        eras = [name for name, _, _ in ERAS]
        print('    %-50s %-8s %-23s %-23s %+7.2f   %s' % (
            row['label'], row['side'], ' '.join(fmt(row[f'cover {e}']) for e in eras), ' '.join(fmt(row[f'partial {e}']) for e in eras),
            row['cover t'], ' '.join(fmt(row[f'win resid {e}']) for e in eras)))
    passed, near = m1_verdict(mk)
    print('\nM1 verdict: %d of %d ratings pass (same sign in every era, |pooled t| >= %.1f, adds to the production correction in every era): %s' % (
        len(passed), len(mk), M1_T, ', '.join('%s (%s)' % (cat.loc[p, 'label'], cat.loc[p, 'side']) for p in passed) or 'none'))
    print('    same sign in every era with |pooled t| >= 2 but short of the bar (watch list only): %s' % (
        ', '.join('%s (%s, t=%+.1f)' % (cat.loc[p, 'label'], cat.loc[p, 'side'], mk.loc[p, 'cover t']) for p in near) or 'none'))

    print('\nM2: does the market overreact to stats that do not repeat?  Spearman correlation, across stats with a better '
          'direction, of split-half reliability with the direction-signed cover correlation')
    directed = full[full['better'].notna()]
    signs = np.where(directed['better'].astype(bool), 1.0, -1.0)
    results = []
    for name, _, _ in ERAS:
        signed = directed[f'cover {name}'] * signs
        ok = signed.notna() & directed['split_half'].notna()
        rho, p = spearmanr(directed.loc[ok, 'split_half'], signed[ok])
        results.append(rho)
        print('    %-10s rho %+.3f (p %.3f, %d stats)' % (name, rho, p, int(ok.sum())))
    print('M2 verdict: %s' % ('PASS: positive in every era' if all(r > 0 for r in results) else 'fail: not positive in every era'))
    print('    diagnostic, within one side only (offense stats are the more stable ones):')
    for side in ('offense', 'defense'):
        part = directed[directed['side'] == side]
        part_signs = np.where(part['better'].astype(bool), 1.0, -1.0)
        cells = []
        for name, _, _ in ERAS:
            signed = part[f'cover {name}'] * part_signs
            ok = signed.notna() & part['split_half'].notna()
            rho, p = spearmanr(part.loc[ok, 'split_half'], signed[ok])
            cells.append('%s rho %+.3f (p %.3f, n=%d)' % (name, rho, p, int(ok.sum())))
        print('      %-8s %s' % (side, ' | '.join(cells)))
    stability_composite(f, full)
    turnover_luck(f, rows)
    starting_hypothesis(full)


def turnover_luck(f, rows):
    """M4: last game's turnover margin (home minus away) against the cover margin."""
    played = rows.sort_values(['team', 'season', 'gameday']).copy()
    played['to_margin'] = played['to_allowed'] - played['to']
    played['prev_to_margin'] = played.groupby(['team', 'season'])['to_margin'].shift(1)
    prev = played.set_index(['game_id', 'team'])['prev_to_margin']
    home = prev.reindex(pd.MultiIndex.from_arrays([f['game_id'], f['home_team']])).to_numpy()
    away = prev.reindex(pd.MultiIndex.from_arrays([f['game_id'], f['away_team']])).to_numpy()
    x = pd.Series(home - away, index=f.index)
    print('\nM4: last game\'s turnover margin, home minus away (takeaways minus giveaways), against the cover margin')
    covers, partials = [], []
    for name, lo, hi in ERAS:
        era = f['season'].between(lo, hi) & x.notna()
        covers.append(corr(x[era], f.loc[era, 'resid']))
        partials.append(partial_corr(x[era], f.loc[era, 'resid'], f.loc[era, 'pred']))
        print('    %-10s cover r %+.3f | partial r given the production correction %+.3f | win-residual r %+.3f | n=%d' % (
            name, covers[-1], partials[-1], corr(x[era], f.loc[era, 'win_resid']), int(era.sum())))
    ok = x.notna()
    slope, t = slope_t(x[ok], f.loc[ok, 'resid'])
    print('    pooled: %+.2f points of cover margin per turnover of last-game margin, robust t %+.2f' % (slope, t))
    passed = all(v < 0 for v in covers) and all(v < 0 for v in partials) and t <= -3.0
    print('M4 verdict: %s' % ('PASS: the market overreacts to last week\'s turnovers' if passed else
                              'fail: no consistent overreaction to last week\'s turnovers'))


def stability_composite(f, full):
    """M3: the most reliable third of the directed ratings minus the least reliable third, against the cover margin."""
    every_era = [feature for feature in full.index if f.loc[f['season'] <= 2005, feature].std() > 1e-9]
    directed = full.loc[every_era]
    directed = directed[directed['better'].notna() & directed['split_half'].notna()]
    order = directed['split_half'].sort_values()
    third = len(order) // 3
    low, high = list(order.index[:third]), list(order.index[-third:])

    def signed_mean(features):
        parts = [(1.0 if bool(full.loc[c, 'better']) else -1.0) * (f[c] - f[c].mean()) / f[c].std() for c in features]
        return pd.concat(parts, axis=1).mean(axis=1)

    s = signed_mean(high) - signed_mean(low)
    print('\nM3: S = the most reliable third of %d directed ratings minus the least reliable third (signed so better = positive)' % len(order))
    print('    most reliable (%d): %s' % (third, ', '.join('%s (%s, %.2f)' % (full.loc[c, 'label'], full.loc[c, 'side'][:3], full.loc[c, 'split_half']) for c in reversed(high))))
    print('    least reliable (%d): %s' % (third, ', '.join('%s (%s, %.2f)' % (full.loc[c, 'label'], full.loc[c, 'side'][:3], full.loc[c, 'split_half']) for c in low)))
    covers, partials = [], []
    for name, lo, hi in ERAS:
        era = f['season'].between(lo, hi)
        covers.append(corr(s[era], f.loc[era, 'resid']))
        partials.append(partial_corr(s[era], f.loc[era, 'resid'], f.loc[era, 'pred']))
        print('    %-10s cover r %+.3f | partial r given the production correction %+.3f | correction alone r %+.3f | n=%d' % (
            name, covers[-1], partials[-1], corr(f.loc[era, 'pred'], f.loc[era, 'resid']), int(era.sum())))
    design = np.column_stack([np.ones(len(f)), f['pred']])
    left = f['resid'].to_numpy(float) - design @ np.linalg.lstsq(design, f['resid'].to_numpy(float), rcond=None)[0]
    slope, t = slope_t(s, left)
    passed = all(v > 0 for v in covers) and all(v > 0 for v in partials) and t >= 3.0
    print('    pooled: %+.2f points of cover margin per unit of S beyond the correction, robust t %+.2f' % (slope, t))
    print('M3 verdict: %s' % ('PASS: S adds to the production correction in every era' if passed else
                              'fail: S does not add to the production correction in every era'))


def starting_hypothesis(full):
    """The owner's original stats (yards, first downs, turnovers) on all four questions."""
    print('\nthe starting hypothesis (yards, first downs, turnovers) on all four questions:')
    print('    %-34s %-8s %13s %11s %13s %7s %7s %8s' % ('stat', 'side', 'same-game r', 'split-half', 'pre-game r', 'worth', 'priced', 'cover t'))
    for feature in ('off__yds', 'def__yds', 'off__fd', 'def__fd', 'off__to', 'def__to'):
        row = full.loc[feature]
        print('    %-34s %-8s %+13.3f %11.2f %+13.3f %+7.2f %+7.2f %+8.2f' % (
            row['label'], row['side'], row['r_win'], row['split_half'], row['pre_r_win'], row['worth'], row['priced'], row['cover t']))


if __name__ == '__main__':
    main()
