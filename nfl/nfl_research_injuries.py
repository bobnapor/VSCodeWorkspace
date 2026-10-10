"""Do injury reports (especially quarterback injuries) help beat the closing spread?

Layers the pre-game injury features from nfl_injury_features.py onto the walk-forward framework in
nfl_research_ats.py, then looks directly at games where a starting quarterback is on the injury report.
"""

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import nfl_research_ats as base
from nfl_injury_features import build_injury_features
from nflverse_data import load_games, load_injuries, load_player_weeks, load_team_weeks

INJURY_FIRST_SEASON = 2009
FIRST_TEST = 2012
INJURY_COLUMNS = ['qb_injury_edge', 'qb_rating_diff', 'h_qb_out', 'a_qb_out', 'h_qb_questionable', 'a_qb_questionable',
                  'out_ol_diff', 'out_dl_diff', 'out_lb_diff', 'out_db_diff', 'out_skill_diff']


def load_features():
    games = base.normalize_teams(load_games())
    team_weeks = load_team_weeks(base.FIRST_SEASON, base.LAST_SEASON)
    team_games = base.build_team_games(games, team_weeks)
    ratings = base.add_ratings(base.add_season_to_date(team_games))
    features = base.build_game_features(games, ratings)
    injuries = load_injuries(INJURY_FIRST_SEASON, base.LAST_SEASON)
    player_weeks = load_player_weeks(INJURY_FIRST_SEASON, base.LAST_SEASON)
    injury_features = build_injury_features(games, injuries, player_weeks)
    return features.merge(injury_features, on='game_id', how='left')


def side_results(frame, bet_home):
    """Win flags for betting the home side (bet_home True) or the away side, pushes dropped."""
    d = frame[frame['resid'] != 0]
    win = (d['resid'] > 0) if bet_home else (d['resid'] < 0)
    return win


def line(label, wins):
    n = len(wins)
    rate = wins.mean() if n else float('nan')
    se = math.sqrt(0.25 / n) if n else float('nan')
    print('%-52s n=%5d  win=%.3f +-%.3f  ROI@-110=%+.3f' % (label, n, rate, se, rate * (100 / 110) - (1 - rate) if n else float('nan')))


def qb_subset_tests(f):
    f = f[(f['season'] >= FIRST_TEST) & f['h_qb_out'].notna()]
    print('games', len(f), '| home QB out/doubtful', int(f['h_qb_out'].sum()), '| away QB out/doubtful', int(f['a_qb_out'].sum()),
          '| either', int(f['qb_out_any'].sum()), '| home q', int(f['h_qb_questionable'].sum()), '| away q', int(f['a_qb_questionable'].sum()))
    print('QB-out games by season:', f.groupby('season')['qb_out_any'].sum().to_dict())
    print('\n-- does the closing spread already price a QB being out? --')
    x = f[['qb_injury_edge', 'qb_rating_diff', 'spread_line']].copy()
    for name, cols in (('resid ~ qb_injury_edge', ['qb_injury_edge']),
                       ('resid ~ qb_injury_edge + qb_rating_diff', ['qb_injury_edge', 'qb_rating_diff'])):
        X = np.column_stack([np.ones(len(f))] + [f[c].values for c in cols])
        y = f['resid'].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        cov = np.linalg.inv(X.T @ X) * resid.var(ddof=X.shape[1])
        print('%-42s' % name, 'coef', beta[1:].round(2), 't', (beta[1:] / np.sqrt(np.diag(cov))[1:]).round(2))
    print('(coef is points of cover margin per 1.0 of EPA-per-dropback cost; >0 means the market under-reacts to QB injuries)')

    print('\n-- ATS on games where a QB is out/doubtful --')
    home_out = f[(f['h_qb_out'] == 1) & (f['a_qb_out'] == 0)]
    away_out = f[(f['a_qb_out'] == 1) & (f['h_qb_out'] == 0)]
    line('home QB out: bet the healthy (away) side', side_results(home_out, False))
    line('home QB out: bet the injured (home) side', side_results(home_out, True))
    line('away QB out: bet the healthy (home) side', side_results(away_out, True))
    line('away QB out: bet the injured (away) side', side_results(away_out, False))
    healthy = pd.concat([side_results(home_out, False), side_results(away_out, True)])
    injured = pd.concat([side_results(home_out, True), side_results(away_out, False)])
    line('QB out (either side): bet the healthy side', healthy)
    line('QB out (either side): bet the injured side', injured)
    big_home = home_out[home_out['abs_spread'] >= 4]
    big_away = away_out[away_out['abs_spread'] >= 4]
    line('  ... spread 4+: bet the healthy side', pd.concat([side_results(big_home, False), side_results(big_away, True)]))
    line('  ... spread 4+: bet the injured side', pd.concat([side_results(big_home, True), side_results(big_away, False)]))
    star_home = home_out[home_out['h_qb_rating'] > 0.05]
    star_away = away_out[away_out['a_qb_rating'] > 0.05]
    line('good QB (rating > 0.05) out: bet the healthy side', pd.concat([side_results(star_home, False), side_results(star_away, True)]))
    line('good QB (rating > 0.05) out: bet the injured side', pd.concat([side_results(star_home, True), side_results(star_away, False)]))
    for season in sorted(f['season'].unique()):
        s = f[f['season'] == season]
        ho, ao = s[(s['h_qb_out'] == 1) & (s['a_qb_out'] == 0)], s[(s['a_qb_out'] == 1) & (s['h_qb_out'] == 0)]
        w = pd.concat([side_results(ho, False), side_results(ao, True)])
        if len(w) >= 8:
            line('   %d healthy side' % season, w)


def ridge_residual(features, columns, alpha=300.0, first_test=FIRST_TEST):
    outputs = []
    usable = features.dropna(subset=columns)
    for season in range(first_test, base.LAST_SEASON + 1):
        train = usable[(usable['season'] < season) & (usable['season'] > INJURY_FIRST_SEASON + 1)]
        test = usable[usable['season'] == season]
        if train.empty or test.empty:
            continue
        scaler = StandardScaler().fit(train[columns])
        model = Ridge(alpha=alpha).fit(scaler.transform(train[columns]), train['resid'])
        out = test.copy()
        out['pred'] = model.predict(scaler.transform(test[columns]))
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def main():
    features = load_features()
    print('games', len(features), 'with injury features', int(features['h_qb_out'].notna().sum()))
    qb_subset_tests(features)

    models = [
        ('ridge: spread + QB/injury features only', ridge_residual(features, ['spread_line'] + INJURY_COLUMNS)),
        ('boosted: base features (same train window)',
         base.walk_forward_boosted(features, first_test=FIRST_TEST, first_train=INJURY_FIRST_SEASON + 1)),
        ('boosted: base + injury features',
         base.walk_forward_boosted(features, extra_columns=INJURY_COLUMNS, first_test=FIRST_TEST, first_train=INJURY_FIRST_SEASON + 1)),
    ]
    for title, predictions in models:
        base.report(predictions, title, ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 99)))
    return features, models


if __name__ == '__main__':
    main()
