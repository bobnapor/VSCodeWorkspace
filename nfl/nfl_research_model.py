"""Walk-forward comparison of margin models: which stats help predict NFL results?

Each test season is predicted by a model fitted only on earlier seasons. Reports how close the projections
get to the real margin (MAE, RMSE, winner picked) next to the closing spread, which is the benchmark to
beat, plus the against-the-spread record.
"""

import sys

import numpy as np
import pandas as pd

import nfl_model as nm
import nfl_research_ats as ats

FIRST_TEST = 2014


def evaluate(table, columns, first_test=FIRST_TEST, alpha=nm.RIDGE_ALPHA):
    outputs = []
    for season in range(first_test, int(table['season'].max()) + 1):
        train = nm.training_frame(table, season)
        test = table[(table['season'] == season) & table['result'].notna() & table['spread_line'].notna()
                     & (table['game_type'] == 'REG')]
        if train.empty or test.empty:
            continue
        out = test.copy()
        out['pred_margin'] = nm.predict(nm.fit(train, columns, alpha), out)
        outputs.append(out)
    frame = pd.concat(outputs, ignore_index=True)
    frame['err'] = frame['pred_margin'] - frame['result']
    return frame


def summarize(label, frame):
    err = frame['err']
    winner = ((frame['pred_margin'] > 0) == (frame['result'] > 0))[frame['result'] != 0].mean()
    print('%-34s n=%5d  MAE=%.3f  RMSE=%.3f  winner=%.3f' % (label, len(frame), err.abs().mean(), np.sqrt((err ** 2).mean()), winner))


def paired(label, a, b):
    """Mean difference in absolute error, a - b (negative means a is closer), with its standard error."""
    diff = a['err'].abs().values - b['err'].abs().values
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print('%-46s %+.3f points MAE  (t=%.1f)' % (label, diff.mean(), diff.mean() / se))


def ats_line(label, frame):
    f = frame.copy()
    f['pred'] = f['pred_margin'] - f['spread_line']
    f['resid'] = f['result'] - f['spread_line']
    f['abs_spread'] = f['spread_line'].abs()
    print('--', label)
    ats.ats_summary(f, 'all games')
    ats.ats_summary(f[f['abs_spread'] >= 4], 'abs spread >= 4')
    for t in (2, 4):
        ats.ats_summary(f[(f['abs_spread'] >= 4) & (f['pred'].abs() >= t)], 'abs spread >= 4 & edge >= %d' % t)


def old_production_model(first_test=FIRST_TEST):
    """The previous production model: linear regression of the margin on season-to-date yards and turnovers."""
    from nflverse_data import load_games, load_team_weeks

    games = ats.normalize_teams(load_games())
    team_weeks = load_team_weeks(ats.FIRST_SEASON, ats.LAST_SEASON)
    team_games = ats.build_team_games(games, team_weeks)
    features = ats.build_game_features(games, ats.add_ratings(ats.add_season_to_date(team_games)))
    frame = ats.walk_forward_margin(features, first_test=first_test)
    frame['err'] = frame['pred_margin'] - frame['result']
    return frame


def main():
    table = nm.load_game_table()
    print('games', len(table), 'seasons', table['season'].min(), table['season'].max())
    stat_sets = {
        'yards + turnovers': ('yds', 'to'),
        '+ first downs': ('yds', 'to', 'fd'),
        '+ points, plays (production)': nm.PRODUCTION_METRICS,
        '+ EPA': nm.PRODUCTION_METRICS + ('epa_pp',),
    }
    results = {}
    for name, metrics in stat_sets.items():
        results[name] = evaluate(table, nm.feature_columns(metrics, use_qb=True, use_elo=True))
    results['production without first downs'] = evaluate(
        table, nm.feature_columns(tuple(m for m in nm.PRODUCTION_METRICS if m != 'fd'), use_qb=True, use_elo=True))
    results['production without Elo'] = evaluate(table, nm.feature_columns(nm.PRODUCTION_METRICS, use_qb=True, use_elo=False))
    results['production without QB'] = evaluate(table, nm.feature_columns(nm.PRODUCTION_METRICS, use_qb=False, use_elo=True))
    results['Elo only'] = evaluate(table, ['elo_diff', 'neutral'])
    production = results['+ points, plays (production)']

    old = old_production_model()
    old = old[old['game_id'].isin(production['game_id'])]
    market = production.copy()
    market['pred_margin'] = market['spread_line']
    market['err'] = market['pred_margin'] - market['result']
    same = production[production['game_id'].isin(old['game_id'])]

    print('\n=== how close each model gets to the real home margin, test seasons %d-%d ===' % (
        production['season'].min(), production['season'].max()))
    summarize('closing spread (benchmark)', market)
    summarize('previous production model', old)
    for name, frame in results.items():
        summarize(name, frame)

    print('\n=== does each piece help? (paired by game; negative = closer to the real margin) ===')
    paired('new production model vs previous model', same.sort_values('game_id'), old.sort_values('game_id'))
    paired('first downs added to yards + turnovers', results['+ first downs'], results['yards + turnovers'])
    paired('first downs removed from the production model', results['production without first downs'], production)
    paired('Elo removed', results['production without Elo'], production)
    paired('QB features removed', results['production without QB'], production)
    paired('EPA added to the production model', results['+ EPA'], production)
    paired('production model vs closing spread', production, market)

    print('\n=== against the spread ===')
    ats_line('production model', production)
    return table, results


if __name__ == '__main__':
    main()
