"""Walk-forward test of whether any model can beat NFL closing spreads, using nflverse data.

Features for each game use only information available before kickoff: exponentially weighted team ratings
from earlier games (carried over from prior seasons with a discount), rest, and quarterback changes.
The model predicts how far the actual home margin lands from the closing spread; the sign of that
prediction is the side to bet. Each test season is predicted by a model trained only on earlier seasons.
"""

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nflverse_data import load_games, load_team_weeks

FIRST_SEASON = 2004
FIRST_TEST_SEASON = 2012
LAST_SEASON = 2026


def build_team_games(games, team_weeks):
    """One row per team per regular-season game with that game's EPA per play on offense and allowed."""
    tw = team_weeks[team_weeks['season_type'] == 'REG'].copy()
    tw['plays'] = tw['attempts'] + tw['carries'] + tw['sacks_suffered']
    tw['epa'] = tw['passing_epa'].fillna(0) + tw['rushing_epa'].fillna(0)
    tw['turnovers'] = (tw['passing_interceptions'].fillna(0) + tw['sack_fumbles_lost'].fillna(0)
                       + tw['rushing_fumbles_lost'].fillna(0) + tw['receiving_fumbles_lost'].fillna(0))
    offense = tw[['game_id', 'team', 'opponent_team', 'plays', 'epa', 'passing_yards', 'rushing_yards', 'turnovers']].rename(
        columns={'plays': 'off_plays', 'epa': 'off_epa', 'passing_yards': 'pass_yds', 'rushing_yards': 'rush_yds',
                 'turnovers': 'to_lost'})
    defense = tw[['game_id', 'team', 'plays', 'epa', 'passing_yards', 'rushing_yards', 'turnovers']].rename(
        columns={'team': 'opponent_team', 'plays': 'def_plays', 'epa': 'def_epa', 'passing_yards': 'def_pass_yds',
                 'rushing_yards': 'def_rush_yds', 'turnovers': 'to_won'}
    )
    team_games = offense.merge(defense, on=['game_id', 'opponent_team'], how='left')
    team_games['off_epa_pp'] = team_games['off_epa'] / team_games['off_plays']
    team_games['def_epa_pp'] = team_games['def_epa'] / team_games['def_plays']

    g = games[games['game_type'] == 'REG'][['game_id', 'season', 'week', 'gameday', 'home_team', 'away_team', 'home_score', 'away_score']]
    team_games = team_games.merge(g, on='game_id', how='inner')
    is_home = team_games['team'] == team_games['home_team']
    team_games['margin'] = np.where(is_home, team_games['home_score'] - team_games['away_score'],
                                    team_games['away_score'] - team_games['home_score'])
    return team_games.sort_values(['team', 'gameday', 'week']).reset_index(drop=True)


def add_ratings(team_games, decay=0.88, prior_weight=2.0, season_carry=0.5):
    """Pre-game exponentially weighted ratings per team (recorded before each game updates them)."""
    metrics = ('off_epa_pp', 'def_epa_pp', 'margin')
    out = {m: [] for m in metrics}
    out['games_played'] = []
    for team, rows in team_games.groupby('team', sort=False):
        num = {m: 0.0 for m in metrics}
        den = 0.0
        last_season = None
        played = 0
        for _, row in rows.iterrows():
            if last_season is not None and row['season'] != last_season:
                num = {m: v * season_carry for m, v in num.items()}
                den *= season_carry
                played = 0
            last_season = row['season']
            for m in metrics:
                out[m].append(num[m] / (den + prior_weight))
            out['games_played'].append(played)
            for m in metrics:
                value = row[m]
                num[m] = num[m] * decay + (0.0 if pd.isna(value) else value)
            den = den * decay + 1.0
            played += 1
    ratings = team_games.copy()
    for m in metrics:
        ratings['r_' + m] = out[m]
    ratings['games_played'] = out['games_played']
    return ratings


def build_game_features(games, ratings):
    g = games[games['game_type'] == 'REG'].copy()
    g = g[g['spread_line'].notna() & g['home_score'].notna()]
    keep = ['game_id', 'team', 'r_off_epa_pp', 'r_def_epa_pp', 'r_margin', 'games_played'] + ['s_' + m for m in STYLE_METRICS]
    home = ratings[keep].rename(columns={'team': 'home_team', **{c: 'h_' + c for c in keep[2:]}})
    away = ratings[keep].rename(columns={'team': 'away_team', **{c: 'a_' + c for c in keep[2:]}})
    g = g.merge(home, on=['game_id', 'home_team']).merge(away, on=['game_id', 'away_team'])

    # starting quarterback differs from the team's previous game
    g['home_qb_key'] = list(zip(g['season'], g['gameday'], g['home_team']))
    g['away_qb_key'] = list(zip(g['season'], g['gameday'], g['away_team']))
    qb_rows = pd.concat([
        g[['season', 'gameday', 'home_team', 'home_qb_id']].rename(columns={'home_team': 'team', 'home_qb_id': 'qb'}),
        g[['season', 'gameday', 'away_team', 'away_qb_id']].rename(columns={'away_team': 'team', 'away_qb_id': 'qb'}),
    ]).sort_values(['team', 'gameday'])
    qb_rows['prev_qb'] = qb_rows.groupby('team')['qb'].shift(1)
    qb_rows['qb_changed'] = ((qb_rows['qb'] != qb_rows['prev_qb']) & qb_rows['prev_qb'].notna()).astype(int)
    changed = {(r.season, r.gameday, r.team): r.qb_changed for r in qb_rows.itertuples()}
    g['h_qb_changed'] = [changed.get(k, 0) for k in g['home_qb_key']]
    g['a_qb_changed'] = [changed.get(k, 0) for k in g['away_qb_key']]

    g['d_epa'] = (g['h_r_off_epa_pp'] - g['h_r_def_epa_pp']) - (g['a_r_off_epa_pp'] - g['a_r_def_epa_pp'])
    g['d_margin'] = g['h_r_margin'] - g['a_r_margin']
    g['rest_diff'] = (g['home_rest'] - g['away_rest']).clip(-4, 4)
    g['div_game'] = g['div_game'].fillna(0)
    g['qb_change_diff'] = g['h_qb_changed'] - g['a_qb_changed']
    g['abs_spread'] = g['spread_line'].abs()
    g['resid'] = g['result'] - g['spread_line']  # >0: home covered
    return g


STYLE_METRICS = ('pass_yds', 'rush_yds', 'to_lost', 'def_pass_yds', 'def_rush_yds', 'to_won')


def add_season_to_date(team_games, prior_weight=4.0):
    """Pre-game season-to-date per-game averages (what the production model sees), shrunk toward last
    season's full-year average early in the year."""
    out = {m: [] for m in STYLE_METRICS}
    for team, rows in team_games.groupby('team', sort=False):
        prior = None
        season_sum = {m: 0.0 for m in STYLE_METRICS}
        count = 0
        last_season = None
        for _, row in rows.iterrows():
            if row['season'] != last_season:
                if last_season is not None and count:
                    prior = {m: season_sum[m] / count for m in STYLE_METRICS}
                season_sum = {m: 0.0 for m in STYLE_METRICS}
                count = 0
                last_season = row['season']
            for m in STYLE_METRICS:
                current = season_sum[m]
                if prior is None:
                    out[m].append(current / count if count else np.nan)
                else:
                    out[m].append((current + prior_weight * prior[m]) / (count + prior_weight))
            for m in STYLE_METRICS:
                value = row[m]
                season_sum[m] += 0.0 if pd.isna(value) else value
            count += 1
    result = team_games.copy()
    for m in STYLE_METRICS:
        result['s_' + m] = out[m]
    return result


STYLE_FEATURES = ['h_s_' + m for m in STYLE_METRICS] + ['a_s_' + m for m in STYLE_METRICS]


def walk_forward_margin(features, columns=STYLE_FEATURES, first_test=FIRST_TEST_SEASON, last_test=LAST_SEASON):
    """Production-style model: linear regression of the home margin on season-to-date team stats, trained on
    earlier seasons. The pick is the side of the closing spread the predicted margin lands on."""
    from sklearn.linear_model import LinearRegression

    usable = features.dropna(subset=columns)
    outputs = []
    for season in range(first_test, last_test + 1):
        train = usable[(usable['season'] < season) & (usable['season'] > FIRST_SEASON + 1)]
        test = usable[usable['season'] == season]
        if train.empty or test.empty:
            continue
        model = LinearRegression().fit(train[columns], train['result'])
        out = test.copy()
        out['pred_margin'] = model.predict(test[columns])
        out['pred'] = out['pred_margin'] - out['spread_line']
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


FEATURES = ['d_epa', 'd_margin', 'rest_diff', 'qb_change_diff', 'spread_line', 'div_game']


def walk_forward(features, columns=FEATURES, alpha=500.0, first_test=FIRST_TEST_SEASON, last_test=LAST_SEASON):
    """Out-of-sample prediction of the market residual for each test season, trained on earlier seasons only."""
    outputs = []
    for season in range(first_test, last_test + 1):
        train = features[(features['season'] < season) & (features['season'] > FIRST_SEASON)]
        test = features[features['season'] == season]
        if train.empty or test.empty:
            continue
        scaler = StandardScaler().fit(train[columns])
        model = Ridge(alpha=alpha).fit(scaler.transform(train[columns]), train['resid'])
        pred = model.predict(scaler.transform(test[columns]))
        out = test.copy()
        out['pred'] = pred
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def walk_forward_boosted(features, extra_columns=(), first_test=FIRST_TEST_SEASON, last_test=LAST_SEASON,
                         first_train=FIRST_SEASON):
    """Gradient boosting on every pre-game feature (ratings, rest, quarterback changes, weather, total, spread)
    predicting the market residual, with shallow trees and strong regularization to limit overfitting."""
    from sklearn.ensemble import GradientBoostingRegressor

    data = features.copy()
    data['dome'] = data['roof'].isin(['dome', 'closed']).astype(int)
    for side in ('h', 'a'):
        data[f'{side}_net_epa'] = data[f'{side}_r_off_epa_pp'] - data[f'{side}_r_def_epa_pp']
    columns = FEATURES + ['total_line', 'wind', 'temp', 'dome', 'h_net_epa', 'a_net_epa', 'h_r_off_epa_pp', 'h_r_def_epa_pp',
                          'a_r_off_epa_pp', 'a_r_def_epa_pp', 'h_qb_changed', 'a_qb_changed', 'h_games_played'] + list(extra_columns)
    outputs = []
    for season in range(first_test, last_test + 1):
        train = data[(data['season'] < season) & (data['season'] > first_train)]
        test = data[data['season'] == season]
        if train.empty or test.empty:
            continue
        model = GradientBoostingRegressor(max_depth=3, learning_rate=0.03, n_estimators=150, min_samples_leaf=80,
                                          subsample=0.7, random_state=0)
        fill = train[columns].median()
        model.fit(train[columns].fillna(fill), train['resid'])
        out = test.copy()
        out['pred'] = model.predict(test[columns].fillna(fill))
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def ats_summary(df, label):
    d = df[df['resid'] != 0]
    picks_home = d['pred'] > 0
    win = np.where(picks_home, d['resid'] > 0, d['resid'] < 0)
    n = len(d)
    rate = win.mean() if n else float('nan')
    se = math.sqrt(0.25 / n) if n else float('nan')
    roi = rate * (100 / 110) - (1 - rate)
    print('%-44s n=%5d  ATS=%.3f +-%.3f  ROI@-110=%+.3f' % (label, n, rate, se, roi))
    return win


def normalize_teams(games):
    """Franchise moves: the stats files use the current abbreviations (LV, LAC, LA) for every season."""
    renames = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}
    games = games.copy()
    for column in ('home_team', 'away_team'):
        games[column] = games[column].replace(renames)
    return games


def report(predictions, title, edge_bins):
    predictions = predictions.copy()
    predictions['abs_pred'] = predictions['pred'].abs()
    print('\n=== %s: walk-forward ATS, test seasons %d-%d ===' % (title, predictions['season'].min(), predictions['season'].max()))
    ats_summary(predictions, 'all games')
    print('by season:')
    for season, g in predictions.groupby('season'):
        ats_summary(g, str(season))
    print('by size of the model\'s disagreement with the spread (points):')
    for lo, hi in edge_bins:
        ats_summary(predictions[(predictions['abs_pred'] >= lo) & (predictions['abs_pred'] < hi)], 'edge [%.1f,%.1f)' % (lo, hi))
    print('by market spread size:')
    for lo, hi in ((0, 3), (3, 4), (4, 7), (7, 10), (10, 40)):
        ats_summary(predictions[(predictions['abs_spread'] >= lo) & (predictions['abs_spread'] < hi)], 'abs spread [%d,%d)' % (lo, hi))
    big = predictions[predictions['abs_spread'] >= 3.5]
    ats_summary(big, 'abs spread >= 3.5')
    for t in (2, 4, 6):
        ats_summary(big[big['abs_pred'] >= t], 'abs spread >= 3.5 & edge >= %d' % t)
    four_plus = predictions[predictions['abs_spread'] >= 4]
    ats_summary(four_plus, 'abs spread >= 4')
    for t in (1, 2, 4, 6):
        ats_summary(four_plus[four_plus['abs_pred'] >= t], 'abs spread >= 4 & edge >= %d' % t)
    print('by season, abs spread >= 4:')
    for season, g in four_plus.groupby('season'):
        ats_summary(g, str(season))
    return predictions


def main():
    games = normalize_teams(load_games())
    team_weeks = load_team_weeks(FIRST_SEASON, LAST_SEASON)
    seasons = sorted(team_weeks['season'].unique())
    print('team-week seasons loaded: %d-%d' % (seasons[0], seasons[-1]))
    team_games = build_team_games(games, team_weeks)
    ratings = add_ratings(add_season_to_date(team_games))
    features = build_game_features(games, ratings)
    print('games with features:', len(features), 'seasons', features['season'].min(), features['season'].max())
    print('market sanity: mean(result - spread_line) = %.3f, corr(result, spread_line) = %.3f' % (
        features['resid'].mean(), np.corrcoef(features['result'], features['spread_line'])[0, 1]))

    ratings_predictions = walk_forward(features)
    report(ratings_predictions, 'EPA ratings + rest + QB change + market', ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 99)))
    style_predictions = walk_forward_margin(features)
    report(style_predictions, 'production-style yards/turnovers regression', ((0, 2), (2, 4), (4, 6), (6, 9), (9, 99)))
    boosted_predictions = walk_forward_boosted(features)
    report(boosted_predictions, 'gradient boosting on all features', ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 99)))
    return ratings_predictions, style_predictions, boosted_predictions, features


if __name__ == '__main__':
    main()
