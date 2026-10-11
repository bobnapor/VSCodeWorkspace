"""NFL home-margin model built on nflverse game-by-game data.

Every team gets a pre-game rating for each stat, offense and defense, as a recency-weighted average of its
earlier games (shrunk toward the league average early in a season and carried partly across seasons). A
ridge regression turns the two teams' ratings, rest, neutral-site and quarterback information into a
projected home margin. Everything a game's features use was known before kickoff, so the same features
can be built for past games (training and backtesting) and for upcoming ones (projections).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nfl_injury_features import TEAM_FULL_NAMES, TEAM_RENAMES, build_injury_features
from nflverse_data import load_games, load_injuries, load_player_weeks, load_team_weeks

DATA_FIRST_SEASON = 2004   # team ratings warm up from here
TRAIN_FIRST_SEASON = 2010  # first season used for fitting (quarterback features start in 2009)
TRAIN_SEASONS = 12         # most recent seasons used for fitting

# Stats rated for every team, offense (what it produces) and defense (what it allows)
RATED_METRICS = ('fd', 'yds', 'plays', 'epa_pp', 'to', 'pts')
RATING_DECAY = 0.90        # weight of an earlier game relative to the next one; ~7-game half-life
SEASON_CARRY = 0.6         # how much of last season's rating carries into the new season
PRIOR_GAMES = 3.0          # pseudo-games of league-average performance that shrink early-season ratings
RIDGE_ALPHA = 300.0
OPPONENT_ADJUSTMENT = 0.6  # share of each opponent's deviation from average removed from a team's stats
ELO_K = 0.07               # Elo update per point of surprise
ELO_REGRESSION = 0.3       # share of Elo rating lost each offseason
ELO_HOME_FIELD = 2.0       # points

PRODUCTION_METRICS = ('yds', 'to', 'fd', 'pts', 'plays')  # net yards, turnovers, first downs, points, plays
CONTEXT_FEATURES = ('neutral', 'rest_diff')
QB_FEATURES = ('qb_rating_diff', 'qb_injury_edge')


def normalize_teams(games):
    games = games.copy()
    for column in ('home_team', 'away_team'):
        games[column] = games[column].replace(TEAM_RENAMES)
    return games


def team_game_stats(team_weeks):
    """Offensive output for each team in each game."""
    tw = team_weeks.copy()
    tw['plays'] = tw['attempts'] + tw['carries'] + tw['sacks_suffered']
    tw['fd'] = tw['passing_first_downs'].fillna(0) + tw['rushing_first_downs'].fillna(0)
    tw['yds'] = tw['passing_yards'] + tw['rushing_yards'] + tw['sack_yards_lost'].fillna(0)
    tw['to'] = (tw['passing_interceptions'].fillna(0) + tw['sack_fumbles_lost'].fillna(0)
                + tw['rushing_fumbles_lost'].fillna(0) + tw['receiving_fumbles_lost'].fillna(0))
    tw['epa_pp'] = (tw['passing_epa'].fillna(0) + tw['rushing_epa'].fillna(0)) / tw['plays']
    return tw[['game_id', 'team', 'fd', 'yds', 'plays', 'epa_pp', 'to']]


def build_team_rows(games, team_weeks):
    """One row per team per game (played or scheduled) with what the team produced and allowed."""
    base = games[['game_id', 'season', 'week', 'gameday', 'game_type', 'home_team', 'away_team', 'home_score', 'away_score']]
    home = base.assign(team=base['home_team'], opp=base['away_team'], pts=base['home_score'], is_home=1)
    away = base.assign(team=base['away_team'], opp=base['home_team'], pts=base['away_score'], is_home=0)
    rows = pd.concat([home, away], ignore_index=True)
    stats = team_game_stats(team_weeks)
    rows = rows.merge(stats, on=['game_id', 'team'], how='left')
    allowed = stats.rename(columns={'team': 'opp', **{m: m + '_allowed' for m in RATED_METRICS if m != 'pts'}})
    rows = rows.merge(allowed, on=['game_id', 'opp'], how='left')
    opp_pts = rows[['game_id', 'team', 'pts']].rename(columns={'team': 'opp', 'pts': 'pts_allowed'})
    rows = rows.merge(opp_pts, on=['game_id', 'opp'], how='left')
    return rows.sort_values(['team', 'gameday', 'week']).reset_index(drop=True)


def add_pregame_ratings(rows, decay=RATING_DECAY, carry=SEASON_CARRY, prior_games=PRIOR_GAMES, adjust=OPPONENT_ADJUSTMENT):
    """Rating of each team before each game; a game's own stats only affect later games.

    Games are processed in date order so each team's stats can be adjusted for the strength of the opponent
    it produced them against (an offense is credited less for gaining yards against a weak defense).
    `adjust` is how much of the opponent's deviation from average is removed: 0 for none, 1 for all.
    """
    league_mean = {}
    for m in RATED_METRICS:
        league_mean[m] = rows[m].mean()
        league_mean[m + '_allowed'] = rows[m + '_allowed'].mean()
    names = list(league_mean)
    n = len(rows)
    out = {name: np.empty(n) for name in names}
    games_played = np.zeros(n)
    values = {name: rows[name].to_numpy(dtype=float) for name in names}
    teams = rows['team'].to_numpy()
    seasons = rows['season'].to_numpy()

    state = {}  # team -> [weighted sums, weight, last season, games played this season]
    game_rows = rows.groupby('game_id', sort=False).indices
    order = rows.drop_duplicates('game_id').sort_values(['gameday', 'week'])['game_id'].tolist()
    for game_id in order:
        pair = game_rows[game_id]
        if len(pair) != 2:
            continue
        for i in pair:
            s = state.setdefault(teams[i], [{name: 0.0 for name in names}, 0.0, None, 0])
            if s[2] is not None and s[2] != seasons[i]:
                s[0] = {name: v * carry for name, v in s[0].items()}
                s[1] *= carry
                s[3] = 0
            s[2] = seasons[i]
            for name in names:
                out[name][i] = (s[0][name] + prior_games * league_mean[name]) / (s[1] + prior_games)
            games_played[i] = s[3]
        i, j = pair
        if np.isnan(values['yds'][i]) or np.isnan(values['yds'][j]):
            continue  # not played yet
        for me, other in ((i, j), (j, i)):
            s = state[teams[me]]
            for name in names:
                value = values[name][me]
                if np.isnan(value):
                    value = league_mean[name]
                else:
                    # own offense is judged against what the opponent's defense normally allows, and vice versa
                    counterpart = name + '_allowed' if not name.endswith('_allowed') else name[:-len('_allowed')]
                    value -= adjust * (out[counterpart][other] - league_mean[counterpart])
                s[0][name] = s[0][name] * decay + value
            s[1] = s[1] * decay + 1.0
            s[3] += 1

    rated = rows.copy()
    for name in names:
        rated['r_' + name] = out[name]
    rated['games_played'] = games_played
    return rated


def add_elo(games, k=ELO_K, regress=ELO_REGRESSION, hfa=ELO_HOME_FIELD):
    """Pre-game margin-based Elo (in points) for every game, played or scheduled."""
    ratings = {}
    last_season = {}
    margin_cap = 24.0
    records = []
    for g in games.sort_values(['gameday', 'week']).itertuples():
        for team in (g.home_team, g.away_team):
            if last_season.get(team) is not None and last_season[team] != g.season:
                ratings[team] = ratings.get(team, 0.0) * (1.0 - regress)
            last_season[team] = g.season
        home, away = ratings.get(g.home_team, 0.0), ratings.get(g.away_team, 0.0)
        site = 0.0 if g.location == 'Neutral' else hfa
        records.append((g.game_id, home, away))
        if not pd.isna(g.result):
            error = max(-margin_cap, min(margin_cap, g.result)) - (home - away + site)
            ratings[g.home_team] = home + k * error
            ratings[g.away_team] = away - k * error
    elo = pd.DataFrame(records, columns=['game_id', 'h_elo', 'a_elo'])
    elo['elo_diff'] = elo['h_elo'] - elo['a_elo']
    return elo


def build_game_table(games, rated_rows, qb_features=None, elo=None):
    """One row per game: both teams' pre-game ratings plus rest, neutral site, Elo and quarterback features."""
    rating_columns = ['r_' + m for m in RATED_METRICS] + ['r_' + m + '_allowed' for m in RATED_METRICS]
    keep = ['game_id', 'team'] + rating_columns + ['games_played']
    home = rated_rows[keep].rename(columns={'team': 'home_team', **{c: 'h_' + c for c in rating_columns + ['games_played']}})
    away = rated_rows[keep].rename(columns={'team': 'away_team', **{c: 'a_' + c for c in rating_columns + ['games_played']}})
    table = games.merge(home, on=['game_id', 'home_team']).merge(away, on=['game_id', 'away_team'])
    table['neutral'] = (table['location'] == 'Neutral').astype(int)
    table['rest_diff'] = (table['home_rest'] - table['away_rest']).clip(-4, 4).fillna(0)
    if qb_features is not None:
        table = table.merge(qb_features[['game_id', 'qb_rating_diff', 'qb_injury_edge']], on='game_id', how='left')
    if elo is not None:
        table = table.merge(elo[['game_id', 'elo_diff']], on='game_id', how='left')
    return table


def feature_columns(metrics, use_qb=True, use_elo=False):
    columns = []
    for m in metrics:
        columns += [f'h_r_{m}', f'h_r_{m}_allowed', f'a_r_{m}', f'a_r_{m}_allowed']
    columns += list(CONTEXT_FEATURES)
    if use_qb:
        columns += list(QB_FEATURES)
    if use_elo:
        columns.append('elo_diff')
    return columns


def load_inputs(refresh=False, last_season=None):
    """Raw nflverse inputs: games, per-team stat rows, quarterback features."""
    games = normalize_teams(load_games(refresh=refresh))
    games = games[games['season'] >= DATA_FIRST_SEASON]
    if last_season is not None:
        games = games[games['season'] <= last_season]
    last = int(games['season'].max())
    team_weeks = load_team_weeks(DATA_FIRST_SEASON, last, refresh=refresh)
    qb_features = build_injury_features(
        games, load_injuries(2009, last, refresh=refresh), load_player_weeks(2009, last, refresh=refresh))
    return {'games': games, 'rows': build_team_rows(games, team_weeks), 'qb_features': qb_features}


def assemble_table(inputs, decay=RATING_DECAY, carry=SEASON_CARRY, prior_games=PRIOR_GAMES, adjust=OPPONENT_ADJUSTMENT,
                   elo_k=ELO_K, elo_regress=ELO_REGRESSION):
    rated = add_pregame_ratings(inputs['rows'], decay, carry, prior_games, adjust)
    elo = add_elo(inputs['games'], k=elo_k, regress=elo_regress)
    return build_game_table(inputs['games'], rated, inputs['qb_features'], elo)


def load_game_table(refresh=False, last_season=None, **rating_settings):
    """The full game table, with features for played and scheduled games, from nflverse."""
    return assemble_table(load_inputs(refresh, last_season), **rating_settings)


def _final_regular_season_week(season):
    return 18 if season >= 2021 else 17


def training_frame(table, before_season, before_week=None, seasons=TRAIN_SEASONS):
    """Completed games available before a given point: earlier seasons, plus earlier weeks of `before_season`.

    The last regular-season week of each year is left out because starters often rest.
    """
    played = table[table['result'].notna() & (table['season'] >= TRAIN_FIRST_SEASON)]
    in_window = played['season'] >= before_season - seasons
    earlier = (played['season'] < before_season) | ((played['season'] == before_season) & (played['week'] < (before_week or 0)))
    resting = (played['game_type'] == 'REG') & (played['week'] == played['season'].map(_final_regular_season_week))
    return played[in_window & earlier & ~resting]


def fit(train, columns, alpha=RIDGE_ALPHA):
    fill = train[columns].median()
    scaler = StandardScaler().fit(train[columns].fillna(fill))
    model = Ridge(alpha=alpha).fit(scaler.transform(train[columns].fillna(fill)), train['result'])
    return {'model': model, 'scaler': scaler, 'fill': fill, 'columns': columns}


def predict(fitted, frame):
    X = fitted['scaler'].transform(frame[fitted['columns']].fillna(fitted['fill']))
    return fitted['model'].predict(X)


def bootstrap_projection(train, frame, columns, runs=200, alpha=RIDGE_ALPHA, seed=0, progress=None):
    """Projected home margin for `frame` from `runs` ridge regressions, each fitted to a bootstrap resample of `train`.

    Each run draws len(train) training games with replacement, fills missing values with the sample's medians,
    standardizes, fits the ridge and projects every game in `frame`; the mean across runs is the projection and the
    spread across runs its uncertainty.  Same steps and random draws as fit()/predict() run in a loop, done in numpy so
    thousands of runs take seconds.  `progress(done, runs)` is called about every 10% of the runs.
    Returns an array of shape (runs, len(frame)).
    """
    rng = np.random.default_rng(seed)
    X = train[columns].to_numpy(dtype=float)
    y = train['result'].to_numpy(dtype=float)
    X_new = frame[columns].to_numpy(dtype=float)
    n, p = X.shape
    penalty = alpha * np.eye(p)
    # medians are only needed where something is missing; other columns' fill values are never used
    needs_fill = np.isnan(X).any(axis=0) | np.isnan(X_new).any(axis=0)
    every = max(1, runs // 10)
    predictions = np.empty((runs, len(frame)))
    for i in range(runs):
        rows = rng.integers(0, n, n)
        sample, target = X[rows], y[rows]
        fill = np.zeros(p)
        if needs_fill.any():
            fill[needs_fill] = np.nanmedian(sample[:, needs_fill], axis=0)
            sample = np.where(np.isnan(sample), fill, sample)
        mean, scale = sample.mean(axis=0), sample.std(axis=0)
        scale[scale < 10 * np.finfo(float).eps] = 1.0          # as StandardScaler treats constant columns
        z = (sample - mean) / scale
        z_mean, y_mean = z.mean(axis=0), target.mean()
        zc = z - z_mean
        coef = np.linalg.solve(zc.T @ zc + penalty, zc.T @ (target - y_mean))
        intercept = y_mean - z_mean @ coef
        predictions[i] = ((np.where(np.isnan(X_new), fill, X_new) - mean) / scale) @ coef + intercept
        if progress and ((i + 1) % every == 0 or i + 1 == runs):
            progress(i + 1, runs)
    return predictions
