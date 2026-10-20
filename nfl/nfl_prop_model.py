"""Player prop projections: a pre-game forecast and an outcome distribution for each player and prop market.

Uses only free nflverse data known before kickoff: player box scores, the closing spread and total (the game the market
expects), the official injury reports and the schedule's starting quarterbacks.  Each market in nfl_prop_lines.MARKETS
has its own model, a gradient-boosted tree ensemble with a Poisson loss that forecasts the player's average result from
  - his recent and season-long production and opportunity (attempts, carries, targets, and his share of the team's),
  - the game environment: the team's implied points from the spread and total and how that compares with what it
    usually scores, the spread (game script), home field and a roof,
  - the opponent's recent yards and catches allowed (to the player's position group for receiving), against the league,
  - teammates ruled Out or Doubtful, by the share of the team's targets and carries they leave behind,
  - the starting quarterback's recent passing and whether he is new (for receivers and runners).
The chance of a result over or under a line comes from what happened in past games with a similar forecast, all of
them forecast out of sample (OutcomeDistribution).  As in nfl_signal.py, a season is forecast by models fitted on the
TRAIN_SEASONS seasons before it.
"""

import logging
import os

import numpy as np
import pandas as pd

try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401  (needed before scikit-learn 1.0)
except ImportError:
    pass
from sklearn.ensemble import HistGradientBoostingRegressor

import nflverse_data as nd
from nfl_injury_features import TEAM_RENAMES
from nfl_prop_lines import MARKETS
from nflverse_data import CACHE_DIR

MODEL_VERSION = 1             # bump after changing features or settings, so cached forecasts are rebuilt
DATA_FIRST_SEASON = 2006      # player histories warm up from here
TRAIN_SEASONS = 10            # each model is fitted on the seasons just before the one it forecasts
FIRST_FORECAST_SEASON = 2018  # first season forecast out of sample (by a model fitted on 2008-2017)
DIST_SEASONS = 6              # seasons of out-of-sample forecasts behind each outcome distribution
NEIGHBORS = {'qb': 400, 'rusher': 1200, 'receiver': 2000}  # past games with the nearest forecasts behind each chance
MIN_HALF_WINDOW = 100         # at the ends of the forecast range the window keeps at least this many games a side
FAST_HALFLIFE, SLOW_HALFLIFE = 1.5, 6.0                     # in games
TREE_SETTINGS = dict(loss='poisson', learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=100,
                     l2_regularization=1.0, early_stopping=False, random_state=0)

STATS = ('attempts', 'completions', 'passing_yards', 'passing_tds', 'passing_interceptions', 'sacks_suffered',
         'carries', 'rushing_yards', 'targets', 'receptions', 'receiving_yards', 'receiving_air_yards')
PLAYER_COLUMNS = ('player_id', 'player_display_name', 'position', 'season', 'week', 'season_type', 'game_id', 'team',
                  'opponent_team') + STATS
OFFENSE_POSITIONS = ('QB', 'RB', 'FB', 'WR', 'TE')
POSITION_CODES = {'QB': 0, 'RB': 1, 'FB': 1, 'WR': 2, 'TE': 3}
RECEIVING_GROUPS = {'RB': 'rb', 'FB': 'rb', 'WR': 'wr', 'TE': 'te'}
SHARES = {'tgt_share': ('targets', 'targets'), 'carry_share': ('carries', 'carries'),
          'ay_share': ('receiving_air_yards', 'receiving_air_yards')}
ALLOWED = ('attempts', 'completions', 'passing_yards', 'passing_tds', 'passing_interceptions', 'sacks_suffered',
           'carries', 'rushing_yards', 'pts') + tuple(f'{stat}_{group}' for group in ('rb', 'wr', 'te')
                                                       for stat in ('targets', 'receptions', 'receiving_yards'))

MARKET_GROUPS = {'pass_yds': 'qb', 'pass_cmp': 'qb', 'pass_td': 'qb', 'pass_int': 'qb', 'pass_att': 'qb',
                 'pass_rush_yds': 'qb', 'rush_att': 'rusher', 'rush_yds': 'rusher', 'rec': 'receiver',
                 'rec_yds': 'receiver', 'rush_rec_yds': 'receiver'}
COUNT_MARKETS = ('pass_cmp', 'pass_td', 'pass_int', 'pass_att', 'rush_att', 'rec')   # whole-number results

KINDS = ('fast', 'slow', 'season')
CONTEXT = ['n_games', 'n_season', 'is_home', 'indoor', 'team_spread', 'total_line', 'implied_pts', 'implied_vs_usual',
           'team_s_pts', 'opp_pts', 'own_questionable']


def _p(kind, stat):
    return f'p_{kind}_{stat}'


FEATURES = {
    'qb': CONTEXT + [_p(k, s) for s in ('attempts', 'completions', 'passing_yards', 'passing_tds',
                                         'passing_interceptions', 'sacks_suffered', 'carries', 'rushing_yards')
                     for k in KINDS]
    + ['team_s_attempts', 'team_s_carries', 'opp_passing_yards', 'opp_completions', 'opp_attempts', 'opp_passing_tds',
       'opp_passing_interceptions', 'opp_sacks_suffered', 'opp_rushing_yards', 'vac_tgt_share'],
    'rusher': CONTEXT + [_p(k, s) for s in ('carries', 'rushing_yards', 'targets', 'receptions', 'carry_share')
                         for k in KINDS]
    + ['pos', 'is_starting_qb', 'team_s_carries', 'team_s_attempts', 'opp_rushing_yards', 'opp_carries',
       'vac_carry_share', 'vac_tgt_share', 'qb_s_passing_yards', 'qb_changed'],
    'receiver': CONTEXT + [_p(k, s) for s in ('targets', 'receptions', 'receiving_yards', 'receiving_air_yards',
                                               'carries', 'rushing_yards', 'tgt_share', 'ay_share', 'carry_share')
                           for k in KINDS]
    + ['pos', 'team_s_attempts', 'team_s_carries', 'opp_rec_yds_grp', 'opp_rec_grp', 'opp_tgt_grp',
       'opp_passing_yards', 'opp_rushing_yards', 'vac_tgt_share', 'vac_carry_share', 'qb_s_passing_yards',
       'qb_s_attempts', 'qb_n_games', 'qb_changed'],
}


# ---------------------------------------------------------------- data

def load_inputs(last_season, refresh=False):
    """nflverse player games (offense), schedule and injury reports; `refresh` re-downloads the latest season."""
    weeks = nd.load_player_weeks(DATA_FIRST_SEASON, last_season, refresh, columns=PLAYER_COLUMNS)
    games = nd.load_games(refresh)
    injuries = nd.load_injuries(max(2009, DATA_FIRST_SEASON), last_season, refresh)
    return weeks, games, injuries


def outcome(frame, market):
    """The player's result for a market (the sum of the market's stat columns); NaN for games not played yet."""
    columns = list(MARKETS[market][2])
    values = frame[columns]
    return values.sum(axis=1, min_count=len(columns))


def with_openers(games, openers):
    """The schedule with each game's opening spread and total in place of the closing ones, where ESPN has them.

    For the early-week variant: when props open, the game lines are the opening ones; games without an opener keep
    the closing numbers (all seasons before 2024, which only the models' training uses).
    """
    games = games.merge(openers[['game_id', 'open_spread_line', 'open_total_line']], on='game_id', how='left')
    games['spread_line'] = games['open_spread_line'].fillna(games['spread_line'])
    games['total_line'] = games['open_total_line'].fillna(games['total_line'])
    return games.drop(columns=['open_spread_line', 'open_total_line'])


def _time_key(season, week):
    return season.astype('int64') * 100 + week.astype('int64')


def _prior_ewm(frame, by, columns, halflife):
    """Recency-weighted mean of each column over each group's earlier rows only (frame sorted in time within groups)."""
    keys = [frame[b] for b in by]
    shifted = frame.groupby(by, sort=False)[columns].shift(1)
    out = shifted.groupby(keys, sort=False).ewm(halflife=halflife, ignore_na=True).mean()
    return out.reset_index(level=list(range(len(by))), drop=True).reindex(frame.index)


def _prior_mean_within(frame, by, columns):
    """Mean of each column over each group's earlier rows, skipping blanks (NaN for a group's first row)."""
    values = frame[columns]
    filled, present = values.fillna(0.0), values.notna().astype(float)
    keys = [frame[b] for b in by]
    sums = filled.groupby(keys, sort=False).cumsum() - filled
    counts = present.groupby(keys, sort=False).cumsum() - present
    return sums / counts.replace(0, np.nan)


def _post_ewm(frame, by, columns, halflife):
    """Recency-weighted mean of each column over each group's rows up to and including the row."""
    keys = [frame[b] for b in by]
    out = frame[columns].groupby(keys, sort=False).ewm(halflife=halflife, ignore_na=True).mean()
    return out.reset_index(level=list(range(len(by))), drop=True).reindex(frame.index)


def team_games(games):
    """One row per team per game (played or scheduled): context the market knew, points, and the starting QB."""
    g = games.copy()
    for column in ('home_team', 'away_team'):
        g[column] = g[column].replace(TEAM_RENAMES)
    g['kickoff_key'] = g['gameday'].astype(str) + ' ' + g['gametime'].fillna('13:00').astype(str)
    g['indoor'] = g['roof'].isin(['dome', 'closed']).astype(float)
    neutral = (g['location'] == 'Neutral').to_numpy()
    base = g[['game_id', 'season', 'week', 'game_type', 'kickoff_key', 'total_line', 'indoor']]
    home = base.assign(team=g['home_team'], opp=g['away_team'], is_home=np.where(neutral, 0.5, 1.0),
                       team_spread=g['spread_line'], pts=g['home_score'], pts_allowed=g['away_score'],
                       starter=g['home_qb_id'])
    away = base.assign(team=g['away_team'], opp=g['home_team'], is_home=np.where(neutral, 0.5, 0.0),
                       team_spread=-g['spread_line'], pts=g['away_score'], pts_allowed=g['home_score'],
                       starter=g['away_qb_id'])
    rows = pd.concat([home, away], ignore_index=True)
    rows['implied_pts'] = (rows['total_line'] + rows['team_spread']) / 2
    rows = rows.sort_values(['team', 'kickoff_key']).reset_index(drop=True)
    rows['qb_changed'] = (rows['starter'] != rows.groupby('team')['starter'].shift(1)).astype(float)
    return rows


def _team_features(tg, offense):
    """Team rows with the team's recent volume and points and, as a defense, what it has allowed (vs the league)."""
    tg = tg.merge(offense, on=['game_id', 'team'], how='left')
    allowed = offense.rename(columns={'team': 'opp', **{c: f'allowed_{c}' for c in offense.columns
                                                          if c not in ('game_id', 'team')}})
    tg = tg.merge(allowed, on=['game_id', 'opp'], how='left')
    tg['allowed_pts'] = tg['pts_allowed']
    tg = tg.sort_values(['team', 'kickoff_key']).reset_index(drop=True)
    own = _prior_ewm(tg, ['team'], ['attempts', 'carries', 'pts'], SLOW_HALFLIFE)
    tg['team_s_attempts'], tg['team_s_carries'], tg['team_s_pts'] = own['attempts'], own['carries'], own['pts']
    allowed_columns = [f'allowed_{c}' for c in ALLOWED]
    prior = _prior_ewm(tg, ['team'], allowed_columns, SLOW_HALFLIFE)
    # each season is judged against the league average of the season before (known before it starts)
    played = tg[tg['pts'].notna()]
    # every season gets a row (one with no game final yet, e.g. before the opener, takes the season before's average)
    league = played.groupby('season')[allowed_columns].mean().reindex(sorted(tg['season'].unique()))
    previous = league.shift(1).fillna(league)
    norm = previous.reindex(tg['season']).to_numpy()
    for i, column in enumerate(allowed_columns):
        tg[f'def_{column[len("allowed_"):]}'] = prior[column].to_numpy() / norm[:, i]
    return tg


def _offense_totals(pw):
    """Per team-game offense totals, with receiving split by position group."""
    totals = pw.groupby(['game_id', 'team'])[list(STATS)].sum(min_count=1)
    rec = pw[pw['position'].isin(RECEIVING_GROUPS)].assign(group=lambda f: f['position'].map(RECEIVING_GROUPS))
    split = rec.pivot_table(index=['game_id', 'team'], columns='group', values=['targets', 'receptions', 'receiving_yards'],
                            aggfunc='sum', fill_value=0)
    split.columns = [f'{stat}_{group}' for stat, group in split.columns]
    return totals.join(split, how='left').reset_index()


def _vacated_shares(frame, injuries):
    """Per (season, week, team): the recent target and carry shares of players ruled Out or Doubtful that week."""
    columns = ['vac_tgt_share', 'vac_carry_share']
    if injuries is None or injuries.empty:
        return pd.DataFrame(columns=['season', 'week', 'team'] + columns)
    out = injuries[injuries['report_status'].isin(['Out', 'Doubtful'])].copy()
    out['team'] = out['team'].replace(TEAM_RENAMES)
    out = out.dropna(subset=['gsis_id', 'season', 'week']).drop_duplicates(['season', 'week', 'gsis_id'])
    out['tkey'] = _time_key(out['season'], out['week'])
    played = frame[frame['played']].copy()
    # each player's shares after his latest game (recency-weighted, including that game)
    post = _post_ewm(played, ['player_id'], ['tgt_share', 'carry_share'], SLOW_HALFLIFE)
    latest = pd.DataFrame({'player_id': played['player_id'], 'tkey': played['tkey'],
                           'tgt': post['tgt_share'], 'carry': post['carry_share']}).sort_values('tkey')
    out = pd.merge_asof(out.sort_values('tkey'), latest, on='tkey', left_by='gsis_id', right_by='player_id',
                        allow_exact_matches=False)
    vacated = out.groupby(['season', 'week', 'team'])[['tgt', 'carry']].sum().reset_index()
    return vacated.rename(columns={'tgt': 'vac_tgt_share', 'carry': 'vac_carry_share'})


def build_frame(player_weeks, games, injuries=None, upcoming=None, early=False):
    """Every offensive player-game with its pre-game features, plus `upcoming` games to forecast.

    `upcoming` has player_id, player_display_name, position, season, week, game_id, team, opponent_team for games not
    played yet (their stats stay NaN).  Rows from played games have `played` True.  `early` builds the early-week
    variant: no injury-report features (the week's report is not out when props open); pass `games` through
    with_openers for the opening game lines.
    """
    pw = player_weeks[player_weeks['position'].isin(OFFENSE_POSITIONS)].copy()
    pw['team'] = pw['team'].replace(TEAM_RENAMES)
    pw['opponent_team'] = pw['opponent_team'].replace(TEAM_RENAMES)
    for stat in STATS:
        pw[stat] = pw[stat].fillna(0.0)
    pw['played'] = True
    if upcoming is not None and len(upcoming):
        extra = upcoming.copy()
        extra['played'] = False
        extra['season_type'] = extra.get('season_type', 'REG')
        pw = pd.concat([pw, extra[[c for c in extra.columns if c in pw.columns]]], ignore_index=True)

    tg = _team_features(team_games(games), _offense_totals(pw[pw['played']]))
    context = tg[['game_id', 'team', 'kickoff_key', 'is_home', 'indoor', 'team_spread', 'total_line', 'implied_pts',
                  'team_s_attempts', 'team_s_carries', 'team_s_pts', 'starter', 'qb_changed']]
    frame = pw.merge(context, on=['game_id', 'team'], how='left')
    defense = tg[['game_id', 'team'] + [c for c in tg.columns if c.startswith('def_')]].rename(columns={'team': 'opponent_team'})
    frame = frame.merge(defense, on=['game_id', 'opponent_team'], how='left')
    frame = frame.sort_values(['player_id', 'kickoff_key']).reset_index(drop=True)
    frame['tkey'] = _time_key(frame['season'], frame['week'])

    team_totals = frame[frame['played']].groupby(['game_id', 'team'])[['targets', 'carries', 'receiving_air_yards']].transform('sum')
    for share, (stat, total) in SHARES.items():
        frame[share] = np.nan
        denominator = team_totals[total].where(team_totals[total] > 0)
        frame.loc[frame['played'], share] = frame.loc[frame['played'], stat] / denominator
    tracked = list(STATS) + list(SHARES)
    for kind, halflife in (('fast', FAST_HALFLIFE), ('slow', SLOW_HALFLIFE)):
        prior = _prior_ewm(frame, ['player_id'], tracked, halflife)
        for column in tracked:
            frame[_p(kind, column)] = prior[column].astype('float32')
    season_prior = _prior_mean_within(frame, ['player_id', 'season'], tracked)
    for column in tracked:
        frame[_p('season', column)] = season_prior[column].astype('float32')
    frame['n_games'] = frame.groupby('player_id').cumcount().astype(float)
    frame['n_season'] = frame.groupby(['player_id', 'season']).cumcount().astype(float)

    frame['pos'] = frame['position'].map(POSITION_CODES).astype(float)
    frame['is_starting_qb'] = (frame['player_id'] == frame['starter']).astype(float)
    frame['implied_vs_usual'] = frame['implied_pts'] / frame['team_s_pts']
    frame['opp_pts'] = frame['def_pts']
    for stat in ('attempts', 'completions', 'passing_yards', 'passing_tds', 'passing_interceptions', 'sacks_suffered',
                 'carries', 'rushing_yards'):
        frame[f'opp_{stat}'] = frame[f'def_{stat}']
    group = frame['position'].map(RECEIVING_GROUPS)
    for name, stat in (('opp_rec_yds_grp', 'receiving_yards'), ('opp_rec_grp', 'receptions'), ('opp_tgt_grp', 'targets')):
        frame[name] = np.nan
        for g in ('rb', 'wr', 'te'):
            mask = group == g
            frame.loc[mask, name] = frame.loc[mask, f'def_{stat}_{g}']

    qb = frame[frame['is_starting_qb'] == 1][['game_id', 'team', _p('slow', 'passing_yards'), _p('slow', 'attempts'), 'n_games']]
    qb = qb.drop_duplicates(['game_id', 'team']).rename(columns={
        _p('slow', 'passing_yards'): 'qb_s_passing_yards', _p('slow', 'attempts'): 'qb_s_attempts', 'n_games': 'qb_n_games'})
    frame = frame.merge(qb, on=['game_id', 'team'], how='left')

    frame = frame.merge(_vacated_shares(frame, injuries), on=['season', 'week', 'team'], how='left')
    for column in ('vac_tgt_share', 'vac_carry_share'):
        frame[column] = frame[column].fillna(0.0)
    frame['own_questionable'] = 0.0
    if injuries is not None and not injuries.empty:
        questionable = injuries[injuries['report_status'] == 'Questionable'][['season', 'week', 'gsis_id']].drop_duplicates()
        flagged = frame[['season', 'week', 'player_id']].merge(
            questionable.rename(columns={'gsis_id': 'player_id'}).assign(q=1.0), on=['season', 'week', 'player_id'], how='left')
        frame['own_questionable'] = flagged['q'].fillna(0.0).to_numpy()
    if early:
        frame[['vac_tgt_share', 'vac_carry_share', 'own_questionable']] = 0.0
    return frame


# ---------------------------------------------------------------- models

def population(frame, group):
    """Rows a market's model is fitted on and forecasts: the players books post such props for."""
    if group == 'qb':
        return frame['is_starting_qb'] == 1
    if group == 'rusher':
        return frame['position'].isin(['RB', 'FB']) | (frame['is_starting_qb'] == 1) | (frame[_p('slow', 'carries')] >= 2)
    return frame['position'].isin(['RB', 'FB', 'WR', 'TE'])


def fit_models(frame, season, markets=None):
    """One fitted model per market, trained on the TRAIN_SEASONS seasons before `season`."""
    train = frame[frame['played'] & (frame['season'] >= season - TRAIN_SEASONS) & (frame['season'] < season)]
    models = {}
    for market in markets or MARKETS:
        group = MARKET_GROUPS[market]
        rows = train[population(train, group)]
        y = np.clip(outcome(rows, market).to_numpy(), 0, None)
        models[market] = HistGradientBoostingRegressor(**TREE_SETTINGS).fit(rows[FEATURES[group]].to_numpy(np.float32), y)
    return models


def forecast(models, rows, market):
    """Forecast mean result for `rows` (which must be in the market's population)."""
    if rows.empty:
        return np.array([])
    return models[market].predict(rows[FEATURES[MARKET_GROUPS[market]]].to_numpy(np.float32))


def season_forecasts(frame, season, models=None, played_only=True):
    """Out-of-sample forecasts of every player-game of `season` in each market's population, with the results."""
    models = models or fit_models(frame, season)
    rows = frame[(frame['season'] == season) & (frame['played'] if played_only else True)]
    parts = []
    for market in MARKETS:
        sub = rows[population(rows, MARKET_GROUPS[market])]
        parts.append(pd.DataFrame({'season': sub['season'].to_numpy(), 'week': sub['week'].to_numpy(),
                                   'game_id': sub['game_id'].to_numpy(), 'player_id': sub['player_id'].to_numpy(),
                                   'market': market, 'forecast': forecast(models, sub, market),
                                   'result': outcome(sub, market).to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def forecast_history(frame, seasons, refresh=False, variant=''):
    """Out-of-sample forecasts for complete past seasons, cached in data_cache (rebuilt when MODEL_VERSION changes).

    `variant` names a separate cache for a model variant built from a different frame (e.g. 'early').
    """
    name = f'prop_forecasts_{variant}_v{MODEL_VERSION}.csv' if variant else f'prop_forecasts_v{MODEL_VERSION}.csv'
    path = os.path.join(CACHE_DIR, name)
    cached = pd.read_csv(path) if os.path.exists(path) and not refresh else pd.DataFrame(columns=['season'])
    missing = [s for s in seasons if s not in set(cached['season'])]
    for season in missing:
        logging.info(f"Forecasting {season} out of sample{' (' + variant + ' variant)' if variant else ''} "
                     f"(models fitted on {season - TRAIN_SEASONS}-{season - 1})")
        cached = pd.concat([cached, season_forecasts(frame, season)], ignore_index=True)
        cached.to_csv(path, index=False)
    return cached[cached['season'].isin(seasons)].reset_index(drop=True)


class OutcomeDistribution:
    """Chances of a result over or under a line, from past games whose (out-of-sample) forecast was closest.

    The past results are scaled by forecast / past forecast for yardage markets, so a neighbour forecast at 60 yards
    that ended at 90 counts as 97.5 for a forecast of 65; whole-number markets use the results as they were.  The
    window holds up to `neighbors` games, centred on the forecast.
    """

    def __init__(self, forecasts, results, neighbors, scale):
        keep = ~(np.isnan(forecasts) | np.isnan(results))
        order = np.argsort(np.asarray(forecasts)[keep])
        self.mu = np.maximum(np.asarray(forecasts)[keep][order], 0.05)
        self.y = np.asarray(results)[keep][order]
        self.log_mu = np.log(self.mu)
        self.k = min(neighbors, len(self.mu))
        self.scale = scale

    def _sample(self, mu):
        # the window stays centred on the forecast, shrinking near the ends of the range rather than sliding inward,
        # which would lean on forecasts all lower (or all higher) than this one
        mu = max(float(mu), 0.05)
        n = len(self.mu)
        i = int(np.searchsorted(self.log_mu, np.log(mu)))
        half = max(min(self.k // 2, i, n - i), MIN_HALF_WINDOW)
        window = slice(max(0, i - half), min(n, i + half))
        return self.y[window] * (mu / self.mu[window]) if self.scale else self.y[window]

    def chances(self, mu, line):
        """(chance over, chance under) of `line`; the rest is a push."""
        sample = self._sample(mu)
        return float(np.mean(sample > line)), float(np.mean(sample < line))

    def median(self, mu):
        return float(np.median(self._sample(mu)))


def distributions(history, season):
    """OutcomeDistribution per market from the DIST_SEASONS seasons of forecasts before `season`."""
    pool = history[(history['season'] < season) & (history['season'] >= season - DIST_SEASONS)]
    out = {}
    for market in MARKETS:
        rows = pool[pool['market'] == market]
        out[market] = OutcomeDistribution(rows['forecast'].to_numpy(float), rows['result'].to_numpy(float),
                                          NEIGHBORS[MARKET_GROUPS[market]], scale=market not in COUNT_MARKETS)
    return out


# ---------------------------------------------------------------- plain-words reason

def reason(row, market):
    """A one-line summary of what drives a forecast, from the player's own numbers and the game's."""
    def num(column, digits=1):
        value = row.get(column)
        return None if value is None or pd.isna(value) else round(float(value), digits)

    parts = []
    group = MARKET_GROUPS[market]
    if group == 'qb':
        att, ypg = num(_p('slow', 'attempts')), num(_p('slow', 'passing_yards'), 0)
        if att is not None:
            parts.append(f"{att:g} attempts and {ypg:g} passing yds a game lately")
        opp = num('opp_passing_yards', 2)
        if opp is not None:
            parts.append(f"opponent allows {opp:.0%} of the league's passing yds")
    elif group == 'rusher' and market != 'rush_rec_yds':
        car, share, ypc = num(_p('slow', 'carries')), num(_p('slow', 'carry_share'), 2), None
        if car:
            ypc = round(float(row.get(_p('slow', 'rushing_yards'), 0)) / car, 1)
            parts.append(f"{car:g} carries a game lately" + (f" ({share:.0%} of the team's)" if share is not None else '')
                         + f", {ypc:g} yds a carry")
        opp = num('opp_rushing_yards', 2)
        if opp is not None:
            parts.append(f"opponent allows {opp:.0%} of the league's rushing yds")
    else:
        tgt, share = num(_p('slow', 'targets')), num(_p('slow', 'tgt_share'), 2)
        if tgt:
            ypt = round(float(row.get(_p('slow', 'receiving_yards'), 0)) / tgt, 1)
            parts.append(f"{tgt:g} targets a game lately" + (f" ({share:.0%} of the team's)" if share is not None else '')
                         + f", {ypt:g} yds a target")
        opp = num('opp_rec_yds_grp', 2)
        if opp is not None:
            parts.append(f"opponent allows {opp:.0%} of the league's receiving yds to {row.get('position')}s")
    implied, usual = num('implied_pts'), num('team_s_pts')
    if implied is not None and usual is not None:
        parts.append(f"team implied {implied:g} pts (usually scores {usual:g})")
    vac = num('vac_tgt_share', 2) if group != 'rusher' else num('vac_carry_share', 2)
    if vac and vac >= 0.05:
        parts.append(f"{vac:.0%} of the team's {'targets' if group != 'rusher' else 'carries'} out injured")
    if group != 'qb' and row.get('qb_changed') == 1 and num('qb_n_games') is not None and num('qb_n_games') < 20:
        parts.append("new starting QB")
    if row.get('own_questionable') == 1:
        parts.append("listed Questionable")
    return '; '.join(parts)
