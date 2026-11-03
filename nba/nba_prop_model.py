"""NBA player prop projections: a pre-game forecast and an outcome distribution for each player and prop market.

From ESPN's box scores (nba_data) and the research table (nba_lab), everything known before tip-off.  Each market of
nba_prop_lines.MARKETS has its own model, a gradient-boosted tree ensemble with a Poisson loss, forecasting the player's
result from
  - his recent and season-long production and opportunity: minutes, points, rebounds (offensive, defensive),
    assists, threes made and tried, shots, free throws, steals, blocks, turnovers, fouls, how often he starts (fast and
    slow recency-weighted averages over his games played, and his season so far), per-minute rates, games played, and
    how many of his team's games he has just missed (a player back from an absence often plays fewer minutes),
  - the game: his team's implied points from the spread and total, the spread itself (blowouts shorten starters'
    minutes), home, rest and back-to-backs,
  - his team's and the opponent's ratings (pace, efficiency, shooting, rebounding, assists, turnovers, blocks, fouls
    allowed),
  - teammates out: the minutes and shots of the team's rotation players who miss the game (vacated minutes and shots go
    to the others), and of the opponent's.
Two variants: 'full' knows who plays tonight and the closing game lines (as at tip-off), 'early' knows only the absences
that already held at the team's last game and the opening game lines when ESPN has them (as in the morning).

The chance of a result over or under a line comes from what happened in past games with the nearest out-of-sample
forecasts (OutcomeDistribution).  A season is forecast by models fitted on the TRAIN_SEASONS seasons before it.
"""

import os

import numpy as np
import pandas as pd

try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401  (needed before scikit-learn 1.0)
except ImportError:
    pass
from sklearn.ensemble import HistGradientBoostingRegressor

import nba_data
import nba_lab
from nba_common import CACHE_DIR, REGULAR_SEASON
from nba_prop_lines import MARKETS

MODEL_VERSION = 1
DATA_FIRST_SEASON = 2012
TRAIN_SEASONS = 6
FIRST_FORECAST_SEASON = 2020
DIST_SEASONS = 4
NEIGHBORS = 4000
MIN_HALF_WINDOW = 200
FAST_HALFLIFE, SLOW_HALFLIFE = 3.0, 12.0          # in games played
ROTATION_MINUTES, TEAM_DECAY = 12.0, 0.9          # rotation: averaging 12+ minutes over recent team games
MIN_PRIOR_GAMES, MIN_SLOW_MINUTES = 3, 8.0        # the players modelled: some history and a real role
TREE_SETTINGS = dict(loss='poisson', learning_rate=0.05, max_iter=250, max_leaf_nodes=31, min_samples_leaf=200,
                     l2_regularization=1.0, early_stopping=False, random_state=0)
STATS = ('minutes', 'points', 'rebounds', 'offensive_rebounds', 'defensive_rebounds', 'assists',
         'three_point_field_goals_made', 'three_point_field_goals_attempted', 'field_goals_attempted',
         'free_throws_attempted', 'steals', 'blocks', 'turnovers', 'fouls', 'starter')
SHORT = {'minutes': 'min', 'points': 'pts', 'rebounds': 'reb', 'offensive_rebounds': 'oreb',
         'defensive_rebounds': 'dreb', 'assists': 'ast', 'three_point_field_goals_made': 'fg3m',
         'three_point_field_goals_attempted': 'fg3a', 'field_goals_attempted': 'fga', 'free_throws_attempted': 'fta',
         'steals': 'stl', 'blocks': 'blk', 'turnovers': 'tov', 'fouls': 'pf', 'starter': 'start'}
COUNT_MARKETS = ('reb', 'ast', 'fg3m', 'ra', 'stl', 'blk', 'tov')     # small whole numbers: results not rescaled
# the markets forecast: steals, blocks and turnovers are left out (lines of 0.5-2.5 that books price by moving the
# odds, not the line, so a win rate at the line says little; nba_research_props.py)
FORECAST_MARKETS = ('pts', 'reb', 'ast', 'fg3m', 'pra', 'pr', 'pa', 'ra')
TEAM_RATINGS = ('pace_own', 'ortg_own', 'three_rate_own', 'ast_rate_own', 'orb_own', 'tov_own')
OPP_RATINGS = ('pace_own', 'ortg_allowed', 'efg_allowed', 'three_rate_allowed', 'three_pct_allowed', 'ast_rate_allowed',
               'orb_allowed', 'tov_allowed', 'ftr_allowed', 'blk_allowed', 'fouls_own', 'margin_own')


def _kinds():
    return [f'{kind}_{SHORT[s]}' for kind in ('fast', 'slow', 'season') for s in STATS]


RATES = ['rate_pts', 'rate_reb', 'rate_ast', 'rate_fg3m', 'rate_fg3a', 'rate_fga', 'rate_fta', 'rate_stl', 'rate_blk',
         'rate_tov', 'rate_oreb', 'rate_dreb']
CONTEXT = ['n_games', 'n_season', 'missed', 'home', 'rest', 'b2b', 'team_spread', 'total', 'implied_pts',
           'vac_min', 'vac_fga', 'opp_vac_min'] + [f'team_{r}' for r in TEAM_RATINGS] + [f'opp_{r}' for r in OPP_RATINGS]
FEATURES = _kinds() + RATES + CONTEXT


# ---------------------------------------------------------------- the frame

def outcome(frame, market):
    """The result of `market` for each player-game (the sum of its box-score columns)."""
    return sum(frame[c] for c in MARKETS[market][2])


def team_absences(schedule, box, out_ids=None):
    """Per (game_id, team_id): the average minutes and shots of the team's rotation players who miss the game (vac_min,
    vac_fga) and of those already missing the team's previous game too (vac_min_known, vac_fga_known), known before
    the day's injury news.  `box` holds the player box rows; for a game with no rows yet, the players in `out_ids`
    (ESPN ids, from the injury list) are the ones who miss it, and without `out_ids` its tonight's absences are blank."""
    played = box[box['minutes'] > 0]
    by_game = {key: dict(zip(g['athlete_id'], zip(g['minutes'], g['field_goals_attempted'])))
               for key, g in played.groupby(['game_id', 'team_id'])}
    minutes, shots, last_out, this_season = {}, {}, {}, {}
    out_ids = None if out_ids is None else set(out_ids)
    rows = []
    for g in schedule.sort_values(['start', 'game_id']).itertuples(index=False):
        for team in (g.home_id, g.away_id):
            roster = minutes.setdefault(team, {})
            fga = shots.setdefault(team, {})
            average = {p: m * (1 - TEAM_DECAY) for p, m in roster.items()}
            rotation = {p for p, m in average.items() if m >= ROTATION_MINUTES}
            tonight = by_game.get((g.game_id, team))
            if tonight is not None:
                out = rotation - set(tonight)
            elif out_ids is not None:
                # as the backtests count who actually played: out are the players on the injury list and those who
                # have not played for the team this season (gone since)
                season_players = this_season.get(team, (None, set()))
                current = season_players[1] if season_players[0] == g.season else set()
                out = {p for p in rotation if p in out_ids or p not in current}
            else:
                out = None
            known = rotation & last_out.get(team, set())
            per_fga = {p: fga.get(p, 0.0) * (1 - TEAM_DECAY) for p in rotation}
            rows.append({'game_id': g.game_id, 'team_id': team,
                         'vac_min': sum(average[p] for p in out) if out is not None else np.nan,
                         'vac_fga': sum(per_fga[p] for p in out) if out is not None else np.nan,
                         'vac_min_known': sum(average[p] for p in known),
                         'vac_fga_known': sum(per_fga[p] for p in known)})
            if tonight is None:
                continue
            last_out[team] = rotation - set(tonight)
            season_players = this_season.get(team)
            if season_players is None or season_players[0] != g.season:
                season_players = this_season[team] = (g.season, set())
            season_players[1].update(tonight)
            for p in list(roster):
                roster[p] *= TEAM_DECAY
                fga[p] = fga.get(p, 0.0) * TEAM_DECAY
                if roster[p] < 0.5 and p not in tonight:
                    roster.pop(p)
                    fga.pop(p, None)
            for p, (m, a) in tonight.items():
                roster[p] = roster.get(p, 0.0) + m
                fga[p] = fga.get(p, 0.0) + a
    return pd.DataFrame(rows)


def team_context(table):
    """Per (game_id, team_id): home, rest, back-to-back, the team's spread, the total and implied points (closing and
    opening lines), the team's and the opponent's ratings."""
    rows = []
    for side, other, team_col in (('h', 'a', 'home_id'), ('a', 'h', 'away_id')):
        sign = 1 if side == 'h' else -1
        part = pd.DataFrame({'game_id': table['game_id'], 'team_id': table[team_col],
                             'season': table['season'], 'start': table['start'],
                             'home': (table['home'] == 1) & (side == 'h'), 'rest': table[f'{side}_rest'],
                             'b2b': table[f'{side}_b2b']})
        for which, spread, total in (('close', 'close_home_spread', 'close_total'),
                                     ('open', 'open_home_spread', 'open_total')):
            part[f'team_spread_{which}'] = sign * table[spread]
            part[f'total_{which}'] = table[total]
            part[f'implied_pts_{which}'] = table[total] / 2 - sign * table[spread] / 2
        for rating in TEAM_RATINGS:
            part[f'team_{rating}'] = table[f'{side}_r_{rating}']
        for rating in OPP_RATINGS:
            part[f'opp_{rating}'] = table[f'{other}_r_{rating}']
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _prior_ewm(frame, columns, halflife):
    shifted = frame.groupby('athlete_id')[columns].shift()
    return shifted.groupby(frame['athlete_id']).transform(lambda s: s.ewm(halflife=halflife, ignore_na=True).mean())


def build_frame(table, box, upcoming=None, out_ids=None):
    """One row per player-game played (and per `upcoming` player-game, rows with athlete_id, game_id, team_id) with the
    FEATURES of both variants ('full' and the '_known'/'_open' inputs of 'early') and the results.  `out_ids` (ESPN ids
    of players out, from the injury list) gives tonight's absences for games not played yet."""
    box = box[(box['season_type'] == REGULAR_SEASON) & box['team_id'].isin(nba_lab.FRANCHISES)].copy()
    box['athlete_id'] = pd.to_numeric(box['athlete_id'], errors='coerce')
    box = box[box['athlete_id'].notna()]
    box['athlete_id'] = box['athlete_id'].astype('int64')
    box['minutes'] = pd.to_numeric(box['minutes'], errors='coerce').fillna(0)
    box['starter'] = box['starter'].astype(str).str.lower().eq('true').astype(float)
    schedule = table[['game_id', 'season', 'start', 'home_id', 'away_id']]
    absences = team_absences(schedule, box, out_ids)
    frame = box[box['minutes'] > 0]
    if upcoming is not None and len(upcoming):
        frame = pd.concat([frame, upcoming.assign(minutes=np.nan)], ignore_index=True)
    frame = frame.drop(columns=['season'], errors='ignore').merge(schedule[['game_id', 'season', 'start']], on='game_id')
    frame = frame.sort_values(['athlete_id', 'start']).reset_index(drop=True)
    stats = list(STATS)
    for column in stats:
        frame[column] = pd.to_numeric(frame[column], errors='coerce').astype(np.float32)
    new = {}
    for kind, halflife in (('fast', FAST_HALFLIFE), ('slow', SLOW_HALFLIFE)):
        ewm = _prior_ewm(frame, stats, halflife)
        for column in stats:
            new[f'{kind}_{SHORT[column]}'] = ewm[column].astype(np.float32)
    shifted = frame.groupby('athlete_id')[stats].shift()
    season_key = [frame['athlete_id'], frame['season']]
    sums = shifted.groupby(season_key).cumsum()
    counts = shifted[stats[0]].notna().groupby(season_key).cumsum()
    for column in stats:
        new[f'season_{SHORT[column]}'] = (sums[column] / counts.replace(0, np.nan)).astype(np.float32)
    new['n_games'] = frame.groupby('athlete_id').cumcount()
    new['n_season'] = frame.groupby(season_key).cumcount()
    minutes = new['slow_min'].where(new['slow_min'] > 0)
    for rate in ('pts', 'reb', 'ast', 'fg3m', 'fg3a', 'fga', 'fta', 'stl', 'blk', 'tov', 'oreb', 'dreb'):
        new[f'rate_{rate}'] = (36 * new[f'slow_{rate}'] / minutes).astype(np.float32)
    frame = pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)
    # games of his team since his last game (a player back from an absence)
    order = pd.concat([schedule[['game_id', 'start', 'season']].assign(team_id=schedule['home_id']),
                       schedule[['game_id', 'start', 'season']].assign(team_id=schedule['away_id'])])
    order = order.sort_values('start')
    order['team_game'] = order.groupby(['team_id', 'season']).cumcount()
    frame = frame.merge(order[['game_id', 'team_id', 'team_game']], on=['game_id', 'team_id'], how='left')
    previous = frame.groupby('athlete_id')[['team_game', 'team_id', 'season']].shift()
    same = previous['team_id'].eq(frame['team_id']) & previous['season'].eq(frame['season'])
    frame['missed'] = (frame['team_game'] - previous['team_game'] - 1).where(same).astype(np.float32)
    frame = frame.merge(team_context(table), on=['game_id', 'team_id', 'season', 'start'], how='left')
    frame = frame.merge(absences, on=['game_id', 'team_id'], how='left')
    opp = absences.rename(columns={'team_id': 'opp_id', 'vac_min': 'opp_vac_min', 'vac_fga': 'opp_vac_fga',
                                   'vac_min_known': 'opp_vac_min_known', 'vac_fga_known': 'opp_vac_fga_known'})
    sides = table.set_index('game_id')
    home, away = frame['game_id'].map(sides['home_id']), frame['game_id'].map(sides['away_id'])
    frame['opp_id'] = np.where(frame['team_id'] == home, away, home)
    frame = frame.merge(opp, on=['game_id', 'opp_id'], how='left')
    results = {f'y_{market}': outcome(frame, market).astype(np.float32) for market in MARKETS}
    results['played'] = frame['minutes'] > 0
    return pd.concat([frame, pd.DataFrame(results, index=frame.index)], axis=1)


def variant_matrix(rows, variant):
    """The FEATURES matrix of `rows` for 'full' (tonight's absences, closing lines) or 'early' (absences known at the
    team's last game, opening lines when there are any)."""
    x = rows.copy()
    if variant == 'full':
        x['team_spread'], x['total'], x['implied_pts'] = x['team_spread_close'], x['total_close'], x['implied_pts_close']
    else:
        for name in ('team_spread', 'total', 'implied_pts'):
            x[name] = x[f'{name}_open'].where(x[f'{name}_open'].notna(), x[f'{name}_close'])
        x['vac_min'], x['vac_fga'], x['opp_vac_min'] = x['vac_min_known'], x['vac_fga_known'], x['opp_vac_min_known']
    return x[FEATURES].to_numpy(np.float32)


def population(frame):
    """The player-games the models learn from and forecast: some history and a real role, as books post props for."""
    return (frame['n_games'] >= MIN_PRIOR_GAMES) & (frame['slow_min'] >= MIN_SLOW_MINUTES)


def fit_models(frame, season, variant, markets=FORECAST_MARKETS):
    """One model per market, fitted on the TRAIN_SEASONS seasons before `season`."""
    train = frame[frame['played'] & population(frame) & (frame['season'] >= season - TRAIN_SEASONS)
                  & (frame['season'] < season)]
    x = variant_matrix(train, variant)
    return {market: HistGradientBoostingRegressor(**TREE_SETTINGS).fit(x, np.clip(train[f'y_{market}'], 0, None))
            for market in markets}


def season_forecasts(frame, season, variant, models=None):
    """Out-of-sample forecasts of every played player-game of `season` in the population, with the results."""
    models = models or fit_models(frame, season, variant)
    rows = frame[(frame['season'] == season) & frame['played'] & population(frame)]
    x = variant_matrix(rows, variant)
    parts = [pd.DataFrame({'season': season, 'game_id': rows['game_id'].to_numpy(),
                           'athlete_id': rows['athlete_id'].to_numpy(), 'market': market,
                           'forecast': model.predict(x), 'result': rows[f'y_{market}'].to_numpy()})
             for market, model in models.items()]
    return pd.concat(parts, ignore_index=True)


def forecast_history(frame, seasons, variant, refresh=False):
    """Out-of-sample forecasts for `seasons`, cached in data_cache per variant (rebuilt when MODEL_VERSION changes;
    the current season is always forecast again, as its games keep coming)."""
    from nba_common import current_season
    path = os.path.join(CACHE_DIR, f'nba_prop_forecasts_{variant}_v{MODEL_VERSION}.csv')
    cached = pd.read_csv(path) if os.path.exists(path) and not refresh else pd.DataFrame(columns=['season'])
    cached = cached[cached['season'] != current_season()]
    for season in [s for s in seasons if s not in set(cached['season'])]:
        print(f'Forecasting {season} ({variant}) with models fitted on {season - TRAIN_SEASONS}-{season - 1}',
              flush=True)
        cached = pd.concat([cached, season_forecasts(frame, season, variant)], ignore_index=True)
        cached[cached['season'] != current_season()].to_csv(path, index=False)
    return cached[cached['season'].isin(seasons)].reset_index(drop=True)


def load_frame(last_season, refresh=False, upcoming=None, table=None, out_ids=None):
    """The research table and the player frame from DATA_FIRST_SEASON to `last_season`."""
    table = table if table is not None else nba_lab.default_table(refresh=refresh)
    box = nba_data.load_player_box(range(DATA_FIRST_SEASON, last_season + 1), refresh)
    return build_frame(table[table['season'] <= last_season], box, upcoming, out_ids)


def season_models(frame, season, variant):
    """The models that forecast `season` (fitted on the TRAIN_SEASONS seasons before it), fitted once and kept in
    data_cache, so a later run does not refit them."""
    import pickle
    path = os.path.join(CACHE_DIR, f'nba_prop_models_{season}_{variant}_v{MODEL_VERSION}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    models = fit_models(frame, season, variant)
    with open(path + '.saving', 'wb') as f:
        pickle.dump(models, f)
    os.replace(path + '.saving', path)
    return models


class OutcomeDistribution:
    """Chances of a result over or under a line, from past games whose (out-of-sample) forecast was closest.

    Results of the larger markets are scaled by forecast / past forecast, so a neighbour forecast at 20 points that
    ended at 26 counts as 27.3 for a forecast of 21; small whole-number markets (COUNT_MARKETS) use the results as they
    were.  The window holds up to `neighbors` games, centred on the forecast."""

    def __init__(self, forecasts, results, neighbors, scale):
        keep = ~(np.isnan(forecasts) | np.isnan(results))
        order = np.argsort(np.asarray(forecasts)[keep])
        self.mu = np.maximum(np.asarray(forecasts)[keep][order], 0.05)
        self.y = np.asarray(results)[keep][order]
        self.log_mu = np.log(self.mu)
        self.k = min(neighbors, len(self.mu))
        self.scale = scale

    def _sample(self, mu):
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
    """OutcomeDistribution per forecast market from the DIST_SEASONS seasons of forecasts before `season`."""
    pool = history[(history['season'] < season) & (history['season'] >= season - DIST_SEASONS)]
    return {market: OutcomeDistribution(rows['forecast'].to_numpy(float), rows['result'].to_numpy(float), NEIGHBORS,
                                        scale=market not in COUNT_MARKETS)
            for market, rows in ((m, pool[pool['market'] == m]) for m in FORECAST_MARKETS)}
