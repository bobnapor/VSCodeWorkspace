"""Research framework for beating the NFL closing spread.

Builds one game table (1999 on) with pre-game team ratings on many stats, market prices, weather and
context, plus walk-forward evaluation tools.  Each season is predicted by models trained only on earlier
seasons, and every model is scored the same way:

* MAE of the projected margin (the closing spread is the benchmark),
* informativeness: does the projection explain anything about the result *beyond* the spread?
  (the coefficient on projection minus spread when regressing the cover margin on it),
* the against-the-spread record, split into a discovery period and a holdout period.
"""

import math

import numpy as np
import pandas as pd

import nfl_model as nm
from nfl_pbp_features import load_pbp_team_games
from nflverse_data import load_games, load_team_weeks

FIRST_SEASON = 1999
FIRST_TEST = 2006
DISCOVERY_END = 2017   # tune and select on test seasons up to here; confirm once on the later seasons
LAST_SEASON = 2026


# ---------------------------------------------------------------- ratings

def rate_teams(rows, pairs, decay=0.90, carry=0.6, prior_games=3.0, adjust=0.6):
    """Pre-game rating for every (own column, allowed column) pair, in date order, opponent-adjusted.

    `rows` has one row per team per game.  A team's offensive rating for a stat is a recency-weighted average
    of its earlier games, corrected for how generous the opponents' defenses usually are (and vice versa).
    """
    names = [c for pair in pairs for c in pair]
    counterpart = {}
    for own, allowed in pairs:
        counterpart[own], counterpart[allowed] = allowed, own
    league = {name: rows[name].mean() for name in names}
    n = len(rows)
    out = {name: np.empty(n) for name in names}
    games_played = np.zeros(n)
    values = {name: rows[name].to_numpy(dtype=float) for name in names}
    teams = rows['team'].to_numpy()
    seasons = rows['season'].to_numpy()
    played = rows['played'].to_numpy()

    state = {}
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
                out[name][i] = (s[0][name] + prior_games * league[name]) / (s[1] + prior_games)
            games_played[i] = s[3]
        i, j = pair
        if not (played[i] and played[j]):
            continue
        for me, other in ((i, j), (j, i)):
            s = state[teams[me]]
            for name in names:
                value = values[name][me]
                if np.isnan(value):
                    value = league[name]
                else:
                    value -= adjust * (out[counterpart[name]][other] - league[counterpart[name]])
                s[0][name] = s[0][name] * decay + value
            s[1] = s[1] * decay + 1.0
            s[3] += 1

    rated = rows[['game_id', 'team']].copy()
    for name in names:
        rated['r_' + name] = out[name]
    rated['games_played'] = games_played
    return rated


# ---------------------------------------------------------------- market and context features

def american_to_prob(odds):
    odds = pd.to_numeric(odds, errors='coerce')
    return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def add_market_features(table):
    t = table.copy()
    ph, pa = american_to_prob(t['home_moneyline']), american_to_prob(t['away_moneyline'])
    t['ml_home_prob'] = ph / (ph + pa)
    t['ml_logit'] = np.log(t['ml_home_prob'] / (1 - t['ml_home_prob']))
    sh, sa = american_to_prob(t['home_spread_odds']), american_to_prob(t['away_spread_odds'])
    t['spread_home_prob'] = sh / (sh + sa)           # > 0.5: the price leans toward the home side covering
    t['juice_lean'] = t['spread_home_prob'] - 0.5
    t['total_line'] = t['total_line']
    t['home_implied_total'] = (t['total_line'] + t['spread_line']) / 2
    t['away_implied_total'] = (t['total_line'] - t['spread_line']) / 2
    absolute = t['spread_line'].abs()
    t['on_key_number'] = absolute.isin([3.0, 7.0, 6.0, 10.0, 14.0]).astype(int)
    t['half_point'] = ((absolute % 1) == 0.5).astype(int)
    t['dist_to_3'] = (absolute - 3).abs()
    t['dist_to_7'] = (absolute - 7).abs()
    t['abs_spread'] = absolute
    t['resid'] = t['result'] - t['spread_line']      # > 0: the home team covered
    return t


def add_context_features(table):
    t = table.copy()
    t['wind'] = pd.to_numeric(t['wind'], errors='coerce')
    t['temp'] = pd.to_numeric(t['temp'], errors='coerce')
    t['dome'] = t['roof'].isin(['dome', 'closed']).astype(int)
    t['cold'] = (t['temp'] < 40).astype(int).where(t['dome'] == 0, 0)
    t['wind_high'] = (t['wind'] >= 15).astype(int).where(t['dome'] == 0, 0)
    t['turf'] = t['surface'].fillna('').str.contains('turf|astro|field', case=False).astype(int)
    t['div_game'] = t['div_game'].fillna(0)
    t['primetime'] = pd.to_datetime(t['gametime'], format='%H:%M', errors='coerce').dt.hour.ge(19).astype(int)
    t['thursday'] = (t['weekday'] == 'Thursday').astype(int)
    t['monday'] = (t['weekday'] == 'Monday').astype(int)
    t['week_early'] = (t['week'] <= 4).astype(int)
    t['short_rest_home'] = (t['home_rest'] < 6).astype(int)
    t['short_rest_away'] = (t['away_rest'] < 6).astype(int)
    t['long_rest_home'] = (t['home_rest'] >= 10).astype(int)
    t['long_rest_away'] = (t['away_rest'] >= 10).astype(int)
    for side in ('home', 'away'):
        # games the head coach has coached this team so far (a new coach is a different team)
        key = t[f'{side}_team'] + '|' + t[f'{side}_coach'].fillna('?')
        t[f'{side}_coach_games'] = t.assign(_k=key).sort_values(['gameday', 'week']).groupby('_k').cumcount().reindex(t.index)
    t['coach_games_diff'] = (t['home_coach_games'].clip(upper=80) - t['away_coach_games'].clip(upper=80))
    return t


# ---------------------------------------------------------------- the game table

BOX_PAIRS = [('fd', 'fd_allowed'), ('yds', 'yds_allowed'), ('plays', 'plays_allowed'), ('to', 'to_allowed'),
             ('pts', 'pts_allowed'), ('epa_pp', 'epa_pp_allowed')]
PBP_STATS = ['epa_pp', 'epa_pass', 'epa_rush', 'success', 'success_pass', 'success_rush', 'early_epa', 'early_success', 'late_epa',
             'comp_epa', 'comp_success', 'explosive', 'sack_rate', 'int_rate', 'fumble_rate', 'fumbles_lost', 'third_rate',
             'first_downs', 'ypa', 'ypc', 'air_yards', 'cpoe', 'pass_rate', 'pass_oe', 'td_per_drive', 'drives', 'penalties',
             'penalty_yards', 'st_epa_for', 'st_epa_against', 'plays']
PBP_PAIRS = [('o_' + s, 'd_' + s) for s in PBP_STATS]
ALL_PAIRS = BOX_PAIRS + PBP_PAIRS


def default_table(rating_params=None, refresh=False):
    """The full game table with every rated stat, cached on disk."""
    import os
    import pickle

    from nflverse_data import CACHE_DIR
    tag = '_'.join(f'{k}{v}' for k, v in sorted((rating_params or {}).items())) or 'default'
    path = os.path.join(CACHE_DIR, f'lab_table_{tag}.pkl')
    if os.path.exists(path) and not refresh:
        with open(path, 'rb') as f:
            return pickle.load(f)
    table = build_table(ALL_PAIRS, rating_params)
    with open(path, 'wb') as f:
        pickle.dump(table, f)
    return table


def off_def_columns(pairs):
    return [f'off__{own}' for own, _ in pairs] + [f'def__{own}' for own, _ in pairs]


BOX_FEATURES = off_def_columns([('fd', 'fd_allowed'), ('yds', 'yds_allowed'), ('plays', 'plays_allowed'), ('to', 'to_allowed'), ('pts', 'pts_allowed')])
PBP_FEATURES = off_def_columns(PBP_PAIRS)
BASIC_CONTEXT = ['neutral', 'rest_diff', 'elo_diff']


def build_table(pairs_spec, rating_params=None, include_pbp=True, refresh=False):
    """Game table with pre-game ratings for `pairs_spec` (list of (own, allowed) column names), market and context.

    `refresh=True` re-downloads the latest season's files first (for live use); research runs use the cache.
    """
    rating_params = rating_params or {}
    nm.DATA_FIRST_SEASON = FIRST_SEASON
    games = nm.normalize_teams(load_games(refresh=refresh))
    games = games[games['season'] >= FIRST_SEASON]
    team_weeks = load_team_weeks(FIRST_SEASON, int(games['season'].max()), refresh=refresh)
    rows = nm.build_team_rows(games, team_weeks)
    if include_pbp:
        pbp = load_pbp_team_games(FIRST_SEASON, int(games['season'].max()), refresh_latest=refresh)
        own = pbp.drop(columns=['season', 'week'])
        allowed = own.rename(columns={'team': 'opp', **{c: 'd_' + c[2:] for c in own.columns if c.startswith('o_')}})
        rows = rows.merge(own, on=['game_id', 'team'], how='left').merge(allowed, on=['game_id', 'opp'], how='left')
    rows['played'] = rows['pts'].notna() & rows['pts_allowed'].notna()
    rated = rate_teams(rows, pairs_spec, **rating_params)

    keep = ['game_id', 'team', 'games_played'] + ['r_' + c for pair in pairs_spec for c in pair]
    home = rated[keep].rename(columns={'team': 'home_team', **{c: 'h_' + c for c in keep[2:]}})
    away = rated[keep].rename(columns={'team': 'away_team', **{c: 'a_' + c for c in keep[2:]}})
    table = games.merge(home, on=['game_id', 'home_team']).merge(away, on=['game_id', 'away_team'])
    for own_col, allowed_col in pairs_spec:
        table[f'off__{own_col}'] = table[f'h_r_{own_col}'] - table[f'a_r_{own_col}']          # offense, home minus away
        table[f'def__{own_col}'] = table[f'h_r_{allowed_col}'] - table[f'a_r_{allowed_col}']  # allowed, home minus away
    table['neutral'] = (table['location'] == 'Neutral').astype(int)
    table['rest_diff'] = (table['home_rest'] - table['away_rest']).clip(-4, 4).fillna(0)
    table['elo_diff'] = nm.add_elo(games).set_index('game_id').loc[table['game_id'], 'elo_diff'].to_numpy()
    if include_pbp:
        # how each game actually went, measured on plays: a less noisy picture of performance than the score
        rows = rows.assign(epa_total=rows['o_epa_pp'] * rows['o_plays'], success_total=rows['o_success'] * rows['o_plays'])
        actual = rows[['game_id', 'team', 'epa_total', 'success_total', 'yds', 'fd', 'to']]
        home_actual = actual.rename(columns={'team': 'home_team', **{c: 'h_act_' + c for c in actual.columns[2:]}})
        away_actual = actual.rename(columns={'team': 'away_team', **{c: 'a_act_' + c for c in actual.columns[2:]}})
        table = table.merge(home_actual, on=['game_id', 'home_team'], how='left').merge(away_actual, on=['game_id', 'away_team'], how='left')
        for name in ('epa_total', 'success_total', 'yds', 'fd', 'to'):
            table[f'perf_{name}'] = table[f'h_act_{name}'] - table[f'a_act_{name}']
    from nfl_injury_features import build_injury_features
    from nflverse_data import load_injuries, load_player_weeks
    last = int(games['season'].max())
    qb = build_injury_features(games, load_injuries(2009, last, refresh=refresh), load_player_weeks(2009, last, refresh=refresh))
    table = table.merge(qb[['game_id', 'qb_rating_diff', 'qb_injury_edge', 'qb_out_any']], on='game_id', how='left')
    table = add_market_features(table)
    return add_context_features(table)


# ---------------------------------------------------------------- walk-forward evaluation

def training_games(table, season, window=14, first_train=2002):
    played = table[table['result'].notna() & (table['game_type'] == 'REG') & (table['season'] >= first_train)]
    return played[(played['season'] < season) & (played['season'] >= season - window)]


def test_games(table, season):
    return table[(table['season'] == season) & table['result'].notna() & table['spread_line'].notna() & (table['game_type'] == 'REG')]


def walk_forward(table, fit_predict, first_test=FIRST_TEST, last_test=LAST_SEASON, **kwargs):
    """Out-of-sample predictions.  fit_predict(train, test) returns a Series/array aligned to test rows."""
    outputs = []
    for season in range(first_test, last_test + 1):
        train, test = training_games(table, season, **kwargs), test_games(table, season)
        if train.empty or test.empty:
            continue
        out = test.copy()
        out['pred'] = np.asarray(fit_predict(train, test), dtype=float)
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def ridge_margin(columns, alpha=300.0, target='result'):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    def fit_predict(train, test):
        spread = train[columns].std()
        usable = [c for c in columns if spread[c] > 1e-6]  # stats missing in early seasons are constant there
        fill = train[usable].median()
        scaler = StandardScaler().fit(train[usable].fillna(fill))
        model = Ridge(alpha=alpha).fit(scaler.transform(train[usable].fillna(fill)), train[target])
        return model.predict(scaler.transform(test[usable].fillna(fill)))
    return fit_predict


def sk_resid(factory, columns, scale=True, target='resid'):
    """Walk-forward adapter for any scikit-learn regressor (fit on the cover margin by default)."""
    from sklearn.preprocessing import StandardScaler

    def fit_predict(train, test):
        spread = train[columns].std()
        usable = [c for c in columns if spread[c] > 1e-6]
        fill = train[usable].median()
        x_train, x_test = train[usable].fillna(fill), test[usable].fillna(fill)
        if scale:
            scaler = StandardScaler().fit(x_train)
            x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        return factory().fit(x_train, train[target]).predict(x_test)
    return fit_predict


# ---------------------------------------------------------------- scoring

def cover_stats(frame, pred_col='pred', is_margin=True):
    """ATS record of betting the side `pred` points to.  pred is a projected margin if is_margin, else a residual."""
    edge = frame[pred_col] - frame['spread_line'] if is_margin else frame[pred_col]
    d = frame[frame['resid'] != 0]
    e = edge.loc[d.index]
    win = np.where(e > 0, d['resid'] > 0, d['resid'] < 0)
    return win, e


def ats_line(label, frame, pred_col='pred', is_margin=True, min_edge=0.0, mask=None):
    f = frame if mask is None else frame[mask]
    win, e = cover_stats(f, pred_col, is_margin)
    keep = np.abs(e.to_numpy()) >= min_edge
    w = win[keep]
    n = len(w)
    rate = w.mean() if n else float('nan')
    se = math.sqrt(0.25 / n) if n else float('nan')
    print('%-52s n=%5d ATS=%.3f +-%.3f  ROI@-110=%+.3f' % (label, n, rate, se, rate * (100 / 110) - (1 - rate) if n else float('nan')))
    return rate, n


def information_test(frame, pred_col='pred', is_margin=True):
    """Slope of the cover margin on the model's disagreement with the market, with a robust t-statistic.

    A slope of 0 means the model says nothing the spread didn't already know; 1 means the model is the better
    estimate.  Returns (slope, t).
    """
    edge = (frame[pred_col] - frame['spread_line']) if is_margin else frame[pred_col]
    x, y = edge.to_numpy(float), frame['resid'].to_numpy(float)
    x_c = x - x.mean()
    slope = (x_c * (y - y.mean())).sum() / (x_c ** 2).sum()
    resid = (y - y.mean()) - slope * x_c
    se = math.sqrt(((x_c ** 2) * (resid ** 2)).sum()) / (x_c ** 2).sum()   # HC0
    return slope, slope / se


def scorecard(label, frame, pred_col='pred', is_margin=True):
    """One-line summary of a model over the discovery and holdout periods."""
    parts = []
    for name, f in (('disc', frame[frame['season'] <= DISCOVERY_END]), ('hold', frame[frame['season'] > DISCOVERY_END])):
        if f.empty:
            continue
        margin = f[pred_col] if is_margin else f['spread_line'] + f[pred_col]
        mae = (margin - f['result']).abs().mean()
        market = (f['spread_line'] - f['result']).abs().mean()
        slope, t = information_test(f, pred_col, is_margin)
        win, _ = cover_stats(f, pred_col, is_margin)
        parts.append('%s: MAE %.3f (mkt %.3f) info slope %+.2f t=%+.1f ATS %.3f n=%d' % (name, mae, market, slope, t, win.mean(), len(win)))
    print('%-34s | %s' % (label, ' | '.join(parts)))
