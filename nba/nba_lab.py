"""The NBA research table: one row per regular-season game with everything known before tip-off (team ratings from box
scores, rest, travel, altitude, who is playing), the closing lines and the result; and the walk-forward harness.

Team ratings (rate_teams).  Every team-game gives one observation of each stat in STATS, offense ('own') and defense
('allowed', the opponent's own), per possession where it matters: points per 100 possessions, the four factors
(effective FG%, turnovers per possession, offensive rebound rate, free throws per field-goal attempt), the three-point
rate and percentage, assists per basket, points in the paint and on the fast break per 100, steals, blocks and fouls, and
the pace (possessions) and margin.  A team's pre-game rating is a recency-weighted average of its earlier games (DECAY
per game), each corrected by OPP_WEIGHT of the opponent's deviation from the league average (a good defense lowers the
scoring it allows, so scoring against it counts for more), shrunk toward the league mean by PRIOR_GAMES pseudo-games
and carried into a new season at CARRY.  The league mean itself drifts (scoring rose from about 100 to 115 points a team
over these seasons), so it is a slow running average.  Everything is built in date order from earlier games only, and
games on the same day do not see each other.

Context (add_context).  Days since the team's last game (a back-to-back is 1), games in the 3 days before, kilometres
travelled from its last game and the time zones crossed, the altitude of the arena (Denver, Utah, Mexico City), home,
away or neutral site.

Availability (availability).  A player's value is his recent Game Score (Hollinger's box-score composite) per game
played, and his part in the rotation his recent minutes (decayed, counting games he missed as 0).  Before each game a
team's rotation is the players averaging ROTATION_MINUTES or more lately; 'missing' is the value of rotation players who
do not play, 'returning' that of players who play but were out of the rotation (back from an absence), 'playing' the
value of those who play.  The backtests know who played, as the closing line mostly did; for games not played yet (the
tracker) the players who have played for the team this season and are not on ESPN's injury list as out play.

The table (game_table) is home minus away for each rating (d_*) and home plus away (s_*, for totals), with the context,
closing lines (nba_lines.closing_lines), the result and the cover and over margins (cover = margin + home spread,
positive when the home team covered; over = points - total).
"""

import math
import os

import numpy as np
import pandas as pd

import nba_data
import nba_lines
from nba_common import CACHE_DIR, REGULAR_SEASON

FRANCHISES = set(range(1, 31))      # ESPN's ids of the 30 teams (All-Star and exhibition teams have others)
FIRST_SEASON = 2012                 # box scores from here, so ratings are warm by the first season with lines (2014)
DECAY, OPP_WEIGHT, PRIOR_GAMES, CARRY = 0.95, 0.6, 5.0, 0.5
LEAGUE_DECAY = 0.9995               # per team-game: the league mean follows scoring over about half a season
ROTATION_MINUTES, PLAYER_DECAY = 12.0, 0.9
TABLE_VERSION = 1
# (stat, has an allowed side); per possession where the count depends on pace
STATS = (('ortg', True), ('efg', True), ('tov', True), ('orb', True), ('ftr', True), ('three_rate', True),
         ('three_pct', True), ('ast_rate', True), ('paint', True), ('fastbreak', True), ('stl', True), ('blk', True),
         ('fouls', True), ('pace', False), ('margin', False))
# venue city: (latitude, longitude, elevation in metres)
CITIES = {
    'Atlanta': (33.757, -84.396, 300), 'Boston': (42.366, -71.062, 5), 'Brooklyn': (40.683, -73.975, 10),
    'Charlotte': (35.225, -80.839, 230), 'Chicago': (41.881, -87.674, 180), 'Cleveland': (41.496, -81.688, 200),
    'Dallas': (32.790, -96.810, 140), 'Denver': (39.749, -105.008, 1609), 'Detroit': (42.341, -83.055, 190),
    'Auburn Hills': (42.697, -83.245, 290), 'Oakland': (37.750, -122.203, 5), 'San Francisco': (37.768, -122.388, 5),
    'Houston': (29.751, -95.362, 15), 'Indianapolis': (39.764, -86.155, 220), 'Los Angeles': (34.043, -118.267, 90),
    'Inglewood': (33.945, -118.341, 30), 'Memphis': (35.138, -90.051, 80), 'Miami': (25.781, -80.188, 2),
    'Milwaukee': (43.045, -87.917, 190), 'Minneapolis': (44.979, -93.276, 260), 'New Orleans': (29.949, -90.082, 2),
    'New York': (40.751, -73.993, 10), 'Newark': (40.734, -74.171, 10), 'Oklahoma City': (35.463, -97.515, 370),
    'Orlando': (28.539, -81.384, 30), 'Philadelphia': (39.901, -75.172, 10), 'Phoenix': (33.446, -112.071, 330),
    'Portland': (45.532, -122.667, 15), 'Sacramento': (38.580, -121.500, 10), 'San Antonio': (29.427, -98.437, 200),
    'Austin': (30.281, -97.732, 150), 'Toronto': (43.643, -79.379, 80), 'Salt Lake City': (40.768, -111.901, 1288),
    'Washington': (38.898, -77.021, 20), 'Lake Buena Vista': (28.370, -81.519, 30), 'Tampa': (27.943, -82.452, 5),
    'Mexico City': (19.404, -99.096, 2240), 'Las Vegas': (36.103, -115.178, 610), 'London': (51.503, 0.003, 10),
    'Paris': (48.839, 2.379, 35), 'Manchester': (53.488, -2.244, 40), 'Berlin': (52.508, 13.443, 35),
}


# ---------------------------------------------------------------- team-game stats

def _per_game(box):
    """The box score's counts under short names, with turnovers including team turnovers."""
    tov = box['total_turnovers'].where(box['total_turnovers'].notna(), box['turnovers'])
    return pd.DataFrame({
        'game_id': box['game_id'], 'team_id': box['team_id'], 'opp_id': box['opponent_team_id'],
        'pts': box['team_score'], 'fgm': box['field_goals_made'], 'fga': box['field_goals_attempted'],
        'fg3m': box['three_point_field_goals_made'], 'fg3a': box['three_point_field_goals_attempted'],
        'ftm': box['free_throws_made'], 'fta': box['free_throws_attempted'], 'orb': box['offensive_rebounds'],
        'drb': box['defensive_rebounds'], 'ast': box['assists'], 'stl': box['steals'], 'blk': box['blocks'],
        'tov': tov, 'pf': box['fouls'], 'paint': box['points_in_paint'], 'fastbreak': box['fast_break_points']})


def team_games(seasons, refresh=False):
    """One row per team per final regular-season game between two franchises: the counts, possessions and the
    observed STATS (own; the allowed side is the opponent's row)."""
    box = nba_data.load_team_box(seasons, refresh)
    box = box[(box['season_type'] == REGULAR_SEASON) & box['team_id'].isin(FRANCHISES)
              & box['opponent_team_id'].isin(FRANCHISES)]
    own = _per_game(box).drop_duplicates(['game_id', 'team_id'])
    opp = own.drop(columns=['opp_id']).add_prefix('o_').rename(columns={'o_game_id': 'game_id', 'o_team_id': 'opp_id'})
    g = own.merge(opp, on=['game_id', 'opp_id'], how='inner')
    team_poss = g['fga'] - g['orb'] + g['tov'] + 0.44 * g['fta']
    opp_poss = g['o_fga'] - g['o_orb'] + g['o_tov'] + 0.44 * g['o_fta']
    g['poss'] = (team_poss + opp_poss) / 2
    safe = lambda a, b: a / b.where(b > 0)                                            # noqa: E731
    g['ortg'] = 100 * safe(g['pts'], g['poss'])
    g['efg'] = safe(g['fgm'] + 0.5 * g['fg3m'], g['fga'])
    g['tov'] = safe(g['tov'], g['poss'])
    g['orb'] = safe(g['orb'], g['orb'] + g['o_drb'])
    g['ftr'] = safe(g['ftm'], g['fga'])
    g['three_rate'] = safe(g['fg3a'], g['fga'])
    g['three_pct'] = safe(g['fg3m'], g['fg3a'])
    g['ast_rate'] = safe(g['ast'], g['fgm'])
    g['paint'] = 100 * safe(g['paint'], g['poss'])
    g['fastbreak'] = 100 * safe(g['fastbreak'], g['poss'])
    g['stl'] = 100 * safe(g['stl'], g['poss'])
    g['blk'] = safe(g['blk'], g['o_fga'] - g['o_fg3a'])
    g['fouls'] = 100 * safe(g['pf'], g['poss'])
    g['pace'] = g['poss']
    g['margin'] = g['pts'] - g['o_pts']
    return g[['game_id', 'team_id', 'opp_id', 'pts', 'o_pts', 'poss'] + [s for s, _ in STATS]]


# ---------------------------------------------------------------- ratings

def stat_columns():
    """The rated columns: each stat's own rating and, where it has one, its allowed rating."""
    return [f'{s}_own' for s, _ in STATS] + [f'{s}_allowed' for s, allowed in STATS if allowed]


def rate_teams(schedule, stats, decay=DECAY, opp_weight=OPP_WEIGHT, prior_games=PRIOR_GAMES, carry=CARRY):
    """Pre-game ratings for every (game, team) of `schedule` (one row per game: game_id, season, start, home_id,
    away_id), in date order, from the games in `stats` (team_games) played before it.  Games without stats yet
    (upcoming) get ratings and change nothing.  Returns one row per (game_id, team_id): r_<stat>_own, r_<stat>_allowed,
    and gp, the team's games so far this season."""
    names = [s for s, _ in STATS]
    allowed = np.array([a for _, a in STATS])
    # a stat without an allowed side is corrected by the opponent's own rating: a fast opponent speeds the game up
    # (pace counts for less), a strong one makes a margin worth more
    single_sign = np.array([1.0 if name == 'margin' else -1.0 for name in names])
    k = len(names)
    obs = {}
    for row in stats[['game_id', 'team_id'] + names].itertuples(index=False):
        obs[(row[0], row[1])] = np.array(row[2:], dtype=float)
    league_sum, league_w = np.zeros(k), 0.0
    state = {}          # team -> [w, own sum (k), allowed sum (k), season, games this season]
    out = []
    games = schedule.sort_values(['start', 'game_id'])
    days = games['start'].dt.tz_convert('America/New_York').dt.date.values
    pending = []        # same-day games: rated before any of that day's updates

    def flush():
        nonlocal league_sum, league_w
        for team, own_obs, allowed_obs in pending:
            st = state[team]
            st[0] = decay * st[0] + 1
            st[1] = decay * st[1] + own_obs
            st[2] = decay * st[2] + allowed_obs
            st[4] += 1
        pending.clear()

    last_day = None
    for i, game in enumerate(games.itertuples(index=False)):
        if days[i] != last_day:
            flush()
            last_day = days[i]
        mean = league_sum / league_w if league_w else np.full(k, np.nan)
        ratings = {}
        for team in (game.home_id, game.away_id):
            st = state.get(team)
            if st is None:
                st = state[team] = [0.0, np.zeros(k), np.zeros(k), game.season, 0]
            if st[3] != game.season:
                st[0] *= carry
                st[1] *= carry
                st[2] *= carry
                st[3], st[4] = game.season, 0
            fill = np.where(np.isnan(mean), 0.0, mean)
            own = (st[1] + prior_games * fill) / (st[0] + prior_games)
            alw = (st[2] + prior_games * fill) / (st[0] + prior_games)
            ratings[team] = (own, alw, st[4])
            out.append((game.game_id, team, st[4], own, alw))
        for team, opp in ((game.home_id, game.away_id), (game.away_id, game.home_id)):
            seen, against = obs.get((game.game_id, team)), obs.get((game.game_id, opp))
            if seen is None or against is None:
                continue
            fill = np.where(np.isnan(mean), seen, mean)
            opp_own, opp_alw, _ = ratings[opp]
            # what the team's offense did, corrected for the defense it faced, and what its defense allowed, corrected
            # for the offense it faced; a missing stat counts as the team's own rating (no change)
            own_obs = np.where(allowed, seen - opp_weight * (opp_alw - fill),
                               seen + single_sign * opp_weight * (opp_own - fill))
            alw_obs = against - opp_weight * (opp_own - fill)
            own_obs = np.where(np.isnan(own_obs), ratings[team][0], own_obs)
            alw_obs = np.where(np.isnan(alw_obs) | ~allowed, ratings[team][1], alw_obs)
            pending.append((team, own_obs, alw_obs))
            league_sum = LEAGUE_DECAY * league_sum + np.where(np.isnan(seen), fill, seen)
            league_w = LEAGUE_DECAY * league_w + 1
    flush()
    rows = []
    for game_id, team, gp, own, alw in out:
        row = {'game_id': game_id, 'team_id': team, 'gp': gp}
        for j, name in enumerate(names):
            row[f'r_{name}_own'] = own[j]
            if allowed[j]:
                row[f'r_{name}_allowed'] = alw[j]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- context

def _distance_km(a, b):
    if a is None or b is None:
        return np.nan
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(min(1.0, h)))


def add_context(schedule):
    """Per (game, team): rest, back-to-back, games in the 3 days before, travel and time zones crossed since the last
    game (from the home arena at a season's start), the arena's altitude and the team's side (home/away/neutral)."""
    games = schedule.sort_values(['start', 'game_id'])
    home_city = (games[~games['neutral_site']].groupby(['season', 'home_id'])['venue_address_city']
                 .agg(lambda c: c.mode().iat[0] if c.notna().any() else None).to_dict())
    rows = []
    for g in games.itertuples(index=False):
        city = g.venue_address_city if isinstance(g.venue_address_city, str) and g.venue_address_city in CITIES \
            else home_city.get((g.season, g.home_id))
        for team, side in ((g.home_id, 'home'), (g.away_id, 'away')):
            rows.append({'game_id': g.game_id, 'team_id': team, 'season': g.season, 'start': g.start, 'city': city,
                         'side': 'neutral' if g.neutral_site else side})
    ctx = pd.DataFrame(rows).sort_values(['team_id', 'start'])
    day = ctx['start'].dt.tz_convert('America/New_York').dt.normalize()
    ctx['day'] = day
    prev_day = ctx.groupby('team_id')['day'].shift()
    prev_season = ctx.groupby('team_id')['season'].shift()
    new_season = prev_season.ne(ctx['season'])
    ctx['rest'] = ((ctx['day'] - prev_day).dt.days).where(~new_season).clip(upper=5)
    ctx['b2b'] = (ctx['rest'] == 1).astype(float)
    ctx['games_3d'] = 0.0
    for lag in (1, 2):
        lagged = ctx.groupby('team_id')['day'].shift(lag)
        same = ctx.groupby('team_id')['season'].shift(lag).eq(ctx['season'])
        ctx['games_3d'] += (((ctx['day'] - lagged).dt.days <= 3) & same).astype(float)
    prev_city = ctx.groupby('team_id')['city'].shift()
    start_city = [home_city.get((s, t)) for s, t in zip(ctx['season'], ctx['team_id'])]
    prev_city = prev_city.where(~new_season, pd.Series(start_city, index=ctx.index))
    ctx['travel_km'] = [_distance_km(CITIES.get(p), CITIES.get(c)) for p, c in zip(prev_city, ctx['city'])]
    ctx['tz_shift'] = [((CITIES[c][1] - CITIES[p][1]) / 15) if p in CITIES and c in CITIES else np.nan
                       for p, c in zip(prev_city, ctx['city'])]
    ctx['altitude'] = [CITIES[c][2] / 1000 if c in CITIES else np.nan for c in ctx['city']]
    return ctx.drop(columns=['day', 'start', 'season'])


# ---------------------------------------------------------------- availability

def game_score(box):
    """Hollinger's Game Score for each player-game."""
    return (box['points'] + 0.4 * box['field_goals_made'] - 0.7 * box['field_goals_attempted']
            - 0.4 * (box['free_throws_attempted'] - box['free_throws_made']) + 0.7 * box['offensive_rebounds']
            + 0.3 * box['defensive_rebounds'] + box['steals'] + 0.7 * box['assists'] + 0.7 * box['blocks']
            - 0.4 * box['fouls'] - box['turnovers'])


def availability(schedule, players, lineups=None, out_ids=None):
    """Per (game, team): the pre-game value of the rotation players who miss the game ('missing'), of rotation-calibre
    players back from an absence ('returning') and of all who play ('playing'), in Game Score per game, and the share of
    the rotation's minutes that is missing.  `players` is the player box score; for games not played yet, `lineups`
    ({(game_id, team_id): set of athlete ids who play}) or else `out_ids` (ESPN ids of players out, from the injury
    list: everyone else who has played for the team lately plays) stands in for it."""
    box = players[(players['season_type'] == REGULAR_SEASON) & players['team_id'].isin(FRANCHISES)].copy()
    box['minutes'] = pd.to_numeric(box['minutes'], errors='coerce').fillna(0)
    box['gs'] = game_score(box.fillna(0))
    played = box[box['minutes'] > 0]
    by_game = {key: dict(zip(g['athlete_id'], zip(g['minutes'], g['gs']))) for key, g in
               played.groupby(['game_id', 'team_id'])}
    team_minutes = {}               # team -> {player: decayed minutes per team game, 0 for games he missed}
    value, weight, mpg = {}, {}, {}  # player -> decayed sums of Game Score, games and minutes, over games he played
    this_season = {}                # team -> (season, players who have played for it this season)
    out_ids = None if out_ids is None else {float(p) for p in out_ids}
    rows = []
    for g in schedule.sort_values(['start', 'game_id']).itertuples(index=False):
        for team in (g.home_id, g.away_id):
            tonight = by_game.get((g.game_id, team))
            final = tonight is not None
            if tonight is None and lineups is not None and (g.game_id, team) in lineups:
                tonight = {p: (None, None) for p in lineups[(g.game_id, team)]}
            if tonight is None and out_ids is not None and team in team_minutes:
                # as the backtests count who actually played: the players who have played for the team this season
                # (so not those who left it) and are not out
                season_players = this_season.get(team, (None, set()))
                current = season_players[1] if season_players[0] == g.season else set()
                tonight = {p: (None, None) for p, m in team_minutes[team].items()
                           if m * (1 - PLAYER_DECAY) >= 1.0 and p in current and float(p) not in out_ids}
            if tonight is None:
                rows.append({'game_id': g.game_id, 'team_id': team})
                continue
            roster = team_minutes.setdefault(team, {})
            average = {p: m * (1 - PLAYER_DECAY) for p, m in roster.items()}
            rotation = {p for p, m in average.items() if m >= ROTATION_MINUTES}
            out = rotation - set(tonight)
            back = [p for p in tonight if p not in rotation and weight.get(p, 0) >= 3
                    and mpg[p] / weight[p] >= ROTATION_MINUTES]
            per_game = {p: value[p] / weight[p] for p in set(tonight) | out if weight.get(p)}
            rotation_minutes = sum(average[p] for p in rotation)
            rows.append({'game_id': g.game_id, 'team_id': team, 'missing': sum(per_game.get(p, 0) for p in out),
                         'returning': sum(per_game.get(p, 0) for p in back),
                         'playing': sum(per_game.get(p, 0) for p in tonight),
                         'missing_minutes': sum(average[p] for p in out) / rotation_minutes if rotation_minutes
                         else np.nan})
            if not final:
                continue
            season_players = this_season.get(team)
            if season_players is None or season_players[0] != g.season:
                season_players = this_season[team] = (g.season, set())
            season_players[1].update(tonight)
            for p in list(roster):
                roster[p] *= PLAYER_DECAY
                if roster[p] < 0.5 and p not in tonight:
                    del roster[p]
            for p, (m, gs) in tonight.items():
                roster[p] = roster.get(p, 0.0) + m
                value[p] = PLAYER_DECAY * value.get(p, 0.0) + gs
                weight[p] = PLAYER_DECAY * weight.get(p, 0.0) + 1
                mpg[p] = PLAYER_DECAY * mpg.get(p, 0.0) + m
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the table

def game_table(seasons=None, refresh=False, upcoming=None, lineups=None, out_ids=None):
    """One row per regular-season game between franchises of `seasons` (default FIRST_SEASON to the latest with box
    scores), with home-minus-away (d_) and home-plus-away (s_) ratings, context, availability, closing lines and
    results.  `upcoming` (schedule rows not played yet) are rated too, for the tracker; their availability comes
    from `lineups` or `out_ids` (availability)."""
    from nba_common import current_season
    seasons = list(seasons or range(FIRST_SEASON, current_season() + 1))
    schedule = nba_data.load_schedule(seasons, refresh)
    schedule = schedule[(schedule['season_type'] == REGULAR_SEASON) & schedule['home_id'].isin(FRANCHISES)
                        & schedule['away_id'].isin(FRANCHISES)]
    if upcoming is not None:
        schedule = pd.concat([schedule, upcoming[~upcoming['game_id'].isin(schedule['game_id'])]], ignore_index=True)
    stats = team_games(seasons, refresh)
    ratings = rate_teams(schedule, stats)
    ctx = add_context(schedule)
    players = nba_data.load_player_box(seasons, refresh)
    avail = availability(schedule, players, lineups, out_ids)
    per_team = ratings.merge(ctx, on=['game_id', 'team_id'], how='left')
    per_team = per_team.merge(avail, on=['game_id', 'team_id'], how='left')
    rated = [c for c in per_team.columns if c.startswith('r_')]
    keep = rated + ['gp', 'rest', 'b2b', 'games_3d', 'travel_km', 'tz_shift', 'altitude', 'missing', 'returning',
                    'playing', 'missing_minutes']
    home = per_team[['game_id', 'team_id'] + keep].add_prefix('h_')
    home = home.rename(columns={'h_game_id': 'game_id', 'h_team_id': 'home_id'})
    away = per_team[['game_id', 'team_id'] + keep].add_prefix('a_')
    away = away.rename(columns={'a_game_id': 'game_id', 'a_team_id': 'away_id'})
    table = schedule.merge(home, on=['game_id', 'home_id'], how='left')
    table = table.merge(away, on=['game_id', 'away_id'], how='left')
    for column in rated:
        table['d_' + column[2:]] = table['h_' + column] - table['a_' + column]
        table['s_' + column[2:]] = table['h_' + column] + table['a_' + column]
    for column in ('rest', 'b2b', 'games_3d', 'travel_km', 'tz_shift', 'missing', 'returning', 'playing',
                   'missing_minutes'):
        table['d_' + column] = table['h_' + column] - table['a_' + column]
    table['home'] = (~table['neutral_site']).astype(float)
    lines = nba_lines.closing_lines([s for s in seasons if s >= nba_lines.FIRST_LINES_SEASON])
    table = table.merge(lines, on='game_id', how='left')
    final = table['status_type_completed'].fillna(False).astype(bool)
    table['margin'] = (table['home_score'] - table['away_score']).where(final)
    table['points'] = (table['home_score'] + table['away_score']).where(final)
    table['cover'] = table['margin'] + table['close_home_spread']
    table['over'] = table['points'] - table['close_total']
    return table.sort_values(['start', 'game_id']).reset_index(drop=True)


def default_table(refresh=False):
    """The full research table, cached in data_cache (rebuilt when TABLE_VERSION changes or on `refresh`)."""
    path = os.path.join(CACHE_DIR, f'nba_table_v{TABLE_VERSION}.pkl')
    if os.path.exists(path) and not refresh:
        return pd.read_pickle(path)
    table = game_table()
    table.to_pickle(path)
    return table


# ---------------------------------------------------------------- walk-forward

def walk_forward(table, fit_predict, first_test, last_test, train_from=None):
    """Score each season from first_test to last_test with a model fitted only on earlier seasons: fit_predict(train,
    test) returns predictions for test's rows.  Returns the test rows with a 'pred' column."""
    out = []
    for season in range(first_test, last_test + 1):
        train = table[(table['season'] < season) & ((table['season'] >= train_from) if train_from else True)]
        test = table[table['season'] == season]
        if test.empty or train.empty:
            continue
        out.append(test.assign(pred=fit_predict(train, test)))
    return pd.concat(out) if out else table.iloc[0:0].assign(pred=[])
