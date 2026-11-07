"""The college football research table: one row per game with everything known before kickoff (team ratings from the
play-by-play, rest, home field), the closing lines and the result; and the walk-forward harness.

Team-game stats (team_stats).  From the reduced play-by-play (cfb_data.team_games: sums and counts per team-game), each
team-game gives one observation of each stat in STATS, offense ('own') and defense ('allowed', the opponent's own):
points per drive, EPA per play (all, passes, runs, competitive plays with the offense's win probability 10-90%, early
downs), success rates, explosive plays, yards per play, line yards and stuffs per run, success on third and fourth down,
sacks per pass, havoc per play, turnovers and scoring opportunities (drives reaching the 40) per drive, points per
opportunity, starting field position, special-teams EPA, penalty yards, points, and the pace (scrimmage plays) and
margin.

Ratings (rate_teams), as in the NBA suite.  A team's pre-game rating is a recency-weighted average of its earlier games
(DECAY per game), each corrected by OPP_WEIGHT of the opponent's deviation from the league average (gaining 6 yards a
play against a defense that allows 7 counts for less), shrunk toward its tier's mean by PRIOR_GAMES pseudo-games, and
carried into a new season at CARRY.  College schedules differ far more than the NFL's or NBA's (an FBS team can play a
top-10 team one week and an FCS team the next), so the correction is full (OPP_WEIGHT 1.0, against 0.6 there).  Teams
outside FBS are shrunk toward the mean of non-FBS teams, not the league's, so an FCS team with two games is not taken
for an average FBS one.  Everything is built in date order from earlier games only; games on the same day do not see
each other.

The table (game_table) is home minus away for each rating (d_*) and home plus away (s_*, for totals), with the rest days,
games played this season, home field, conference game, closing lines (cfb_lines.closing_lines), the result, and the
cover and over margins (cover = margin + home spread, positive when the home team covered; over = points - total).
Only games between two FBS teams are in the table (FCS games still feed the ratings).
"""

import os

import numpy as np
import pandas as pd

import cfb_cfbd
import cfb_data
import cfb_lines
from cfb_common import CACHE_DIR, FBS, REGULAR_SEASON, current_season

FIRST_SEASON = cfb_data.FIRST_PBP_SEASON        # ratings warm up on 2004-2005; lines start in 2006
DECAY, OPP_WEIGHT, PRIOR_GAMES, CARRY = 0.90, 1.0, 3.0, 0.5
LEAGUE_DECAY = 0.9995
MAX_REST = 21
TABLE_VERSION = 2
# (stat, has an allowed side); built from the team-game sums by team_stats
STATS = (('ppd', True), ('epa', True), ('pass_epa', True), ('rush_epa', True), ('comp_epa', True),
         ('early_epa', True), ('success', True), ('pass_success', True), ('rush_success', True),
         ('late_success', True), ('explosive', True), ('ypp', True), ('line_yards', True), ('stuff', True),
         ('sack_rate', True), ('havoc', True), ('turnovers', True), ('opp_rate', True), ('pts_per_opp', True),
         ('field_pos', True), ('st_epa', True), ('penalty_yards', True), ('points', True), ('plays', False),
         ('margin', False))
CONTEXT = ('rest', 'gp')


# ---------------------------------------------------------------- team-game stats

def team_stats(sums, games):
    """One row per team per final game: the observed STATS (own; the allowed side is the opponent's row).  `sums` and
    `games` are cfb_data.load_team_games' outputs."""
    safe = lambda a, b: a / b.where(b > 0)                                            # noqa: E731
    s = sums.copy()
    out = pd.DataFrame({'game_id': s['game_id'], 'team_id': s['team_id']})
    out['ppd'] = safe(s['drive_points'], s['n_drives'])
    out['epa'] = safe(s['epa'], s['n_plays'])
    out['pass_epa'] = safe(s['pass_epa'], s['n_pass'])
    out['rush_epa'] = safe(s['rush_epa'], s['n_rush'])
    out['comp_epa'] = safe(s['comp_epa'], s['n_comp'])
    out['early_epa'] = safe(s['early_epa'], s['n_early'])
    out['success'] = safe(s['success'], s['n_plays'])
    out['pass_success'] = safe(s['pass_success'], s['n_pass'])
    out['rush_success'] = safe(s['rush_success'], s['n_rush'])
    out['late_success'] = safe(s['late_success'], s['n_late'])
    out['explosive'] = safe(s['explosive'], s['n_plays'])
    out['ypp'] = safe(s['yards'], s['n_plays'])
    out['line_yards'] = safe(s['line_yards'], s['n_rush'])
    out['stuff'] = safe(s['stuffed'], s['n_rush'])
    out['sack_rate'] = safe(s['sacks'], s['n_pass'])
    out['havoc'] = safe(s['havoc'], s['n_plays'])
    out['turnovers'] = safe(s['turnovers'], s['n_drives'])
    out['opp_rate'] = safe(s['n_opportunities'], s['n_drives'])
    out['pts_per_opp'] = safe(s['opportunity_points'], s['n_opportunities'])
    out['field_pos'] = safe(s['start_sum'], s['n_drives'])
    out['st_epa'] = s['st_epa']
    out['penalty_yards'] = s['penalty_yards']
    out['plays'] = s['n_plays']
    g = games.drop_duplicates('game_id')
    score = pd.concat([
        pd.DataFrame({'game_id': g['game_id'], 'team_id': g['home_id'], 'points': g['home_score'],
                      'margin': g['home_score'] - g['away_score']}),
        pd.DataFrame({'game_id': g['game_id'], 'team_id': g['away_id'], 'points': g['away_score'],
                      'margin': g['away_score'] - g['home_score']})])
    out = out.merge(score, on=['game_id', 'team_id'], how='inner')
    return out[['game_id', 'team_id'] + [name for name, _ in STATS]]


def stat_columns():
    """The rated columns: each stat's own rating and, where it has one, its allowed rating."""
    return [f'{s}_own' for s, _ in STATS] + [f'{s}_allowed' for s, allowed in STATS if allowed]


# ---------------------------------------------------------------- ratings

def rate_teams(schedule, stats, tiers, decay=DECAY, opp_weight=OPP_WEIGHT, prior_games=PRIOR_GAMES, carry=CARRY):
    """Pre-game ratings for every (game, team) of `schedule` (one row per game: game_id, season, start, home_id,
    away_id), in date order, from the games in `stats` (team_stats) played before it.  `tiers` is {(season, team_id):
    True for FBS}.  Games without stats yet (upcoming) get ratings and change nothing.  Returns one row per (game_id,
    team_id): r_<stat>_own, r_<stat>_allowed, and gp, the team's games so far this season."""
    names = [s for s, _ in STATS]
    allowed = np.array([a for _, a in STATS])
    # a stat without an allowed side is corrected by the opponent's own rating: a fast opponent speeds the game up
    # (plays count for less), a strong one makes a margin worth more
    single_sign = np.array([1.0 if name == 'margin' else -1.0 for name in names])
    k = len(names)
    obs = {}
    for row in stats[['game_id', 'team_id'] + names].itertuples(index=False):
        obs[(row[0], row[1])] = np.array(row[2:], dtype=float)
    league = {True: [np.zeros(k), np.zeros(k)], False: [np.zeros(k), np.zeros(k)]}   # tier -> [sum, weight]
    state = {}          # team -> [w, own sum (k), allowed sum (k), season, games this season]
    out = []
    games = schedule.sort_values(['start', 'game_id'])
    days = games['start'].dt.tz_convert('America/New_York').dt.date.values
    pending = []        # same-day games: rated before any of that day's updates (teams and the league means)

    def mean_of(tier=None):
        if tier is None:
            total = league[True][0] + league[False][0]
            weight = league[True][1] + league[False][1]
        else:
            total, weight = league[tier]
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(weight > 0, total / np.where(weight > 0, weight, 1), np.nan)

    def flush():
        for team, own_obs, allowed_obs, tier, seen in pending:
            st = state[team]
            st[0] = decay * st[0] + 1
            st[1] = decay * st[1] + own_obs
            st[2] = decay * st[2] + allowed_obs
            st[4] += 1
            present = ~np.isnan(seen)
            league[tier][0] = LEAGUE_DECAY * league[tier][0] + np.where(present, seen, 0.0)
            league[tier][1] = LEAGUE_DECAY * league[tier][1] + present
        pending.clear()

    last_day = None
    for i, game in enumerate(games.itertuples(index=False)):
        if days[i] != last_day:
            flush()
            last_day = days[i]
        overall = mean_of()
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
            tier_mean = mean_of(tiers.get((game.season, team), False))
            prior = np.where(np.isnan(tier_mean), overall, tier_mean)
            prior = np.where(np.isnan(prior), 0.0, prior)
            own = (st[1] + prior_games * prior) / (st[0] + prior_games)
            alw = (st[2] + prior_games * prior) / (st[0] + prior_games)
            ratings[team] = (own, alw, st[4])
            out.append((game.game_id, team, st[4], own, alw))
        for team, opp in ((game.home_id, game.away_id), (game.away_id, game.home_id)):
            seen, against = obs.get((game.game_id, team)), obs.get((game.game_id, opp))
            if seen is None or against is None:
                continue
            fill = np.where(np.isnan(overall), seen, overall)
            opp_own, opp_alw, _ = ratings[opp]
            # what the team's offense did, corrected for the defense it faced, and what its defense allowed, corrected
            # for the offense it faced; a missing stat counts as the team's own rating (no change)
            own_obs = np.where(allowed, seen - opp_weight * (opp_alw - fill),
                               seen + single_sign * opp_weight * (opp_own - fill))
            alw_obs = against - opp_weight * (opp_own - fill)
            own_obs = np.where(np.isnan(own_obs), ratings[team][0], own_obs)
            alw_obs = np.where(np.isnan(alw_obs) | ~allowed, ratings[team][1], alw_obs)
            pending.append((team, own_obs, alw_obs, tiers.get((game.season, team), False), seen))
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

def add_context(schedule):
    """Days since each team's last game (MAX_REST for a season's first game or after a long break), per (game, team)."""
    sides = pd.concat([schedule[['game_id', 'season', 'start', 'home_id']].rename(columns={'home_id': 'team_id'}),
                       schedule[['game_id', 'season', 'start', 'away_id']].rename(columns={'away_id': 'team_id'})])
    sides = sides.sort_values(['team_id', 'start'])
    previous = sides.groupby(['team_id', 'season'])['start'].shift()
    sides['rest'] = ((sides['start'] - previous).dt.total_seconds() / 86400).round().clip(upper=MAX_REST)
    sides['rest'] = sides['rest'].fillna(MAX_REST)
    return sides[['game_id', 'team_id', 'rest']]


# ---------------------------------------------------------------- the table

def fbs_tiers(seasons, refresh=False):
    """{(season, team_id): True for FBS, False otherwise}."""
    groups = cfb_data.load_groups(seasons, refresh)
    return {(int(s), int(t)): sub == FBS for s, t, sub in zip(groups['season'], groups['team_id'],
                                                              groups['subdivision_id'])}


def game_table(seasons=None, refresh=False):
    """One row per game between two FBS teams in `seasons` (default FIRST_SEASON to the current season), regular season
    and bowls (season_type), with home-minus-away (d_) and home-plus-away (s_) ratings, context, closing lines and
    results.  Seasons from cfb_lines.CFBD_FROM add CollegeFootballData's games (upcoming ones too) and lines."""
    seasons = list(seasons or range(FIRST_SEASON, current_season() + 1))
    sums, pbp_games = cfb_data.load_team_games(seasons, refresh_latest=refresh)
    schedule = cfb_data.load_schedule(seasons, refresh)
    recent = [s for s in seasons if s >= cfb_lines.CFBD_FROM]
    if recent:
        extra = pd.concat([cfb_cfbd.games(s, refresh=refresh and s == seasons[-1]) for s in recent], ignore_index=True)
        extra = extra[~extra['game_id'].isin(schedule['game_id'])]
        schedule = pd.concat([schedule, extra[[c for c in schedule.columns if c in extra.columns]]], ignore_index=True)
    schedule = schedule[schedule['home_id'].notna() & schedule['away_id'].notna()].copy()
    schedule['home_id'] = schedule['home_id'].astype(int)
    schedule['away_id'] = schedule['away_id'].astype(int)
    schedule = schedule[schedule['start'].notna()]
    tiers = fbs_tiers(seasons, refresh)
    stats = team_stats(sums, pbp_games)
    ratings = rate_teams(schedule, stats, tiers)
    per_team = ratings.merge(add_context(schedule), on=['game_id', 'team_id'], how='left')
    rated = [c for c in per_team.columns if c.startswith('r_')]
    keep = rated + list(CONTEXT)
    home = per_team[['game_id', 'team_id'] + keep].add_prefix('h_').rename(
        columns={'h_game_id': 'game_id', 'h_team_id': 'home_id'})
    away = per_team[['game_id', 'team_id'] + keep].add_prefix('a_').rename(
        columns={'a_game_id': 'game_id', 'a_team_id': 'away_id'})
    fbs = [tiers.get((s, h), False) and tiers.get((s, a), False)
           for s, h, a in zip(schedule['season'], schedule['home_id'], schedule['away_id'])]
    table = schedule[fbs].merge(home, on=['game_id', 'home_id'], how='left').merge(away, on=['game_id', 'away_id'],
                                                                                  how='left')
    derived = {}
    for column in rated:
        derived['d_' + column[2:]] = table['h_' + column] - table['a_' + column]
        derived['s_' + column[2:]] = table['h_' + column] + table['a_' + column]
    for column in CONTEXT:
        derived['d_' + column] = table['h_' + column] - table['a_' + column]
        derived['sum_' + column] = table['h_' + column] + table['a_' + column]
    derived['home'] = (~table['neutral_site']).astype(float)
    derived['conference'] = table['conference_competition'].astype(float)
    table = pd.concat([table, pd.DataFrame(derived, index=table.index)], axis=1)
    pbp_cols = pbp_games[['game_id', 'home_id', 'away_id', 'home_score', 'away_score', 'espn_home_spread',
                          'espn_total', 'season']].drop_duplicates('game_id')
    lines = cfb_lines.closing_lines(pbp_cols[pbp_cols['season'] < cfb_lines.CFBD_FROM])
    if recent:
        lines = pd.concat([lines] + [cfb_lines.current_lines(s, refresh=refresh and s == seasons[-1]) for s in recent],
                          ignore_index=True)
    table = table.merge(lines.drop(columns=['season']), on='game_id', how='left')
    # a line is the home team's as its source sees it; flip it where the schedule has the teams the other way round
    swapped = table['line_home_id'].notna() & table['line_home_id'].ne(table['home_id'])
    for column in ('close_home_spread', 'pinnacle_home_spread', 'open_home_spread', 'espn_home_spread',
                   'dk_home_spread'):
        table.loc[swapped, column] = -table.loc[swapped, column]
    final = table['home_score'].notna() & table['away_score'].notna() & table['status'].astype(str).str.contains(
        'FINAL', case=False, na=False)
    table['margin'] = (table['home_score'] - table['away_score']).where(final)
    table['points'] = (table['home_score'] + table['away_score']).where(final)
    table['cover'] = table['margin'] + table['close_home_spread']
    table['over'] = table['points'] - table['close_total']
    table['regular'] = table['season_type'] == REGULAR_SEASON
    return table.sort_values(['start', 'game_id']).reset_index(drop=True)


def default_table(refresh=False):
    """The full research table, cached in data_cache (rebuilt when TABLE_VERSION changes or on `refresh`)."""
    path = os.path.join(CACHE_DIR, f'cfb_table_v{TABLE_VERSION}.pkl')
    if os.path.exists(path) and not refresh:
        return pd.read_pickle(path)
    table = game_table(refresh=refresh)
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
