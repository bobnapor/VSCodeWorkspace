"""Context features for the graph research (cfb_research_graph.py), from CollegeFootballData (one call per season per
endpoint, cached in data_cache).  Each is computed per team from what is known before kickoff, then entered home
minus away (d_*).  A team or game the API does not cover gets a blank (the model's training average).

PRIOR (what the market knows in August):
    talent      the mean recruiting-class points (247Sports composite) of the team's last four classes
    ret         the share of last season's offensive production (PPA) returning; ret_pass, the passing share alone
                (mostly the quarterback).  CollegeFootballData has it from 2014.
    *_e         each of those times early = max(0, 1 - games played / 8) (both teams' average): a prior should count
                most before the season's own games pile up.
TRAVEL (the visitor's trip, at the game's stadium; the home team's stadium when the API has no venue):
    travel      log(1 + km from the team's home stadium)
    tz_east     hours of clock moved east (an Oregon team at noon in Ohio); tz_west, moved west
    body_early  kickoff before 11 AM by the team's home clock; body_late, at or after 9 PM by it
    altitude    km the stadium sits above the team's home stadium, beyond the first 0.6 km (Laramie, Boulder, Provo)
SPOT (AP poll, the latest one published before the game; week 1 is the preseason poll):
    lookahead   the team's next game (within 9 days) is against a ranked team, and this week's opponent is unranked
    letdown     the team's last game was against a ranked team, and this week's opponent is unranked
    off_upset   the team won its last game outright as an underdog of 7+ points (closing line; FBS games only)
"""

import math
import os

import numpy as np
import pandas as pd

import cfb_cfbd
from cfb_common import CACHE_DIR

CONTEXT_VERSION = 1
FAMILIES = {
    'PRIOR': ['d_talent', 'd_ret', 'd_ret_pass', 'd_talent_e', 'd_ret_e', 'd_ret_pass_e'],
    'TRAVEL': ['d_travel', 'd_tz_east', 'd_tz_west', 'd_body_early', 'd_body_late', 'd_altitude'],
    'SPOT': ['d_lookahead', 'd_letdown', 'd_off_upset'],
}
EARLY_GAMES = 8
ALTITUDE_FREE_M = 600
LAST = pd.Timestamp('2100-01-01', tz='UTC')


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _places():
    """({team id: (lat, lon, elevation m, tz, school)}, {venue id: (lat, lon, elevation m, tz)})."""
    teams, venues = {}, {}
    for v in cfb_cfbd._get('/venues', {}, 'cfbd_venues.json'):
        venues[v['id']] = (_float(v.get('latitude')), _float(v.get('longitude')), _float(v.get('elevation')),
                           v.get('timezone'))
    for t in cfb_cfbd._get('/teams', {}, 'cfbd_teams_all.json'):
        loc = t.get('location') or {}
        place = (_float(loc.get('latitude')), _float(loc.get('longitude')), _float(loc.get('elevation')),
                 loc.get('timezone'))
        teams[t['id']] = place + (t.get('school'),)
        if loc.get('id') is not None:
            old = venues.get(loc['id'])
            # the team's own stadium: fill what the venue list lacks
            venues[loc['id']] = tuple(o if (o is not None and not (isinstance(o, float) and math.isnan(o))) else n
                                      for o, n in zip(old, place)) if old else place
    return teams, venues


def _games(season):
    out = []
    for kind in cfb_cfbd.SEASON_TYPES:
        for g in cfb_cfbd._get('/games', {'year': season, 'seasonType': kind}, f'cfbd_games_{season}_{kind}.json'):
            out.append(dict(g, kind=kind))
    return out


def _polls(season):
    """[(kind, week, {ranked team ids})] of the AP poll."""
    found = []
    for week in cfb_cfbd._get('/rankings', {'year': season}, f'cfbd_rankings_{season}.json'):
        for poll in week.get('polls') or []:
            if poll.get('poll') == 'AP Top 25':
                found.append((week.get('seasonType'), week.get('week'),
                              {r.get('teamId') for r in poll.get('ranks') or [] if r.get('teamId') is not None}))
    return found


def _ranked_before(polls, kind, week):
    """The ranked set of the latest AP poll published before a game of (kind, week)."""
    regular = sorted((w, s) for k, w, s in polls if k == 'regular')
    if not regular:
        return None
    if kind != 'regular':
        return regular[-1][1]
    before = [s for w, s in regular if w <= week]
    return before[-1] if before else None


def _haversine(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def _offset(tz, moment):
    if not tz or moment is None or pd.isna(moment):
        return float('nan')
    return moment.tz_convert(tz).utcoffset().total_seconds() / 3600


def _clock(tz, moment):
    if not tz or moment is None or pd.isna(moment):
        return float('nan')
    local = moment.tz_convert(tz)
    return local.hour + local.minute / 60


def team_features(table):
    """One row per (game_id, team_id) of `table`'s games with every per-team feature."""
    seasons = sorted(int(s) for s in table['season'].unique())
    teams, venues = _places()
    school_id = {v[4]: k for k, v in teams.items() if v[4]}
    lines = {g: (h, s) for g, h, s in zip(table['game_id'], table['home_id'], table['close_home_spread'])}
    wanted = set(table['game_id'])
    rows = []
    for season in seasons:
        classes = {}
        for year in range(season - 3, season + 1):
            for r in cfb_cfbd._get('/recruiting/teams', {'year': year}, f'cfbd_recruiting_{year}.json'):
                classes.setdefault(school_id.get(r.get('team')), []).append(_float(r.get('points')))
        talent = {t: np.nanmean(p) for t, p in classes.items() if t is not None}
        returning = {}
        for r in cfb_cfbd._get('/player/returning', {'year': season}, f'cfbd_returning_{season}.json'):
            returning[school_id.get(r.get('team'))] = (_float(r.get('percentPPA')), _float(r.get('percentPassingPPA')))
        polls = _polls(season)
        games = [g for g in _games(season) if g.get('homeId') is not None and g.get('awayId') is not None]
        for g in games:
            g['start'] = pd.Timestamp(g['startDate']) if g.get('startDate') else pd.NaT
        games.sort(key=lambda g: (g['start'] if not pd.isna(g['start']) else LAST, g['id']))
        by_team = {}
        for g in games:
            for t in (g['homeId'], g['awayId']):
                by_team.setdefault(t, []).append(g)
        for team, schedule in by_team.items():
            home_place = teams.get(team, (np.nan, np.nan, np.nan, None, None))
            for i, g in enumerate(schedule):
                if g['id'] not in wanted:
                    continue
                opp = g['awayId'] if g['homeId'] == team else g['homeId']
                venue = venues.get(g.get('venueId'))
                if venue is None and not g.get('neutralSite'):
                    venue = teams.get(g['homeId'], (np.nan, np.nan, np.nan, None))[:4]
                venue = venue or (np.nan, np.nan, np.nan, None)
                venue_tz = venue[3] or (teams.get(g['homeId'], (None,) * 4)[3] if not g.get('neutralSite') else None)
                start = g['start'] if not g.get('startTimeTBD') else pd.NaT
                travel = (math.log1p(_haversine(home_place, venue))
                          if not any(math.isnan(x) for x in (home_place[0], home_place[1], venue[0], venue[1]))
                          else np.nan)
                east = _offset(venue_tz, g['start']) - _offset(home_place[3], g['start'])
                body = _clock(home_place[3], start)
                climb = venue[2] - home_place[2]
                ranked_now = _ranked_before(polls, g['kind'], g['week'])
                opp_ranked = ranked_now is not None and opp in ranked_now
                lookahead = letdown = off_upset = 0.0
                if i + 1 < len(schedule):
                    nxt = schedule[i + 1]
                    nxt_opp = nxt['awayId'] if nxt['homeId'] == team else nxt['homeId']
                    soon = (not pd.isna(nxt['start']) and not pd.isna(g['start'])
                            and (nxt['start'] - g['start']).days <= 9)
                    lookahead = float(soon and ranked_now is not None and nxt_opp in ranked_now and not opp_ranked)
                if i > 0:
                    prev = schedule[i - 1]
                    prev_opp = prev['awayId'] if prev['homeId'] == team else prev['homeId']
                    ranked_then = _ranked_before(polls, prev['kind'], prev['week'])
                    letdown = float(ranked_then is not None and prev_opp in ranked_then and not opp_ranked)
                    line = lines.get(prev['id'])
                    pts = (prev.get('homePoints'), prev.get('awayPoints'))
                    if line and line[1] == line[1] and None not in pts:
                        spread = line[1] if line[0] == team else -line[1]
                        won = (pts[0] > pts[1]) if prev['homeId'] == team else (pts[1] > pts[0])
                        off_upset = float(spread >= 7 and won)
                ret = returning.get(team, (np.nan, np.nan))
                rows.append({'game_id': g['id'], 'team_id': team, 'talent': talent.get(team, np.nan),
                             'ret': ret[0], 'ret_pass': ret[1], 'travel': travel,
                             'tz_east': max(0.0, east) if east == east else np.nan,
                             'tz_west': max(0.0, -east) if east == east else np.nan,
                             'body_early': float(body < 11) if body == body else np.nan,
                             'body_late': float(body >= 21) if body == body else np.nan,
                             'altitude': max(0.0, climb - ALTITUDE_FREE_M) / 1000 if climb == climb else np.nan,
                             'lookahead': lookahead, 'letdown': letdown, 'off_upset': off_upset})
    return pd.DataFrame(rows)


def add_features(table):
    """`table` with every FAMILIES column (cached per CONTEXT_VERSION)."""
    path = os.path.join(CACHE_DIR, f'cfb_context_v{CONTEXT_VERSION}.pkl')
    if os.path.exists(path):
        per_team = pd.read_pickle(path)
    else:
        per_team = team_features(table)
        per_team.to_pickle(path)
    names = [c for c in per_team.columns if c not in ('game_id', 'team_id')]
    home = per_team.rename(columns={c: 'h_' + c for c in names}).rename(columns={'team_id': 'home_id'})
    away = per_team.rename(columns={c: 'a_' + c for c in names}).rename(columns={'team_id': 'away_id'})
    out = table.merge(home, on=['game_id', 'home_id'], how='left').merge(away, on=['game_id', 'away_id'], how='left')
    for c in names:
        out['d_' + c] = out['h_' + c] - out['a_' + c]
    early = (1 - (out['h_gp'] + out['a_gp']) / 2 / EARLY_GAMES).clip(lower=0)
    for c in ('talent', 'ret', 'ret_pass'):
        out[f'd_{c}_e'] = out['d_' + c] * early
    return out


def coverage(table):
    """Share of the table's regular-season games with each feature, by season (a check, printed by discovery)."""
    cols = [c for f in FAMILIES.values() for c in f]
    return table[table['regular']].groupby('season')[cols].apply(lambda x: x.notna().mean())
