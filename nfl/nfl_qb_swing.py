"""The quarterback swing behind the pick rule's QB skip (nfl_signal.QB_DOWNGRADE).

A team's ratings (nfl_lab.rate_teams) are a weighted average of its earlier games: each game's weight decays 0.90 per
later game, and everything is multiplied by 0.6 at a new season.  With the same weights over the starters of those
games, every team carries an embedded quarterback: the weighted average of those starters' ratings (the projection
model's quarterback rating, nfl_injury_features.build_qb_quality, every rating as of the day of the game).  The swing
is today's starter's rating minus the embedded one, in passing EPA per dropback: about 0 when the starter played the
games behind the ratings, -0.10 to -0.30 the week a backup replaces a starter, nonzero after an offseason change too.

nfl_research_qb.py (pre-registered, October 21, 2026) found the pick model's underdog picks on a team with a swing of
-0.05 or worse near break-even (51.6%, against 55.5% with no downgrade on either side, lower in every era), while
backing the starter out of the model did not help, so the pick rule skips those picks.

Played games use the starter who played.  The week being predicted uses projected starters (projected_starters):
the schedule's listed starter when he is available, else the last game's starter when he is, else unknown, counted
at replacement level (so a pick on that team is skipped).  A quarterback on a reserve list in the latest weekly
roster or listed Out or Doubtful on the week's injury report is unavailable.
"""

import numpy as np
import pandas as pd

from nfl_injury_features import (OUT_STATUSES, REPLACEMENT_QB_EPA, build_qb_quality, normalize_team_column, qb_rating_before,
                                 reserve_list, week_key)

DECAY, CARRY = 0.90, 0.6   # nfl_lab.rate_teams defaults, as in production
FIRST_SEASON = 1999
QB_COLUMNS = ['player_id', 'position', 'season', 'week', 'season_type', 'attempts', 'sacks_suffered', 'passing_epa']


def projected_starters(table, season, week, injuries=None, rosters=None):
    """{(game_id, 'home' or 'away'): (qb_id or None, name, how)} for the unplayed games of `season` `week`, where `how` is
    'listed', 'last game' or 'unknown'.  `table` has the schedule's starters (listed ones for unplayed games)."""
    reserve = reserve_list(rosters, season, week)
    out = set()
    if injuries is not None and not injuries.empty:
        inj = injuries[(injuries['game_type'] == 'REG') & (injuries['season'] == season) & (injuries['week'] == week)]
        inj = normalize_team_column(inj[(inj['position'] == 'QB') & inj['report_status'].isin(OUT_STATUSES)], 'team')
        out = set(zip(inj['team'], inj['gsis_id']))

    def available(team, qb):
        return isinstance(qb, str) and (team, qb) not in reserve and (team, qb) not in out

    starters, last = {}, {}
    for g in table.sort_values(['gameday', 'week']).itertuples():
        if g.season == season and g.week == week and pd.isna(g.result):
            for side, team, qb, name in (('home', g.home_team, g.home_qb_id, g.home_qb_name),
                                         ('away', g.away_team, g.away_qb_id, g.away_qb_name)):
                previous = last.get(team)
                if available(team, qb):
                    starters[(g.game_id, side)] = (qb, name, 'listed')
                elif previous and available(team, previous[0]):
                    starters[(g.game_id, side)] = (previous[0], previous[1], 'last game')
                else:
                    starters[(g.game_id, side)] = (None, None, 'unknown')
        elif pd.notna(g.result):
            for team, qb, name in ((g.home_team, g.home_qb_id, g.home_qb_name), (g.away_team, g.away_qb_id, g.away_qb_name)):
                if isinstance(qb, str):
                    last[team] = (qb, name)
    return starters


def swings(table, history, starters=None):
    """game_id, qb_swing_home, qb_swing_away for every game in `table`; `starters` (projected_starters) sets the starter
    of the games it covers, None meaning unknown (counted at replacement level)."""
    games = table.sort_values(['gameday', 'week']).reset_index(drop=True)
    keys = [week_key(s, w) for s, w in zip(games['season'], games['week'])]
    state = {}   # team -> [{qb: weight}, season]
    result = {'home': np.full(len(games), np.nan), 'away': np.full(len(games), np.nan)}
    for i, g in enumerate(games.itertuples()):
        for side, team, qb in (('home', g.home_team, g.home_qb_id), ('away', g.away_team, g.away_qb_id)):
            if starters and (g.game_id, side) in starters:
                qb = starters[(g.game_id, side)][0]
            weights, season = state.setdefault(team, [{}, None])
            if season is not None and season != g.season:
                weights = {q: w * CARRY for q, w in weights.items()}
                state[team][0] = weights
            state[team][1] = g.season
            total = sum(weights.values())
            if total <= 0:
                result[side][i] = 0.0
                continue
            embedded = sum(w * qb_rating_before(history, q, keys[i]) for q, w in weights.items()) / total
            today = qb_rating_before(history, qb, keys[i]) if isinstance(qb, str) else REPLACEMENT_QB_EPA
            result[side][i] = today - embedded
        if pd.notna(g.result):
            for team, qb in ((g.home_team, g.home_qb_id), (g.away_team, g.away_qb_id)):
                weights = {q: w * DECAY for q, w in state[team][0].items()}
                if isinstance(qb, str):
                    weights[qb] = weights.get(qb, 0.0) + 1.0
                state[team][0] = weights
    return games[['game_id']].assign(qb_swing_home=result['home'], qb_swing_away=result['away'])


def with_swings(table, season, week, refresh=False):
    """(`table` with qb_swing_home / qb_swing_away, projected starters of `season` `week`), from nflverse player stats,
    injury reports and weekly rosters (cached; `refresh` re-downloads the latest season)."""
    from nflverse_data import load_injuries, load_player_weeks, load_weekly_rosters
    history = build_qb_quality(load_player_weeks(FIRST_SEASON, season, refresh=refresh, columns=QB_COLUMNS))
    try:
        injuries = load_injuries(season, season, refresh=refresh)
    except Exception:
        injuries = None
    try:
        rosters = load_weekly_rosters(season, season, refresh=refresh)
    except Exception:
        rosters = None
    starters = projected_starters(table, season, week, injuries, rosters)
    return table.merge(swings(table, history, starters), on='game_id', how='left'), starters
