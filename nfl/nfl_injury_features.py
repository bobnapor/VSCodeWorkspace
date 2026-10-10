"""Pre-game injury features from the official weekly injury reports.

Everything here is known before kickoff: the reports are published during the week and the quarterback
quality ratings only use games played before the one being predicted.
"""

import bisect

import numpy as np
import pandas as pd

TEAM_RENAMES = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA', 'JAC': 'JAX', 'WSH': 'WAS'}
REPLACEMENT_QB_EPA = -0.08       # per dropback, a typical backup
STARTER_PRIOR_DROPBACKS = 150.0  # shrinks a young quarterback's rating toward replacement level
QB_GAME_DECAY = 0.97             # older games count a little less
OUT_STATUSES = ('Out', 'Doubtful')

POSITION_GROUPS = {
    'ol': ('T', 'G', 'C', 'OL', 'OT', 'OG'),
    'dl': ('DE', 'DT', 'NT', 'DL'),
    'lb': ('LB', 'OLB', 'ILB', 'MLB'),
    'db': ('CB', 'S', 'SS', 'FS', 'DB'),
    'skill': ('RB', 'FB', 'HB', 'WR', 'TE'),
}


def normalize_team_column(frame, column):
    frame = frame.copy()
    frame[column] = frame[column].replace(TEAM_RENAMES)
    return frame


def week_key(season, week):
    return int(season) * 100 + int(week)


def build_qb_quality(player_weeks):
    """Per quarterback, a time-ordered list of (game key, rating after that game).

    The rating is exponentially weighted passing EPA per dropback shrunk toward replacement level.
    """
    qb = player_weeks[(player_weeks['position'] == 'QB') & (player_weeks['season_type'] == 'REG')].copy()
    qb['dropbacks'] = qb['attempts'].fillna(0) + qb['sacks_suffered'].fillna(0)
    qb = qb[qb['dropbacks'] >= 1].sort_values(['season', 'week'])
    history = {}
    state = {}
    for row in qb.itertuples():
        epa_sum, db_sum = state.get(row.player_id, (0.0, 0.0))
        epa_sum = epa_sum * QB_GAME_DECAY + (0.0 if pd.isna(row.passing_epa) else row.passing_epa)
        db_sum = db_sum * QB_GAME_DECAY + row.dropbacks
        state[row.player_id] = (epa_sum, db_sum)
        rating = (epa_sum + STARTER_PRIOR_DROPBACKS * REPLACEMENT_QB_EPA) / (db_sum + STARTER_PRIOR_DROPBACKS)
        history.setdefault(row.player_id, ([], []))
        history[row.player_id][0].append(week_key(row.season, row.week))
        history[row.player_id][1].append(rating)
    return history


def qb_rating_before(history, qb_id, key):
    """The quarterback's rating from games strictly before `key`, or replacement level if he has none."""
    if qb_id not in history:
        return REPLACEMENT_QB_EPA
    keys, ratings = history[qb_id]
    i = bisect.bisect_left(keys, key)
    return ratings[i - 1] if i else REPLACEMENT_QB_EPA


def build_injury_features(games, injuries, player_weeks):
    """One row per regular-season game with injury features for both teams, keyed by game_id."""
    games = games[games['game_type'] == 'REG'].copy()
    history = build_qb_quality(player_weeks)

    inj = normalize_team_column(injuries[injuries['game_type'] == 'REG'], 'team')
    out_qbs = set()
    questionable_qbs = set()
    qb_inj = inj[inj['position'] == 'QB']
    for row in qb_inj.itertuples():
        key = (int(row.season), int(row.week), row.team, row.gsis_id)
        if row.report_status in OUT_STATUSES:
            out_qbs.add(key)
        elif row.report_status == 'Questionable':
            questionable_qbs.add(key)

    out_counts = {}
    flagged = inj[inj['report_status'].isin(OUT_STATUSES)]
    for group, positions in POSITION_GROUPS.items():
        counts = flagged[flagged['position'].isin(positions)].groupby(['season', 'week', 'team']).size()
        out_counts[group] = counts.to_dict()

    # each team's previous starting quarterback, looking across seasons
    long = pd.concat([
        games[['game_id', 'season', 'week', 'gameday', 'home_team', 'home_qb_id']].rename(columns={'home_team': 'team', 'home_qb_id': 'qb'}),
        games[['game_id', 'season', 'week', 'gameday', 'away_team', 'away_qb_id']].rename(columns={'away_team': 'team', 'away_qb_id': 'qb'}),
    ]).sort_values(['team', 'gameday'])
    long['prev_qb'] = long.groupby('team')['qb'].shift(1)
    prev_qb = {(r.game_id, r.team): r.prev_qb for r in long.itertuples()}

    rows = []
    for game in games.itertuples():
        row = {'game_id': game.game_id}
        key = week_key(game.season, game.week)
        for side, team in (('h', game.home_team), ('a', game.away_team)):
            starter = prev_qb.get((game.game_id, team))
            starter = starter if isinstance(starter, str) else None
            rating = qb_rating_before(history, starter, key) if starter else REPLACEMENT_QB_EPA
            lookup = (int(game.season), int(game.week), team, starter)
            out = lookup in out_qbs
            questionable = lookup in questionable_qbs
            row[f'{side}_qb_rating'] = rating
            row[f'{side}_qb_out'] = int(out)
            row[f'{side}_qb_questionable'] = int(questionable)
            # points-scale cost: how much worse the team is without its usual starter
            weight = 1.0 if out else (0.35 if questionable else 0.0)
            row[f'{side}_qb_cost'] = weight * (rating - REPLACEMENT_QB_EPA)
            for group in POSITION_GROUPS:
                row[f'{side}_out_{group}'] = out_counts[group].get((int(game.season), int(game.week), team), 0)
        rows.append(row)
    features = pd.DataFrame(rows)
    features['qb_rating_diff'] = features['h_qb_rating'] - features['a_qb_rating']
    features['qb_injury_edge'] = features['a_qb_cost'] - features['h_qb_cost']  # > 0 favors the home team
    features['qb_out_any'] = ((features['h_qb_out'] + features['a_qb_out']) > 0).astype(int)
    for group in POSITION_GROUPS:
        features[f'out_{group}_diff'] = features[f'a_out_{group}'] - features[f'h_out_{group}']  # > 0 favors home
    return features


TEAM_FULL_NAMES = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys', 'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'KC': 'Kansas City Chiefs',
    'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers', 'LA': 'Los Angeles Rams', 'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings', 'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers', 'SF': 'San Francisco 49ers',
    'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders',
}


def qb_injury_flags(games, injuries, season, week):
    """Quarterback status notes for one week's games: {(home team, away team): text}, full team names.

    Flags a team's previous starter if the injury report lists him Out, Doubtful or Questionable, and names
    the starter the schedule expects when that isn't the previous starter. Games with nothing to flag get no entry.
    """
    games = games[games['game_type'] == 'REG'].copy()
    for column in ('home_team', 'away_team'):
        games[column] = games[column].replace(TEAM_RENAMES)
    long = pd.concat([
        games[['game_id', 'gameday', 'home_team', 'home_qb_id', 'home_qb_name']].rename(
            columns={'home_team': 'team', 'home_qb_id': 'qb', 'home_qb_name': 'qb_name'}),
        games[['game_id', 'gameday', 'away_team', 'away_qb_id', 'away_qb_name']].rename(
            columns={'away_team': 'team', 'away_qb_id': 'qb', 'away_qb_name': 'qb_name'}),
    ]).sort_values(['team', 'gameday'])
    long['prev_qb'] = long.groupby('team')['qb'].shift(1)
    long['prev_qb_name'] = long.groupby('team')['qb_name'].shift(1)
    previous = {(r.game_id, r.team): r for r in long.itertuples()}

    inj = injuries[(injuries['game_type'] == 'REG') & (injuries['season'] == season) & (injuries['week'] == week)]
    inj = normalize_team_column(inj, 'team')
    qb_status = {}
    for row in inj[inj['position'] == 'QB'].itertuples():
        if row.report_status in ('Out', 'Doubtful', 'Questionable'):
            qb_status[(row.team, row.gsis_id)] = row.report_status

    flags = {}
    for game in games[(games['season'] == season) & (games['week'] == week)].itertuples():
        notes = []
        for team in (game.home_team, game.away_team):
            row = previous.get((game.game_id, team))
            if row is None:
                continue
            status = qb_status.get((team, row.prev_qb)) if isinstance(row.prev_qb, str) else None
            if status:
                notes.append(f'{team}: {row.prev_qb_name} {status.upper()}')
            if isinstance(row.qb, str) and isinstance(row.prev_qb, str) and row.qb != row.prev_qb:
                notes.append(f'{team}: expected starter {row.qb_name} (last game {row.prev_qb_name})')
        if notes:
            flags[(TEAM_FULL_NAMES.get(game.home_team, game.home_team), TEAM_FULL_NAMES.get(game.away_team, game.away_team))] = '; '.join(notes)
    return flags

