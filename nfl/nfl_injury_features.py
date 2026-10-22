"""Pre-game injury features from the official weekly injury reports.

Everything here is known before kickoff: the reports are published during the week and the quarterback
quality ratings only use games played before the one being predicted.  Players on injured reserve drop off the
reports, so for games not yet played the weekly rosters' reserve lists fill that gap (reserve_list).
"""

import bisect
import re

import numpy as np
import pandas as pd

TEAM_RENAMES = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA', 'JAC': 'JAX', 'WSH': 'WAS'}
REPLACEMENT_QB_EPA = -0.08       # per dropback, a typical backup
STARTER_PRIOR_DROPBACKS = 150.0  # shrinks a young quarterback's rating toward replacement level
QB_GAME_DECAY = 0.97             # older games count a little less
OUT_STATUSES = ('Out', 'Doubtful')
PRACTICE_WORDS = {'Did Not Participate In Practice': 'did not practice', 'Limited Participation in Practice': 'limited in practice',
                  'Full Participation in Practice': 'full practice'}
NAME_SUFFIXES = ('jr', 'sr', 'ii', 'iii', 'iv', 'v')

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


def reserve_list(rosters, season, week=None):
    """{(team, gsis_id): 'injured reserve' or 'a reserve list'} from the latest nflverse weekly roster of `season` (up to
    `week` when given).  A player placed on injured reserve drops off the weekly injury report, so without this a
    starter whose season just ended looks available."""
    if rosters is None or rosters.empty:
        return {}
    r = rosters[rosters['season'] == season]
    if week is not None:
        r = r[r['week'] <= week]
    if r.empty:
        return {}
    latest = normalize_team_column(r[r['week'] == r['week'].max()], 'team')
    on_reserve = latest[(latest['status'] == 'RES') & latest['gsis_id'].notna()]
    return {(row.team, row.gsis_id): 'injured reserve' if row.status_description_abbr == 'R01' else 'a reserve list'
            for row in on_reserve.itertuples()}


def build_injury_features(games, injuries, player_weeks, reserve=None):
    """One row per regular-season game with injury features for both teams, keyed by game_id.

    `reserve` (from reserve_list) marks a team's previous starter out of games not yet played when he is on a reserve
    list; played games are left as the injury reports had them, so the training data does not change."""
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
            out = lookup in out_qbs or bool(reserve and pd.isna(game.result) and (team, starter) in reserve)
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


def qb_injury_flags(games, injuries, season, week, rosters=None, live=None):
    """Quarterback status notes for one week's games: {(home team, away team): text}, full team names.

    Three quarterbacks per team: the last game's starter (the one the projection uses), the starter the schedule lists
    (nflverse's guess, which can lag the news) and the usual starter (most starts in this season's games so far, ties
    going to the one who started first).  The
    word on each comes from the latest weekly roster (`rosters`, nflverse_data.load_weekly_rosters: injured reserve,
    which drops a player off the injury report), this week's injury report (the designation, else practice
    participation) and ESPN's live injury list (`live`, nfl_sheet.fetch_espn_injuries: news before the reports).  Notes:
    the last starter's status; a different listed starter, shown as out instead when he is on a reserve list or listed
    Out or Doubtful; and a usual starter who neither started the last game nor is listed, with how long he has been
    out, when anything is reported on him.  Games with nothing to note get no entry.
    """
    games = games[games['game_type'] == 'REG'].copy()
    for column in ('home_team', 'away_team'):
        games[column] = games[column].replace(TEAM_RENAMES)
    long = pd.concat([
        games[['game_id', 'season', 'gameday', 'result', 'home_team', 'home_qb_id', 'home_qb_name']].rename(
            columns={'home_team': 'team', 'home_qb_id': 'qb', 'home_qb_name': 'qb_name'}),
        games[['game_id', 'season', 'gameday', 'result', 'away_team', 'away_qb_id', 'away_qb_name']].rename(
            columns={'away_team': 'team', 'away_qb_id': 'qb', 'away_qb_name': 'qb_name'}),
    ]).sort_values(['team', 'gameday'])
    long['prev_qb'] = long.groupby('team')['qb'].shift(1)
    long['prev_qb_name'] = long.groupby('team')['qb_name'].shift(1)
    previous = {(r.game_id, r.team): r for r in long.itertuples()}
    played = long[(long['season'] == season) & long['result'].notna() & long['qb'].notna()]

    inj = injuries[(injuries['game_type'] == 'REG') & (injuries['season'] == season) & (injuries['week'] == week)]
    inj = normalize_team_column(inj, 'team')
    qb_status, practice = {}, {}
    for row in inj[inj['position'] == 'QB'].itertuples():
        if row.report_status in ('Out', 'Doubtful', 'Questionable'):
            qb_status[(row.team, row.gsis_id)] = row.report_status
        if getattr(row, 'practice_status', None) in PRACTICE_WORDS:
            practice[(row.team, row.gsis_id)] = PRACTICE_WORDS[row.practice_status]
    reserve = reserve_list(rosters, season, week)

    def word_on(team, qb, name, returning=False):
        """(official, ESPN) words on one quarterback: a reserve list, else this week's designation or practice participation
        (for a starter only a missed practice; any practice for a usual starter on his way back), and ESPN's latest when
        not Active."""
        if (team, qb) in reserve:
            return f'ON {reserve[(team, qb)].upper()}', ''
        official = qb_status[(team, qb)].upper() if (team, qb) in qb_status else practice.get((team, qb), '')
        if not returning and official in ('limited in practice', 'full practice'):
            official = ''
        news, espn = (live or {}).get((team, name_key(name))), ''
        if news and news[0] and news[0] != 'Active':
            status, injury, comment, when = news
            comment = re.sub(r',[^,]*\breports?\.?$', '.', str(comment or '').strip())
            espn = (f'(ESPN {_espn_date(when)}, {status.lower()}: {_clip(comment)})' if comment
                    else f'(ESPN {_espn_date(when)}, {status.lower()}' + (f', {injury.lower()})' if injury else ')'))
        return official, espn

    def said(*words):
        return ' '.join(w for w in words if w)

    def usual_starter(team, gameday):
        """(id, name, games since his last start) for the most starts in the team's games so far this season, ties going
        to the one who started first (the season's own starter), or None."""
        earlier = played[(played['team'] == team) & (played['gameday'] < gameday)]
        if earlier.empty:
            return None
        order = list(earlier['qb'])
        last = {qb: i for i, qb in enumerate(order)}
        best = max(last, key=lambda q: (order.count(q), -order.index(q)))
        return best, earlier['qb_name'].iloc[last[best]], len(order) - 1 - last[best]

    flags = {}
    for game in games[(games['season'] == season) & (games['week'] == week)].itertuples():
        notes = []
        for team in (game.home_team, game.away_team):
            row = previous.get((game.game_id, team))
            if row is None:
                continue
            prev = row.prev_qb if isinstance(row.prev_qb, str) else None
            listed = row.qb if isinstance(row.qb, str) else None
            if prev:
                words = said(*word_on(team, prev, row.prev_qb_name))
                if (team, prev) in reserve:
                    notes.append(f'{team}: {row.prev_qb_name} {words} (started the last game; the projection counts him out)')
                elif words:
                    notes.append(f'{team}: {row.prev_qb_name} {words}')
            if listed and prev and listed != prev:
                words = said(*word_on(team, listed, row.qb_name))
                if (team, listed) in reserve or qb_status.get((team, listed)) in OUT_STATUSES:
                    notes.append(f'{team}: {row.qb_name} {words}, though the schedule lists him; '
                                 f'{row.prev_qb_name} started the last game and is the one the projection uses')
                else:
                    notes.append(f'{team}: expected starter {said(row.qb_name, words)} (last game {row.prev_qb_name})')
            usual = usual_starter(team, game.gameday)
            if usual and usual[0] not in (prev, listed):
                qb, name, missed = usual
                official, espn = word_on(team, qb, name, returning=True)
                # with ESPN's list at hand, a usual starter nobody reports on (benched, or healthy and not starting) is left out
                if official or espn or not live:
                    notes.append(f'{team}: usual starter {name} has missed the last {f"{missed} games" if missed > 1 else "game"}'
                                 + (f', {official}' if official else '') + (f' {espn}' if espn else '')
                                 + (f'; the projection uses {row.prev_qb_name}' if prev else ''))
        if notes:
            flags[(TEAM_FULL_NAMES.get(game.home_team, game.home_team), TEAM_FULL_NAMES.get(game.away_team, game.away_team))] = '; '.join(notes)
    return flags


def name_key(name):
    """A name in lower case without punctuation or suffixes, to match ESPN's player names to nflverse's."""
    words = re.sub(r'[^a-z ]', '', str(name).lower()).split()
    return ' '.join(w for w in words if w not in NAME_SUFFIXES)


def _espn_date(stamp):
    try:
        return pd.Timestamp(stamp).tz_convert('America/New_York').strftime('%b %d').replace(' 0', ' ')
    except (TypeError, ValueError):
        return ''


def _clip(text, limit=150):
    return text if len(text) <= limit else text[:limit].rsplit(' ', 1)[0] + '...'

