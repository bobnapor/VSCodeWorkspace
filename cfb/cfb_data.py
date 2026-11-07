"""Free college football data, as sportsdataverse publishes ESPN's (the college counterpart of nflverse), cached in
cfb/data_cache.

  schedule      every game: week, season type, neutral site, conference game, both teams (ESPN ids) and the score
  groups        each team's subdivision (FBS, FCS, ...) and conference by season
  play-by-play  every play since 2004 with ESPN's expected points (EPA), success, explosive plays, havoc, line yards,
                drives and ESPN's own spread and total for the game.  A season is about 550 MB, so it is reduced to one
                row per team per game of counts and sums (team_games) and the raw file is deleted, as the NFL suite
                does with nflverse's play-by-play.  This 32-bit Python cannot hold a season's 507 columns at once, so the
                file is read PBP_CHUNK plays at a time.

Seasons are numbered by the year they start (2025 = fall 2025, bowls included).  The latest season is downloaded
again when asked (refresh); older ones come from the cache; a failed download falls back to the cached copy.  Teams
are keyed by ESPN's team id.
"""

import os
import time

import numpy as np
import pandas as pd
import requests

from cfb_common import CACHE_DIR

BASE_URL = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download'
FIRST_PBP_SEASON = 2004
PBP_CHUNK = 10000
PBP_VERSION = 2                 # bump when team_games' columns change, to rebuild every season's aggregate
COMPETITIVE_WP = (0.10, 0.90)   # plays with the offense's win probability inside this band are not garbage time
# drive results as ESPN labels them: 'MADE FG' in 2004, 'RUSHING TD', 'PASSING TD' and 'FG GOOD' to 2013, 'TD' and 'FG'
OFFENSE_TD = ('TD', 'RUSHING TD', 'PASSING TD')
FIELD_GOAL = ('FG', 'FG GOOD', 'MADE FG')
UNFINISHED_DRIVES = ('END OF HALF', 'END OF GAME', 'END OF 4TH QUARTER', 'NOT PROVIDED')

SCHEDULE_COLUMNS = ('game_id', 'season', 'week', 'season_type', 'game_date', 'neutral_site', 'conference_competition',
                    'home_id', 'away_id', 'home_team', 'away_team', 'home_abbreviation', 'away_abbreviation',
                    'home_score', 'away_score', 'venue', 'status')
GROUP_COLUMNS = ('season', 'team_id', 'team_name', 'subdivision_id', 'conference_id')
PBP_COLUMNS = ('game_id', 'season', 'week', 'seasonType', 'pos_team_id', 'homeTeamId', 'awayTeamId',
               'homeFinalScore', 'awayFinalScore', 'homeTeamSpread', 'overUnder', 'down', 'scrimmage_play',
               'kneel_down', 'pass', 'rush', 'sack', 'EPA', 'EPA_success', 'EPA_explosive', 'statYardage', 'havoc',
               'line_yards', 'stuffed_run', 'first_down_created', 'is_pos_team_turnover', 'wp_before', 'sp',
               'penalty_flag', 'penalty_declined', 'yds_penalty', 'penalty_team_id', 'drive.id', 'drive.result',
               'scoring_opp', 'start.yardsToEndzone')
GAME_COLUMNS = ['game_id', 'season', 'week', 'season_type', 'home_id', 'away_id', 'home_score', 'away_score',
                'espn_home_spread', 'espn_total']


def cached_download(url, filename, refresh=False):
    """The cached copy of `url` in data_cache (downloaded when missing or when `refresh`, streamed to disk); a failed
    download falls back to the cached copy when there is one."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if refresh or not os.path.exists(path):
        temporary = path + '.download'
        try:
            with requests.get(url, stream=True, timeout=300, headers={'User-Agent': 'Mozilla/5.0'}) as response:
                response.raise_for_status()
                with open(temporary, 'wb') as f:
                    for block in response.iter_content(1 << 20):
                        f.write(block)
        except requests.RequestException:
            if os.path.exists(temporary):
                os.remove(temporary)
            if os.path.exists(path):
                return path
            raise
        os.replace(temporary, path)
    return path


def _read_seasons(tag, name, seasons, columns, refresh):
    """One CSV a season from a release `tag`, only `columns`; `refresh` downloads the latest season again."""
    seasons = sorted(int(s) for s in seasons)
    wanted, frames = set(columns), []
    for season in seasons:
        filename = name.format(season=season)
        try:
            path = cached_download(f'{BASE_URL}/{tag}/{filename}', filename, bool(refresh) and season == seasons[-1])
        except requests.RequestException:
            continue
        frames.append(pd.read_csv(path, low_memory=False, usecols=lambda c: c in wanted))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns))
    return frame.reindex(columns=[c for c in columns if c in frame.columns])


def load_schedule(seasons, refresh=False):
    """Every game of `seasons`, one row each."""
    games = _read_seasons('espn_cfb_schedules', 'cfb_schedule_{season}.csv', seasons, SCHEDULE_COLUMNS, refresh)
    games = games.drop_duplicates('game_id')
    games['start'] = pd.to_datetime(games['game_date'], utc=True, errors='coerce')
    games['neutral_site'] = games['neutral_site'].fillna(False).astype(bool)
    games['conference_competition'] = games['conference_competition'].fillna(False).astype(bool)
    return games.sort_values(['start', 'game_id']).reset_index(drop=True)


def load_groups(seasons, refresh=False):
    """Each team's subdivision ('cfb:fbs', 'cfb:fcs', ...) and conference, one row per (season, team_id)."""
    groups = _read_seasons('cfb_groups', 'cfb_team_group_seasons_{season}.csv', seasons, GROUP_COLUMNS, refresh)
    return groups.drop_duplicates(['season', 'team_id'])


# ---------------------------------------------------------------- play-by-play, reduced to team-games

def _flag(column):
    """A boolean column as read from CSV (True/False, 'TRUE', 1.0 or blank) as 0/1 floats."""
    if column.dtype == bool:
        return column.astype(float)
    return column.astype(str).str.lower().isin(('true', '1', '1.0')).astype(float)


def _offense(chunk):
    """Sums and counts of one chunk's scrimmage plays, per (game_id, team_id) of the offense."""
    scrimmage = (_flag(chunk['scrimmage_play']) > 0) & (_flag(chunk['kneel_down']) == 0) & chunk['EPA'].notna() \
        & chunk['pos_team_id'].notna()
    p = chunk[scrimmage]
    epa, success, explosive = p['EPA'].astype(float), _flag(p['EPA_success']), _flag(p['EPA_explosive'])
    is_pass = _flag(p['pass'])
    is_rush = _flag(p['rush']) * (1 - is_pass)
    wp = p['wp_before'].astype(float)
    competitive = ((wp >= COMPETITIVE_WP[0]) & (wp <= COMPETITIVE_WP[1])).astype(float)
    down = p['down'].astype(float)
    early, late = down.isin((1, 2)).astype(float), down.isin((3, 4)).astype(float)
    frame = pd.DataFrame({
        'game_id': p['game_id'], 'team_id': p['pos_team_id'],
        'n_plays': 1.0, 'epa': epa, 'success': success, 'explosive': explosive,
        'yards': p['statYardage'].astype(float).fillna(0),
        'n_pass': is_pass, 'pass_epa': epa * is_pass, 'pass_success': success * is_pass,
        'n_rush': is_rush, 'rush_epa': epa * is_rush, 'rush_success': success * is_rush,
        'line_yards': p['line_yards'].astype(float).fillna(0) * is_rush, 'stuffed': _flag(p['stuffed_run']) * is_rush,
        'n_comp': competitive, 'comp_epa': epa * competitive, 'comp_success': success * competitive,
        'n_early': early, 'early_epa': epa * early, 'early_success': success * early,
        'n_late': late, 'late_success': success * late,
        'sacks': _flag(p['sack']), 'havoc': _flag(p['havoc']), 'first_downs': _flag(p['first_down_created']),
        'turnovers': _flag(p['is_pos_team_turnover']),
    })
    return frame.groupby(['game_id', 'team_id']).sum()


def _special(chunk):
    """Special-teams EPA per (game_id, team_id) of the team with the ball (kicking or returning, as ESPN credits it)."""
    sp = chunk[(_flag(chunk['sp']) > 0) & chunk['EPA'].notna() & chunk['pos_team_id'].notna()]
    return pd.DataFrame({'game_id': sp['game_id'], 'team_id': sp['pos_team_id'], 'st_epa': sp['EPA'].astype(float)}) \
        .groupby(['game_id', 'team_id']).sum()


def _penalties(chunk):
    """Accepted penalties and their yards per (game_id, team_id) of the penalized team."""
    pen = chunk[(_flag(chunk['penalty_flag']) > 0) & (_flag(chunk['penalty_declined']) == 0)
                & chunk['penalty_team_id'].notna()]
    return pd.DataFrame({'game_id': pen['game_id'], 'team_id': pen['penalty_team_id'], 'penalties': 1.0,
                         'penalty_yards': pen['yds_penalty'].astype(float).abs().fillna(0)}) \
        .groupby(['game_id', 'team_id']).sum()


def _drives(chunk):
    """One row per (game_id, team_id, drive): its result, whether it reached a scoring opportunity, where it started
    (yards to the end zone) and its scrimmage plays."""
    d = chunk[chunk['drive.id'].notna() & chunk['pos_team_id'].notna()]
    return pd.DataFrame({'game_id': d['game_id'], 'team_id': d['pos_team_id'], 'drive': d['drive.id'],
                         'result': d['drive.result'], 'opportunity': _flag(d['scoring_opp']),
                         'start': d['start.yardsToEndzone'].astype(float), 'plays': _flag(d['scrimmage_play'])}) \
        .groupby(['game_id', 'team_id', 'drive'], sort=False) \
        .agg(result=('result', 'first'), opportunity=('opportunity', 'max'), start=('start', 'first'),
             plays=('plays', 'sum'))


def _drive_sums(drives):
    """Per team-game drive totals: drives that ended in a result (not the clock), points from touchdowns (7) and field
    goals (3), scoring opportunities (reaching the opponent's 40) and the points on them, and the summed starting field
    position (yards to the end zone)."""
    d = drives.groupby(level=['game_id', 'team_id', 'drive'], sort=False) \
        .agg(result=('result', 'first'), opportunity=('opportunity', 'max'), start=('start', 'first'),
             plays=('plays', 'sum')).reset_index()
    d = d[(d['plays'] > 0) & d['result'].notna()]
    result = d['result'].astype(str).str.strip().str.upper()
    d = d[~result.isin(UNFINISHED_DRIVES)]
    result = result[d.index]
    points = np.where(result.isin(OFFENSE_TD), 7.0, np.where(result.isin(FIELD_GOAL), 3.0, 0.0))
    frame = pd.DataFrame({'game_id': d['game_id'], 'team_id': d['team_id'], 'n_drives': 1.0, 'drive_points': points,
                          'n_opportunities': d['opportunity'], 'opportunity_points': points * d['opportunity'],
                          'start_sum': d['start'].fillna(d['start'].mean())})
    return frame.groupby(['game_id', 'team_id']).sum()


def aggregate_pbp_file(path):
    """(team-game sums, one row per game) from one season's play-by-play CSV, read PBP_CHUNK plays at a time."""
    header = pd.read_csv(path, nrows=0).columns
    use = [c for c in PBP_COLUMNS if c in header]
    offense, special, penalties, drives, games = [], [], [], [], []
    for chunk in pd.read_csv(path, usecols=use, low_memory=False, chunksize=PBP_CHUNK):
        for column in PBP_COLUMNS:
            if column not in chunk:
                chunk[column] = np.nan
        offense.append(_offense(chunk))
        special.append(_special(chunk))
        penalties.append(_penalties(chunk))
        drives.append(_drives(chunk))
        games.append(chunk.drop_duplicates('game_id')[['game_id', 'season', 'week', 'seasonType', 'homeTeamId',
                                                       'awayTeamId', 'homeFinalScore', 'awayFinalScore',
                                                       'homeTeamSpread', 'overUnder']])
    sums = pd.concat(offense).groupby(level=[0, 1]).sum()
    sums = sums.join(pd.concat(special).groupby(level=[0, 1]).sum(), how='left')
    sums = sums.join(pd.concat(penalties).groupby(level=[0, 1]).sum(), how='left')
    sums = sums.join(_drive_sums(pd.concat(drives)), how='left')
    sums = sums.fillna({'st_epa': 0.0, 'penalties': 0.0, 'penalty_yards': 0.0}).reset_index()
    games = pd.concat(games).drop_duplicates('game_id')
    # ESPN's homeTeamSpread is positive when the home team is favored; the suite's home spread is negative then
    games = games.rename(columns={'seasonType': 'season_type', 'homeTeamId': 'home_id', 'awayTeamId': 'away_id',
                                  'homeFinalScore': 'home_score', 'awayFinalScore': 'away_score',
                                  'overUnder': 'espn_total'})
    games['espn_home_spread'] = -games.pop('homeTeamSpread')
    return sums, games[GAME_COLUMNS]


def _aggregate_paths(season):
    return (os.path.join(CACHE_DIR, f'pbp_team_{season}_v{PBP_VERSION}.csv'),
            os.path.join(CACHE_DIR, f'pbp_games_{season}_v{PBP_VERSION}.csv'))


def build_season(season, refresh=False, keep_raw=False):
    """Download one season's play-by-play, reduce it to team-games and delete the raw file (unless `keep_raw`)."""
    team_path, games_path = _aggregate_paths(season)
    if not refresh and os.path.exists(team_path) and os.path.exists(games_path):
        return
    filename = f'play_by_play_{season}.csv'
    began = time.time()
    raw = cached_download(f'{BASE_URL}/espn_cfb_pbp/{filename}', filename, refresh)
    downloaded = time.time()
    sums, games = aggregate_pbp_file(raw)
    sums.to_csv(team_path, index=False)
    games.to_csv(games_path, index=False)
    if not keep_raw:
        os.remove(raw)
    print(f'{season}: {len(games):,} games, {len(sums):,} team-games (download {downloaded - began:.0f}s, '
          f'aggregate {time.time() - downloaded:.0f}s)', flush=True)


def load_team_games(seasons, refresh_latest=False):
    """(team-game sums, games) for `seasons` from the reduced play-by-play, building any season missing from the
    cache; `refresh_latest` rebuilds the last season (the one being played)."""
    seasons = sorted(int(s) for s in seasons)
    sums, games = [], []
    for season in seasons:
        build_season(season, refresh=refresh_latest and season == seasons[-1])
        team_path, games_path = _aggregate_paths(season)
        sums.append(pd.read_csv(team_path))
        games.append(pd.read_csv(games_path))
    return pd.concat(sums, ignore_index=True), pd.concat(games, ignore_index=True)


if __name__ == '__main__':
    import sys
    from cfb_common import current_season
    # python cfb_data.py [first [last]]: build the reduced play-by-play for those seasons (default all)
    first = int(sys.argv[1]) if len(sys.argv) > 1 else FIRST_PBP_SEASON
    last = int(sys.argv[2]) if len(sys.argv) > 2 else current_season()
    for year in range(first, last + 1):
        build_season(year, refresh=year == current_season())
