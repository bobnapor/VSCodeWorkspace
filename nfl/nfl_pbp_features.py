"""Per-team, per-game offensive measures built from nflverse play-by-play data, 1999 on.

Each season's play-by-play file (about 100 MB) is downloaded, reduced to one row per team per game, and
the raw file is deleted, so only the small aggregate is kept in nfl/data_cache/.  A team's defensive
measures for a game are simply its opponent's offensive measures in that game.
"""

import os

import numpy as np
import pandas as pd
import requests

from nflverse_data import CACHE_DIR

PBP_URL = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv.gz'
PBP_COLUMNS = [
    'game_id', 'season', 'week', 'season_type', 'posteam', 'defteam', 'play_type', 'epa', 'success', 'yards_gained', 'down',
    'ydstogo', 'pass', 'rush', 'sack', 'interception', 'fumble', 'fumble_lost', 'qb_dropback', 'wp', 'first_down', 'penalty',
    'penalty_team', 'penalty_yards', 'touchdown', 'drive', 'special_teams_play', 'air_yards', 'cpoe', 'qb_kneel', 'qb_spike',
    'qb_scramble', 'play_deleted', 'third_down_converted', 'third_down_failed', 'xpass', 'pass_oe', 'yardline_100',
]
COMPETITIVE_WP = (0.10, 0.90)  # plays with win probability inside this band are not garbage time


def _rate(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def aggregate_pbp(pbp):
    """One row per team per game of offensive measures, from one season's play-by-play."""
    pbp = pbp[pbp['play_deleted'].fillna(0) == 0]
    rows = []
    scrimmage = pbp[((pbp['pass'] == 1) | (pbp['rush'] == 1)) & pbp['epa'].notna() & pbp['posteam'].notna()
                    & (pbp['qb_kneel'].fillna(0) == 0) & (pbp['qb_spike'].fillna(0) == 0)]
    special = pbp[(pbp['special_teams_play'] == 1) & pbp['epa'].notna() & pbp['posteam'].notna()]

    for (game_id, team), g in scrimmage.groupby(['game_id', 'posteam']):
        passes = g[g['qb_dropback'] == 1]
        runs = g[(g['rush'] == 1) & (g['qb_dropback'] != 1)]
        early = g[g['down'] <= 2]
        late = g[g['down'] >= 3]
        comp = g[(g['wp'] >= COMPETITIVE_WP[0]) & (g['wp'] <= COMPETITIVE_WP[1])]
        thirds = g[g['down'] == 3]
        attempts = g[(g['pass'] == 1) & (g['sack'] != 1)]
        drives = g['drive'].nunique()
        opp_penalties = pbp[(pbp['game_id'] == game_id) & (pbp['penalty'] == 1) & (pbp['penalty_team'] == team)]
        st_for = special[(special['game_id'] == game_id) & (special['posteam'] == team)]['epa'].sum()
        st_against = special[(special['game_id'] == game_id) & (special['defteam'] == team)]['epa'].sum()
        rows.append({
            'game_id': game_id, 'team': team, 'season': g['season'].iloc[0], 'week': g['week'].iloc[0],
            'o_plays': len(g), 'o_epa_pp': g['epa'].mean(),
            'o_epa_pass': passes['epa'].mean() if len(passes) else np.nan,
            'o_epa_rush': runs['epa'].mean() if len(runs) else np.nan,
            'o_success': g['success'].mean(),
            'o_success_pass': passes['success'].mean() if len(passes) else np.nan,
            'o_success_rush': runs['success'].mean() if len(runs) else np.nan,
            'o_early_epa': early['epa'].mean() if len(early) else np.nan,
            'o_early_success': early['success'].mean() if len(early) else np.nan,
            'o_late_epa': late['epa'].mean() if len(late) else np.nan,
            'o_comp_epa': comp['epa'].mean() if len(comp) else np.nan,
            'o_comp_success': comp['success'].mean() if len(comp) else np.nan,
            'o_explosive': float(((passes['yards_gained'] >= 20).sum() + (runs['yards_gained'] >= 10).sum()) / max(len(g), 1)),
            'o_sack_rate': _rate((passes['sack'] == 1).sum(), len(passes)),
            'o_int_rate': _rate((attempts['interception'] == 1).sum(), len(attempts)),
            'o_fumble_rate': _rate((g['fumble'] == 1).sum(), len(g)),
            'o_fumbles_lost': float((g['fumble_lost'] == 1).sum()),
            'o_third_rate': _rate((thirds['third_down_converted'] == 1).sum(), len(thirds)),
            'o_first_downs': float((g['first_down'] == 1).sum()),
            'o_ypa': attempts['yards_gained'].mean() if len(attempts) else np.nan,
            'o_ypc': runs['yards_gained'].mean() if len(runs) else np.nan,
            'o_air_yards': attempts['air_yards'].mean() if attempts['air_yards'].notna().any() else np.nan,
            'o_cpoe': attempts['cpoe'].mean() if attempts['cpoe'].notna().any() else np.nan,
            'o_pass_rate': _rate(len(passes), len(g)),
            'o_pass_oe': g['pass_oe'].mean() if g['pass_oe'].notna().any() else np.nan,
            'o_td_per_drive': _rate((g['touchdown'] == 1).sum(), drives),
            'o_drives': float(drives),
            'o_penalties': float(len(opp_penalties)),
            'o_penalty_yards': float(opp_penalties['penalty_yards'].sum()),
            'o_st_epa_for': float(st_for), 'o_st_epa_against': float(st_against),
        })
    return pd.DataFrame(rows)


def _season_path(season):
    return os.path.join(CACHE_DIR, f'pbp_team_{season}.csv')


SCORING_COLUMNS = ['game_id', 'season', 'week', 'home_team', 'away_team', 'posteam', 'defteam', 'td_team', 'touchdown',
                   'field_goal_result', 'extra_point_result', 'two_point_conv_result', 'two_point_attempt', 'safety', 'play_deleted']


def aggregate_scoring(pbp):
    """Scoring events per team per game: touchdowns, field goals, extra points, two-point conversions, safeties.

    6*td + xp + 2*two_pt + 3*fg + 2*safety rebuilds the final score exactly in about 99.6% of team-games
    (the rest are defensive two-point returns).
    """
    pbp = pbp[pbp['play_deleted'].fillna(0) == 0]
    rows = []
    for game_id, g in pbp.groupby('game_id'):
        for team in (g['home_team'].iloc[0], g['away_team'].iloc[0]):
            rows.append({
                'game_id': game_id, 'team': team, 'season': g['season'].iloc[0], 'week': g['week'].iloc[0],
                'td': int(((g['touchdown'] == 1) & (g['td_team'] == team)).sum()),
                'fg': int(((g['field_goal_result'] == 'made') & (g['posteam'] == team)).sum()),
                'fg_att': int((g['field_goal_result'].notna() & (g['posteam'] == team)).sum()),
                'xp': int(((g['extra_point_result'] == 'good') & (g['posteam'] == team)).sum()),
                'xp_att': int((g['extra_point_result'].notna() & (g['posteam'] == team)).sum()),
                'two_pt': int(((g['two_point_conv_result'] == 'success') & (g['posteam'] == team)).sum()),
                'two_pt_att': int((g['two_point_conv_result'].notna() & (g['posteam'] == team)).sum()),
                'safeties': int(((g['safety'] == 1) & (g['defteam'] == team)).sum()),
            })
    return pd.DataFrame(rows)


def build_scoring_season(season, refresh=False):
    path = os.path.join(CACHE_DIR, f'pbp_scoring_{season}.csv')
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    raw = os.path.join(CACHE_DIR, f'_pbp_{season}.csv.gz')
    response = requests.get(PBP_URL.format(year=season), timeout=300)
    response.raise_for_status()
    with open(raw, 'wb') as f:
        f.write(response.content)
    present = pd.read_csv(raw, nrows=1).columns
    pbp = pd.read_csv(raw, usecols=[c for c in SCORING_COLUMNS if c in present], low_memory=False)
    result = aggregate_scoring(pbp)
    result.to_csv(path, index=False)
    os.remove(raw)
    return result


def load_scoring_events(first_season=1999, last_season=2026, refresh_latest=False):
    frames = []
    for season in range(first_season, last_season + 1):
        try:
            frames.append(build_scoring_season(season, refresh=refresh_latest and season == last_season))
        except requests.RequestException:
            continue
    return pd.concat(frames, ignore_index=True)


def build_season(season, refresh=False):
    path = _season_path(season)
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    raw = os.path.join(CACHE_DIR, f'_pbp_{season}.csv.gz')
    response = requests.get(PBP_URL.format(year=season), timeout=300)
    response.raise_for_status()
    with open(raw, 'wb') as f:
        f.write(response.content)
    columns_present = pd.read_csv(raw, nrows=1).columns
    pbp = pd.read_csv(raw, usecols=[c for c in PBP_COLUMNS if c in columns_present], low_memory=False)
    result = aggregate_pbp(pbp)
    result.to_csv(path, index=False)
    os.remove(raw)
    return result


def load_pbp_team_games(first_season=1999, last_season=2026, refresh_latest=False):
    """Offensive measures per team per game for every season, downloading and aggregating what is missing."""
    frames = []
    for season in range(first_season, last_season + 1):
        try:
            frames.append(build_season(season, refresh=refresh_latest and season == last_season))
        except requests.RequestException:
            continue
    return pd.concat(frames, ignore_index=True)


if __name__ == '__main__':
    os.makedirs(CACHE_DIR, exist_ok=True)
    for season in range(1999, 2027):
        frame = build_season(season)
        print(season, len(frame), flush=True)
