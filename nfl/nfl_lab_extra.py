"""Extra pre-game features for the research framework: cover history, overreaction and travel."""

import numpy as np
import pandas as pd

# (latitude, longitude, standard-time UTC offset) of each team's home stadium
TEAM_SITES = {
    'ARI': (33.53, -112.26, -7), 'ATL': (33.76, -84.40, -5), 'BAL': (39.28, -76.62, -5), 'BUF': (42.77, -78.79, -5),
    'CAR': (35.23, -80.85, -5), 'CHI': (41.86, -87.62, -6), 'CIN': (39.10, -84.52, -5), 'CLE': (41.51, -81.70, -5),
    'DAL': (32.75, -97.09, -6), 'DEN': (39.74, -105.02, -7), 'DET': (42.34, -83.05, -5), 'GB': (44.50, -88.06, -6),
    'HOU': (29.68, -95.41, -6), 'IND': (39.76, -86.16, -5), 'JAX': (30.32, -81.64, -5), 'KC': (39.05, -94.48, -6),
    'LV': (36.09, -115.18, -8), 'LAC': (33.95, -118.34, -8), 'LA': (33.95, -118.34, -8), 'MIA': (25.96, -80.24, -5),
    'MIN': (44.97, -93.26, -6), 'NE': (42.09, -71.26, -5), 'NO': (29.95, -90.08, -6), 'NYG': (40.81, -74.07, -5),
    'NYJ': (40.81, -74.07, -5), 'PHI': (39.90, -75.17, -5), 'PIT': (40.45, -80.00, -5), 'SF': (37.40, -121.97, -8),
    'SEA': (47.60, -122.33, -8), 'TB': (27.98, -82.50, -5), 'TEN': (36.17, -86.77, -6), 'WAS': (38.91, -76.86, -5),
}


def add_travel_features(table):
    t = table.copy()
    home = t['home_team'].map(TEAM_SITES)
    away = t['away_team'].map(TEAM_SITES)
    lat1, lon1 = np.radians([x[0] for x in away]), np.radians([x[1] for x in away])
    lat2, lon2 = np.radians([x[0] for x in home]), np.radians([x[1] for x in home])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    t['travel_km'] = 6371 * 2 * np.arcsin(np.sqrt(a))
    t['tz_shift'] = [h[2] - a_[2] for h, a_ in zip(home, away)]     # > 0: the visitor came from a later time zone... eastward
    hour = pd.to_datetime(t['gametime'], format='%H:%M', errors='coerce').dt.hour
    t['early_start'] = (hour <= 13).astype(int)
    t['west_to_east_early'] = ((t['tz_shift'] <= -2) & (t['early_start'] == 1)).astype(int)   # visitor from the west, 1pm ET kickoff
    t['east_to_west'] = (t['tz_shift'] >= 2).astype(int)
    return t


def add_cover_history(table, decay=0.8):
    """Each team's recent record against the spread and its last game's surprise, as of before the game."""
    t = table.sort_values(['gameday', 'week']).copy()
    ewm, last_surprise, last_margin, last_season = {}, {}, {}, {}
    out = {k: [] for k in ('h_ats_ewm', 'a_ats_ewm', 'h_last_surprise', 'a_last_surprise', 'h_last_margin', 'a_last_margin')}
    for g in t.itertuples():
        for side, team in (('h', g.home_team), ('a', g.away_team)):
            if last_season.get(team) is not None and last_season[team] != g.season:
                ewm[team] = ewm.get(team, 0.0) * 0.5
            last_season[team] = g.season
            out[f'{side}_ats_ewm'].append(ewm.get(team, 0.0))
            out[f'{side}_last_surprise'].append(last_surprise.get(team, 0.0))
            out[f'{side}_last_margin'].append(last_margin.get(team, 0.0))
        if not pd.isna(g.result) and not pd.isna(g.spread_line):
            cover = g.result - g.spread_line   # from the home team's point of view
            for team, c, m in ((g.home_team, cover, g.result), (g.away_team, -cover, -g.result)):
                ewm[team] = ewm.get(team, 0.0) * decay + (1 - decay) * np.clip(c, -21, 21)
                last_surprise[team] = float(np.clip(c, -21, 21))
                last_margin[team] = float(np.clip(m, -28, 28))
    for k, v in out.items():
        t[k] = v
    t['ats_ewm_diff'] = t['h_ats_ewm'] - t['a_ats_ewm']
    t['last_surprise_diff'] = t['h_last_surprise'] - t['a_last_surprise']
    t['last_margin_diff'] = t['h_last_margin'] - t['a_last_margin']
    return t.sort_index()
