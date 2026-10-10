"""
starters_cache.py
-----------------
Downloads historical MLB probable/actual starting pitchers from the
MLB Stats API and saves one CSV per season to MLB_STATS_DIR.

Output files:  C:/Users/Bobby/Downloads/MLB_Stats/mlb_starters_<year>.csv
Columns:       date, home_team, away_team,
               home_starter, home_starter_id,
               away_starter, away_starter_id

Usage:
    # Download all supported seasons
    python starters_cache.py

    # Download specific years only
    python starters_cache.py --years 2024 2025

    # Re-download even if file already exists
    python starters_cache.py --years 2025 --force

Notes:
    - The API returns "probablePitcher" — for past games this is the
      actual starter recorded in the schedule endpoint.
    - If a pitcher is TBD the id is empty string.
    - 2020 was a shortened season (60 games).
    - Rate-limit: the script sleeps 0.3s per day to avoid hammering
      the MLB API.
"""

import csv
import sys
import time
import argparse
import os
import warnings
from datetime import date, timedelta

warnings.filterwarnings('ignore')

try:
    import requests
except ImportError:
    print('[ERROR] requests not installed.  Run: pip install requests')
    sys.exit(1)

# ---------------------------------------------------------------------------
# CONFIG — must match mlbpicks.py
# ---------------------------------------------------------------------------
MLB_STATS_DIR = 'C:/Users/Bobby/Downloads/MLB_Stats'

SEASON_DATES = {
    2010: ('2010-04-04', '2010-11-01'),
    2011: ('2011-03-31', '2011-10-28'),
    2012: ('2012-03-28', '2012-10-29'),
    2013: ('2013-03-31', '2013-10-31'),
    2014: ('2014-03-22', '2014-10-30'),
    2015: ('2015-04-05', '2015-11-01'),
    2016: ('2016-04-03', '2016-11-02'),
    2017: ('2017-04-02', '2017-11-01'),
    2018: ('2018-03-29', '2018-10-28'),
    2019: ('2019-03-20', '2019-10-30'),
    2020: ('2020-07-23', '2020-10-27'),
    2021: ('2021-04-01', '2021-11-02'),
    2022: ('2022-04-07', '2022-11-05'),
    2023: ('2023-03-30', '2023-11-01'),
    2024: ('2024-03-20', '2024-10-30'),
    2025: ('2025-03-27', '2025-10-31'),
    2026: ('2026-03-26', '2026-10-31'),
}

FIELDNAMES = [
    'date', 'home_team', 'away_team',
    'home_starter', 'home_starter_id',
    'away_starter', 'away_starter_id',
]

SLEEP_SECS = 0.3   # between API calls


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _get_starters_for_date(game_date):
    """
    Fetch starters for one date.  Returns list of game dicts.
    Raises requests.HTTPError on non-2xx responses.
    """
    url = 'https://statsapi.mlb.com/api/v1/schedule'
    params = {
        'sportId': 1,
        'date': game_date.strftime('%Y-%m-%d'),
        'hydrate': 'probablePitcher,team',
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get('dates', []):
        for game in date_entry.get('games', []):
            home = game['teams']['home']
            away = game['teams']['away']

            home_p = home.get('probablePitcher') or {}
            away_p = away.get('probablePitcher') or {}

            games.append({
                'date': game_date.strftime('%Y-%m-%d'),
                'home_team': home['team']['name'],
                'away_team': away['team']['name'],
                'home_starter':    home_p.get('fullName', 'TBD'),
                'home_starter_id': home_p.get('id', ''),
                'away_starter':    away_p.get('fullName', 'TBD'),
                'away_starter_id': away_p.get('id', ''),
            })
    return games


def get_starters_for_season(year, verbose=True):
    """Pull starters for every day of the season. Returns list of dicts."""
    if year not in SEASON_DATES:
        print(f'[ERROR] No season dates configured for {year}.')
        return []

    start_str, end_str = SEASON_DATES[year]
    start = date.fromisoformat(start_str)
    end   = date.fromisoformat(end_str)

    all_games = []
    current   = start
    total_days = (end - start).days + 1
    day_num   = 0

    while current <= end:
        day_num += 1
        if verbose:
            print(
                f'  [{day_num:>3}/{total_days}] {current}  '
                f'(games so far: {len(all_games)})',
                end='\r',
            )
        try:
            games = _get_starters_for_date(current)
            all_games.extend(games)
        except Exception as exc:
            print(f'\n  [WARN] {current}: {exc}')

        current += timedelta(days=1)
        time.sleep(SLEEP_SECS)

    if verbose:
        print()   # newline after \r progress
    return all_games


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _csv_path(year):
    return os.path.join(MLB_STATS_DIR, f'mlb_starters_{year}.csv')


def save_starters_csv(games, year):
    path = _csv_path(year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(games)
    print(f'  Saved {len(games)} rows -> {path}')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Download MLB probable starters to CSV.'
    )
    parser.add_argument(
        '--years', nargs='+', type=int,
        default=sorted(SEASON_DATES.keys()),
        help='Seasons to download (default: all available)',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-download even if CSV already exists',
    )
    args = parser.parse_args()

    for year in sorted(args.years):
        path = _csv_path(year)
        if os.path.exists(path) and not args.force:
            print(f'[SKIP] {year}: {path} already exists (--force to redo)')
            continue

        print(f'\n=== {year} ===')
        games = get_starters_for_season(year)
        if games:
            save_starters_csv(games, year)
        else:
            print(f'  No games found for {year}.')

    print('\nAll done.')


if __name__ == '__main__':
    main()
