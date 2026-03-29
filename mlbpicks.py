"""
mlbpicks.py
-----------
MLB prediction platform using local baseball-reference.com HTML files.

Predicts for a given date:
  - Run-line pick (home team -1.5)
  - Moneyline pick (straight winner)
  - Over/Under total pick

Usage:
  python mlbpicks.py                         # prompts for date
  python mlbpicks.py --date 2025-08-20
  python mlbpicks.py --date 2025-08-20 --runs 500
  python mlbpicks.py --date 2025-08-20 --save

Data files (saved locally from baseball-reference.com):
  Stats : <MLB_STATS_DIR>/team_stats_<yyyy>.html
  Sched : <MLB_STATS_DIR>/schedule_<yyyy>.html
"""

import re
import sys
import argparse
import warnings
from datetime import datetime, date
from collections import defaultdict

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Comment
from sklearn.linear_model import Ridge
from sklearn.utils import resample

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MLB_STATS_DIR = 'C:/Users/Bobby/Downloads/MLB_Stats'
STATS_FILE_TPL = 'team_stats_{year}.html'
SCHED_FILE_TPL = 'schedule_{year}.html'
PITCHER_FILE_TPL = 'player_pitching_{year}.html'
STARTERS_FILE_TPL = 'mlb_starters_{year}.csv'

DEFAULT_NUM_RUNS = 500
DEFAULT_STATS_YEAR = 2025
DEFAULT_SCHED_YEAR = 2025

# Standard MLB run-line spread
RUN_LINE = 1.5

# ---------------------------------------------------------------------------
# DUMMY PITCHER STATS
# Used until real pitcher_stats_<year>.html files are downloaded.
# Static fallback used only when the pitcher stats HTML file is missing.
# When the file IS present, _DEFAULT is computed dynamically from that
# year's actual starter population (see load_pitcher_stats).
_LEAGUE_AVG_SP = {'era': 4.25, 'whip': 1.30, 'k9': 8.5, 'bb9': 3.1}

# ---------------------------------------------------------------------------
# TEAM NAME NORMALISATION
# Map schedule short names / variants -> canonical batting/pitching name
# ---------------------------------------------------------------------------
NAME_MAP = {
    "Arizona D'Backs": 'Arizona Diamondbacks',
    "D-backs": 'Arizona Diamondbacks',
    "D'Backs": 'Arizona Diamondbacks',
    'ARI': 'Arizona Diamondbacks',
    'ATH': 'Athletics',
    'Oakland Athletics': 'Athletics',
    "Oakland A's": 'Athletics',
    "A's": 'Athletics',
    'BAL': 'Baltimore Orioles',
    'BOS': 'Boston Red Sox',
    'CHC': 'Chicago Cubs',
    'CUB': 'Chicago Cubs',           # odds files
    'CWS': 'Chicago White Sox',
    'CHW': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds',
    'CLE': 'Cleveland Guardians',
    'Cleveland Indians': 'Cleveland Guardians',
    'COL': 'Colorado Rockies',
    'DET': 'Detroit Tigers',
    'FLA': 'Miami Marlins',          # Florida Marlins (pre-2012)
    'HOU': 'Houston Astros',
    'KAN': 'Kansas City Royals',     # odds files
    'KCR': 'Kansas City Royals',
    'KC': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels',
    'ANA': 'Los Angeles Angels',     # Anaheim / odds variants
    'LAD': 'Los Angeles Dodgers',
    'MIA': 'Miami Marlins',
    'MIL': 'Milwaukee Brewers',
    'MIN': 'Minnesota Twins',
    'NYM': 'New York Mets',
    'NYY': 'New York Yankees',
    'OAK': 'Athletics',
    'PHI': 'Philadelphia Phillies',
    'PIT': 'Pittsburgh Pirates',
    'SDP': 'San Diego Padres',
    'SDG': 'San Diego Padres',       # odds files
    'SD': 'San Diego Padres',
    'SEA': 'Seattle Mariners',
    'SFG': 'San Francisco Giants',
    'SFO': 'San Francisco Giants',   # odds files
    'SF': 'San Francisco Giants',
    'STL': 'St. Louis Cardinals',
    'TBR': 'Tampa Bay Rays',
    'TAM': 'Tampa Bay Rays',         # odds files
    'TB': 'Tampa Bay Rays',
    'TEX': 'Texas Rangers',
    'TOR': 'Toronto Blue Jays',
    'WSN': 'Washington Nationals',
    'WAS': 'Washington Nationals',
    'WSH': 'Washington Nationals',
}


def normalize_name(name):
    """Return canonical team name."""
    name = name.strip()
    return NAME_MAP.get(name, name)


# ---------------------------------------------------------------------------
# HTML PARSERS
# ---------------------------------------------------------------------------

def _stats_path(year):
    """
    Return stats file path, trying team_stats_yyyy first,
    then falling back to stats_yyyy (2010-2014 naming).
    """
    import os
    primary = MLB_STATS_DIR + '/' + STATS_FILE_TPL.format(year=year)
    if os.path.exists(primary):
        return primary
    fallback = MLB_STATS_DIR + f'/stats_{year}.html'
    if os.path.exists(fallback):
        return fallback
    return primary  # will fail with clear FileNotFoundError


def _sched_path(year):
    return MLB_STATS_DIR + '/' + SCHED_FILE_TPL.format(year=year)


def _pitcher_path(year):
    return MLB_STATS_DIR + '/' + PITCHER_FILE_TPL.format(year=year)


def _starters_path(year):
    return MLB_STATS_DIR + '/' + STARTERS_FILE_TPL.format(year=year)


def load_starters_csv(year):
    """
    Load mlb_starters_<year>.csv produced by starters_cache.py.

    Returns dict keyed (away_team, home_team, 'YYYY-MM-DD') ->
        {'away_sp': full_name_or_None, 'home_sp': full_name_or_None}

    Team names are normalized so they match the canonical names used
    throughout mlbpicks.py.  Returns empty dict if file not found.
    """
    import os
    path = _starters_path(year)
    if not os.path.exists(path):
        return {}

    lookup = {}
    try:
        df = pd.read_csv(path, dtype=str).fillna('')
    except Exception as exc:
        print(f'  [WARN] Could not load starters CSV {path}: {exc}')
        return {}

    for _, row in df.iterrows():
        away = normalize_name(row.get('away_team', '').strip())
        home = normalize_name(row.get('home_team', '').strip())
        d    = row.get('date', '').strip()
        if not away or not home or not d:
            continue
        away_sp = row.get('away_starter', '').strip() or None
        home_sp = row.get('home_starter', '').strip() or None
        # Treat 'TBD' as unknown
        if away_sp == 'TBD':
            away_sp = None
        if home_sp == 'TBD':
            home_sp = None
        lookup[(away, home, d)] = {
            'away_sp': away_sp,
            'home_sp': home_sp,
        }

    return lookup


def load_pitcher_stats(year):
    """
    Load individual pitcher season stats from player_pitching_<year>.html
    (baseball-reference.com standard pitching table).

    Returns dict: LASTNAME_UPPER -> {era, whip, k9, bb9}
    The '_DEFAULT' key holds the mean of all loaded starters for that
    year and is used as the fallback for unknown pitchers.

    Falls back to {'_DEFAULT': _LEAGUE_AVG_SP} if the file is missing.

    Table id : 'players_standard_pitching'
    Columns used:
        name_display       - full player name (may end with * or +)
        p_earned_run_avg   - ERA
        p_whip             - WHIP
        p_so_per_nine      - K/9
        p_bb_per_nine      - BB/9
        p_ip               - innings pitched
        p_gs               - games started (used to filter starters)
        team_name_abbr     - '2TM'/'3TM' marks aggregate multi-team rows

    Multi-team (traded) players appear twice: once per team with class
    'partial_table' (skipped by _parse_table) and once as an aggregate
    row with no class — so the combined season stats are used.

    Only pitchers with >= MIN_SP_GS games started are indexed.  When
    two pitchers share a last name the one with more IP wins.
    """
    import os
    MIN_SP_GS = 5   # minimum games started to be considered a starter
    _missing = {'_DEFAULT': _LEAGUE_AVG_SP}

    path = _pitcher_path(year)
    if not os.path.exists(path):
        print(
            f'  [INFO] Pitcher stats not found for {year}: {path}'
            f'\n         Using static league-average as default.'
        )
        return _missing

    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
    except OSError as exc:
        print(f'[ERROR] Could not open pitcher stats: {exc}')
        return _missing

    rows = _load_table(soup, 'players_standard_pitching')
    if not rows:
        print(
            f'  [WARN] No pitching rows found in {path}. '
            f'Using static league-average as default.'
        )
        return _missing

    def _f(row, col):
        """Parse float from cell, return NaN on failure."""
        try:
            return float(row.get(col, '') or 'nan')
        except ValueError:
            return float('nan')

    # ip_by_last tracks {LASTNAME: ip} to resolve last-name collisions
    # by keeping the pitcher with more innings
    ip_by_last = {}
    stats = {}

    for row in rows:
        name_raw = row.get('name_display', '').strip()
        if not name_raw:
            continue

        # GS filter — skip relievers
        try:
            gs = int(row.get('p_gs', '0') or '0')
        except ValueError:
            gs = 0
        if gs < MIN_SP_GS:
            continue

        # Innings pitched (for collision resolution)
        try:
            ip = float(row.get('p_ip', '0') or '0')
        except ValueError:
            ip = 0.0

        # Extract last name: "Tarik Skubal*" -> "SKUBAL"
        clean = name_raw.rstrip('*+#').strip()
        parts = clean.split()
        last = parts[-1].upper() if parts else ''
        if not last:
            continue

        # If we've seen this last name before, keep the higher-IP entry
        if last in stats and ip <= ip_by_last.get(last, 0):
            continue

        era  = _f(row, 'p_earned_run_avg')
        whip = _f(row, 'p_whip')
        k9   = _f(row, 'p_so_per_nine')
        bb9  = _f(row, 'p_bb_per_nine')

        # Only store rows where at least ERA parsed successfully
        if np.isnan(era):
            continue

        stats[last] = {
            'era':  era,
            'whip': whip if not np.isnan(whip) else _LEAGUE_AVG_SP['whip'],
            'k9':   k9   if not np.isnan(k9)   else _LEAGUE_AVG_SP['k9'],
            'bb9':  bb9  if not np.isnan(bb9)  else _LEAGUE_AVG_SP['bb9'],
            'gs':   gs,
        }
        ip_by_last[last] = ip

    if not stats:
        return _missing

    # Compute _DEFAULT as the median of all loaded starters for this year.
    # Median is preferred over mean because ERA/WHIP distributions are
    # right-skewed — a few replacement-level pitchers inflate the mean.
    def _median(key):
        return float(np.median([v[key] for v in stats.values()]))

    stats['_DEFAULT'] = {
        'era':  _median('era'),
        'whip': _median('whip'),
        'k9':   _median('k9'),
        'bb9':  _median('bb9'),
    }
    d = stats['_DEFAULT']
    print(
        f'  Pitcher stats: {len(stats) - 1} starters loaded for {year}. '
        f'Default: ERA={d["era"]:.2f} WHIP={d["whip"]:.3f} '
        f'K9={d["k9"]:.1f} BB9={d["bb9"]:.1f}'
    )
    return stats


def _lookup_sp(pitcher_name, pitcher_stats, prev_pitcher_stats=None):
    """
    Look up a starting pitcher by last name (case-insensitive).

    If the pitcher is missing from pitcher_stats OR has fewer than
    MIN_BLEND_GS games started in the current season, his prior-season
    stats (prev_pitcher_stats) are blended in.  The blend weight is
    proportional to how many current-season GS he already has:

        weight_cur  = gs_cur / MIN_BLEND_GS   (capped at 1.0)
        weight_prev = 1 - weight_cur

    So with 0 GS the result is 100% prior season; with 3/5 GS it is
    60% current / 40% prior; at 5+ GS it is 100% current season.

    Falls back to the year's '_DEFAULT' median when both dicts miss him.
    """
    MIN_BLEND_GS = 5   # must match load_pitcher_stats MIN_SP_GS

    def _default(ps):
        return (ps or {}).get('_DEFAULT', _LEAGUE_AVG_SP)

    if not pitcher_name:
        return _default(pitcher_stats)

    last = pitcher_name.strip().split()[-1].upper()
    cur  = (pitcher_stats or {}).get(last)
    prev = (prev_pitcher_stats or {}).get(last)

    if cur is None and prev is None:
        # Not found in either year — average the two years' medians
        d_cur  = _default(pitcher_stats)
        d_prev = _default(prev_pitcher_stats)
        return {
            k: round((d_cur[k] + d_prev[k]) / 2, 4)
            for k in ('era', 'whip', 'k9', 'bb9')
        }

    if cur is None:
        # No current-season data at all — use prior year entirely
        return {k: prev[k] for k in ('era', 'whip', 'k9', 'bb9')}

    gs_cur = cur.get('gs', MIN_BLEND_GS)
    if prev is None or gs_cur >= MIN_BLEND_GS:
        # Enough current-season starts, or no prior data to blend
        return {k: cur[k] for k in ('era', 'whip', 'k9', 'bb9')}

    # Partial-season blend
    w_cur  = gs_cur / MIN_BLEND_GS          # 0.0 – <1.0
    w_prev = 1.0 - w_cur
    return {
        k: round(w_cur * cur[k] + w_prev * prev[k], 4)
        for k in ('era', 'whip', 'k9', 'bb9')
    }


# ---------------------------------------------------------------------------
# MLB STATS API — probable starters
# ---------------------------------------------------------------------------

def fetch_probable_starters(date_str):
    """
    Fetch probable starting pitchers from the MLB Stats API for a given date.

    date_str : 'YYYY-MM-DD'

    Returns starters dict keyed (away_team, home_team) ->
        {'away_sp': full_name_or_None, 'home_sp': full_name_or_None,
         'away_sp_id': int_or_None, 'home_sp_id': int_or_None}

    Team names are normalized via normalize_name() so they match
    the canonical names used throughout mlbpicks.py.

    Returns empty dict on any network / parse error (non-fatal).
    """
    try:
        import requests as _req
    except ImportError:
        print('  [INFO] requests not installed — skipping SP auto-fetch.')
        return {}

    url = 'https://statsapi.mlb.com/api/v1/schedule'
    params = {
        'sportId': 1,
        'date': date_str,
        'hydrate': 'probablePitcher(note),team',
    }

    try:
        resp = _req.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f'  [WARN] Could not fetch probable starters: {exc}')
        return {}

    starters = {}
    for date_entry in data.get('dates', []):
        for game in date_entry.get('games', []):
            try:
                home_raw = game['teams']['home']['team']['name']
                away_raw = game['teams']['away']['team']['name']
            except KeyError:
                continue

            home_team = normalize_name(home_raw)
            away_team = normalize_name(away_raw)

            home_sp_data = game['teams']['home'].get('probablePitcher') or {}
            away_sp_data = game['teams']['away'].get('probablePitcher') or {}

            home_sp = home_sp_data.get('fullName') or None
            away_sp = away_sp_data.get('fullName') or None

            key = (away_team, home_team)
            starters[key] = {
                'away_sp':    away_sp,
                'home_sp':    home_sp,
                'away_sp_id': away_sp_data.get('id'),
                'home_sp_id': home_sp_data.get('id'),
            }

    return starters


def _merge_starters(fetched, from_lines):
    """
    Merge starters from two sources.
    Values in from_lines (manually entered in UI) take priority
    over auto-fetched values so the user can override if needed.
    """
    merged = dict(fetched)
    for key, sp in from_lines.items():
        if key not in merged:
            merged[key] = sp
        else:
            # Manual entry wins for non-empty values
            if sp.get('away_sp'):
                merged[key]['away_sp'] = sp['away_sp']
            if sp.get('home_sp'):
                merged[key]['home_sp'] = sp['home_sp']
    return merged


def _parse_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _parse_table(table):
    """Extract rows as list of dicts from a BeautifulSoup table element."""
    thead = table.find('thead')
    if not thead:
        return []
    headers = [th.get('data-stat', '') for th in thead.find_all('th')]
    tbody = table.find('tbody')
    if not tbody:
        return []
    rows = []
    for tr in tbody.find_all('tr'):
        if tr.get('class'):
            continue  # skip spacer / header rows
        cells = tr.find_all(['td', 'th'])
        if not cells:
            continue
        row = {
            headers[i]: cells[i].text.strip()
            for i in range(min(len(headers), len(cells)))
        }
        rows.append(row)
    return rows


def _load_table(soup, table_id):
    """
    Find a table by id in visible HTML or inside HTML comments.
    Returns list of row dicts.
    """
    # Try visible first
    table = soup.find('table', id=table_id)
    if table:
        return _parse_table(table)

    # Fall back to commented tables (baseball-reference hides some tables)
    for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
        csoup = BeautifulSoup(comment, 'html.parser')
        table = csoup.find('table', id=table_id)
        if table:
            return _parse_table(table)

    return []


def load_team_stats(year):
    """
    Load season batting + pitching stats for all teams.
    Returns dict: canonical_team_name -> {feature: float}
    """
    path = _stats_path(year)
    try:
        with open(path, encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
    except FileNotFoundError:
        print(f'[ERROR] Stats file not found:\n  {path}')
        return {}

    batting_rows = _load_table(soup, 'teams_standard_batting')
    pitching_rows = _load_table(soup, 'teams_standard_pitching')

    # Index pitching by canonical team name
    pitching = {}
    for row in pitching_rows:
        name = normalize_name(row.get('team_name', ''))
        if name:
            pitching[name] = row

    stats = {}
    for row in batting_rows:
        raw_name = row.get('team_name', '')
        if not raw_name or raw_name.lower() in ('', 'lg avg', 'average'):
            continue
        name = normalize_name(raw_name)
        if name.lower() in ('', 'lg avg', 'average'):
            continue

        p = pitching.get(name, {})
        g = max(_parse_float(row.get('G', '1')), 1)

        stats[name] = {
            # Batting
            'runs_per_game': _parse_float(row.get('runs_per_game')),
            'batting_avg': _parse_float(row.get('batting_avg')),
            'obp': _parse_float(row.get('onbase_perc')),
            'slg': _parse_float(row.get('slugging_perc')),
            'ops': _parse_float(row.get('onbase_plus_slugging')),
            'ops_plus': _parse_float(row.get('onbase_plus_slugging_plus')),
            'hr_per_game': _parse_float(row.get('HR', '0')) / g,
            'bb_per_game': _parse_float(row.get('BB', '0')) / g,
            'so_per_game': _parse_float(row.get('SO', '0')) / g,
            # Pitching
            'era': _parse_float(p.get('earned_run_avg')),
            'whip': _parse_float(p.get('whip')),
            'runs_allowed_per_game': _parse_float(
                p.get('runs_allowed_per_game')
            ),
            'k9': _parse_float(p.get('strikeouts_per_nine')),
            'bb9': _parse_float(p.get('bases_on_balls_per_nine')),
            'h9': _parse_float(p.get('hits_per_nine')),
            'era_plus': _parse_float(p.get('earned_run_avg_plus')),
            'fip': _parse_float(p.get('fip')),
        }

    return stats


# ---------------------------------------------------------------------------
# SCHEDULE PARSER
# ---------------------------------------------------------------------------

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
}


def _parse_date_header(text):
    """
    Parse 'Tuesday, March 18, 2025' -> datetime.date.
    Also handles "Today's Games" -> date.today().
    """
    text = text.strip()
    if 'today' in text.lower():
        return date.today()
    if ',' in text:
        text = text.split(',', 1)[1].strip()
    m = re.match(r'(\w+)\s+(\d+),\s+(\d{4})', text)
    if m:
        month = MONTH_MAP.get(m.group(1))
        if month:
            return date(int(m.group(3)), month, int(m.group(2)))
    return None


def _parse_game_para(p_tag):
    """
    Parse a <p class="game"> tag.

    HTML structure (completed game):
      <strong> <a href="/teams/LAD/...">Los Angeles Dodgers</a>
       (4)</strong>
       @
       <a href="/teams/CHC/...">Chicago Cubs</a>
       (1)

    The '@' is on its own line surrounded by newlines/spaces.
    Winner is inside <strong>.
    Away team is LEFT of '@', home team is RIGHT.

    Returns dict or None.
    """
    raw_html = str(p_tag)

    # Find '@' separator — may be ' @ ', '\n @\n', etc.
    at_match = re.search(r'\s@\s', raw_html)
    if not at_match:
        return None

    at_start = at_match.start()
    at_end = at_match.end()
    before_at = raw_html[:at_start]
    after_at = raw_html[at_end:]

    def _team_score(chunk):
        csoup = BeautifulSoup(chunk, 'html.parser')
        link = csoup.find('a', href=re.compile(r'/teams/'))
        team = normalize_name(link.text.strip()) if link else None
        sm = re.search(r'\((\d+)\)', csoup.get_text())
        score = int(sm.group(1)) if sm else None
        return team, score

    away_team, away_score = _team_score(before_at)
    home_team, home_score = _team_score(after_at)

    if not away_team or not home_team:
        return None

    completed = (away_score is not None and home_score is not None)

    # Determine winner from <strong> wrapper
    winner = None
    strong = p_tag.find('strong')
    if strong:
        winner_link = strong.find('a')
        if winner_link:
            wname = normalize_name(winner_link.text.strip())
            winner = wname  # could be away or home

    return {
        'away_team': away_team,
        'home_team': home_team,
        'away_score': away_score,
        'home_score': home_score,
        'winner': winner,
        'completed': completed,
    }


def load_schedule(year):
    """
    Parse all games from the schedule HTML.
    Returns list of dicts: date, away_team, home_team,
                           away_score, home_score, winner, completed

    The schedule HTML has nested divs where outer divs absorb all inner
    game paragraphs via BeautifulSoup's recursive find/find_all.
    Fix: scan all h3 and p tags in document order so each p.game is
    assigned only to the most recent h3 date header.
    """
    path = _sched_path(year)
    try:
        with open(path, encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
    except FileNotFoundError:
        print(f'[ERROR] Schedule file not found:\n  {path}')
        return []

    games = []
    current_date = None

    for tag in soup.find_all(['h3', 'p']):
        if tag.name == 'h3':
            current_date = _parse_date_header(tag.text.strip())
        elif tag.name == 'p' and 'game' in (tag.get('class') or []):
            if current_date is None:
                continue
            info = _parse_game_para(tag)
            if info:
                info['date'] = current_date
                games.append(info)

    return games


def get_games_for_date(schedule, target_date):
    """Return all games (completed or scheduled) on target_date."""
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    return [g for g in schedule if g['date'] == target_date]


def get_completed_before(schedule, target_date):
    """Return completed games strictly before target_date."""
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    return [
        g for g in schedule
        if g['date'] < target_date
        and g['completed']
        and g['away_score'] is not None
        and g['home_score'] is not None
    ]


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    'home_rpg', 'home_ops', 'home_ops_plus', 'home_era', 'home_whip',
    'home_k9', 'home_bb9', 'home_fip', 'home_rapg',
    'away_rpg', 'away_ops', 'away_ops_plus', 'away_era', 'away_whip',
    'away_k9', 'away_bb9', 'away_fip', 'away_rapg',
    'rpg_diff', 'ops_diff', 'era_diff', 'whip_diff',
    # Starting pitcher features (per-game, from pitcher_stats_<year>.html)
    'home_sp_era', 'home_sp_whip', 'home_sp_k9', 'home_sp_bb9',
    'away_sp_era', 'away_sp_whip', 'away_sp_k9', 'away_sp_bb9',
    'sp_era_diff', 'sp_whip_diff',
]


def _s(d, key):
    """Safe float getter — returns 0.0 on NaN."""
    val = d.get(key, np.nan)
    if isinstance(val, float) and np.isnan(val):
        return 0.0
    return float(val)


def build_feature_row(home_stats, away_stats,
                      home_sp=None, away_sp=None):
    """
    Return feature dict for one matchup.

    home_sp / away_sp : optional dicts with keys era, whip, k9, bb9
                        (from load_pitcher_stats).  When None, falls
                        back to the team's season-average pitching stats.
    """
    # SP stats: use provided pitcher data, else team season ERA/WHIP/etc.
    if home_sp is None:
        home_sp = {
            'era':  _s(home_stats, 'era'),
            'whip': _s(home_stats, 'whip'),
            'k9':   _s(home_stats, 'k9'),
            'bb9':  _s(home_stats, 'bb9'),
        }
    if away_sp is None:
        away_sp = {
            'era':  _s(away_stats, 'era'),
            'whip': _s(away_stats, 'whip'),
            'k9':   _s(away_stats, 'k9'),
            'bb9':  _s(away_stats, 'bb9'),
        }

    return {
        'home_rpg':     _s(home_stats, 'runs_per_game'),
        'home_ops':     _s(home_stats, 'ops'),
        'home_ops_plus': _s(home_stats, 'ops_plus'),
        'home_era':     _s(home_stats, 'era'),
        'home_whip':    _s(home_stats, 'whip'),
        'home_k9':      _s(home_stats, 'k9'),
        'home_bb9':     _s(home_stats, 'bb9'),
        'home_fip':     _s(home_stats, 'fip'),
        'home_rapg':    _s(home_stats, 'runs_allowed_per_game'),
        'away_rpg':     _s(away_stats, 'runs_per_game'),
        'away_ops':     _s(away_stats, 'ops'),
        'away_ops_plus': _s(away_stats, 'ops_plus'),
        'away_era':     _s(away_stats, 'era'),
        'away_whip':    _s(away_stats, 'whip'),
        'away_k9':      _s(away_stats, 'k9'),
        'away_bb9':     _s(away_stats, 'bb9'),
        'away_fip':     _s(away_stats, 'fip'),
        'away_rapg':    _s(away_stats, 'runs_allowed_per_game'),
        'rpg_diff':  (
            _s(home_stats, 'runs_per_game')
            - _s(away_stats, 'runs_per_game')
        ),
        'ops_diff':  _s(home_stats, 'ops') - _s(away_stats, 'ops'),
        'era_diff':  _s(away_stats, 'era') - _s(home_stats, 'era'),
        'whip_diff': _s(away_stats, 'whip') - _s(home_stats, 'whip'),
        # Starting pitcher features
        'home_sp_era':  float(home_sp.get('era',  _LEAGUE_AVG_SP['era'])),
        'home_sp_whip': float(home_sp.get('whip', _LEAGUE_AVG_SP['whip'])),
        'home_sp_k9':   float(home_sp.get('k9',   _LEAGUE_AVG_SP['k9'])),
        'home_sp_bb9':  float(home_sp.get('bb9',  _LEAGUE_AVG_SP['bb9'])),
        'away_sp_era':  float(away_sp.get('era',  _LEAGUE_AVG_SP['era'])),
        'away_sp_whip': float(away_sp.get('whip', _LEAGUE_AVG_SP['whip'])),
        'away_sp_k9':   float(away_sp.get('k9',   _LEAGUE_AVG_SP['k9'])),
        'away_sp_bb9':  float(away_sp.get('bb9',  _LEAGUE_AVG_SP['bb9'])),
        'sp_era_diff':  (
            float(away_sp.get('era', _LEAGUE_AVG_SP['era']))
            - float(home_sp.get('era', _LEAGUE_AVG_SP['era']))
        ),
        'sp_whip_diff': (
            float(away_sp.get('whip', _LEAGUE_AVG_SP['whip']))
            - float(home_sp.get('whip', _LEAGUE_AVG_SP['whip']))
        ),
    }


def build_training_data(completed_games, team_stats,
                        pitcher_stats=None, starters_lookup=None):
    """
    Build training matrices from historical results.
    Uses season-to-date team stats (already per-game in the HTML).

    pitcher_stats   : LASTNAME_UPPER -> {era,whip,k9,bb9} (from load_pitcher_stats)
    starters_lookup : (away,home,'YYYY-MM-DD') -> {away_sp,home_sp} (from load_starters_csv)

    Returns X, y_diff (home-away run diff), y_total (combined runs).
    """
    rows, y_diff, y_total = [], [], []
    _ps = pitcher_stats or {'_DEFAULT': _LEAGUE_AVG_SP}
    _sl = starters_lookup or {}

    for g in completed_games:
        ht, at = g['home_team'], g['away_team']
        if ht not in team_stats or at not in team_stats:
            continue
        # Resolve date to string for starters lookup
        d = g.get('date')
        if d is not None:
            date_str = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
        else:
            date_str = ''
        sp_info = _sl.get((at, ht, date_str), {})
        home_sp = _lookup_sp(sp_info.get('home_sp'), _ps)
        away_sp = _lookup_sp(sp_info.get('away_sp'), _ps)
        rows.append(build_feature_row(team_stats[ht], team_stats[at],
                                      home_sp=home_sp, away_sp=away_sp))
        y_diff.append(g['home_score'] - g['away_score'])
        y_total.append(g['home_score'] + g['away_score'])

    if not rows:
        return None, None, None

    X = pd.DataFrame(rows, columns=FEATURE_COLS).fillna(0)
    return X, np.array(y_diff, dtype=float), np.array(y_total, dtype=float)


def build_training_data_multiyear(games_by_year, stats_by_year,
                                  pitcher_stats_by_year=None,
                                  starters_by_year=None):
    """
    Build training matrices across multiple seasons.

    games_by_year        : dict  {year: [game dicts with home/away/scores]}
                           Each game uses the team stats from its OWN year.
    stats_by_year        : dict  {year: team_stats dict}
    pitcher_stats_by_year: dict  {year: pitcher_stats dict}  (optional)
    starters_by_year     : dict  {year: starters_lookup dict} (optional)

    This lets us include e.g. 3 prior full seasons as extra training
    signal.  More recent games will also be included if passed in.

    Returns X, y_diff, y_total  (same contract as build_training_data).
    """
    rows, y_diff_vals, y_total_vals = [], [], []

    for year in sorted(games_by_year):
        team_stats = stats_by_year.get(year)
        if team_stats is None:
            continue
        _ps = (pitcher_stats_by_year or {}).get(year) or {'_DEFAULT': _LEAGUE_AVG_SP}
        _sl = (starters_by_year or {}).get(year) or {}
        for g in games_by_year[year]:
            ht, at = g['home_team'], g['away_team']
            if ht not in team_stats or at not in team_stats:
                continue
            d = g.get('date')
            if d is not None:
                date_str = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
            else:
                date_str = ''
            sp_info = _sl.get((at, ht, date_str), {})
            home_sp = _lookup_sp(sp_info.get('home_sp'), _ps)
            away_sp = _lookup_sp(sp_info.get('away_sp'), _ps)
            rows.append(build_feature_row(team_stats[ht], team_stats[at],
                                          home_sp=home_sp, away_sp=away_sp))
            y_diff_vals.append(g['home_score'] - g['away_score'])
            y_total_vals.append(g['home_score'] + g['away_score'])

    if not rows:
        return None, None, None

    X = pd.DataFrame(rows, columns=FEATURE_COLS).fillna(0)
    return (
        X,
        np.array(y_diff_vals, dtype=float),
        np.array(y_total_vals, dtype=float),
    )



# ---------------------------------------------------------------------------
# PREDICTION ENGINE — Monte Carlo with bootstrap resampling
# ---------------------------------------------------------------------------

def _fit_predict(X_train, y_train, X_pred):
    """Fit Ridge model and return scalar prediction."""
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return float(model.predict(X_pred)[0])


def _run_iteration(X, y_diff, y_total, predict_rows):
    """
    One bootstrap iteration: resample training data, fit 2 models,
    predict all games.  Returns list of (pred_diff, pred_total).
    """
    n = len(X)
    idx = resample(np.arange(n), n_samples=n, replace=True)
    Xb = X.iloc[idx]
    yd = y_diff[idx]
    yt = y_total[idx]

    results = []
    for row in predict_rows:
        Xp = pd.DataFrame([row], columns=FEATURE_COLS).fillna(0)
        results.append((_fit_predict(Xb, yd, Xp), _fit_predict(Xb, yt, Xp)))
    return results


def predict_games(games_today, team_stats, X, y_diff, y_total,
                  num_runs=DEFAULT_NUM_RUNS, run_line=RUN_LINE,
                  run_lines_map=None,
                  pitcher_stats=None, prev_pitcher_stats=None,
                  starters_today=None):
    """
    Run Monte Carlo predictions for all games on target date.

    run_lines_map      : optional dict (away_team, home_team) -> float
                         Per-game run line. Falls back to run_line default.
    pitcher_stats      : dict  LASTNAME_UPPER -> {era, whip, k9, bb9, gs}
                         from load_pitcher_stats(). If None, SP features
                         fall back to team season ERA/WHIP.
    prev_pitcher_stats : dict  same format, from load_pitcher_stats(year-1).
                         Blended in for pitchers with < 5 GS this season.
    starters_today     : dict  (away_team, home_team)
                                  -> {'away_sp': name, 'home_sp': name}
                         Announced starters for today's games. If None or
                         key missing, team season stats are used as SP proxy.

    Returns list of result dicts.
    """
    if run_lines_map is None:
        run_lines_map = {}
    if pitcher_stats is None:
        pitcher_stats = {'_DEFAULT': _LEAGUE_AVG_SP}
    if starters_today is None:
        starters_today = {}

    game_infos = []
    for g in games_today:
        ht, at = g['home_team'], g['away_team']
        if ht not in team_stats:
            print(f'  [WARN] No stats for: {ht}')
            continue
        if at not in team_stats:
            print(f'  [WARN] No stats for: {at}')
            continue

        key = (at, ht)
        sp_info = starters_today.get(key, {})
        home_sp_name = sp_info.get('home_sp') or ''
        away_sp_name = sp_info.get('away_sp') or ''
        home_sp = _lookup_sp(home_sp_name, pitcher_stats, prev_pitcher_stats)
        away_sp = _lookup_sp(away_sp_name, pitcher_stats, prev_pitcher_stats)

        feat = build_feature_row(
            team_stats[ht], team_stats[at],
            home_sp=home_sp, away_sp=away_sp,
        )
        game_infos.append((g, feat, away_sp_name, home_sp_name))

    if not game_infos:
        return []

    predict_rows = [feat for _, feat, _, _ in game_infos]
    all_diffs = defaultdict(list)
    all_totals = defaultdict(list)

    for _ in range(num_runs):
        iteration = _run_iteration(X, y_diff, y_total, predict_rows)
        for i, (pd_val, pt_val) in enumerate(iteration):
            all_diffs[i].append(pd_val)
            all_totals[i].append(pt_val)

    output = []
    for i, (g, _, away_sp_name, home_sp_name) in enumerate(game_infos):
        diffs = np.array(all_diffs[i])
        totals = np.array(all_totals[i])

        mean_diff = float(np.mean(diffs))
        mean_total = float(np.mean(totals))
        median_diff = float(np.median(diffs))
        median_total = float(np.median(totals))

        home_win_pct = float(np.mean(diffs > 0)) * 100
        away_win_pct = 100.0 - home_win_pct

        # Per-game run line (falls back to function default)
        game_rl = run_lines_map.get(
            (g['away_team'], g['home_team']), run_line
        )

        home_cover_pct = float(np.mean(diffs > game_rl)) * 100
        away_cover_pct = float(np.mean(diffs < -game_rl)) * 100
        push_pct = 100.0 - home_cover_pct - away_cover_pct

        # Moneyline pick
        ml_pick = g['home_team'] if mean_diff > 0 else g['away_team']
        ml_conf = max(home_win_pct, away_win_pct)

        # Run-line pick
        if home_cover_pct >= away_cover_pct:
            rl_pick = f"{g['home_team']} -{game_rl}"
            rl_conf = home_cover_pct
        else:
            rl_pick = f"{g['away_team']} +{game_rl}"
            rl_conf = away_cover_pct

        # Projected individual scores
        proj_home = (mean_total + mean_diff) / 2
        proj_away = (mean_total - mean_diff) / 2

        output.append({
            'away_team': g['away_team'],
            'home_team': g['home_team'],
            'away_sp': away_sp_name,
            'home_sp': home_sp_name,
            'mean_diff': round(mean_diff, 2),
            'median_diff': round(median_diff, 2),
            'mean_total': round(mean_total, 2),
            'median_total': round(median_total, 2),
            'proj_home': round(proj_home, 2),
            'proj_away': round(proj_away, 2),
            'home_win_pct': round(home_win_pct, 1),
            'away_win_pct': round(away_win_pct, 1),
            'home_cover_pct': round(home_cover_pct, 1),
            'away_cover_pct': round(away_cover_pct, 1),
            'push_pct': round(push_pct, 1),
            'ml_pick': ml_pick,
            'ml_conf': round(ml_conf, 1),
            'rl_pick': rl_pick,
            'rl_conf': round(rl_conf, 1),
            'run_line': game_rl,
            # Store raw arrays for O/U calculation vs external line
            '_diffs': diffs,
            '_totals': totals,
        })

    return output


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def _tier(pct):
    if pct >= 70:
        return 'HIGH'
    elif pct >= 60:
        return 'MEDIUM'
    return 'LOW'


def _implied_prob(american_odds):
    """Convert American moneyline odds to implied probability (0-100)."""
    try:
        o = float(american_odds)
    except (TypeError, ValueError):
        return None
    if o > 0:
        return round(100.0 / (o + 100) * 100, 1)
    else:
        return round(-o / (-o + 100) * 100, 1)


def print_picks(results, ou_lines=None, ml_odds=None, predict_date=None,
                num_runs=DEFAULT_NUM_RUNS, training_games=0):
    """
    Print formatted prediction table.

    ou_lines: optional dict keyed by (away_team, home_team) -> float line
              e.g. {('New York Mets', 'Washington Nationals'): 8.5}
    """
    date_str = (
        predict_date.strftime('%Y-%m-%d')
        if hasattr(predict_date, 'strftime') else str(predict_date)
    )
    print(f'\n{"=" * 70}')
    print(f'  MLB PREDICTIONS  |  {date_str}')
    print(
        f'  Training games : {training_games}   '
        f'Monte Carlo iterations : {num_runs}'
    )
    print(f'{"=" * 70}')

    if not results:
        print('  No games predicted.')
        return

    for r in results:
        away, home = r['away_team'], r['home_team']
        print(f'\n  {away} (away)  @  {home} (home)')
        away_sp = r.get('away_sp') or 'TBD'
        home_sp = r.get('home_sp') or 'TBD'
        print(f'  SP: {away_sp:<28}  vs  {home_sp}')
        print(f'  {"-" * 60}')

        # Moneyline — show book implied prob + model edge if ML odds available
        odds_pair = (ml_odds or {}).get((away, home))
        if odds_pair:
            away_ml_val, home_ml_val = odds_pair
            away_imp = _implied_prob(away_ml_val)
            home_imp = _implied_prob(home_ml_val)
            # Edge = model win% minus book implied prob (positive = value)
            home_edge = (
                round(r['home_win_pct'] - home_imp, 1)
                if home_imp is not None else None
            )
            away_edge = (
                round(r['away_win_pct'] - away_imp, 1)
                if away_imp is not None else None
            )

            def _fmt_odds(val):
                if val is None:
                    return '   N/A'
                return f'{val:+.0f}' if val != 0 else '  EVEN'

            def _fmt_edge(val):
                if val is None:
                    return ''
                sign = '+' if val >= 0 else ''
                return f'  edge {sign}{val:.1f}%'

            away_odds_str = _fmt_odds(away_ml_val)
            home_odds_str = _fmt_odds(home_ml_val)

            print(
                f'  MONEYLINE  : {r["ml_pick"]:<32} '
                f'{r["ml_conf"]:5.1f}%  [{_tier(r["ml_conf"])}]'
            )
            print(
                f'    Book ML  : {away} {away_odds_str}  '
                f'(implied {away_imp:.1f}%){_fmt_edge(away_edge)}'
            )
            print(
                f'               {home} {home_odds_str}  '
                f'(implied {home_imp:.1f}%){_fmt_edge(home_edge)}'
            )
        else:
            print(
                f'  MONEYLINE  : {r["ml_pick"]:<32} '
                f'{r["ml_conf"]:5.1f}%  [{_tier(r["ml_conf"])}]'
            )

        # Run-line
        print(
            f'  RUN LINE   : {r["rl_pick"]:<32} '
            f'{r["rl_conf"]:5.1f}%  [{_tier(r["rl_conf"])}]'
        )

        # Over/Under
        proj_total = r['mean_total']
        ou_line = (ou_lines or {}).get((away, home))
        if ou_line is not None:
            totals = r['_totals']
            over_pct = float(np.mean(totals > ou_line)) * 100
            under_pct = 100.0 - over_pct
            ou_pick = 'OVER' if over_pct >= 50 else 'UNDER'
            ou_conf = max(over_pct, under_pct)
            print(
                f'  OVER/UNDER : {ou_pick} {ou_line:<28} '
                f'{ou_conf:5.1f}%  [{_tier(ou_conf)}]  '
                f'(proj: {proj_total:.1f})'
            )
        else:
            print(
                f'  OVER/UNDER : Projected total = {proj_total:.1f}  '
                f'(provide --ou_line or ou_lines dict for pick)'
            )

        # Projected score
        print(
            f'  PROJ SCORE : {home} {r["proj_home"]:.1f}  '
            f'{away} {r["proj_away"]:.1f}'
        )

    print(f'\n{"=" * 70}')
    print('  Tiers: HIGH >= 70%  |  MEDIUM >= 60%  |  LOW < 60%')
    print(f'{"=" * 70}\n')


def save_picks_csv(results, predict_date, ou_lines=None):
    """Save picks to CSV under C:/Users/Bobby/."""
    date_str = (
        predict_date.strftime('%Y%m%d')
        if hasattr(predict_date, 'strftime') else str(predict_date)
    )
    filepath = f'C:/Users/Bobby/mlb_picks_{date_str}.csv'

    rows = []
    for r in results:
        away, home = r['away_team'], r['home_team']
        ou_line = (ou_lines or {}).get((away, home))
        totals = r['_totals']
        if ou_line is not None:
            over_pct = round(float(np.mean(totals > ou_line)) * 100, 1)
            ou_pick = 'OVER' if over_pct >= 50 else 'UNDER'
            ou_conf = max(over_pct, 100 - over_pct)
        else:
            ou_pick = ''
            ou_conf = None
            ou_line = None

        rows.append({
            'date': date_str,
            'away_team': away,
            'home_team': home,
            'ml_pick': r['ml_pick'],
            'ml_conf': r['ml_conf'],
            'ml_tier': _tier(r['ml_conf']),
            'rl_pick': r['rl_pick'],
            'rl_conf': r['rl_conf'],
            'rl_tier': _tier(r['rl_conf']),
            'ou_line': ou_line,
            'ou_pick': ou_pick,
            'ou_conf': ou_conf,
            'proj_home_score': r['proj_home'],
            'proj_away_score': r['proj_away'],
            'proj_total': r['mean_total'],
            'home_win_pct': r['home_win_pct'],
            'away_win_pct': r['away_win_pct'],
            'home_cover_pct': r['home_cover_pct'],
            'away_cover_pct': r['away_cover_pct'],
        })

    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f'  Picks saved to: {filepath}')
    return filepath


# ---------------------------------------------------------------------------
# LINES CSV LOADER  (written by mlb_lines_ui.py)
# ---------------------------------------------------------------------------

def load_lines_csv(filepath, predict_date):
    """
    Read mlb_lines.csv produced by mlb_lines_ui.py.

    CSV columns: date, away_team, home_team, run_line,
                 away_ml, home_ml, ou_line,
                 away_sp, home_sp   (optional — starting pitcher last names)

    Returns:
        games_from_lines : list of game dicts
        ou_lines         : dict  (away, home) -> float or None
        ml_odds          : dict  (away, home) -> (away_ml, home_ml)
        run_lines        : dict  (away, home) -> float (away perspective)
        starters         : dict  (away, home)
                                   -> {'away_sp': str, 'home_sp': str}
    """
    if isinstance(predict_date, str):
        predict_date = datetime.strptime(predict_date, '%Y-%m-%d').date()

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f'[ERROR] Lines file not found: {filepath}')
        return [], {}, {}, {}, {}

    # Filter to the prediction date
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df[df['date'] == predict_date]

    if df.empty:
        print(
            f'[WARN] No lines found for {predict_date} in {filepath}'
        )
        return [], {}, {}, {}, {}

    games_from_lines = []
    ou_lines = {}
    ml_odds = {}
    run_lines = {}
    starters = {}

    for _, row in df.iterrows():
        away = normalize_name(str(row.get('away_team', '')).strip())
        home = normalize_name(str(row.get('home_team', '')).strip())
        if not away or not home:
            continue

        key = (away, home)
        games_from_lines.append({
            'away_team': away,
            'home_team': home,
            'away_score': None,
            'home_score': None,
            'winner': None,
            'completed': False,
            'date': predict_date,
        })

        # Run line (away perspective: +1.5 means away is dog, -1.5 favored)
        try:
            run_lines[key] = float(row.get('run_line', 1.5))
        except (ValueError, TypeError):
            run_lines[key] = 1.5

        # O/U line
        try:
            ou_val = row.get('ou_line', '')
            ou_lines[key] = float(ou_val) if str(ou_val).strip() else None
        except (ValueError, TypeError):
            ou_lines[key] = None

        # Moneyline odds (American format, e.g. -150, +130)
        try:
            away_ml = float(row.get('away_ml', ''))
        except (ValueError, TypeError):
            away_ml = None
        try:
            home_ml = float(row.get('home_ml', ''))
        except (ValueError, TypeError):
            home_ml = None
        ml_odds[key] = (away_ml, home_ml)

        # Starting pitchers (last name, case-insensitive)
        away_sp = str(row.get('away_sp', '') or '').strip()
        home_sp = str(row.get('home_sp', '') or '').strip()
        starters[key] = {
            'away_sp': away_sp or None,
            'home_sp': home_sp or None,
        }

    print(
        f'  Loaded {len(games_from_lines)} game(s) from lines file '
        f'for {predict_date}.'
    )
    return games_from_lines, ou_lines, ml_odds, run_lines, starters


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='MLB picks: moneyline, run-line, over/under'
    )
    parser.add_argument(
        '--date', type=str, default=None,
        help='Predict date YYYY-MM-DD (default: prompt)'
    )
    parser.add_argument(
        '--runs', type=int, default=DEFAULT_NUM_RUNS,
        help=f'Monte Carlo iterations (default: {DEFAULT_NUM_RUNS})'
    )
    parser.add_argument(
        '--stats_year', type=int, default=DEFAULT_STATS_YEAR,
        help=f'Year for team stats HTML (default: {DEFAULT_STATS_YEAR})'
    )
    parser.add_argument(
        '--sched_year', type=int, default=DEFAULT_SCHED_YEAR,
        help=f'Year for schedule HTML (default: {DEFAULT_SCHED_YEAR})'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save picks to CSV'
    )
    parser.add_argument(
        '--lines', type=str, default=None,
        help=(
            'Path to mlb_lines.csv from mlb_lines_ui.py. '
            'Provides run lines, ML odds, and O/U lines. '
            'When supplied, only games in this file are predicted '
            'and the schedule HTML is not used for game selection.'
        )
    )
    parser.add_argument(
        '--prior_years', type=int, default=3,
        help=(
            'Number of full prior seasons to include in training data. '
            'e.g. --prior_years 3 on a 2026 date uses full 2023/2024/2025 '
            'seasons plus 2026 games up to the prediction date. '
            'Set to 0 to use current season only. (default: 3)'
        )
    )
    args = parser.parse_args()

    # ---- Determine prediction date ----
    if args.date:
        try:
            predict_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print('[ERROR] --date must be YYYY-MM-DD')
            sys.exit(1)
    else:
        raw = input('Enter prediction date (YYYY-MM-DD): ').strip()
        try:
            predict_date = datetime.strptime(raw, '%Y-%m-%d').date()
        except ValueError:
            print('[ERROR] Invalid date format.')
            sys.exit(1)

    num_runs = args.runs

    # ---- Load data ----
    print(f'\nLoading team stats ({args.stats_year})...')
    team_stats = load_team_stats(args.stats_year)
    if not team_stats:
        sys.exit(1)
    print(f'  Loaded {len(team_stats)} teams.')

    print(f'Loading schedule ({args.sched_year})...')
    schedule = load_schedule(args.sched_year)
    if not schedule:
        sys.exit(1)
    print(f'  Loaded {len(schedule)} games.')

    # ---- Lines file (from mlb_lines_ui.py) ----
    ou_lines = None
    ml_odds = {}
    run_lines_map = {}
    starters_today = {}
    if args.lines:
        print(f'Loading lines from {args.lines}...')
        (games_from_lines, ou_lines,
         ml_odds, run_lines_map, starters_today) = (
            load_lines_csv(args.lines, predict_date)
        )
        if not games_from_lines:
            print('[WARN] No games loaded from lines file.')
        games_today = games_from_lines
    else:
        games_today = get_games_for_date(schedule, predict_date)
        if not games_today:
            # Fall back: build game list from the MLB Stats API (handles
            # dates not yet in the downloaded schedule HTML)
            print(f'  [INFO] No games in schedule HTML for {predict_date}. '
                  'Fetching from MLB Stats API...')
            api_starters = fetch_probable_starters(
                predict_date.strftime('%Y-%m-%d')
            )
            if api_starters:
                for (away_team, home_team) in api_starters:
                    games_today.append({
                        'date':       predict_date,
                        'away_team':  away_team,
                        'home_team':  home_team,
                        'away_score': None,
                        'home_score': None,
                        'completed':  False,
                    })
                print(f'  {len(games_today)} game(s) fetched from API.')
            else:
                print(f'\n[INFO] No games found for {predict_date}.')
                sys.exit(0)
        else:
            print(f'  {len(games_today)} game(s) on {predict_date}.')

    if not games_today:
        print('[INFO] No games to predict.')
        sys.exit(0)

    # ---- Auto-fetch probable starters from MLB Stats API ----
    date_str = predict_date.strftime('%Y-%m-%d')
    print(f'\nFetching probable starters for {date_str}...')
    fetched_starters = fetch_probable_starters(date_str)
    if fetched_starters:
        # Merge: manual UI entries override auto-fetched values
        starters_today = _merge_starters(fetched_starters, starters_today)
        found = sum(
            1 for s in starters_today.values()
            if s.get('away_sp') or s.get('home_sp')
        )
        print(f'  Probable starters fetched for {found} game(s).')
        for (at, ht), sp in sorted(starters_today.items()):
            a = sp.get('away_sp') or 'TBD'
            h = sp.get('home_sp') or 'TBD'
            print(f'    {at:<28} {a:<24}  vs  {ht:<28} {h}')
    else:
        print('  No starters returned (API unavailable or off-season).')

    if not games_today:
        print('[INFO] No games to predict.')
        sys.exit(0)

    # ---- Training data ----
    completed = get_completed_before(schedule, predict_date)
    print(
        f'  Training on {len(completed)} completed games '
        f'before {predict_date}.'
    )
    if len(completed) < 10:
        print(
            '[WARN] Very few completed games — '
            'predictions may be unreliable.'
        )

    if args.prior_years > 0:
        # Load full prior seasons and combine with current-season games
        games_by_year = {args.sched_year: completed}
        stats_by_year = {args.sched_year: team_stats}
        pitcher_stats_by_year = {}
        starters_by_year = {
            args.sched_year: load_starters_csv(args.sched_year)
        }
        for offset in range(1, args.prior_years + 1):
            py = args.sched_year - offset
            prior_stats = load_team_stats(py)
            prior_sched = load_schedule(py)
            if prior_stats and prior_sched:
                # All completed games from that full season
                prior_completed = [
                    g for g in prior_sched
                    if g['completed'] and g['away_score'] is not None
                ]
                games_by_year[py] = prior_completed
                stats_by_year[py] = prior_stats
                pitcher_stats_by_year[py] = load_pitcher_stats(py)
                starters_by_year[py] = load_starters_csv(py)
                print(
                    f'  Prior season {py}: '
                    f'{len(prior_completed)} completed games loaded.'
                )
            else:
                print(f'  [WARN] Prior season {py}: data not found, skipping.')
        total_games = sum(len(v) for v in games_by_year.values())
        print(f'  Total training games (all years): {total_games}')
        X, y_diff, y_total = build_training_data_multiyear(
            games_by_year, stats_by_year,
            pitcher_stats_by_year=pitcher_stats_by_year,
            starters_by_year=starters_by_year,
        )
    else:
        X, y_diff, y_total = build_training_data(
            completed, team_stats,
            pitcher_stats=load_pitcher_stats(args.stats_year),
            starters_lookup=load_starters_csv(args.sched_year),
        )

    if X is None or len(X) == 0:
        print(
            '[ERROR] Could not build training data. '
            'Check team name mapping.'
        )
        sys.exit(1)

    # ---- Pitcher stats ----
    print(f'\nLoading pitcher stats ({args.stats_year})...')
    pitcher_stats = load_pitcher_stats(args.stats_year)

    # Load prior-year stats for early-season blending
    prev_year = args.stats_year - 1
    print(f'Loading prior-year pitcher stats ({prev_year}) for blending...')
    prev_pitcher_stats = load_pitcher_stats(prev_year)

    # Report which starters were found / not found / blended
    MIN_BLEND_GS = 5
    if starters_today:
        for (at, ht), sp_info in starters_today.items():
            for side, name in [
                ('Away', sp_info.get('away_sp')),
                ('Home', sp_info.get('home_sp')),
            ]:
                if not name:
                    continue
                last = name.strip().split()[-1].upper()
                cur  = pitcher_stats.get(last)
                prev = prev_pitcher_stats.get(last)
                if cur is not None and cur.get('gs', MIN_BLEND_GS) >= MIN_BLEND_GS:
                    status = f'found ({cur["gs"]} GS, current season)'
                elif cur is not None and prev is not None:
                    gs = cur.get('gs', 0)
                    w  = gs / MIN_BLEND_GS
                    status = (
                        f'blended ({gs} GS cur / {prev["gs"]} GS prev, '
                        f'{w:.0%} cur + {1-w:.0%} prior)'
                    )
                elif cur is None and prev is not None:
                    status = f'prior season only ({prev["gs"]} GS)'
                else:
                    status = 'NOT FOUND (using avg)'
                print(f'  {side} SP {name}: {status}')

    # ---- Predict ----
    print(f'\nRunning {num_runs} Monte Carlo iterations...')
    results = predict_games(
        games_today, team_stats, X, y_diff, y_total,
        num_runs=num_runs,
        run_lines_map=run_lines_map,
        pitcher_stats=pitcher_stats,
        prev_pitcher_stats=prev_pitcher_stats,
        starters_today=starters_today,
    )

    # ---- Display ----
    print_picks(
        results,
        ou_lines=ou_lines,
        ml_odds=ml_odds,
        predict_date=predict_date,
        num_runs=num_runs,
        training_games=len(completed),
    )

    if args.save:
        save_picks_csv(results, predict_date, ou_lines=ou_lines)


if __name__ == '__main__':
    main()
