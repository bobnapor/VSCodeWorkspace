"""The yearly projections workbook, and the data that fills it: spreads from sportsbooks and quarterback injury flags.

The workbook has three tabs, rebuilt from scratch every run after the games already in it are read back (by header
text), so nothing typed in is lost and older layouts convert on their own:
  This Week   what to bet now: the season scoreboard, this week's underdog picks and favorite picks with the model's
              reasons, the teaser legs and every game of the week.  Its cells point into the Season Log, so an edit
              there shows up here.
  Season Log  one row per game: the running record and the place to type spreads, final margins and notes (yellow).
              Detail columns are grouped at the right; click the + above them to show them.
  Model Info  the editable settings, the backtest tables behind Hist. Win %, and how to read the workbook.
The file is written to a temporary copy first and then swapped in, so a failed save never leaves a half-written log.
"""

import logging
import math
import os
import re
import shutil
import time
from datetime import datetime

import pandas as pd
import requests

# Spreads that sit exactly on a key number (used by the research scripts; the pick rule does not depend on them)
KEY_LINES = (3.0, 6.0, 7.0, 10.0, 14.0)
# Wong teaser legs (see nfl_signal.py): underdogs of +1.5/+2.5 and favorites of -7.5/-8.5, teased 6 points
TEASER_DOG_LINES = (1.5, 2.5)
TEASER_FAV_LINES = (7.5, 8.5)
TEASER_POINTS = 6
# Favorite picks (see nfl_signal.py): night games, kickoff at this hour Eastern or later
NIGHT_KICKOFF_HOUR = 19
PLAYOFF_WEEK_NAMES = {'WC': 'WildCard', 'DIV': 'Division', 'CON': 'ConfChamp', 'SB': 'SuperBowl'}

SHEET_THIS_WEEK, SHEET_LOG, SHEET_INFO = 'This Week', 'Season Log', 'Model Info'

# Season Log columns: what you look at first, then results, notes and reasons, then the detail columns (grouped, collapsed)
LOG_LAYOUT = (
    ('week', 'Week'), ('away', 'Away'), ('home', 'Home'), ('spread', 'Home Spread'),
    ('pick', 'Pick'), ('fav_pick', 'Favorite Pick'), ('teaser', 'Teaser Leg'), ('dog_correction', 'Lean to Underdog (pts)'),
    ('hit_rate', 'Hist. Win %'), ('actual', 'Final Margin (Home)'), ('result', 'Pick Result'), ('fav_result', 'Fav Pick Result'),
    ('teaser_result', 'Teaser Result'), ('clv', 'Line vs Close'), ('qb_flag', 'QB Note'), ('notes', 'Notes'), ('reason', 'Why the Model Leans'),
    ('fair_line', 'Fair Line (Home)'), ('fair_vs_spread', 'Fair Line vs Spread'), ('side', 'Model Side'),
    ('covering', 'Covering Team'), ('side_result', 'Model Side Result'), ('close', 'Closing Spread (Home)'),
    ('correction', 'Model Correction (Home)'), ('kickoff', 'Kickoff (ET)'), ('fav_clv', 'Fav Line vs Close'), ('year', 'Year'),
    ('line_written', 'Line Written (Home)'),
)


def _column_letter(index):
    """Excel column letters for a 1-based column number (A ... Z, AA, ...)."""
    letters = ''
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


COL = {key: index for index, (key, _) in enumerate(LOG_LAYOUT, start=1)}
LETTER = {key: _column_letter(index) for key, index in COL.items()}
LAST_VISIBLE = COL['reason']
LAST_COLUMN = len(LOG_LAYOUT)
LOG_WIDTHS = {'week': 6, 'away': 20, 'home': 20, 'spread': 8, 'pick': 15, 'fav_pick': 15, 'teaser': 15, 'dog_correction': 9,
              'hit_rate': 8, 'actual': 9, 'result': 8, 'fav_result': 8, 'teaser_result': 8, 'clv': 8, 'qb_flag': 30, 'notes': 24,
              'reason': 90, 'fair_line': 9, 'fair_vs_spread': 9, 'side': 12, 'covering': 12, 'side_result': 9, 'close': 9,
              'correction': 10, 'kickoff': 9, 'fav_clv': 9, 'year': 6, 'line_written': 9}
# The reason shows its headline (first line) only, so every game stays one row high; the full text is in the cell
ONE_LINE_FIELDS = ('reason',)
LOG_ROW_HEIGHT = 17
# Names the columns went by in earlier layouts, for reading older workbooks
LEGACY_HEADERS = {
    'Home Team': 'home', 'Away Team': 'away', 'Spread (as Home)': 'spread', 'Actual Score Difference (as Home)': 'actual',
    'Fair Line (as Home)': 'fair_line', 'QB Injury Flag': 'qb_flag', 'Model Correction (as Home)': 'correction',
    'Closing Spread (as Home)': 'close', 'Underdog Signal': 'pick', 'Underdog-Side Correction': 'dog_correction',
    'Correction Toward Underdog': 'dog_correction', 'Cover Probability (Side to Cover)': 'hit_rate',
}
READ_HEADERS = {**{header: key for key, header in LOG_LAYOUT}, **LEGACY_HEADERS}
REQUIRED_FIELDS = ('week', 'year', 'home', 'away', 'spread', 'actual', 'notes')
MANUAL_INPUT_FIELDS = ('spread', 'actual', 'notes')

# Model Info: settings in column B, then the hit-rate table, the teaser table and the favorite-pick table below them
INFO_SETTINGS = {
    'threshold': ('Minimum lean to underdog (pts)', 0.0, 5),
    'first_week': ('First pick week', 5, 6),
    'last_week': ('Last pick week', 16, 7),
    'fav_threshold': ('Minimum lean to favorite (pts)', 0.5, 8),
}
LEGACY_SETTING_LABELS = {'Minimum correction toward underdog (pts)': 'threshold'}
HIT_TABLE_HEADERS = ('Lean to underdog from (pts)', 'Win %', 'Games', 'Which picks')
LEGACY_HIT_TABLE_HEADER = 'Correction toward underdog from'
HIT_TABLE_FIRST_ROW, HIT_TABLE_ROWS = 13, 5
HIT_TABLE_NOTES = {-99: 'Model leans the favorite (see Favorite picks)', 0: 'Underdog, lean under 0.25', 0.25: 'Underdog, lean 0.25-0.5',
                   0.5: 'Underdog, lean 0.5-1', 1: 'Underdog, lean 1 or more'}
TEASER_HEADERS = ('Teaser leg (weeks 5-16)', 'Win % since 1999', 'Win % since 2018')
TEASER_TABLE_FIRST_ROW = 22
FAV_TABLE_HEADERS = ('Favorite picks, night games (weeks 5-16)', 'Win %', 'Games')
FAV_TABLE_FIRST_ROW, FAV_TABLE_ROWS = 28, 4
FORMAT_LAST_ROW = 1000

# Colors
NAVY, WHITE, GRAY_TEXT = '1F4E78', 'FFFFFF', '595959'
MANUAL_HEADER, MANUAL_CELL, DETAIL_HEADER = 'FFD966', 'FFF8E1', '7F7F7F'
WEEK_BAND, CARD_FILL, SECTION_FILL = 'EEF3F8', 'F2F2F2', 'DDEBF7'
PICK_FILL, FAV_FILL, TEASER_FILL, WIN_FILL, LOSS_FILL = 'BDD7EE', 'E4DFEC', 'FCE4D6', 'C6EFCE', 'FFC7CE'
SIGNED_GENERAL = '+General;-General;0'


def translate_week_number(week_str, year):
    """Numeric week for a week label; playoff rounds follow the regular season (18 games from 2021, 17 before)."""
    if str(week_str).isdigit():
        return int(week_str)
    offset = 18 if int(year) >= 2021 else 17
    rounds = {'WildCard': 1, 'Division': 2, 'ConfChamp': 3, 'SuperBowl': 4}
    return offset + rounds[week_str] if week_str in rounds else None


def get_last_regular_season_week(year):
    return 18 if int(year) >= 2021 else 17


def get_current_season_year(today=None):
    """NFL season year for today's date; January and February still belong to the season that just ended."""
    today = today or datetime.now()
    return today.year if today.month >= 3 else today.year - 1


def anchored_fair_line(spread, correction):
    """Fair line for the home side: the spread-implied margin moved by the correction, in half points.

    The move is at least half a point whenever there is a correction, so Fair Line vs Spread never rounds to zero
    and the Model Side always matches the side the correction favors.
    """
    move = math.copysign(max(0.5, round(abs(correction) * 2) / 2), correction) if correction else 0.0
    return -spread + move + 0.0


def fair_line_as_home(projected_difference):
    """Projected home margin rounded to the nearest half point, same sign as the projection.

    +0.6 (home wins by 0.6) gives +0.5; -9.2 (home loses by 9.2) gives -9.0.
    """
    return round(projected_difference * 2) / 2 + 0.0


def get_desktop_path():
    """Desktop folder, honoring OneDrive redirection."""
    import ctypes
    from ctypes import wintypes
    buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
    CSIDL_DESKTOPDIRECTORY = 0x10
    ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_DESKTOPDIRECTORY, None, 0, buf)
    return buf.value


DRAFTKINGS_BOOK_ID = 68  # "DK NJ" on Action Network
ACTION_NETWORK_SCOREBOARD_URL = 'https://api.actionnetwork.com/web/v2/scoreboard/nfl'

# Books to take a spread from, in order of preference: DraftKings first, then the consensus line, then
# other sportsbooks, with the opening line last. Only used for games DraftKings has no line for.
SPREAD_BOOK_PREFERENCE = (
    (DRAFTKINGS_BOOK_ID, 'DraftKings NJ'),
    (1534, 'DraftKings PA'),
    (15, 'Consensus'),
    (69, 'FanDuel NJ'),
    (79, 'Action Network book 79'),
    (71, 'Action Network book 71'),
    (75, 'Action Network book 75'),
    (74, 'Action Network book 74'),
    (972, 'Action Network book 972'),
    (123, 'Action Network book 123'),
    (30, 'Opening line'),
)


def fetch_draftkings_spreads(year, week, max_retries=3):
    """Point spreads for one regular-season week from Action Network's public scoreboard feed.

    Uses the DraftKings line when there is one and otherwise the first available book in
    SPREAD_BOOK_PREFERENCE, logging each game that had to use a different book.

    Returns {(home_team, away_team): spread as home} where a negative number means the home team is favored,
    matching the Home Spread column. A game that has kicked off (under way or finished) returns its closing line:
    the feed then also carries live in-game lines, which are ignored. Returns {} when the feed is unavailable or has
    no lines for that season (Action Network has none before 2023), so the sheet is left blank.
    """
    params = {'bookIds': ','.join(str(book_id) for book_id, _ in SPREAD_BOOK_PREFERENCE), 'season': year, 'week': week}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}
    for attempt in range(max_retries):
        try:
            response = requests.get(ACTION_NETWORK_SCOREBOARD_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            games = response.json().get('games', [])
            break
        except (requests.RequestException, ValueError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                logging.warning(f"Could not fetch spreads for {year} week {week}: {e}")
                return {}

    spreads = {}
    for game in games:
        names = {team['id']: team['full_name'] for team in game.get('teams', [])}
        home = names.get(game.get('home_team_id'))
        away = names.get(game.get('away_team_id'))
        if not (home and away):
            continue
        for book_id, book_name in SPREAD_BOOK_PREFERENCE:
            lines = game.get('markets', {}).get(str(book_id), {}).get('event', {}).get('spread', [])
            # once a game kicks off the feed adds live in-game lines; the pre-game line it keeps is the closing line
            home_line = next((line['value'] for line in lines if line.get('side') == 'home' and not line.get('is_live')), None)
            if home_line is None:
                continue
            spreads[(home, away)] = float(home_line)
            if book_id != DRAFTKINGS_BOOK_ID:
                logging.info(f"No DraftKings spread for {away} at {home} (week {week}); using {book_name}: {home_line}")
            break
    return spreads


_qb_flag_data = {}
ESPN_INJURIES_URL = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries'


def fetch_espn_injuries():
    """ESPN's live injury list, {(team, name key): (status, injury, short comment, time)}, or {} if unavailable.  It
    carries practice news and injured reserve before nflverse's copy of the official reports catches up."""
    from nfl_injury_features import TEAM_RENAMES, name_key
    try:
        response = requests.get(ESPN_INJURIES_URL, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        teams = response.json().get('injuries', [])
    except (requests.RequestException, ValueError) as e:
        logging.warning(f"Could not load ESPN's injury list ({e}); QB notes use the nflverse reports only")
        return {}
    live = {}
    for team in teams:
        for item in team.get('injuries', []):
            athlete = item.get('athlete') or {}
            code = (athlete.get('team') or {}).get('abbreviation')
            live[(TEAM_RENAMES.get(code, code), name_key(athlete.get('displayName')))] = (
                item.get('status'), (item.get('details') or {}).get('type'), item.get('shortComment'), item.get('date'))
    return live


def fetch_qb_injury_flags(year, week):
    """Quarterback notes for one week's games from nflverse's injury reports and weekly rosters and ESPN's live injury
    list, or {} if unavailable.

    Downloads fresh schedule, injury and roster files once per run, since the reports update through the week (the
    rosters carry injured reserve, which drops a player off the reports; ESPN carries practice news before the reports
    reach nflverse).  See nfl_injury_features.qb_injury_flags for exactly what gets noted.
    """
    global _qb_flag_data
    try:
        if _qb_flag_data.get('year') != year:
            from nflverse_data import load_games, load_injuries, load_weekly_rosters
            try:
                rosters = load_weekly_rosters(year, year, refresh=True)
            except Exception as e:
                logging.warning(f"Could not load the {year} weekly rosters ({e}); QB notes will not show injured reserve")
                rosters = None
            _qb_flag_data = {'year': year, 'games': load_games(refresh=True), 'injuries': load_injuries(year, year, refresh=True),
                             'rosters': rosters, 'live': fetch_espn_injuries()}
        from nfl_injury_features import qb_injury_flags
        flags = qb_injury_flags(_qb_flag_data['games'], _qb_flag_data['injuries'], year, week, _qb_flag_data['rosters'],
                                _qb_flag_data['live'])
        for (home, away), note in flags.items():
            logging.info(f"QB note, week {week}, {away} at {home}: {note}")
        return flags
    except Exception as e:
        logging.warning(f"Could not load quarterback injury data for {year} week {week}: {e}")
        return {}


def _logged_dk_lines(year):
    """{(week, home, away): DraftKings home spreads the runs logged in line_log.csv} for `year`, full team names.  Lets a
    workbook from before the Line Written column tell a spread this code wrote from one typed in."""
    from nfl_injury_features import TEAM_FULL_NAMES
    from nflverse_data import CACHE_DIR
    path = os.path.join(CACHE_DIR, 'line_log.csv')
    try:
        log = pd.read_csv(path)
    except (OSError, ValueError):
        return {}
    log = log[(log['season'] == int(year)) & log['dk_home_spread'].notna()]
    lines = {}
    for row in log.itertuples():
        key = (int(row.week), TEAM_FULL_NAMES.get(row.home_team, row.home_team), TEAM_FULL_NAMES.get(row.away_team, row.away_team))
        lines.setdefault(key, set()).add(float(row.dk_home_spread))
    return lines


# ---------------------------------------------------------------- reading an existing workbook

def _field_columns(ws):
    """{field: column} for the columns a sheet has, matched by header text so any earlier layout can be read."""
    fields = {}
    for col in range(1, min(ws.max_column, 40) + 1):
        key = READ_HEADERS.get(ws.cell(1, col).value)
        if key and key not in fields:
            fields[key] = col
    return fields


def _find_log_sheet(wb):
    """The sheet holding the games: the Season Log, or in an older workbook the first sheet with the basic columns."""
    if SHEET_LOG in wb.sheetnames:
        return wb[SHEET_LOG]
    for ws in wb.worksheets:
        if all(field in _field_columns(ws) for field in REQUIRED_FIELDS):
            return ws
    return None


def _cells(wb, max_row=60, max_col=40):
    for ws in (wb.worksheets if wb is not None else []):
        for row in ws.iter_rows(min_row=1, max_row=max_row, max_col=max_col):
            for cell in row:
                if cell.value is not None:
                    yield ws, cell


def _settings_from(wb):
    """The editable settings the workbook holds, found by their labels on any sheet; defaults where missing."""
    labels = {label: key for key, (label, _default, _row) in INFO_SETTINGS.items()}
    labels.update(LEGACY_SETTING_LABELS)
    found = {}
    for ws, cell in _cells(wb):
        if cell.value in labels and labels[cell.value] not in found:
            value = ws.cell(cell.row, cell.column + 1).value
            if value not in (None, ''):
                found[labels[cell.value]] = value
    return {key: found.get(key, default) for key, (_label, default, _row) in INFO_SETTINGS.items()}


def _hit_rate_table_from(wb):
    """The hit-rate table the workbook already holds, [(lean from, rate, games)], for runs that cannot recompute it."""
    for ws, cell in _cells(wb):
        if cell.value in (HIT_TABLE_HEADERS[0], LEGACY_HIT_TABLE_HEADER) and ws.cell(cell.row, cell.column + 2).value == 'Games':
            rows = [tuple(ws.cell(r, cell.column + i).value for i in range(3)) for r in range(cell.row + 1, cell.row + 1 + HIT_TABLE_ROWS)]
            return [r for r in rows if all(isinstance(v, (int, float)) for v in r)]
    return []


def _teaser_table_from(wb):
    """The teaser-leg table the workbook already holds, [(label, rate since 1999, rate since 2018)]."""
    for ws, cell in _cells(wb):
        if cell.value == TEASER_HEADERS[0]:
            rows = [tuple(ws.cell(r, cell.column + i).value for i in range(3)) for r in (cell.row + 1, cell.row + 2)]
            return [r for r in rows if isinstance(r[1], (int, float))]
    return []


def _favorite_table_from(wb):
    """The favorite-pick table the workbook already holds, [(era, rate, games)]."""
    for ws, cell in _cells(wb):
        if cell.value == FAV_TABLE_HEADERS[0]:
            rows = [tuple(ws.cell(r, cell.column + i).value for i in range(3)) for r in range(cell.row + 1, cell.row + 1 + FAV_TABLE_ROWS)]
            return [r for r in rows if isinstance(r[1], (int, float))]
    return []


# ---------------------------------------------------------------- small helpers

def _nickname(cell):
    """Excel formula text for the team nickname (last word of the name) in `cell`, so derived columns stay short."""
    return f'TRIM(RIGHT(SUBSTITUTE({cell}," ",REPT(" ",30)),30))'


def _nick(name):
    return str(name).split()[-1]


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fill(color):
    from openpyxl.styles import PatternFill
    return PatternFill(fill_type='solid', start_color=color, end_color=color)


def _lean_to_dog(record):
    """The model's lean toward the underdog at the record's spread (rounded like the sheet), or None."""
    spread, correction = record['spread'], record['correction']
    if not (_number(spread) and _number(correction)) or spread == 0:
        return None
    return round(correction if spread > 0 else -correction, 3)


def _in_pick_weeks(record, settings):
    week = record['week']
    return isinstance(week, int) and settings['first_week'] <= week <= settings['last_week']


def _is_pick(record, settings):
    lean = _lean_to_dog(record)
    return _in_pick_weeks(record, settings) and lean is not None and lean > 0 and lean >= settings['threshold']


def _kickoff_text(value):
    """A kickoff as 'HH:MM' text: as written by the runner, or from a cell Excel turned into a time."""
    if hasattr(value, 'hour') and hasattr(value, 'minute'):
        return f'{value.hour:02d}:{value.minute:02d}'
    return value if isinstance(value, str) and value else None


def _night_game(kickoff):
    """Whether an 'HH:MM' Eastern kickoff is at night (as nfl_signal.is_night_game)."""
    try:
        return int(str(kickoff).split(':')[0]) >= NIGHT_KICKOFF_HOUR
    except ValueError:
        return False


def _is_favorite_pick(record, settings):
    """As nfl_signal.is_favorite_pick: the pick weeks, a night game, and a lean to the favorite of at least the minimum."""
    lean = _lean_to_dog(record)
    return (_in_pick_weeks(record, settings) and lean is not None and lean < 0 and -lean >= settings['fav_threshold']
            and _night_game(record.get('kickoff')))


def _teaser_leg(record, settings):
    """(team nickname, teased spread text, 'Underdog'/'Favorite') for a Wong teaser leg, or None."""
    spread = record['spread']
    if not _number(spread) or not _in_pick_weeks(record, settings):
        return None
    line, home_is_dog = abs(spread), spread > 0
    if line in TEASER_DOG_LINES:
        return _nick(record['home'] if home_is_dog else record['away']), f'+{line + TEASER_POINTS:g}', 'Underdog'
    if line in TEASER_FAV_LINES:
        return _nick(record['away'] if home_is_dog else record['home']), f'-{line - TEASER_POINTS:g}', 'Favorite'
    return None


def _split_reason(text):
    """(headline without its 'Titans by 0.7: ' lead, detail lines) of a reason from nfl_reasons; ('', []) without one."""
    if not text:
        return '', []
    lines = str(text).split('\n')
    lead = re.match(r'^[^:\n]+ by (under )?[0-9.]+: ', lines[0])
    return (lines[0][lead.end():] if lead else lines[0]), [line for line in lines[1:] if line.strip()]


def _lean_text(record):
    """'Titans by 0.7 (underdog)' for the side the model leans to, or '' without a correction."""
    correction = record['correction']
    if not _number(correction):
        return ''
    if abs(correction) < 0.05:
        return 'No real lean'
    team = record['home'] if correction >= 0 else record['away']
    text = f'{_nick(team)} by {abs(correction):.1f}'
    spread = record['spread']
    if _number(spread) and spread != 0:
        team_is_dog = (spread > 0) == (correction >= 0)
        text += ' (underdog)' if team_is_dog else ' (favorite)'
    return text


# ---------------------------------------------------------------- the three tabs

def _write_info(ws, settings, hit_rate_table, teaser_table, through_season, favorite_table=()):
    """Model Info: settings (yellow, editable), the backtest tables and how to read the workbook."""
    from openpyxl.styles import Alignment, Font

    ws.sheet_view.showGridLines = False
    for letter, width in zip('ABCD', (34, 14, 12, 46)):
        ws.column_dimensions[letter].width = width
    ws['A1'] = 'Model Info and Settings'
    ws['A1'].font = Font(size=16, bold=True, color=NAVY)
    ws['A2'] = f'Tables cover every regular-season game from 1999 through {through_season}, weeks 5-16, each scored by a model that never saw it. Recomputed every run.'
    ws['A2'].font = Font(italic=True, color=GRAY_TEXT)

    def section(row, title):
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
        cell = ws.cell(row, 1, title)
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = _fill(NAVY)

    def header(row, titles):
        for col, title in enumerate(titles, start=1):
            cell = ws.cell(row, col, title)
            cell.font = Font(bold=True)
            cell.fill = _fill(SECTION_FILL)
            cell.alignment = Alignment(horizontal='center' if col > 1 else 'left')

    section(4, 'SETTINGS  (edit the yellow cells; the next run keeps them)')
    for key, (label, _default, row) in INFO_SETTINGS.items():
        ws.cell(row, 1, label).font = Font(bold=True)
        cell = ws.cell(row, 2, settings[key])
        cell.fill = _fill(MANUAL_HEADER)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
    ws.cell(9, 1, 'A Pick is the underdog whenever the model leans to it by more than the minimum, in the pick weeks.').font = Font(italic=True, color=GRAY_TEXT)
    ws.cell(10, 1, 'A Favorite Pick is the favorite in a night game (kickoff 7 pm Eastern or later) when the model leans to it by at '
                   'least its minimum, in the pick weeks.').font = Font(italic=True, color=GRAY_TEXT)

    section(HIT_TABLE_FIRST_ROW - 2, "HOW OFTEN THE MODEL'S SIDE COVERED  (Hist. Win % in the Season Log)")
    header(HIT_TABLE_FIRST_ROW - 1, HIT_TABLE_HEADERS)
    for r, (edge, rate, games) in enumerate(hit_rate_table[:HIT_TABLE_ROWS], start=HIT_TABLE_FIRST_ROW):
        ws.cell(r, 1, edge).alignment = Alignment(horizontal='left')
        ws.cell(r, 2, rate).number_format = '0.0%'
        ws.cell(r, 2).alignment = Alignment(horizontal='center')
        ws.cell(r, 3, games).alignment = Alignment(horizontal='center')
        ws.cell(r, 4, HIT_TABLE_NOTES.get(edge, ''))
    ws.cell(HIT_TABLE_FIRST_ROW + HIT_TABLE_ROWS, 1, 'Break-even at -110 odds is 52.4%. A lean of -99 means the model favors the favorite.').font = Font(italic=True, color=GRAY_TEXT)

    section(TEASER_TABLE_FIRST_ROW - 2, 'TEASER LEGS  (6 points through 3 and 7; pair two legs in one teaser)')
    header(TEASER_TABLE_FIRST_ROW - 1, TEASER_HEADERS)
    for r, (label, all_rate, recent_rate) in enumerate(teaser_table, start=TEASER_TABLE_FIRST_ROW):
        ws.cell(r, 1, label)
        for col, value in ((2, all_rate), (3, recent_rate)):
            ws.cell(r, col, value).number_format = '0.0%'
            ws.cell(r, col).alignment = Alignment(horizontal='center')
    ws.cell(TEASER_TABLE_FIRST_ROW + 2, 1, 'A 2-team teaser needs 72.4% per leg at -110 and 73.9% at -120.').font = Font(italic=True, color=GRAY_TEXT)

    section(FAV_TABLE_FIRST_ROW - 2, 'FAVORITE PICKS  (night games where the model leans to the favorite by the minimum or more)')
    header(FAV_TABLE_FIRST_ROW - 1, FAV_TABLE_HEADERS)
    for r, (era, rate, games) in enumerate(list(favorite_table)[:FAV_TABLE_ROWS], start=FAV_TABLE_FIRST_ROW):
        ws.cell(r, 1, era).alignment = Alignment(horizontal='left')
        ws.cell(r, 2, rate).number_format = '0.0%'
        ws.cell(r, 2).alignment = Alignment(horizontal='center')
        ws.cell(r, 3, games).alignment = Alignment(horizontal='center')
    ws.cell(FAV_TABLE_FIRST_ROW + FAV_TABLE_ROWS, 1, 'At the default 0.5-point minimum. Thinner evidence than the underdog picks: '
                                                     'expect about 55%.').font = Font(italic=True, color=GRAY_TEXT)

    guide_row = FAV_TABLE_FIRST_ROW + FAV_TABLE_ROWS + 2
    section(guide_row, 'HOW TO READ THIS WORKBOOK')
    guide = (
        'This Week: the season scoreboard, then this week\'s picks with the reason for each, the teaser legs and every game.',
        'Pick: the underdog to bet, at the spread shown. The model leans to it; about 55% of these covered in every era tested.',
        'Favorite Pick: the favorite to lay points with, at the spread shown, in a Thursday, Sunday or Monday night game when the',
        '     model leans to it by 0.5 or more (56-65% in every era since 1999, about one a week; expect about 55%).',
        'Lean: how many points the model thinks the spread is off toward that team. Hist. Win %: how often that size of lean covered.',
        'Why: the themes behind the lean (special teams, pace, defense, ...), strongest first, then each one\'s points of lean and its',
        '     main rating with both teams\' numbers (season-to-date, adjusted for opponents and recency). "(worse)": the team is',
        '     worse there and it still helps them, because the line has historically overrated that gap. "Steady" effects have',
        '     pointed the same way across the training seasons; effects that flipped are weaker evidence.',
        'Teaser Leg: a published 6-point teaser leg (about 75% of legs won); pair two of them in one teaser.',
        'Season Log: yellow columns are typed by hand.  Home Spread fills in from DraftKings and follows the line until',
        '     kickoff; a spread you type yourself (the line you bet) is kept, and so are final margins and notes.',
        '     Why the Model Leans shows the headline; select the cell to read the whole reason in the formula bar.',
        '     Line vs Close: points your pick got versus the closing line (positive = you beat the close).',
        '     Detail columns (fair line, model side, closing spread, model correction, kickoff, the favorite picks\' line versus',
        '     the close, and the line this code last wrote) are grouped at the right: click + to show.',
        'Settings above: change a minimum lean or the pick weeks; the formulas pick up the change at once.',
    )
    for i, line in enumerate(guide, start=guide_row + 1):
        ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=4)
        ws.cell(i, 1, line).alignment = Alignment(wrap_text=True, vertical='top')


def _write_log(ws, records, settings):
    """Season Log: one row per game; returns {game key: row number} for the This Week references."""
    from openpyxl.formatting.rule import CellIsRule, FormulaRule
    from openpyxl.styles import Alignment, Border, Font, Side
    from openpyxl.utils import get_column_letter

    info = f"'{SHEET_INFO}'!"
    setting = {key: f'{info}$B${row}' for key, (_label, _default, row) in INFO_SETTINGS.items()}
    hit_from = f'{info}$A${HIT_TABLE_FIRST_ROW}:$A${HIT_TABLE_FIRST_ROW + HIT_TABLE_ROWS - 1}'
    hit_rate = f'{info}$B${HIT_TABLE_FIRST_ROW}:$B${HIT_TABLE_FIRST_ROW + HIT_TABLE_ROWS - 1}'

    ws.freeze_panes = 'D2'
    for key, title in LOG_LAYOUT:
        cell = ws.cell(1, COL[key], title)
        detail = COL[key] > LAST_VISIBLE
        manual = key in MANUAL_INPUT_FIELDS
        cell.font = Font(bold=True, color='000000' if manual else WHITE)
        cell.fill = _fill(MANUAL_HEADER if manual else DETAIL_HEADER if detail else NAVY)
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        ws.column_dimensions[get_column_letter(COL[key])].width = LOG_WIDTHS[key]
    ws.row_dimensions[1].height = 32

    centered = {'week', 'spread', 'pick', 'fav_pick', 'teaser', 'dog_correction', 'hit_rate', 'actual', 'result', 'fav_result',
                'teaser_result', 'clv', 'fair_line', 'fair_vs_spread', 'side', 'covering', 'side_result', 'close', 'correction',
                'kickoff', 'fav_clv', 'year', 'line_written'}
    formats = {'spread': SIGNED_GENERAL, 'close': SIGNED_GENERAL, 'actual': SIGNED_GENERAL, 'fair_line': '+0.0;-0.0;0.0',
               'fair_vs_spread': '+0.0;-0.0;0.0', 'dog_correction': '+0.0;-0.0;0.0', 'hit_rate': '0%', 'clv': '+0.0;-0.0;0.0',
               'correction': '+0.00;-0.00;0.00', 'fav_clv': '+0.0;-0.0;0.0', 'line_written': SIGNED_GENERAL}
    outline = Side(style='medium', color=NAVY)
    row_of = {}

    def week_group(record):
        return (int(record['year']), translate_week_number(str(record['week']), record['year']) or 99)

    groups = [week_group(r) for r in records]
    band = {}
    for g in groups:
        band.setdefault(g, len(band) % 2 == 1)

    for r, record in enumerate(records, start=2):
        c = {key: f'{letter}{r}' for key, letter in LETTER.items()}
        home_name, away_name = _nickname(c['home']), _nickname(c['away'])
        spread, actual, dog = c['spread'], c['actual'], c['dog_correction']
        in_weeks = f'AND(ISNUMBER({c["week"]}),{c["week"]}>={setting["first_week"]},{c["week"]}<={setting["last_week"]})'
        values = {
            'week': record['week'], 'away': record['away'], 'home': record['home'], 'spread': record['spread'],
            'actual': record['actual'], 'reason': record.get('reason'), 'qb_flag': record.get('qb_flag'), 'notes': record['notes'],
            'fair_line': record['fair_line'], 'close': record.get('close'), 'correction': record['correction'], 'year': record['year'],
            'line_written': record.get('line_written'),
            # the correction measured toward the underdog (positive = the model likes the underdog more than the spread does)
            'dog_correction': f'=IF(OR({spread}="",{c["correction"]}="",{spread}=0),"",ROUND(IF({spread}>0,{c["correction"]},-{c["correction"]}),3))',
            'pick': f'=IF({dog}="","",IF(AND({in_weeks},{dog}>0,{dog}>={setting["threshold"]}),'
                    f'IF({spread}>0,{home_name}&" +"&ABS({spread}),{away_name}&" +"&ABS({spread})),""))',
            'hit_rate': f'=IF({dog}="","",IFERROR(LOOKUP({dog},{hit_from},{hit_rate}),""))',
            'fair_vs_spread': f'={c["fair_line"]}+{spread}',
            'side': f'=IF({c["fair_vs_spread"]}>0,{home_name},{away_name})',
            'covering': f'=IF(OR({spread}="",{actual}=""),"",IF({actual}+{spread}=0,"Push",IF(({actual}+{spread})>0,{home_name},{away_name})))',
            'side_result': f'=IF({c["covering"]}="","",IF({c["covering"]}="Push","Push",IF({c["covering"]}={c["side"]},"Win","Loss")))',
            'result': f'=IF({c["pick"]}="","",{c["side_result"]})',
            'clv': f'=IF(OR({c["pick"]}="",{c["close"]}="",{spread}=""),"",IF({spread}>0,{spread}-{c["close"]},{c["close"]}-{spread}))',
            # favorite picks: night games where the model leans to the favorite (a negative lean to the underdog) by the minimum
            'kickoff': record.get('kickoff'),
            'fav_pick': f'=IF(OR({dog}="",{c["kickoff"]}=""),"",IF(AND({in_weeks},{dog}<0,-{dog}>={setting["fav_threshold"]},'
                        f'IFERROR(VALUE(LEFT({c["kickoff"]},2)),0)>={NIGHT_KICKOFF_HOUR}),'
                        f'IF({spread}<0,{home_name}&" -"&ABS({spread}),{away_name}&" -"&ABS({spread})),""))',
            'fav_result': f'=IF({c["fav_pick"]}="","",{c["side_result"]})',
            'fav_clv': f'=IF(OR({c["fav_pick"]}="",{c["close"]}="",{spread}=""),"",IF({spread}<0,{spread}-{c["close"]},{c["close"]}-{spread}))',
        }
        dog_leg = 'OR(' + ','.join(f'ABS({spread})={v:g}' for v in TEASER_DOG_LINES) + ')'
        fav_leg = 'OR(' + ','.join(f'ABS({spread})={v:g}' for v in TEASER_FAV_LINES) + ')'
        values['teaser'] = (f'=IF(OR({spread}="",NOT({in_weeks})),"",IF({dog_leg},IF({spread}>0,{home_name},{away_name})&" +"&(ABS({spread})+{TEASER_POINTS}),'
                            f'IF({fav_leg},IF({spread}>0,{away_name},{home_name})&" -"&(ABS({spread})-{TEASER_POINTS}),"")))')
        home_teased = f'OR(AND({dog_leg},{spread}>0),AND({fav_leg},{spread}<0))'
        values['teaser_result'] = (f'=IF(OR({c["teaser"]}="",{actual}=""),"",IF({home_teased},IF({actual}+{spread}+{TEASER_POINTS}>0,"Win","Loss"),'
                                   f'IF({actual}+{spread}<{TEASER_POINTS},"Win","Loss")))')

        group = week_group(record)
        first = r == 2 or groups[r - 3] != group
        last = r == len(records) + 1 or groups[r - 1] != group
        for key, _title in LOG_LAYOUT:
            col = COL[key]
            cell = ws.cell(r, col, values[key])
            if key in formats:
                cell.number_format = formats[key]
            if key in ONE_LINE_FIELDS:
                cell.alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
            else:
                cell.alignment = Alignment(horizontal='center' if key in centered else 'left', vertical='center')
            if key in MANUAL_INPUT_FIELDS:
                cell.fill = _fill(MANUAL_CELL)
            elif band[group] and col <= LAST_VISIBLE:
                cell.fill = _fill(WEEK_BAND)
            if col <= LAST_VISIBLE:
                cell.border = Border(left=outline if col == 1 else None, right=outline if col == LAST_VISIBLE else None,
                                     top=outline if first else None, bottom=outline if last else None)
        ws.cell(r, COL['pick']).font = Font(bold=True)
        ws.cell(r, COL['fav_pick']).font = Font(bold=True)
        ws.row_dimensions[r].height = LOG_ROW_HEIGHT
        row_of[(str(record['week']), str(record['year']), record['home'], record['away'])] = r

    # conditional formatting: picks blue, favorite picks lilac, teaser legs orange, wins green, losses red, line value green/red
    last_row = FORMAT_LAST_ROW
    for key, color in (('pick', PICK_FILL), ('fav_pick', FAV_FILL), ('teaser', TEASER_FILL)):
        ws.conditional_formatting.add(f'{LETTER[key]}2:{LETTER[key]}{last_row}',
                                      FormulaRule(formula=[f'LEN(${LETTER[key]}2)>0'], fill=_fill(color)))
    for key in ('result', 'fav_result', 'teaser_result', 'side_result'):
        for outcome, color in (('Win', WIN_FILL), ('Loss', LOSS_FILL)):
            ws.conditional_formatting.add(f'{LETTER[key]}2:{LETTER[key]}{last_row}',
                                          FormulaRule(formula=[f'${LETTER[key]}2="{outcome}"'], fill=_fill(color)))
    for key in ('clv', 'fav_clv'):
        ws.conditional_formatting.add(f'{LETTER[key]}2:{LETTER[key]}{last_row}', CellIsRule(operator='greaterThan', formula=['0'], fill=_fill(WIN_FILL)))
        ws.conditional_formatting.add(f'{LETTER[key]}2:{LETTER[key]}{last_row}', CellIsRule(operator='lessThan', formula=['0'], fill=_fill(LOSS_FILL)))

    ws.auto_filter.ref = f'A1:{get_column_letter(LAST_VISIBLE)}{max(len(records) + 1, 2)}'
    # the detail columns form one collapsed outline group (click + above the column after them to show them)
    for col in range(LAST_VISIBLE + 1, LAST_COLUMN + 1):
        dimension = ws.column_dimensions[get_column_letter(col)]
        dimension.outline_level = 1
        dimension.hidden = True
    ws.column_dimensions[get_column_letter(LAST_COLUMN + 1)].collapsed = True
    return row_of


def _write_this_week(ws, records, row_of, week, year, settings, teaser_table, favorite_table=()):
    """This Week: the scoreboard, this week's underdog and favorite picks with reasons, teaser legs and every game, all
    pointing into the log."""
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Border, Font, Side

    ws.sheet_view.showGridLines = False
    for letter, width in zip('ABCDEF', (25, 16, 20, 15, 24, 13)):
        ws.column_dimensions[letter].width = width
    log = f"'{SHEET_LOG}'!"
    span = lambda key: f'{log}${LETTER[key]}$2:${LETTER[key]}${FORMAT_LAST_ROW}'
    thin = Side(style='thin', color='BFBFBF')

    def ref(key, row, blank_if_empty=True):
        cell = f'{log}{LETTER[key]}{row}'
        return f'=IF({cell}="","",{cell})' if blank_if_empty else f'={cell}'

    def merged(row, text, font=None, fill=None, height=None, wrap=False, vertical='center', indent=None):
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
        cell = ws.cell(row, 1, text)
        if font:
            cell.font = font
        if fill:
            cell.fill = _fill(fill)
        cell.alignment = Alignment(wrap_text=wrap, vertical=vertical, horizontal='left',
                                   indent=indent if indent is not None else 1 if fill else 0)
        if height:
            ws.row_dimensions[row].height = height
        return cell

    def header(row, titles):
        for col, title in enumerate(titles, start=1):
            cell = ws.cell(row, col, title)
            cell.font = Font(bold=True, color=NAVY)
            cell.fill = _fill(SECTION_FILL)
            cell.alignment = Alignment(horizontal='left' if col == 1 else 'center', vertical='center', wrap_text=True)
            cell.border = Border(bottom=thin)

    title = f'{year} Week {week}' if str(week).isdigit() else f'{year} {week}'
    merged(1, f'NFL Picks  -  {title}', Font(size=20, bold=True, color=NAVY), height=34)
    merged(2, f'Updated {datetime.now():%A %B %d, %I:%M %p}.  Straight picks are underdogs the model leans to (about 55% '
              f'against the spread in every era tested); favorite picks are night-game favorites it leans to by '
              f'{settings["fav_threshold"]}+ points (56-65% in every era, thinner evidence: expect about 55%); teaser legs '
              f'follow the published 6-point rule (about 75% of legs).',
           Font(italic=True, color=GRAY_TEXT), height=45, wrap=True)

    # scoreboard
    count = lambda key, outcome: f'COUNTIF({span(key)},"{outcome}")'
    number = translate_week_number(str(week), year)
    previous = number - 1 if number else None
    if previous and settings['first_week'] <= previous <= settings['last_week']:
        last_label = f'Picks week {previous}'
        last_week = (f'=COUNTIFS({span("week")},{previous},{span("result")},"Win")&"-"&'
                     f'COUNTIFS({span("week")},{previous},{span("result")},"Loss")')
    else:
        last_label, last_week = 'Picks last week', '-'
    cards = (
        ('Picks this season', f'={count("result", "Win")}&"-"&{count("result", "Loss")}&"-"&{count("result", "Push")}', None),
        ('Pick win %', f'=IFERROR({count("result", "Win")}/({count("result", "Win")}+{count("result", "Loss")}),"-")', '0.0%'),
        (last_label, last_week, None),
        ('Teaser legs', f'={count("teaser_result", "Win")}&"-"&{count("teaser_result", "Loss")}', None),
        ('Teaser leg win %', f'=IFERROR({count("teaser_result", "Win")}/({count("teaser_result", "Win")}+{count("teaser_result", "Loss")}),"-")', '0.0%'),
        ('Avg line vs close', f'=IFERROR(AVERAGE({span("clv")}),"-")', '+0.00;-0.00;0.00'),
    )
    if previous and settings['first_week'] <= previous <= settings['last_week']:
        fav_last_label = f'Fav picks week {previous}'
        fav_last_week = (f'=COUNTIFS({span("week")},{previous},{span("fav_result")},"Win")&"-"&'
                         f'COUNTIFS({span("week")},{previous},{span("fav_result")},"Loss")')
    else:
        fav_last_label, fav_last_week = 'Fav picks last week', '-'
    both = lambda outcome: f'({count("result", outcome)}+{count("fav_result", outcome)})'
    fav_cards = (
        ('Favorite picks', f'={count("fav_result", "Win")}&"-"&{count("fav_result", "Loss")}&"-"&{count("fav_result", "Push")}', None),
        ('Favorite pick win %', f'=IFERROR({count("fav_result", "Win")}/({count("fav_result", "Win")}+{count("fav_result", "Loss")}),"-")', '0.0%'),
        (fav_last_label, fav_last_week, None),
        ('All straight picks', f'={both("Win")}&"-"&{both("Loss")}&"-"&{both("Push")}', None),
        ('All picks win %', f'=IFERROR({both("Win")}/({both("Win")}+{both("Loss")}),"-")', '0.0%'),
        ('Fav line vs close', f'=IFERROR(AVERAGE({span("fav_clv")}),"-")', '+0.00;-0.00;0.00'),
    )
    for top_row, row_cards in ((4, cards), (6, fav_cards)):
        for col, (label, formula, number_format) in enumerate(row_cards, start=1):
            top, bottom = ws.cell(top_row, col, label), ws.cell(top_row + 1, col, formula)
            top.font = Font(size=9, bold=True, color=GRAY_TEXT)
            bottom.font = Font(size=15, bold=True, color=NAVY)
            for cell in (top, bottom):
                cell.fill = _fill(CARD_FILL if top_row == 4 else FAV_FILL)
                cell.alignment = Alignment(horizontal='center', vertical='center')
            if number_format:
                bottom.number_format = number_format
        ws.row_dimensions[top_row + 1].height = 26
    ws.cell(8, 1, 'Records count only the picks; wins and losses at the spread, pushes refunded.').font = Font(size=9, italic=True, color=GRAY_TEXT)

    def why(row, record):
        """The reason under a pick: a bold headline, then its detail lines; returns the row after the block."""
        headline, details = _split_reason(record.get('reason'))
        merged(row, 'Why: ' + (headline or 'no reason recorded'), Font(size=10, bold=True, color=NAVY), height=17, indent=1)
        block_end = row
        if details:
            row += 1
            merged(row, '\n'.join(details), Font(size=9, color=GRAY_TEXT), wrap=True, height=12.75 * len(details) + 5,
                   vertical='top', indent=3)
            block_end = row
        for col in range(1, 7):
            ws.cell(block_end, col).border = Border(bottom=thin)
        return row + 1

    this_week = [r for r in records if str(r['week']) == str(week) and str(r['year']) == str(year)]
    picks = sorted((r for r in this_week if _is_pick(r, settings)), key=lambda r: -_lean_to_dog(r))
    row = 10
    merged(row, f"THIS WEEK'S PICKS ({len(picks)})  -  bet the underdog at the number shown",
           Font(bold=True, size=12, color=WHITE), NAVY, height=22)
    row += 1
    header(row, ('Game', 'Pick', 'Lean (pts)', 'Hist. Win %', 'QB note', 'Result'))
    row += 1
    first_result_row = row
    if not picks:
        reason = ('picks are made in weeks %s-%s' % (settings['first_week'], settings['last_week'])
                  if not any(_in_pick_weeks(r, settings) for r in this_week) else 'the model does not lean to any underdog')
        merged(row, f'No picks this week ({reason}).', Font(italic=True, color=GRAY_TEXT))
        row += 1
    for record in picks:
        r = row_of[(str(record['week']), str(record['year']), record['home'], record['away'])]
        cells = (f'{_nick(record["away"])} @ {_nick(record["home"])}', ref('pick', r), ref('dog_correction', r), ref('hit_rate', r),
                 ref('qb_flag', r), ref('result', r))
        for col, value in enumerate(cells, start=1):
            cell = ws.cell(row, col, value)
            cell.alignment = Alignment(horizontal='left' if col in (1, 5) else 'center', vertical='center', wrap_text=col == 5)
        ws.cell(row, 1).font = Font(bold=True)
        ws.cell(row, 2).font = Font(bold=True, size=12)
        ws.cell(row, 2).fill = _fill(PICK_FILL)
        ws.cell(row, 3).number_format = '+0.0;-0.0;0.0'
        ws.cell(row, 4).number_format = '0%'
        qb_lines = math.ceil(len(str(record.get('qb_flag') or '')) / 26)
        ws.row_dimensions[row].height = max(22, 15 * qb_lines + 4)
        row = why(row + 1, record)

    row += 1
    favorites = sorted((r for r in this_week if _is_favorite_pick(r, settings)), key=_lean_to_dog)
    merged(row, f'FAVORITE PICKS ({len(favorites)})  -  lay the points with the favorite at the number shown (night games)',
           Font(bold=True, size=12, color=WHITE), NAVY, height=22)
    row += 1
    header(row, ('Game', 'Favorite pick', 'Lean (pts)', 'Hist. Win %', 'Kickoff and QB note', 'Result'))
    row += 1
    favorite_rate = list(favorite_table)[-1][1] if favorite_table else None
    if not favorites:
        reason = ('picks are made in weeks %s-%s' % (settings['first_week'], settings['last_week'])
                  if not any(_in_pick_weeks(r, settings) for r in this_week)
                  else 'the model does not lean to a favorite in a night game by %s or more' % settings['fav_threshold'])
        merged(row, f'No favorite picks this week ({reason}).', Font(italic=True, color=GRAY_TEXT))
        row += 1
    for record in favorites:
        r = row_of[(str(record['week']), str(record['year']), record['home'], record['away'])]
        lean_cell, qb_cell = f'{log}{LETTER["dog_correction"]}{r}', f'{log}{LETTER["qb_flag"]}{r}'
        cells = (f'{_nick(record["away"])} @ {_nick(record["home"])}', ref('fav_pick', r), f'=IF({lean_cell}="","",-{lean_cell})',
                 favorite_rate, f'="Kickoff {record["kickoff"]} ET"&IF({qb_cell}="","","; "&{qb_cell})', ref('fav_result', r))
        for col, value in enumerate(cells, start=1):
            cell = ws.cell(row, col, value)
            cell.alignment = Alignment(horizontal='left' if col in (1, 5) else 'center', vertical='center', wrap_text=col == 5)
        ws.cell(row, 1).font = Font(bold=True)
        ws.cell(row, 2).font = Font(bold=True, size=12)
        ws.cell(row, 2).fill = _fill(FAV_FILL)
        ws.cell(row, 3).number_format = '+0.0;-0.0;0.0'
        ws.cell(row, 4).number_format = '0%'
        qb_lines = math.ceil((len(str(record.get('qb_flag') or '')) + 20) / 26)
        ws.row_dimensions[row].height = max(22, 15 * qb_lines + 4)
        row = why(row + 1, record)

    row += 1
    legs = [(r, _teaser_leg(r, settings)) for r in this_week]
    legs = [(r, leg) for r, leg in legs if leg]
    merged(row, f'TEASER LEGS ({len(legs)})  -  6-point teaser legs; pair two legs in one teaser', Font(bold=True, size=12, color=WHITE), NAVY, height=22)
    row += 1
    header(row, ('Game', 'Teaser leg', 'Leg type', 'Win % since 1999', 'Win % since 2018', 'Result'))
    row += 1
    rates = {('Underdog' if str(label).startswith('Underdog') else 'Favorite'): (a, b) for label, a, b in teaser_table}
    if not legs:
        merged(row, 'No teaser legs this week.', Font(italic=True, color=GRAY_TEXT))
        row += 1
    for record, (team, teased, kind) in legs:
        r = row_of[(str(record['week']), str(record['year']), record['home'], record['away'])]
        all_rate, recent_rate = rates.get(kind, (None, None))
        cells = (f'{_nick(record["away"])} @ {_nick(record["home"])}', ref('teaser', r), kind, all_rate, recent_rate, ref('teaser_result', r))
        for col, value in enumerate(cells, start=1):
            cell = ws.cell(row, col, value)
            cell.alignment = Alignment(horizontal='left' if col == 1 else 'center', vertical='center')
            cell.border = Border(bottom=thin)
        ws.cell(row, 2).font = Font(bold=True)
        ws.cell(row, 2).fill = _fill(TEASER_FILL)
        for col in (4, 5):
            ws.cell(row, col).number_format = '0.0%'
        row += 1

    row += 1
    merged(row, f'EVERY GAME THIS WEEK ({len(this_week)})', Font(bold=True, size=12, color=WHITE), NAVY, height=22)
    row += 1
    header(row, ('Game', 'Home spread', 'Model leans', 'Pick', 'Teaser leg', 'Pick result'))
    row += 1
    first_game_row = row

    def either(first, second, r):
        a, b = f'{log}{LETTER[first]}{r}', f'{log}{LETTER[second]}{r}'
        return f'=IF({a}<>"",{a},IF({b}<>"",{b},""))'

    for record in this_week:
        r = row_of[(str(record['week']), str(record['year']), record['home'], record['away'])]
        cells = (f'{_nick(record["away"])} @ {_nick(record["home"])}', ref('spread', r), _lean_text(record), either('pick', 'fav_pick', r),
                 ref('teaser', r), either('result', 'fav_result', r))
        for col, value in enumerate(cells, start=1):
            cell = ws.cell(row, col, value)
            cell.alignment = Alignment(horizontal='left' if col == 1 else 'center', vertical='center')
            cell.border = Border(bottom=thin)
        ws.cell(row, 2).number_format = SIGNED_GENERAL
        row += 1
    if this_week:
        # a favorite pick lays points ("Chiefs -3.5"), an underdog pick takes them ("Bills +3")
        ws.conditional_formatting.add(f'D{first_game_row}:D{row - 1}', FormulaRule(
            formula=[f'ISNUMBER(SEARCH(" -",$D{first_game_row}))'], fill=_fill(FAV_FILL), stopIfTrue=True))
        for col_letter, color in (('D', PICK_FILL), ('E', TEASER_FILL)):
            ws.conditional_formatting.add(f'{col_letter}{first_game_row}:{col_letter}{row - 1}',
                                          FormulaRule(formula=[f'LEN(${col_letter}{first_game_row})>0'], fill=_fill(color)))
    for col_letter in ('F',):
        for outcome, color in (('Win', WIN_FILL), ('Loss', LOSS_FILL)):
            ws.conditional_formatting.add(f'{col_letter}{first_result_row}:{col_letter}{row}',
                                          FormulaRule(formula=[f'${col_letter}{first_result_row}="{outcome}"'], fill=_fill(color)))

    row += 1
    for note in ('Lean: points the model thinks the spread is off toward that team.  Hist. Win %: how often the model\'s side '
                 'covered at that size of lean (1999 to date); for favorite picks, how often every favorite pick since 1999 covered.',
                 'Why: the themes that moved the model, strongest first, then for each its points of lean and its main rating '
                 'with both teams\' numbers.  "(worse)" means the team is worse there and it still helps them: the line has '
                 'historically overrated that gap.  Steady effects pointed the same way in the earlier and later training '
                 'years; effects that flipped are weaker evidence.',
                 'Every cell here points into the Season Log; type spreads, final margins and notes there.  Settings and '
                 'backtests: Model Info.'):
        merged(row, note, Font(size=9, italic=True, color=GRAY_TEXT), wrap=True, height=12 * math.ceil(len(note) / 135) + 5)
        row += 1


# ---------------------------------------------------------------- the writer

def write_projections_spreadsheet(rows, year, output_dir=None, actual_lookup=None, spread_fetcher=None, qb_flag_fetcher=None,
                                  hit_rate_table=None, teaser_table=None, closing_lookup=None, current_week=None, favorite_table=None):
    """Write projections to <Desktop>/<year>_nfl_projections.xlsx, appending to it if it already exists.

    The workbook is the season's running log: each run adds its week's games in week order (a week run after later
    weeks is inserted after the weeks before it), keeps every spread, note and recorded correction or reason, refreshes
    final margins (`actual_lookup`, {(week, year, home, away): difference}, played games only, so 0 is a tie) and closing spreads (`closing_lookup`) for
    games that have been played, fills blank spreads from `spread_fetcher(year, week)` (DraftKings via Action Network
    in production) and moves a spread this code wrote (remembered in Line Written) with the line until the game is played,
    while a spread typed into the sheet is never replaced, and refreshes QB notes for unplayed games from
    `qb_flag_fetcher(year, week)`.  The fair line is the spread moved by the model correction, so the model side always
    agrees with the pick.  `rows` may carry a 'reason' (the plain-words explanation), kept once recorded like the
    correction, and a 'kickoff' ('HH:MM' Eastern, which decides favorite picks), refreshed whenever a row brings one
    since the league can move a game into or out of a night slot.  `current_week` is the week the This Week tab shows
    (default: the week of the rows passed in).  `favorite_table` is nfl_signal.favorite_backtest's [(era, rate, games)].

    Returns (output_path, rows_added, actuals_updated, spreads_filled).
    """
    from openpyxl import Workbook, load_workbook

    output_dir = output_dir or get_desktop_path()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{year}_nfl_projections.xlsx')

    old = load_workbook(output_path) if os.path.exists(output_path) else None
    log_sheet = _find_log_sheet(old) if old is not None else None
    if old is not None and log_sheet is None:
        backup = output_path.replace('.xlsx', f'_unreadable_{datetime.now():%Y%m%d_%H%M%S}.xlsx')
        shutil.copy2(output_path, backup)
        logging.warning(f"{output_path} has no games sheet this code can read; copied it to {backup} and starting fresh")
    if old is not None and log_sheet is not None and SHEET_LOG not in old.sheetnames:
        backup = output_path.replace('.xlsx', '_before_new_layout.xlsx')
        if not os.path.exists(backup):
            shutil.copy2(output_path, backup)
            logging.info(f"Converting the workbook to the three-tab layout; the old one is saved as {backup}")
    settings = _settings_from(old)
    # a table that couldn't be recomputed this run keeps what the workbook already holds
    hit_rate_table = [r for r in (hit_rate_table or []) if r[1] is not None] or _hit_rate_table_from(old)
    teaser_table = [(label, all_rate, recent) for label, all_rate, _n, recent, _m in (teaser_table or [])] or _teaser_table_from(old)
    favorite_table = [r for r in (favorite_table or []) if r[1] is not None] or _favorite_table_from(old)

    def week_value(week):
        return int(week) if str(week).isdigit() else week

    def game_key(week, game_year, home, away):
        return (str(week_value(week)), str(game_year), home, away)

    def played_actual(actual):
        # Unplayed games carry a placeholder difference of 0 in `rows`
        return None if actual in (0, None) or pd.isna(actual) else int(actual)

    actuals = {}
    for key, actual in (actual_lookup or {}).items():
        if actual is not None and not pd.isna(actual):  # only played games are looked up, so a 0 here is a tie
            actuals[game_key(*key)] = int(actual)
    for row in rows:
        if played_actual(row['actual']) is not None:
            actuals[game_key(row['week'], row['year'], row['home_team'], row['away_team'])] = played_actual(row['actual'])
    closes = {game_key(*key): spread for key, spread in (closing_lookup or {}).items() if spread is not None and not pd.isna(spread)}

    # Read the games already in the workbook
    records, existing, actuals_updated = [], {}, 0
    if log_sheet is not None:
        fields = _field_columns(log_sheet)
        for r in range(2, log_sheet.max_row + 1):
            value = lambda field: log_sheet.cell(r, fields[field]).value if field in fields else None
            if not value('home'):
                continue
            record = {field: value(field) for field in ('week', 'year', 'home', 'away', 'fair_line', 'spread', 'actual', 'notes',
                                                         'qb_flag', 'correction', 'close', 'reason', 'kickoff', 'line_written')}
            record['week'] = week_value(record['week'])
            record['kickoff'] = _kickoff_text(record['kickoff'])
            key = game_key(record['week'], record['year'], record['home'], record['away'])
            if key in actuals and record['actual'] != actuals[key]:
                record['actual'] = actuals[key]
                actuals_updated += 1
            existing[key] = record
            records.append(record)

    added = 0
    for row in rows:
        key = game_key(row['week'], row['year'], row['home_team'], row['away_team'])
        if key in existing:
            # the correction and its reason are recorded once, like the fair line, so the sheet shows what was known
            record = existing[key]
            if record['correction'] in (None, ''):
                record['correction'] = row.get('correction')
            if record.get('reason') in (None, '') and row.get('reason'):
                record['reason'] = row['reason']
            if row.get('kickoff'):
                record['kickoff'] = row['kickoff']
            continue
        if row.get('backfill'):  # corrections for earlier weeks fill games already in the sheet but never add new ones
            continue
        record = {
            'week': week_value(row['week']), 'year': int(row['year']), 'home': row['home_team'], 'away': row['away_team'],
            'fair_line': fair_line_as_home(float(row['projected'])), 'spread': None, 'actual': actuals.get(key), 'notes': None,
            'qb_flag': None, 'correction': row.get('correction'), 'close': None, 'reason': row.get('reason'),
            'kickoff': row.get('kickoff'), 'line_written': None,
        }
        existing[key] = record
        records.append(record)
        added += 1

    for record in records:
        key = game_key(record['week'], record['year'], record['home'], record['away'])
        if key in closes:
            record['close'] = float(closes[key])

    # Quarterback injury notes for games not yet played; reports change through the week so these refresh every run
    if qb_flag_fetcher:
        fetched_flags = {}
        for record in records:
            if record['actual'] not in (None, '') or not str(record['week']).isdigit():
                continue
            week_number = int(record['week'])
            if week_number > get_last_regular_season_week(record['year']):
                continue
            if week_number not in fetched_flags:
                fetched_flags[week_number] = qb_flag_fetcher(int(year), week_number)
            record['qb_flag'] = fetched_flags[week_number].get((record['home'], record['away']))

    # Spreads for games not yet played follow the sportsbook feed until kickoff: a blank spread is filled, and a spread
    # this code wrote (remembered in Line Written) moves with the line, since a pick or teaser leg holds only at the line
    # you can bet.  A spread typed into the sheet (the line you bet) is never replaced.
    spreads_filled = 0
    if spread_fetcher:
        fetched_weeks, logged = {}, None
        for record in records:
            if not str(record['week']).isdigit():
                continue
            has_spread = record['spread'] not in (None, '')
            if has_spread and record['actual'] not in (None, ''):
                continue
            week_number = int(record['week'])
            if week_number > get_last_regular_season_week(record['year']):
                continue
            if week_number not in fetched_weeks:
                fetched_weeks[week_number] = spread_fetcher(int(year), week_number)
            line = fetched_weeks[week_number].get((record['home'], record['away']))
            if line is None:
                continue
            if not has_spread:
                record['spread'] = record['line_written'] = line
                spreads_filled += 1
                continue
            spread = record['spread'] if _number(record['spread']) else None
            if _number(record.get('line_written')):
                written_by_code = spread == record['line_written']
            else:
                # workbooks from before Line Written: a spread equal to a DraftKings line a run logged was written by the code
                logged = _logged_dk_lines(year) if logged is None else logged
                written_by_code = spread in logged.get((week_number, record['home'], record['away']), ())
            if line == spread:
                if written_by_code:
                    record['line_written'] = line
            elif written_by_code:
                logging.info(f"Week {week_number} {record['away']} at {record['home']}: the line moved from {spread:g} to {line:g}, "
                             f"so the sheet now has {line:g} (type the line you bet into Home Spread to keep it)")
                record['spread'] = record['line_written'] = line
            else:
                logging.warning(f"Week {week_number} {record['away']} at {record['home']}: kept the spread typed into the sheet "
                                f"({record['spread']}); the current line is {line:g}, and a pick holds only at the line you can bet")

    # The fair line is the spread moved by the correction, so the model side always agrees with the pick.
    # Games without both numbers keep the stats-only line they were first written with.
    for record in records:
        if _number(record['spread']) and _number(record['correction']):
            record['fair_line'] = anchored_fair_line(record['spread'], record['correction'])

    def season_order(record):
        week_number = translate_week_number(str(record['week']), record['year'])
        return (int(record['year']), week_number if week_number is not None else 99)

    # Stable sort: a week run late slots in after earlier weeks and before later ones, and games
    # within a week keep the order they were first written in
    records.sort(key=season_order)
    if current_week is None:
        fresh = [r for r in rows if not r.get('backfill')]
        current_week = week_value(fresh[0]['week']) if fresh else (records[-1]['week'] if records else 1)

    wb = Workbook()
    this_week = wb.active
    this_week.title = SHEET_THIS_WEEK
    log = wb.create_sheet(SHEET_LOG)
    info = wb.create_sheet(SHEET_INFO)
    _write_info(info, settings, hit_rate_table, teaser_table, int(year), favorite_table)
    row_of = _write_log(log, records, settings)
    _write_this_week(this_week, records, row_of, current_week, int(year), settings, teaser_table, favorite_table)
    wb.active = 0

    # write a temporary copy and swap it in, so a failed save never leaves a half-written workbook
    temporary = output_path + '.saving'
    try:
        wb.save(temporary)
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return output_path, added, actuals_updated, spreads_filled
