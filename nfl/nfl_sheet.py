"""The yearly projections spreadsheet: layout, formulas, spreads from sportsbooks and quarterback injury flags.

Everything that touches the workbook lives here so the model and the weekly runner stay separate from it.
"""

import logging
import math
import os
import time
from datetime import datetime

import pandas as pd
import requests

# The table behind the Backtest ATS Win % column (how often the model's side covered, by size of the correction toward
# the underdog) is computed each run by nfl_signal.backtest_hit_rates and written to the sheet; nothing is typed in.
HIT_RATE_HEADER = 'Backtest ATS Win % (at this edge)'
HIT_RATE_TABLE_ROWS = 5   # rows 2-6, just above the two settings

# Column order: what you act on first (week, teams, spread, the pick), then the model's numbers, then the result,
# with the long text (QB flags, notes) and the bookkeeping columns last so the sheet fits on one screen.
# The sheet is laid out again from these every run and read back by header text, so older layouts migrate.
LAYOUT = (
    ('week', 'Week'), ('home', 'Home Team'), ('away', 'Away Team'), ('spread', 'Spread (as Home)'),
    ('pick', 'Underdog Signal'), ('dog_correction', 'Correction Toward Underdog'), ('hit_rate', HIT_RATE_HEADER),
    ('fair_line', 'Fair Line (as Home)'), ('fair_vs_spread', 'Fair Line vs Spread'), ('side', 'Side to Cover'),
    ('actual', 'Actual Score Difference (as Home)'), ('covering', 'Covering Team'), ('result', 'Result'),
    ('qb_flag', 'QB Injury Flag'), ('notes', 'Notes'), ('year', 'Year'), ('correction', 'Model Correction (as Home)'),
)
COL = {key: index for index, (key, _) in enumerate(LAYOUT, start=1)}
LETTER = {key: chr(64 + index) for key, index in COL.items()}   # fewer than 26 columns, so A-Z is enough
LAST_SHEET_COLUMN = len(LAYOUT)
# Names the columns went by in earlier layouts, for reading older sheets
LEGACY_HEADERS = {'Cover Probability (Side to Cover)': 'hit_rate', 'Underdog-Side Correction': 'dog_correction'}
READ_HEADERS = {**{header: key for key, header in LAYOUT}, **LEGACY_HEADERS}
REQUIRED_FIELDS = ('week', 'year', 'home', 'away', 'fair_line', 'spread', 'actual', 'notes')
# Columns filled in by hand
MANUAL_INPUT_FIELDS = ('spread', 'actual', 'notes')
MANUAL_INPUT_FILL_COLOR = 'FFFF99'
# Settings and the edge/hit-rate lookup table sit one column to the right of the data
TABLE_COLUMNS = (LAST_SHEET_COLUMN + 2, LAST_SHEET_COLUMN + 3, LAST_SHEET_COLUMN + 4)   # correction from, cover rate, games
TABLE_LETTERS = tuple(chr(64 + c) for c in TABLE_COLUMNS)
TABLE_EDGE_HEADER = 'Correction toward underdog from'
# Editable settings under the hit-rate table: key -> (label, default, row)
SETTINGS = {
    'threshold': ('Signal threshold (pts)', 0.5, 7),
    'picks': ('Picks per week', 3, 8),
    'first_week': ('First pick week', 5, 9),
    'last_week': ('Last pick week', 16, 10),
}
SETTINGS_NOTE_ROW = 12
PREVIOUS_DEFAULT_PICKS_PER_WEEK = 4   # sheets that still hold the old default are moved to the new one
MAX_QB_FLAG_WIDTH = 40
MAX_NOTES_WIDTH = 30
MIN_NOTES_WIDTH = 24

PLAYOFF_WEEK_NAMES = {'WC': 'WildCard', 'DIV': 'Division', 'CON': 'ConfChamp', 'SB': 'SuperBowl'}


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
    and Side to Cover always matches the side the correction favors.
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


RESULT_FORMAT_RULES = (
    ('Win', 'C6EFCE'),   # light green
    ('Loss', 'FFC7CE'),  # light red
)
SIGNAL_FILL_COLOR = 'BDD7EE'   # light blue
FORMAT_LAST_ROW = 1000          # formatting and the weekly ranking cover rows well past the current data


def _add_conditional_formatting(ws):
    """Green/red on Covering Team and Result for a win/loss, and light blue on a flagged underdog."""
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import PatternFill

    result_range = f"{LETTER['covering']}2:{LETTER['result']}{FORMAT_LAST_ROW}"
    for result, color in RESULT_FORMAT_RULES:
        ws.conditional_formatting.add(
            result_range,
            FormulaRule(formula=[f'${LETTER["result"]}2="{result}"'], fill=PatternFill(start_color=color, end_color=color, fill_type='solid')),
        )
    pick = LETTER['pick']
    ws.conditional_formatting.add(
        f'{pick}2:{pick}{FORMAT_LAST_ROW}',
        FormulaRule(formula=[f'LEN(${pick}2)>0'], fill=PatternFill(start_color=SIGNAL_FILL_COLOR, end_color=SIGNAL_FILL_COLOR, fill_type='solid')),
    )


def _hit_rate_table_from(ws):
    """The cover-rate table the sheet already holds, [(correction from, cover rate, games)], for runs that can't recompute it.

    Only rows that carry a game count are taken, which leaves out the typed-in tables older versions wrote.
    """
    for row in ws.iter_rows(min_row=1, max_row=1, max_col=40):
        for cell in row:
            if cell.value == TABLE_EDGE_HEADER:
                rows = [tuple(ws.cell(r, cell.column + i).value for i in range(3)) for r in range(2, 2 + HIT_RATE_TABLE_ROWS)]
                return [r for r in rows if isinstance(r[2], (int, float)) and isinstance(r[1], (int, float))]
    return []


def _reset_sheet(ws, settings, hit_rate_table, through_season):
    """Clear the sheet and lay out the header, settings table, formatting and panes from scratch.

    Rewriting it every run is what lets an older layout turn into the current one: the games were already read
    back by header text, and the formulas are written again for wherever the columns now are.
    `hit_rate_table` is [(correction from, cover rate, games)] for the Backtest ATS Win % column.
    """
    from openpyxl.formatting.formatting import ConditionalFormattingList
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    ws.delete_rows(1, ws.max_row)
    ws.conditional_formatting = ConditionalFormattingList()
    ws.column_dimensions.clear()
    ws.row_dimensions.clear()
    ws.freeze_panes = 'A2'

    manual_fill = PatternFill(fill_type='solid', start_color=MANUAL_INPUT_FILL_COLOR, end_color=MANUAL_INPUT_FILL_COLOR)
    for col, (key, header) in enumerate(LAYOUT, start=1):
        cell = ws.cell(1, col, header)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = Border(bottom=Side(style='medium'))
        if key in MANUAL_INPUT_FIELDS:
            cell.fill = manual_fill

    edge_col, rate_col, games_col = TABLE_COLUMNS
    for col, title in ((edge_col, TABLE_EDGE_HEADER), (rate_col, 'Backtest ATS Win %'), (games_col, 'Games')):
        cell = ws.cell(1, col, title)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = Border(bottom=Side(style='medium'))
    for r, (edge, hit_rate, games) in enumerate(hit_rate_table[:HIT_RATE_TABLE_ROWS], start=2):
        ws.cell(r, edge_col, edge)
        ws.cell(r, rate_col, hit_rate).number_format = '0.0%'
        ws.cell(r, games_col, games)
    # editable settings: the correction that flags an underdog on its own, how many top underdogs a week are always
    # flagged, and the weeks picks are made in
    for key, (label, _default, row) in SETTINGS.items():
        ws.cell(row, edge_col, label).font = Font(bold=True)
        ws.cell(row, rate_col, settings[key])
    if hit_rate_table:
        first, last = settings['first_week'], settings['last_week']
        ws.cell(SETTINGS_NOTE_ROW, edge_col, f'Table: walk-forward backtest, 2006-{through_season} regular seasons, weeks {first}-{last}, '
                                              f'recomputed every run. Break-even at -110 is 52.4%.').alignment = Alignment(wrap_text=False)
    _add_conditional_formatting(ws)


def _settings_from(ws):
    """The editable settings the sheet holds, found by their labels since older layouts kept them elsewhere."""
    labels = {label: key for key, (label, _default, _row) in SETTINGS.items()}
    found = {}
    for row in ws.iter_rows(min_row=1, max_row=30, max_col=40):
        for cell in row:
            if cell.value in labels:
                found[labels[cell.value]] = ws.cell(cell.row, cell.column + 1).value
    settings = {key: default if found.get(key) in (None, '') else found[key] for key, (_label, default, _row) in SETTINGS.items()}
    if settings['picks'] == PREVIOUS_DEFAULT_PICKS_PER_WEEK:
        settings['picks'] = SETTINGS['picks'][1]
    return settings


def _field_columns(ws):
    """{field: column} for the columns the sheet has, matched by header text so any earlier layout can be read."""
    fields = {}
    for col in range(1, min(ws.max_column, 40) + 1):
        key = READ_HEADERS.get(ws.cell(1, col).value)
        if key and key not in fields:
            fields[key] = col
    return fields


def _load_or_create_projections_workbook(output_path):
    """Open the existing year workbook, or start a new one.

    Columns are found by header text, so a sheet from any earlier layout is read and then laid out again.
    A file that lacks the basic columns is copied to a timestamped backup instead of being used.
    """
    import shutil
    from openpyxl import Workbook, load_workbook

    def new_workbook():
        wb = Workbook()
        wb.active.title = 'Sheet1'
        return wb

    if not os.path.exists(output_path):
        return new_workbook()

    wb = load_workbook(output_path)
    if all(field in _field_columns(wb.active) for field in REQUIRED_FIELDS):
        return wb

    backup_path = output_path.replace('.xlsx', f'_old_layout_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    shutil.copy2(output_path, backup_path)
    logging.warning(f"{output_path} has a different layout; copied it to {backup_path} and starting a new sheet")
    return new_workbook()


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
    matching the Spread column. Past weeks return the closing line. Returns {} when the feed is unavailable
    or has no lines for that season (Action Network has none before 2023), so the sheet is left blank.
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
            home_line = next((line['value'] for line in lines if line.get('side') == 'home'), None)
            if home_line is None:
                continue
            spreads[(home, away)] = float(home_line)
            if book_id != DRAFTKINGS_BOOK_ID:
                logging.info(f"No DraftKings spread for {away} at {home} (week {week}); using {book_name}: {home_line}")
            break
    return spreads


_qb_flag_data = {}


def fetch_qb_injury_flags(year, week):
    """Quarterback injury notes for one week's games from nflverse injury reports, or {} if unavailable.

    Downloads fresh schedule and injury files once per run, since the reports update through the week.
    See nfl_injury_features.qb_injury_flags for exactly what gets flagged.
    """
    global _qb_flag_data
    try:
        if _qb_flag_data.get('year') != year:
            from nflverse_data import load_games, load_injuries
            _qb_flag_data = {'year': year, 'games': load_games(refresh=True), 'injuries': load_injuries(year, year, refresh=True)}
        from nfl_injury_features import qb_injury_flags
        return qb_injury_flags(_qb_flag_data['games'], _qb_flag_data['injuries'], year, week)
    except Exception as e:
        logging.warning(f"Could not load quarterback injury data for {year} week {week}: {e}")
        return {}


def _nickname(cell):
    """Excel formula text for the team nickname (last word of the name) in `cell`, so derived columns stay short."""
    return f'TRIM(RIGHT(SUBSTITUTE({cell}," ",REPT(" ",30)),30))'


def _autosize_columns(ws, records):
    """Fit every column to what it shows so the whole sheet fits on one screen without scrolling.

    Formula columns hold values that aren't known until Excel calculates, so they are sized from what they can
    show (nicknames, Win/Loss/Push, short numbers). Headers wrap, and the long QB flag and notes text wraps within
    a capped width with the row made taller to fit.
    """
    from textwrap import wrap
    from openpyxl.utils import get_column_letter

    nicknames = [name.split()[-1] for record in records for name in (record['home'], record['away'])]
    longest_nickname = max((len(name) for name in nicknames), default=8)
    formula_samples = {
        'pick': ['x' * longest_nickname + ' +99.5'], 'dog_correction': ['-9.9'], 'hit_rate': ['100%'],
        'fair_vs_spread': ['-99.5'], 'side': ['x' * longest_nickname], 'covering': ['x' * longest_nickname], 'result': ['Push', 'Loss'],
    }
    fair_line_like = ('fair_line', 'correction')

    def shown(value, key):
        if isinstance(value, float) and key in fair_line_like:
            return f'{value:.1f}'
        return str(value)

    widths = {}
    for key, header in LAYOUT:
        col = COL[key]
        if key in formula_samples:
            samples = formula_samples[key]
        else:
            samples = [shown(ws.cell(r, col).value, key) for r in range(2, len(records) + 2) if ws.cell(r, col).value is not None]
        longest = max((len(sample) for sample in samples), default=0)
        header_words = max(len(word) for word in header.split())
        header_width = math.ceil(header_words * 1.15) + 1   # bold text is wider
        width = max(longest + 2, header_width)
        if key == 'qb_flag':
            width = min(width, MAX_QB_FLAG_WIDTH)
        elif key == 'notes':
            width = max(min(width, MAX_NOTES_WIDTH), MIN_NOTES_WIDTH)
        widths[key] = width
        ws.column_dimensions[get_column_letter(col)].width = width
    for col, width in zip(TABLE_COLUMNS, (24.0, 14.0, 8.0)):
        ws.column_dimensions[get_column_letter(col)].width = width

    # Header row tall enough for the longest wrapped header; data rows tall enough for wrapped flags and notes
    header_lines = max(len(wrap(header, width=max(int(widths[key]) - 1, 1))) for key, header in LAYOUT)
    ws.row_dimensions[1].height = 15 * header_lines + 3
    for r, record in enumerate(records, start=2):
        lines = 1
        for key, field in (('qb_flag', 'qb_flag'), ('notes', 'notes')):
            text = record.get(field)
            if text:
                lines = max(lines, len(wrap(str(text), width=max(int(widths[key]) - 1, 1))))
        if lines > 1:
            ws.row_dimensions[r].height = 15 * lines


def write_projections_spreadsheet(rows, year, output_dir=None, actual_lookup=None, spread_fetcher=None, qb_flag_fetcher=None,
                                  hit_rate_table=None):
    """Write projections to <Desktop>/<year>_nfl_projections.xlsx, appending to it if it already exists.

    The workbook acts as a running log for the season: each run adds its week's games in week order, so
    a week run after later weeks is inserted after the weeks before it and ahead of the weeks that follow.
    A game already in the sheet (same week, year and teams) keeps its stats-only fair line until both its spread
    and its model correction are known; from then on the fair line is the spread moved by the correction, so
    Side to Cover agrees with the underdog signal. Actual score differences are refreshed for every game in the sheet
    that has since been played, using `actual_lookup` ({(week, year, home, away): difference}) plus the
    rows passed in. Blank spreads are filled from `spread_fetcher(year, week)` (DraftKings via Action
    Network in production); a spread already in the sheet is never replaced, and notes are never touched.
    Games not yet played get a QB Injury Flag from `qb_flag_fetcher(year, week)` ({(home, away): text}),
    refreshed every run since injury reports change through the week; played games keep their last flag.
    Layout: the columns you act on come first (week, teams, spread, the underdog pick), then the model's numbers,
    then the result, with the long QB flags and notes and the bookkeeping columns last. The sheet is read back by
    header text and laid out again every run, so an older layout migrates; columns that need manual entry are
    highlighted and the rest are formulas.

    Returns (output_path, rows_added, actuals_updated, spreads_filled).
    """
    from openpyxl.styles import Border, PatternFill, Side

    week_outline = Side(style='medium')
    manual_fill = PatternFill(fill_type='solid', start_color=MANUAL_INPUT_FILL_COLOR, end_color=MANUAL_INPUT_FILL_COLOR)

    output_dir = output_dir or get_desktop_path()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{year}_nfl_projections.xlsx')

    wb = _load_or_create_projections_workbook(output_path)
    ws = wb.active
    settings = _settings_from(ws)
    fields = _field_columns(ws)
    # a table that couldn't be recomputed this run keeps what the sheet already holds
    hit_rate_table = [r for r in (hit_rate_table or []) if r[1] is not None] or _hit_rate_table_from(ws)

    def week_value(week):
        return int(week) if str(week).isdigit() else week

    def game_key(week, game_year, home, away):
        return (str(week_value(week)), str(game_year), home, away)

    def played_actual(actual):
        # Unplayed games carry a placeholder difference of 0
        return None if actual in (0, None) or pd.isna(actual) else int(actual)

    actuals = {}
    for key, actual in (actual_lookup or {}).items():
        if played_actual(actual) is not None:
            actuals[game_key(*key)] = played_actual(actual)
    for row in rows:
        if played_actual(row['actual']) is not None:
            actuals[game_key(row['week'], row['year'], row['home_team'], row['away_team'])] = played_actual(row['actual'])

    # Read the existing games so the sheet can be rewritten in week order
    records = []
    existing_keys = set()
    actuals_updated = 0
    def cell_value(r, field):
        return ws.cell(r, fields[field]).value if field in fields else None

    for r in range(2, ws.max_row + 1):
        if not cell_value(r, 'home'):
            continue
        record = {
            'week': cell_value(r, 'week'), 'year': cell_value(r, 'year'),
            'home': cell_value(r, 'home'), 'away': cell_value(r, 'away'),
            'fair_line': cell_value(r, 'fair_line'), 'spread': cell_value(r, 'spread'),
            'actual': cell_value(r, 'actual'), 'notes': cell_value(r, 'notes'),
            'qb_flag': cell_value(r, 'qb_flag'), 'correction': cell_value(r, 'correction'),
        }
        key = game_key(record['week'], record['year'], record['home'], record['away'])
        existing_keys.add(key)
        if key in actuals and record['actual'] != actuals[key]:
            record['actual'] = actuals[key]
            actuals_updated += 1
        records.append(record)

    added = 0
    for row in rows:
        key = game_key(row['week'], row['year'], row['home_team'], row['away_team'])
        if key in existing_keys:
            # the correction is recorded once, like the fair line, so the sheet shows what was known when you bet
            for record in records:
                if game_key(record['week'], record['year'], record['home'], record['away']) == key and record['correction'] in (None, ''):
                    record['correction'] = row.get('correction')
            continue
        if row.get('backfill'):  # corrections for earlier weeks fill games already in the sheet but never add new ones
            continue
        existing_keys.add(key)
        records.append({
            'week': week_value(row['week']), 'year': int(row['year']),
            'home': row['home_team'], 'away': row['away_team'],
            'fair_line': fair_line_as_home(float(row['projected'])), 'spread': None,
            'actual': actuals.get(key), 'notes': None, 'qb_flag': None, 'correction': row.get('correction'),
        })
        added += 1

    # Quarterback injury flags for games not yet played; reports change through the week so these refresh every run
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

    # Fill blank spreads from the sportsbook feed; a spread already in the sheet is never replaced
    spreads_filled = 0
    if spread_fetcher:
        fetched_weeks = {}
        for record in records:
            if record['spread'] not in (None, '') or not str(record['week']).isdigit():
                continue
            week_number = int(record['week'])
            if week_number > get_last_regular_season_week(record['year']):
                continue
            if week_number not in fetched_weeks:
                fetched_weeks[week_number] = spread_fetcher(int(year), week_number)
            line = fetched_weeks[week_number].get((record['home'], record['away']))
            if line is not None:
                record['spread'] = line
                spreads_filled += 1

    # The fair line is the spread moved by the correction, so Side to Cover always agrees with the underdog signal.
    # Games without both numbers keep the stats-only line they were first written with.
    for record in records:
        spread, correction = record['spread'], record['correction']
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (spread, correction)):
            record['fair_line'] = anchored_fair_line(spread, correction)

    def season_order(record):
        week_number = translate_week_number(str(record['week']), record['year'])
        return (int(record['year']), week_number if week_number is not None else 99)

    # Stable sort: a week run late slots in after earlier weeks and before later ones, and games
    # within a week keep the order they were first written in
    records.sort(key=season_order)

    _reset_sheet(ws, settings, hit_rate_table, int(year))

    from openpyxl.styles import Alignment

    centered = ('week', 'year', 'spread', 'dog_correction', 'hit_rate', 'fair_line', 'fair_vs_spread', 'actual', 'result', 'correction')
    wrapped = ('qb_flag', 'notes')
    week_range, year_range, dog_range = (f'${LETTER[key]}$2:${LETTER[key]}${FORMAT_LAST_ROW}' for key in ('week', 'year', 'dog_correction'))
    edge_letter, rate_letter = TABLE_LETTERS[:2]
    last_table_row = HIT_RATE_TABLE_ROWS + 1

    for r, record in enumerate(records, start=2):
        c = {key: f'{letter}{r}' for key, letter in LETTER.items()}
        home_name, away_name = _nickname(c['home']), _nickname(c['away'])
        spread, actual, dog = c['spread'], c['actual'], c['dog_correction']

        ws.cell(r, COL['week'], record['week'])
        ws.cell(r, COL['year'], record['year'])
        ws.cell(r, COL['home'], record['home'])
        ws.cell(r, COL['away'], record['away'])
        ws.cell(r, COL['fair_line'], record['fair_line']).number_format = '0.0'
        # the spread, the actual difference and the notes are entered manually
        ws.cell(r, COL['spread'], record['spread'])
        ws.cell(r, COL['fair_vs_spread'], f"={c['fair_line']}+{spread}")
        ws.cell(r, COL['side'], f"=IF({c['fair_vs_spread']}>0,{home_name},{away_name})")
        ws.cell(r, COL['actual'], record['actual'])
        ws.cell(r, COL['covering'],
                f'=IF(OR({spread}="",{actual}=""),"",IF({actual}+{spread}=0,"Push",IF(({actual}+{spread})>0,{home_name},{away_name})))')
        ws.cell(r, COL['result'],
                f'=IF({c["covering"]}="","",IF({c["covering"]}="Push","Push",IF({c["covering"]}={c["side"]},"Win","Loss")))')
        ws.cell(r, COL['hit_rate'],
                f'=IF({dog}="","",IFERROR(LOOKUP({dog},${edge_letter}$2:${edge_letter}${last_table_row},${rate_letter}$2:${rate_letter}${last_table_row}),""))').number_format = '0%'
        ws.cell(r, COL['notes'], record['notes'])
        ws.cell(r, COL['qb_flag'], record['qb_flag'])
        ws.cell(r, COL['correction'], record['correction']).number_format = '0.0'
        # the correction measured toward the underdog (positive = the model likes the underdog more than the spread does)
        ws.cell(r, COL['dog_correction'],
                f'=IF(OR({spread}="",{c["correction"]}="",{spread}=0),"",ROUND(IF({spread}>0,{c["correction"]},-{c["correction"]}),3))').number_format = '0.0'
        # flag the underdog, shown with its spread, when the correction toward it is positive and either ranks in
        # the week's top picks-per-week or is at least the threshold
        rank_count = f'COUNTIFS({week_range},{c["week"]},{year_range},{c["year"]},{dog_range},">"&{dog})'
        underdog = f'IF({spread}>0,{home_name}&" +"&ABS({spread}),{away_name}&" +"&ABS({spread}))'
        setting = {key: f'${rate_letter}${row}' for key, (_label, _default, row) in SETTINGS.items()}
        in_pick_weeks = f'AND(ISNUMBER({c["week"]}),{c["week"]}>={setting["first_week"]},{c["week"]}<={setting["last_week"]})'
        ws.cell(r, COL['pick'],
                f'=IF({dog}="","",IF(AND({in_pick_weeks},{dog}>0,OR({dog}>={setting["threshold"]},{rank_count}<{setting["picks"]})),{underdog},""))')
        for key in MANUAL_INPUT_FIELDS:
            ws.cell(r, COL[key]).fill = manual_fill

        # Bold outline around each week's block of games, rebuilt every run since rows move
        week_group = season_order(record)
        is_first = r == 2 or season_order(records[r - 3]) != week_group
        is_last = r == len(records) + 1 or season_order(records[r - 1]) != week_group
        for key, col in COL.items():
            cell = ws.cell(r, col)
            cell.alignment = Alignment(horizontal='center' if key in centered else None, vertical='center', wrap_text=key in wrapped)
            cell.border = Border(
                left=week_outline if col == 1 else None,
                right=week_outline if col == LAST_SHEET_COLUMN else None,
                top=week_outline if is_first else None,
                bottom=week_outline if is_last else None,
            )

    _autosize_columns(ws, records)
    wb.save(output_path)
    return output_path, added, actuals_updated, spreads_filled
