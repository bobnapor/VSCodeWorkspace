"""The yearly projections spreadsheet: layout, formulas, spreads from sportsbooks and quarterback injury flags.

Everything that touches the workbook lives here so the model and the weekly runner stay separate from it.
"""

import logging
import os
import time
from datetime import datetime

import pandas as pd
import requests

# How often the model's pick covered against closing spreads in nfl_research_model.py (walk-forward, 2014-2026
# regular seasons, 3,132 games, pushes excluded). Each pair is (smallest edge in points, ATS win rate for edges
# from there to the next row). Every band sits near 50%, and break-even at -110 is 52.4%, so these are hit
# rates and not probabilities of covering.
ATS_HIT_RATE_BY_EDGE = ((0, 0.51), (3, 0.53), (6, 0.49))
HIT_RATE_HEADER = 'Backtest ATS Win % (at this edge)'
OLD_COVER_PROBABILITY_HEADER = 'Cover Probability (Side to Cover)'

SPREADSHEET_HEADERS = [
    'Week', 'Year', 'Home Team', 'Away Team',
    'Fair Line (as Home)', 'Spread (as Home)', 'Fair Line vs Spread',
    'Side to Cover', 'Actual Score Difference (as Home)', 'Covering Team', 'Result',
    HIT_RATE_HEADER, 'Notes',
]
# Columns filled in by hand: Spread (F), Actual Score Difference (I), Notes (M)
MANUAL_INPUT_COLUMNS = (6, 9, 13)
MANUAL_INPUT_FILL_COLOR = 'FFFF99'
# Quarterback injury notes (N), the model's correction to the spread (O) and the underdog signal (P) live outside
# SPREADSHEET_HEADERS so sheets from before they were added still load
QB_FLAG_COLUMN = 14
QB_FLAG_HEADER = 'QB Injury Flag'
CORRECTION_COLUMN = 15
CORRECTION_HEADER = 'Model Correction (as Home)'
SIGNAL_COLUMN = 16
SIGNAL_HEADER = 'Underdog Signal'
LAST_SHEET_COLUMN = SIGNAL_COLUMN
# Settings and the edge/hit-rate lookup table sit to the right of the data, in columns R:S
TABLE_COLUMNS = (18, 19)
THRESHOLD_ROW = 7
DEFAULT_SIGNAL_THRESHOLD = 0.5
SPREADSHEET_COLUMN_WIDTHS = {'R': 16.0, 'S': 18.0}
OLD_TABLE_HEADER = 'Edge (pts) from'

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


RESULT_FORMAT_RANGE = 'J2:K1000'
RESULT_FORMAT_RULES = (
    ('Win', 'C6EFCE'),   # light green
    ('Loss', 'FFC7CE'),  # light red
)


def _ensure_result_formatting(ws):
    """Shade the Covering Team and Result cells green for a win and red for a loss.

    Applied to rows well past the current data so appended games are covered, and only added when the
    sheet doesn't already have the rules so repeated runs don't stack duplicates.
    """
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import PatternFill

    for formatting in ws.conditional_formatting:
        if str(formatting.sqref) == RESULT_FORMAT_RANGE:
            return
    for result, color in RESULT_FORMAT_RULES:
        ws.conditional_formatting.add(
            RESULT_FORMAT_RANGE,
            FormulaRule(formula=[f'$K2="{result}"'], fill=PatternFill(start_color=color, end_color=color, fill_type='solid')),
        )


def _new_projections_workbook():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    manual_fill = PatternFill(fill_type='solid', start_color=MANUAL_INPUT_FILL_COLOR, end_color=MANUAL_INPUT_FILL_COLOR)
    wb = Workbook()
    ws = wb.active
    ws.title = 'Sheet1'

    header_border = Border(bottom=Side(style='medium'))
    for col, title in enumerate(SPREADSHEET_HEADERS, start=1):
        cell = ws.cell(row=1, column=col, value=title)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = header_border
    ws.row_dimensions[1].height = 29.5
    for letter, width in SPREADSHEET_COLUMN_WIDTHS.items():
        ws.column_dimensions[letter].width = width
    for col in MANUAL_INPUT_COLUMNS:
        ws.cell(1, col).fill = manual_fill

    _write_hit_rate_table(ws)
    _ensure_qb_flag_header(ws)
    return wb


def _ensure_qb_flag_header(ws):
    from openpyxl.styles import Alignment, Border, Font, Side

    for column, header in ((QB_FLAG_COLUMN, QB_FLAG_HEADER), (CORRECTION_COLUMN, CORRECTION_HEADER), (SIGNAL_COLUMN, SIGNAL_HEADER)):
        cell = ws.cell(1, column)
        if cell.value != header:
            cell.value = header
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = Border(bottom=Side(style='medium'))


def _write_hit_rate_table(ws):
    """Settings and the edge-size lookup table in R:S; the Backtest ATS Win % column reads the table."""
    from openpyxl.styles import Alignment, Font

    edge_col, rate_col = TABLE_COLUMNS
    for col, title in ((edge_col, 'Edge (pts) from'), (rate_col, 'Backtest ATS Win %')):
        cell = ws.cell(1, col, title)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    for r, (edge, hit_rate) in enumerate(ATS_HIT_RATE_BY_EDGE, start=2):
        ws.cell(r, edge_col, edge)
        ws.cell(r, rate_col, hit_rate).number_format = '0%'
    ws.cell(THRESHOLD_ROW, edge_col, 'Signal threshold (pts)').font = Font(bold=True)
    if ws.cell(THRESHOLD_ROW, rate_col).value in (None, ''):
        ws.cell(THRESHOLD_ROW, rate_col, DEFAULT_SIGNAL_THRESHOLD)  # editable: the correction needed to flag an underdog


def _ensure_signal_formatting(ws):
    """Light-blue highlight on any game the underdog signal flags."""
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import PatternFill

    signal_range = f'P2:P1000'
    for formatting in ws.conditional_formatting:
        if str(formatting.sqref) == signal_range:
            return
    ws.conditional_formatting.add(
        signal_range, FormulaRule(formula=['LEN($P2)>0'], fill=PatternFill(start_color='BDD7EE', end_color='BDD7EE', fill_type='solid'))
    )


def _load_or_create_projections_workbook(output_path):
    """Open the existing year workbook, or start a new one.

    An existing file whose header row doesn't match the current layout is copied to a timestamped
    backup instead of being appended to, so older layouts are never corrupted. The one exception is
    the old Cover Probability column header, which is renamed in place.
    """
    import shutil
    from openpyxl import load_workbook

    if not os.path.exists(output_path):
        return _new_projections_workbook()

    wb = load_workbook(output_path)
    ws = wb.active
    if ws.cell(1, 15).value == OLD_TABLE_HEADER:
        # sheets from before the signal columns kept the lookup table in O:P, which now hold the correction and signal
        for r in range(1, 5):
            ws.cell(r, 15).value = None
            ws.cell(r, 16).value = None
    headers = [ws.cell(1, col).value for col in range(1, len(SPREADSHEET_HEADERS) + 1)]
    if headers[11] == OLD_COVER_PROBABILITY_HEADER:
        headers[11] = HIT_RATE_HEADER
        ws.cell(1, 12).value = HIT_RATE_HEADER
    if headers == SPREADSHEET_HEADERS:
        return wb

    backup_path = output_path.replace('.xlsx', f'_old_layout_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    shutil.copy2(output_path, backup_path)
    logging.warning(f"{output_path} has a different layout; copied it to {backup_path} and starting a new sheet")
    return _new_projections_workbook()


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


def _autosize_columns(ws, records):
    """Fit columns A-M to what is written in them.

    Formula columns hold values that aren't known until Excel calculates, so they are sized from what
    they can show: team names, Win/Loss/Push, numbers and a percentage. Headers wrap onto two lines.
    """
    import math
    from openpyxl.utils import get_column_letter
    team_names = [record[field] for record in records for field in ('home', 'away')]
    formula_samples = {
        7: ['-99.5'],
        8: team_names,
        10: team_names + ['Push'],
        11: ['Push', 'Loss'],
        12: ['100%'],
    }
    for col, header in enumerate(SPREADSHEET_HEADERS, start=1):
        samples = [str(ws.cell(r, col).value) for r in range(2, len(records) + 2) if col not in formula_samples
                   and ws.cell(r, col).value is not None]
        samples += formula_samples.get(col, [])
        content_width = max((len(sample) for sample in samples), default=0)
        header_width = max(max(len(word) for word in header.split()), math.ceil(len(header) / 2))
        width = max(content_width, header_width) + 3
        ws.column_dimensions[get_column_letter(col)].width = min(width, 60)

    flag_lengths = [len(str(record['qb_flag'])) for record in records if record.get('qb_flag')]
    ws.column_dimensions[get_column_letter(QB_FLAG_COLUMN)].width = min(max(flag_lengths, default=0) + 3, 90) if flag_lengths else 16
    ws.column_dimensions[get_column_letter(CORRECTION_COLUMN)].width = 14
    ws.column_dimensions[get_column_letter(SIGNAL_COLUMN)].width = max([len(name) for name in team_names] + [len(SIGNAL_HEADER)], default=16) + 3


def write_projections_spreadsheet(rows, year, output_dir=None, actual_lookup=None, spread_fetcher=None, qb_flag_fetcher=None):
    """Write projections to <Desktop>/<year>_nfl_projections.xlsx, appending to it if it already exists.

    The workbook acts as a running log for the season: each run adds its week's games in week order, so
    a week run after later weeks is inserted after the weeks before it and ahead of the weeks that follow.
    A game already in the sheet (same week, year and teams) keeps its original fair line, since
    that is the line that was bet on. Actual score differences are refreshed for every game in the sheet
    that has since been played, using `actual_lookup` ({(week, year, home, away): difference}) plus the
    rows passed in. Blank spreads are filled from `spread_fetcher(year, week)` (DraftKings via Action
    Network in production); a spread already in the sheet is never replaced, and notes are never touched.
    Games not yet played get a QB Injury Flag from `qb_flag_fetcher(year, week)` ({(home, away): text}),
    refreshed every run since injury reports change through the week; played games keep their last flag.
    Layout follows the '2024 NFL Projections' workbook, with Notes moved to the far right; columns that
    need manual entry are highlighted and the rest are formulas.

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
    _ensure_result_formatting(ws)
    _ensure_signal_formatting(ws)
    _write_hit_rate_table(ws)
    _ensure_qb_flag_header(ws)

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
    for r in range(2, ws.max_row + 1):
        if not ws.cell(r, 3).value:
            continue
        record = {
            'week': ws.cell(r, 1).value, 'year': ws.cell(r, 2).value,
            'home': ws.cell(r, 3).value, 'away': ws.cell(r, 4).value,
            'fair_line': ws.cell(r, 5).value, 'spread': ws.cell(r, 6).value,
            'actual': ws.cell(r, 9).value, 'notes': ws.cell(r, 13).value,
            'qb_flag': ws.cell(r, QB_FLAG_COLUMN).value, 'correction': ws.cell(r, CORRECTION_COLUMN).value,
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

    def season_order(record):
        week_number = translate_week_number(str(record['week']), record['year'])
        return (int(record['year']), week_number if week_number is not None else 99)

    # Stable sort: a week run late slots in after earlier weeks and before later ones, and games
    # within a week keep the order they were first written in
    records.sort(key=season_order)

    for r in range(2, ws.max_row + 1):
        for col in range(1, LAST_SHEET_COLUMN + 1):
            ws.cell(r, col).value = None

    for r, record in enumerate(records, start=2):
        ws.cell(r, 1, record['week'])
        ws.cell(r, 2, record['year'])
        ws.cell(r, 3, record['home'])
        ws.cell(r, 4, record['away'])
        ws.cell(r, 5, record['fair_line']).number_format = '0.0'
        # F (spread), I (actual) and M (notes) are entered manually
        ws.cell(r, 6, record['spread'])
        ws.cell(r, 7, f'=E{r}+F{r}')
        ws.cell(r, 8, f'=IF(G{r}>0,C{r},D{r})')
        ws.cell(r, 9, record['actual'])
        ws.cell(r, 10, f'=IF(OR(F{r}="",I{r}=""),"",IF(I{r}+F{r}=0,"Push",IF((I{r}+F{r})>0,C{r},D{r})))')
        ws.cell(r, 11, f'=IF(J{r}="","",IF(J{r}="Push","Push",IF(J{r}=H{r},"Win","Loss")))')
        last_table_row = len(ATS_HIT_RATE_BY_EDGE) + 1
        edge_letter, rate_letter = 'R', 'S'
        ws.cell(r, 12, f'=IF(F{r}="","",LOOKUP(ABS(G{r}),${edge_letter}$2:${edge_letter}${last_table_row},${rate_letter}$2:${rate_letter}${last_table_row}))').number_format = '0%'
        ws.cell(r, 13, record['notes'])
        ws.cell(r, QB_FLAG_COLUMN, record['qb_flag'])
        ws.cell(r, CORRECTION_COLUMN, record['correction']).number_format = '0.0'
        # flag the underdog when the correction points that way by at least the threshold in S7
        threshold = f'${rate_letter}${THRESHOLD_ROW}'
        ws.cell(r, SIGNAL_COLUMN, f'=IF(OR(F{r}="",O{r}=""),"",IF(AND(F{r}>0,O{r}>={threshold}),C{r},IF(AND(F{r}<0,O{r}<=-{threshold}),D{r},"")))')
        for col in MANUAL_INPUT_COLUMNS:
            ws.cell(r, col).fill = manual_fill

        # Bold outline around each week's block of games, rebuilt every run since rows move
        week_group = season_order(record)
        is_first = r == 2 or season_order(records[r - 3]) != week_group
        is_last = r == len(records) + 1 or season_order(records[r - 1]) != week_group
        for col in range(1, LAST_SHEET_COLUMN + 1):
            ws.cell(r, col).border = Border(
                left=week_outline if col == 1 else None,
                right=week_outline if col == LAST_SHEET_COLUMN else None,
                top=week_outline if is_first else None,
                bottom=week_outline if is_last else None,
            )

    _autosize_columns(ws, records)
    wb.save(output_path)
    return output_path, added, actuals_updated, spreads_filled
