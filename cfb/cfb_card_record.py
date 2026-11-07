"""The record of the college football picks the share cards showed, and of the ones you bet:
CFB\\<season>_cfb_card_record.xlsx.

The card step (cfb_share_card.py, the last step of every daily run) rebuilds it right after the day's card, from the
share log (CFB\\data\\share_card_log.csv: every pick a card showed and the day it did), so a card's picks are in it the
day the card goes out.  Each pick is graded once its game is final, at the line the card showed: the picked team's
margin plus that line, from CollegeFootballData's final scores; landing on the line is a push.

Bet? (yellow) is yours: set it to No for a pick you did not bet (say the line had moved), and it leaves the record at
once (the record at the top is Excel formulas) and stays out, since each run reads Bet? back by the pick's key (the
hidden Season and Key columns) before rebuilding the sheet.  Set it back to Yes to count the pick again.  The workbook is
saved to a temporary copy and swapped in, so a failed save (the workbook open in Excel) leaves the old file whole, and a
workbook that cannot be read is left as it is, so no Bet? mark is lost.

Run it alone (the card step runs it every time):  python cfb_card_record.py
"""

import os
import re
import sys
from datetime import datetime

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

import cfb_cfbd
from cfb_common import cfb_folder, current_season, key_game, read_share_log

SHEET_RECORD = 'Card Record'
BET, NOT_BET = 'Yes', 'No'
LAYOUT = (('pick', 'Pick', 34), ('bet', 'Bet?', 8), ('outcome', 'W/L', 8), ('result', 'Result', 36),
          ('kickoff', 'Kickoff (ET)', 16), ('card', 'Card', 12), ('week', 'Week', 7), ('season', 'Season', 8),
          ('key', 'Key', 30))
COL = {key: i for i, (key, _, _) in enumerate(LAYOUT, start=1)}
LETTER = {key: get_column_letter(i) for key, i in COL.items()}
READ_HEADERS = {header: key for key, header, _ in LAYOUT}
RECORD_COLUMNS = ('Won', 'Lost', 'Push', 'Win %', 'Pending', 'Not bet')
RECORD_TOP = 5
NAVY, GRAY_TEXT, MANUAL_HEADER, MANUAL_CELL, DETAIL_HEADER = '1F4E78', '595959', 'FFD966', 'FFF8E1', '7F7F7F'
WIN_FILL, LOSS_FILL = 'C6EFCE', 'FFC7CE'
NOT_BET_FONT = Font(color='A6A6A6', strike=True)
SPREAD_LINE = r'([+-]\d+(?:\.\d+)?)$'


def fill(color):
    return PatternFill('solid', start_color=color, end_color=color)


def header_cell(cell, color=NAVY, font_color='FFFFFF'):
    cell.font, cell.fill = Font(bold=True, color=font_color), fill(color)
    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)


def record_path(season):
    return os.path.join(cfb_folder(), f'{season}_cfb_card_record.xlsx')


def card_line(pick):
    """The line the card showed, from the share log's text of the pick ('Wake Forest -3'), or None."""
    found = re.search(SPREAD_LINE, str(pick).strip())
    return float(found.group(1)) if found else None


def bet_mark(value):
    """Bet? as typed: No (or n, x, false), else Yes, so a cleared cell counts the pick as bet."""
    text = '' if value is None else str(value).strip().lower()
    return NOT_BET if text in ('no', 'n', 'x', 'false') else BET


def grade(team_id, line, game):
    """(the final score, 'W', 'L' or 'P') for a pick of `team_id` at `line`, or (None, None) until the game is final."""
    if game is None or line is None or game['status'] != 'STATUS_FINAL' or pd.isna(game['home_score']):
        return None, None
    score = f"{game['away_team']} {game['away_score']:.0f} @ {game['home_team']} {game['home_score']:.0f}"
    margin = game['home_score'] - game['away_score']
    margin = margin if team_id == game['home_id'] else -margin
    edge = margin + line
    return score, 'W' if edge > 0 else 'L' if edge < 0 else 'P'


def graded_picks(log, games):
    """One dict per share-log entry with LAYOUT's fields, newest card first and, within a card, by kickoff."""
    by_id = {g['game_id']: g for g in games.to_dict('records')}
    rows = []
    for entry in log.itertuples():
        _, team, game_id = str(entry.key).split('|')
        game = by_id.get(int(game_id))
        result, won = grade(int(team), card_line(entry.pick), game)
        kickoff = game['kickoff'].to_pydatetime() if game is not None and pd.notna(game['kickoff']) else None
        rows.append({'pick': entry.pick, 'outcome': won, 'result': result, 'kickoff': kickoff,
                     'card': pd.Timestamp(entry.shown_on).to_pydatetime(), 'week': int(entry.week),
                     'season': str(entry.season), 'key': entry.key, 'shown_on': entry.shown_on,
                     'order': kickoff or datetime.max})
    rows.sort(key=lambda row: row['order'])
    rows.sort(key=lambda row: row['shown_on'], reverse=True)
    return rows


def read_marks(path):
    """{(season, key): Bet?} from the record's pick table, matched by header text; {} when there is no workbook yet.
    Raises when the workbook cannot be read."""
    if not os.path.exists(path):
        return {}
    workbook = load_workbook(path, read_only=True)
    try:
        marks = {}
        for ws in workbook.worksheets:
            columns = None
            for row in ws.iter_rows(values_only=True):
                if columns is None:
                    if row and {'Season', 'Key', 'Bet?'} <= set(row):
                        columns = {READ_HEADERS[value]: i for i, value in enumerate(row) if value in READ_HEADERS}
                    continue
                season, key, bet = (row[columns[field]] if columns[field] < len(row) else None
                                    for field in ('season', 'key', 'bet'))
                if key:
                    marks[(str(season), str(key))] = bet_mark(bet)
        return marks
    finally:
        workbook.close()


def write_record(path, season, rows, updated_at):
    wb = Workbook()
    ws = wb.active
    ws.title = SHEET_RECORD
    ws['A1'] = f'{season} college football card record'
    ws['A1'].font = Font(bold=True, size=14, color=NAVY)
    stamp = f'{updated_at:%a %b} {updated_at.day} {updated_at:%Y}, {updated_at:%I:%M %p}'.replace(', 0', ', ')
    ws['A2'] = (f"Updated {stamp}.  Every pick a share card has shown, newest card first, graded at the card's line "
                f"once its game is final.")
    ws['A3'] = ("Bet? (yellow): set it to No for a pick you did not bet.  It leaves the record at once and stays out on "
                "later runs; set it back to Yes to count it again.")
    for cell in ('A2', 'A3'):
        ws[cell].font = Font(italic=True, color=GRAY_TEXT)

    header = RECORD_TOP + 5
    first, last = header + 1, max(header + 1, header + len(rows))
    span = lambda key: f'${LETTER[key]}${first}:${LETTER[key]}${last}'                    # noqa: E731
    outcomes, bets = span('outcome'), span('bet')
    header_cell(ws.cell(RECORD_TOP, 1, ''))
    for c, title in enumerate(RECORD_COLUMNS, start=2):
        header_cell(ws.cell(RECORD_TOP, c, title))
    won, lost = (get_column_letter(2 + RECORD_COLUMNS.index(t)) for t in ('Won', 'Lost'))
    for r, label, mine in ((RECORD_TOP + 1, 'Picks you bet', f'{bets},"<>{NOT_BET}",{outcomes}'),
                           (RECORD_TOP + 2, 'Every card pick, bet or not', f'{outcomes}')):
        ws.cell(r, 1, label).font = Font(bold=r == RECORD_TOP + 1, italic=r != RECORD_TOP + 1)
        formulas = {'Won': f'=COUNTIFS({mine},"W")', 'Lost': f'=COUNTIFS({mine},"L")',
                    'Push': f'=COUNTIFS({mine},"P")', 'Win %': f'=IFERROR({won}{r}/({won}{r}+{lost}{r}),"")',
                    'Pending': (f'=COUNTIFS({bets},"<>{NOT_BET}",{outcomes},"")' if r == RECORD_TOP + 1
                                else f'=COUNTIFS({span("key")},"<>",{outcomes},"")'),
                    'Not bet': f'=COUNTIF({bets},"{NOT_BET}")' if r == RECORD_TOP + 1 else None}
        for c, title in enumerate(RECORD_COLUMNS, start=2):
            cell = ws.cell(r, c, formulas[title])
            cell.alignment = Alignment(horizontal='center')
            if title == 'Win %':
                cell.number_format = '0.0%'
    ws.cell(RECORD_TOP + 3, 1, 'Win % leaves out pushes.  Expect about 55% over a season; single seasons have run '
                               '50-63% (2010-2025).').font = Font(italic=True, color=GRAY_TEXT)

    for key, title, width in LAYOUT:
        cell = ws.cell(header, COL[key], title)
        if key == 'bet':
            header_cell(cell, MANUAL_HEADER, '000000')
        else:
            header_cell(cell, DETAIL_HEADER if COL[key] >= COL['season'] else NAVY)
        ws.column_dimensions[LETTER[key]].width = width
    for r, row in enumerate(rows, start=first):
        for key, _, _ in LAYOUT:
            ws.cell(r, COL[key], row.get(key))
        ws.cell(r, COL['bet']).fill = fill(MANUAL_CELL)
        for key in ('bet', 'outcome', 'week'):
            ws.cell(r, COL[key]).alignment = Alignment(horizontal='center')
        ws.cell(r, COL['kickoff']).number_format = 'ddd m/d h:mm AM/PM'
        ws.cell(r, COL['card']).number_format = 'ddd mmm d'
        if row['outcome'] in ('W', 'L'):
            ws.cell(r, COL['outcome']).fill = fill(WIN_FILL if row['outcome'] == 'W' else LOSS_FILL)
    choices = DataValidation(type='list', formula1=f'"{BET},{NOT_BET}"', allow_blank=True, showErrorMessage=True,
                             errorTitle='Bet?', error='Choose Yes, or No for a pick you did not bet.')
    ws.add_data_validation(choices)
    choices.add(f'{LETTER["bet"]}{first}:{LETTER["bet"]}{last}')
    ws.conditional_formatting.add(f'A{first}:{LETTER["week"]}{last}',
                                  FormulaRule(formula=[f'${LETTER["bet"]}{first}="{NOT_BET}"'], font=NOT_BET_FONT))
    ws.auto_filter.ref = f'A{header}:{LETTER["key"]}{last}'
    ws.column_dimensions.group(LETTER['season'], LETTER['key'], hidden=True)
    temporary = path + '.saving'
    try:
        wb.save(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def summary_line(rows):
    """'Card record (picks you bet): 13-9-1, 59.1%; 3 pending; 1 marked not bet'."""
    mine = [row for row in rows if row['bet'] != NOT_BET]
    won, lost, push = (sum(row['outcome'] == kind for row in mine) for kind in ('W', 'L', 'P'))
    pending = sum(row['outcome'] is None for row in mine)
    text = f'Card record (picks you bet): {won}-{lost}' + (f'-{push}' if push else '')
    if won + lost:
        text += f', {won / (won + lost):.1%}'
    notes = [(pending, 'pending'), (len(rows) - len(mine), 'marked not bet')]
    return text + ''.join(f'; {count} {what}' for count, what in notes if count)


def update(season=None, path=None, log_path=None, now=None, refresh=False):
    """Rebuild the record from the share log, keeping every Bet? mark.  Returns (path, summary line), or None when the
    workbook could not be read or saved (the reason is printed, and the old file is left whole).  `refresh` fetches
    CollegeFootballData's games again (the picks run has just done so)."""
    season = season or current_season()
    path = path or record_path(season)
    try:
        marks = read_marks(path)
    except Exception as error:
        print(f'Could not read the card record, so it was left as it is (no Bet? mark is lost): {path} ({error})')
        return None
    log = read_share_log(log_path)
    log = log[log['season'] == str(season)]
    games = cfb_cfbd.games(season, refresh=refresh)
    games['kickoff'] = games['start'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    games = games[games['game_id'].isin({key_game(k) for k in log['key']})]
    rows = graded_picks(log, games)
    for row in rows:
        row['bet'] = marks.get((row['season'], row['key']), BET)
    try:
        write_record(path, season, rows, now or datetime.now())
    except OSError as error:
        print(f'Could not save the card record ({error.strerror or error}; is it open in Excel?), so it was left as it '
              f'was: close it and run the card again (run_all.bat card): {path}')
        return None
    return path, summary_line(rows)


def main():
    found = update(refresh=True)
    if found:
        print(found[1])
        print(f'Card record updated: {found[0]}')
    return 0 if found else 1


if __name__ == '__main__':
    sys.exit(main())
