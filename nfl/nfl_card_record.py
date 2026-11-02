"""The record of the picks the share cards showed, and of the ones you bet: NFL\\<season>_nfl_card_record.xlsx.

The card step (nfl_share_card.py, the last step of every daily run) rebuilds it right after the day's card, from the
share log (NFL\\data\\share_card_log.csv: every pick a card showed and the day it did), so a card's picks are in it the
day the card goes out, and it grades each pick once its game is final, at the line the card showed:
  spreads, teaser legs   the picked team's margin plus the card's line (a teaser leg's teased line), from nflverse's
                         final scores; landing on the line is a push
  game totals            the final total against the card's total (the card's totals are unders)
  player props           the player's stat in the props workbook's Prop Log (graded there from nflverse's stats) against
                         the card's line; Void when he did not play, as DraftKings voids those
It only reads what the card and the flows wrote, sharing no logic with them and changing none of their files: the
projections and props workbooks and the weather log keep their own records of every pick, as before.

Bet? (yellow) is yours: set it to No for a pick you did not bet (say the price was worse than the card's), and it leaves
the record at once (the record at the top is Excel formulas) and stays out, since each run reads Bet? back by the pick's
key (the hidden Season and Key columns) before rebuilding the sheet.  Set it back to Yes to count the pick again.  The
workbook is saved to a temporary copy and swapped in, so a save that fails (the workbook open in Excel) leaves the old
file whole, and a workbook that cannot be read is left as it is, so no Bet? mark is lost.

Run it alone (the card step runs it every time):  python nfl_card_record.py
"""

import os
import re
import sys
from datetime import datetime

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

import nfl_props_sheet as props_sheet
import nfl_sheet as sheet
from nfl_injury_features import TEAM_FULL_NAMES, TEAM_RENAMES
from nflverse_data import load_games

SHEET_RECORD = 'Card Record'
BET, NOT_BET = 'Yes', 'No'
# the kinds of pick, in the order the cards show them: (the share log's kind, the Kind column's name, the record's row)
KINDS = (('underdog', 'Underdog spread', 'Underdog spreads'), ('favorite', 'Favorite spread', 'Favorite spreads'),
         ('teaser', 'Teaser leg', 'Teaser legs'), ('prop', 'Player prop', 'Player props'),
         ('total', 'Game total', 'Game totals (unders)'))
KIND_NAME = {kind: name for kind, name, _ in KINDS}
SECTION = {'underdog': 0, 'favorite': 0, 'teaser': 1, 'prop': 2, 'total': 3}   # the card's sections, in its order
# the pick table: the pick and your Bet? first, then how it did and where it came from, then hidden detail that each run
# reads back to keep Bet?; the record above the table shares its columns (the record's labels under Pick)
LAYOUT = (('pick', 'Pick', 40), ('bet', 'Bet?', 8), ('outcome', 'W/L', 8), ('result', 'Result', 30),
          ('kind', 'Kind', 16), ('game', 'Game', 22), ('kickoff', 'Kickoff (ET)', 16), ('card', 'Card', 12),
          ('week', 'Week', 7), ('season', 'Season', 8), ('key', 'Key', 50))
COL = {key: i for i, (key, _, _) in enumerate(LAYOUT, start=1)}
LETTER = {key: get_column_letter(i) for key, i in COL.items()}
READ_HEADERS = {header: key for key, header, _ in LAYOUT}
CENTERED = ('bet', 'outcome', 'week')
RECORD_COLUMNS = ('Won', 'Lost', 'Push', 'Win %', 'Pending', 'Void', 'Not bet')
RECORD_TOP = 5                                  # the record's header row
NOT_BET_FONT = Font(color='A6A6A6', strike=True)
# where the line is in the share log's text of a pick ('Saints +3.5', 'Jerry Jeudy Over 1.5 Receptions',
# 'Falcons @ Buccaneers under 43.5')
LINE_PATTERNS = {'prop': r'\b(?:Over|Under) (\d+(?:\.\d+)?)\b', 'total': r'\bunder (\d+(?:\.\d+)?)$'}
SPREAD_LINE = r'([+-]\d+(?:\.\d+)?)$'


def record_path(season):
    """NFL\\<season>_nfl_card_record.xlsx."""
    return os.path.join(sheet.nfl_folder(), f'{season}_nfl_card_record.xlsx')


def nickname(code):
    return TEAM_FULL_NAMES.get(code, str(code)).split()[-1]


def kind_of(key):
    """'underdog', 'favorite', 'teaser', 'prop' or 'total', from a share-log key ('spread|underdog|NO|PIT@NO', ...)."""
    parts = str(key).split('|')
    return parts[1] if parts[0] == 'spread' and len(parts) > 1 else parts[0]


def card_line(kind, pick):
    """The line the card showed, from the share log's text of the pick, or None."""
    found = re.search(LINE_PATTERNS.get(kind, SPREAD_LINE), str(pick).strip())
    return float(found.group(1)) if found else None


def outcome(edge):
    """'W', 'L' or 'P' by where the result landed against the line (a positive `edge` is the pick's side of it)."""
    return 'W' if edge > 0 else 'L' if edge < 0 else 'P'


def bet_mark(value):
    """Bet? as typed: No (or n, x, false), else Yes, so a cleared cell counts the pick as bet."""
    text = '' if value is None else str(value).strip().lower()
    return NOT_BET if text in ('no', 'n', 'x', 'false') else BET


def kickoff_of(game):
    if game is None:
        return None
    when = pd.Timestamp(f'{game.gameday} {game.gametime}') if isinstance(game.gametime, str) else pd.Timestamp(game.gameday)
    return when.to_pydatetime()


def grade(kind, key, line, game, logged):
    """(what happened, 'W', 'L', 'P' or 'Void') for a card pick at the card's `line`, or (None, None) until it is
    final: `game` is its nflverse game, `logged` its Prop Log record (props only)."""
    if kind == 'prop':
        if logged is None:
            return None, None
        if logged.get('outcome') == 'Void':
            return 'did not play (void)', 'Void'
        line = line if line is not None else logged.get('line')
        if not sheet._number(logged.get('result')) or not sheet._number(line):
            return None, None
        _, _, prop, side, _ = key.split('|')
        stat = float(logged['result'])
        return f'{stat:g} {prop.lower()}', outcome(stat - line if side == 'Over' else line - stat)
    if game is None or pd.isna(game.result) or line is None:
        return None, None
    score = (f'{nickname(game.away_team)} {game.away_score:.0f} @ {nickname(game.home_team)} '
             f'{game.home_score:.0f}')
    if kind == 'total':
        points = game.home_score + game.away_score
        return f'{score} ({points:.0f} points)', outcome(line - points)
    margin = game.result if key.split('|')[-2] == game.home_team else -game.result
    return score, outcome(margin + line)


def graded_picks(log, games, prop_log):
    """One dict per share-log entry with LAYOUT's fields: newest card first and, within a card, in the card's order (its
    sections, then kickoff)."""
    schedule = {(int(g.week), g.away_team, g.home_team): g for g in games.itertuples()}
    props = {(str(r.get('season')), str(r.get('week')), r.get('player'), r.get('prop'), r.get('side'), r.get('game')): r
             for r in prop_log}
    rows = []
    for entry in log.itertuples():
        kind = kind_of(entry.key)
        if kind not in KIND_NAME:
            continue
        away, _, home = str(entry.key).rsplit('|', 1)[-1].partition('@')
        week = int(entry.week)
        game = schedule.get((week, away, home))
        logged = None
        if kind == 'prop':
            _, player, prop, side, _ = entry.key.split('|')
            logged = props.get((str(entry.season), str(week), player, prop, side, f'{away} at {home}'))
        result, won = grade(kind, entry.key, card_line(kind, entry.pick), game, logged)
        kickoff = kickoff_of(game)
        rows.append({'pick': entry.pick, 'outcome': won, 'result': result, 'kind': KIND_NAME[kind],
                     'game': f'{nickname(away)} @ {nickname(home)}', 'kickoff': kickoff,
                     'card': pd.Timestamp(entry.shown_on).to_pydatetime(), 'week': week, 'season': str(entry.season),
                     'key': entry.key, 'shown_on': entry.shown_on,
                     'order': (SECTION[kind], kickoff or datetime.max, entry.pick)})
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


def _record_row(ws, r, formulas, bold=False):
    """A row of the record: `formulas` by RECORD_COLUMNS title, with Win % from Won and Lost."""
    won, lost = (get_column_letter(2 + RECORD_COLUMNS.index(title)) for title in ('Won', 'Lost'))
    formulas = dict(formulas, **{'Win %': f'=IFERROR({won}{r}/({won}{r}+{lost}{r}),"")'})
    for c, title in enumerate(RECORD_COLUMNS, start=2):
        cell = ws.cell(r, c, formulas.get(title))
        cell.alignment = Alignment(horizontal='center')
        if bold:
            cell.font = Font(bold=True)
        if title == 'Win %':
            cell.number_format = '0.0%'


def write_record(path, season, rows, updated_at):
    """The record (see the module docstring) at `path`, saved through a temporary copy."""
    wb = Workbook()
    ws = wb.active
    ws.title = SHEET_RECORD
    ws['A1'] = f'{season} card record'
    ws['A1'].font = Font(bold=True, size=14, color=props_sheet.NAVY)
    stamp = f'{updated_at:%a %b} {updated_at.day} {updated_at:%Y}, {updated_at:%I:%M %p}'.replace(', 0', ', ')
    ws['A2'] = (f"Updated {stamp}.  Every pick a share card has shown, newest card first, graded at the card's line "
                f"once its game is final.")
    ws['A3'] = ("Bet? (yellow): set it to No for a pick you did not bet, say because the price was worse than the "
                "card's.  It leaves the record at once and stays out on later runs; set it back to Yes to count it again.")
    for cell in ('A2', 'A3'):
        ws[cell].font = Font(italic=True, color=props_sheet.GRAY_TEXT)

    header = RECORD_TOP + len(KINDS) + 5        # the pick table's header row, below the record and its note
    first = header + 1
    last = max(first, header + len(rows))

    def span(key):
        return f'${LETTER[key]}${first}:${LETTER[key]}${last}'
    kinds, outcomes, bets = span('kind'), span('outcome'), span('bet')
    props_sheet._header(ws.cell(RECORD_TOP, 1, 'Picks you bet'))
    for c, title in enumerate(RECORD_COLUMNS, start=2):
        props_sheet._header(ws.cell(RECORD_TOP, c, title))
    for r, (_, name, label) in enumerate(KINDS, start=RECORD_TOP + 1):
        ws.cell(r, 1, label)
        mine = f'{kinds},"{name}",{bets},"<>{NOT_BET}",{outcomes}'
        _record_row(ws, r, {'Won': f'=COUNTIFS({mine},"W")', 'Lost': f'=COUNTIFS({mine},"L")',
                            'Push': f'=COUNTIFS({mine},"P")', 'Pending': f'=COUNTIFS({mine},"")',
                            'Void': f'=COUNTIFS({mine},"Void")',
                            'Not bet': f'=COUNTIFS({kinds},"{name}",{bets},"{NOT_BET}")'})
    total = RECORD_TOP + 1 + len(KINDS)
    ws.cell(total, 1, 'All picks you bet').font = Font(bold=True)
    _record_row(ws, total, {title: f'=SUM({get_column_letter(c)}{RECORD_TOP + 1}:{get_column_letter(c)}{total - 1})'
                            for c, title in enumerate(RECORD_COLUMNS, start=2) if title != 'Win %'}, bold=True)
    every = total + 1
    ws.cell(every, 1, 'Every card pick, bet or not').font = Font(italic=True, color=props_sheet.GRAY_TEXT)
    _record_row(ws, every, {'Won': f'=COUNTIF({outcomes},"W")', 'Lost': f'=COUNTIF({outcomes},"L")',
                            'Push': f'=COUNTIF({outcomes},"P")', 'Pending': f'=COUNTIFS({kinds},"<>",{outcomes},"")',
                            'Void': f'=COUNTIF({outcomes},"Void")'})
    ws.cell(every + 1, 1, "Win % leaves out pushes and voids.  Teaser legs are graded one by one at the teased line (a "
                          "2-leg teaser wins when both its legs win); props by the player's stat in the props workbook's "
                          "Prop Log.").font = Font(italic=True, color=props_sheet.GRAY_TEXT)

    for key, title, width in LAYOUT:
        cell = ws.cell(header, COL[key], title)
        if key == 'bet':
            props_sheet._header(cell, props_sheet.MANUAL_HEADER, '000000')
        else:
            props_sheet._header(cell, props_sheet.DETAIL_HEADER if COL[key] >= COL['season'] else props_sheet.NAVY)
        ws.column_dimensions[LETTER[key]].width = width
    for r, row in enumerate(rows, start=first):
        for key, _, _ in LAYOUT:
            ws.cell(r, COL[key], row.get(key))
        ws.cell(r, COL['bet']).fill = props_sheet._fill(props_sheet.MANUAL_CELL)
        for key in CENTERED:
            ws.cell(r, COL[key]).alignment = Alignment(horizontal='center')
        ws.cell(r, COL['kickoff']).number_format = 'ddd m/d h:mm AM/PM'
        ws.cell(r, COL['card']).number_format = 'ddd mmm d'
        if row['outcome'] in ('W', 'L'):
            color = props_sheet.WIN_FILL if row['outcome'] == 'W' else props_sheet.LOSS_FILL
            ws.cell(r, COL['outcome']).fill = props_sheet._fill(color)
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
    """The record of the picks you bet, for the run's output: 'Card record (picks you bet): 13-9-1, 59.1%; ...'."""
    mine = [row for row in rows if row['bet'] != NOT_BET]
    won, lost, push, void = (sum(row['outcome'] == kind for row in mine) for kind in ('W', 'L', 'P', 'Void'))
    pending = sum(row['outcome'] is None for row in mine)
    text = f'Card record (picks you bet): {won}-{lost}' + (f'-{push}' if push else '')
    if won + lost:
        text += f', {won / (won + lost):.1%}'
    notes = [(pending, 'pending'), (void, 'void'), (len(rows) - len(mine), 'marked not bet')]
    return text + ''.join(f'; {count} {what}' for count, what in notes if count)


def update(season=None, refresh=False, path=None, log_path=None, props_path=None, now=None):
    """Rebuild the record from the share log, keeping every Bet? mark.  Returns (path, summary line), or None when the
    workbook could not be read or saved (the reason is printed, and the old file is left whole).  `refresh` downloads
    nflverse's schedule and scores again (the card step has just done so)."""
    season = season or sheet.get_current_season_year()
    path = path or record_path(season)
    try:
        marks = read_marks(path)
    except Exception as error:
        print(f'Could not read the card record, so it was left as it is (no Bet? mark is lost): {path} ({error})')
        return None
    log = sheet.read_share_log(log_path)
    log = log[log['season'] == str(season)]
    games = load_games(refresh=refresh)
    games = games[games['season'] == int(season)].copy()
    for column in ('home_team', 'away_team'):
        games[column] = games[column].replace(TEAM_RENAMES)
    props_path = props_path or os.path.join(sheet.nfl_folder(), f'{season}_nfl_props.xlsx')
    prop_log = props_sheet.read_log(props_path) if log['key'].str.startswith('prop|').any() else []
    rows = graded_picks(log, games, prop_log)
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
