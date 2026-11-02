"""The NBA game-lines tracker: every game's spread and total at DraftKings with the correction model's lean, tracked
through the season but not bet (nba_research_validate.py: on the untouched 2020-21 to 2025-26 seasons the leans went
52.4% on spreads and 52.5% on overs, about nothing at -110).  The season's results are the next clean test.

Run it by hand, from this folder (python nba_prod_games.py); each run:
  1. rebuilds the research table with the latest box scores and ESPN's injury list (players out tonight count as
     missing) and fits Model B for spreads and totals on every completed season with lines (alphas as validated) and
     Model A for a fair spread;
  2. takes DraftKings' current lines for today's games (Action Network) and scores the games not yet tipped off;
  3. grades earlier games, and writes NBA\\<season>_nba_games.xlsx: Today (today's games and leans), Season Log (every
     game tracked: the line at the last run before tip-off, the lean, the result) and Model Info (the research and the
     tracked record).
A lean is the side the correction favors by SPREAD_LEAN points or more (spreads) and the over when it favors the over
by OVER_LEAN or more (totals), the rules tested in the research.  A game keeps the line and lean of the last run before
its tip-off.
"""

import os
import sys
from datetime import datetime

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import nba_injuries
import nba_lab
import nba_lines
import nba_model
from nba_common import current_season, nba_folder, season_label

SPREAD_ALPHA, TOTAL_ALPHA, MARGIN_ALPHA = 1000, 10000, 3000
SPREAD_LEAN, OVER_LEAN = 0.5, 1.0
NAVY, WHITE, GRAY_TEXT, WIN_FILL, LOSS_FILL = '1F4E78', 'FFFFFF', '595959', 'C6EFCE', 'FFC7CE'
LOG_LAYOUT = (('date', 'Date', 11), ('tipoff', 'Tip-off (ET)', 9), ('away', 'Away', 14), ('home', 'Home', 14),
              ('home_spread', 'Home Spread (DK)', 9), ('total', 'Total (DK)', 8), ('spread_lean', 'Spread Lean', 18),
              ('correction', 'Lean (pts, home +)', 9), ('total_lean', 'Total Lean', 12),
              ('total_correction', 'Lean (pts, over +)', 9), ('fair_spread', 'Fair Spread (Model A)', 10),
              ('score', 'Final', 14), ('spread_result', 'Spread Lean W/L', 8), ('total_result', 'Total Lean W/L', 8),
              ('lines_at', 'Lines As Of', 15), ('game_id', 'game_id', 11), ('season', 'Season', 7))
READ_HEADERS = {header: key for key, header, _ in LOG_LAYOUT}


def workbook_path(season):
    return os.path.join(nba_folder(), f'{season_label(season)}_nba_games.xlsx')


def _fill(color):
    return PatternFill(fill_type='solid', start_color=color, end_color=color)


def _header(cell):
    cell.font = Font(bold=True, color=WHITE)
    cell.fill = _fill(NAVY)
    cell.alignment = Alignment(wrap_text=True, vertical='center', horizontal='center')


def read_log(path):
    """{game_id: row dict} of the Season Log already in the workbook (by header text)."""
    if not os.path.exists(path):
        return {}
    wb = load_workbook(path)
    if 'Season Log' not in wb.sheetnames:
        return {}
    ws = wb['Season Log']
    columns = {READ_HEADERS[ws.cell(1, c).value]: c for c in range(1, ws.max_column + 1)
               if ws.cell(1, c).value in READ_HEADERS}
    rows = {}
    for r in range(2, ws.max_row + 1):
        row = {key: ws.cell(r, c).value for key, c in columns.items()}
        if row.get('game_id'):
            rows[int(row['game_id'])] = row
    return rows


def fit_models(table, season):
    """Model B for spreads and totals on every completed season with lines, Model A for the margin."""
    history = nba_model.add_sums(table[table['season'] < season])
    with_lines = history[history['season'] >= nba_model.FIRST_LINES_SEASON]
    spread = nba_model.RidgeModel(nba_model.SPREAD_FEATURES, SPREAD_ALPHA).fit(
        with_lines[with_lines['cover'].notna()], 'cover')
    total = nba_model.RidgeModel(nba_model.TOTAL_FEATURES, TOTAL_ALPHA).fit(
        with_lines[with_lines['over'].notna()], 'over')
    margin = nba_model.RidgeModel(nba_model.SPREAD_FEATURES, MARGIN_ALPHA).fit(
        history[history['margin'].notna()], 'margin')
    return spread, total, margin


def lean_text(row, correction):
    """'Thunder +4.5' for the side the correction favors by SPREAD_LEAN or more, else ''."""
    if correction is None or pd.isna(correction) or abs(correction) < SPREAD_LEAN or pd.isna(row['home_spread']):
        return ''
    if correction > 0:
        return f"{row['home']} {row['home_spread']:+g}"
    return f"{row['away']} {-row['home_spread']:+g}"


def grade(row):
    """(spread lean result, total lean result) for a logged game with a final score."""
    if not row.get('score') or row.get('home_spread') in (None, ''):
        return None, None
    away_pts, home_pts = (int(x.split()[-1]) for x in str(row['score']).split(' @ '))
    margin, points = home_pts - away_pts, home_pts + away_pts
    spread_result = total_result = None
    if row.get('spread_lean'):
        cover = margin + float(row['home_spread'])
        side = 1 if float(row['correction']) > 0 else -1
        spread_result = 'P' if cover == 0 else 'W' if side * cover > 0 else 'L'
    if row.get('total_lean') and row.get('total') not in (None, ''):
        diff = points - float(row['total'])
        total_result = 'P' if diff == 0 else 'W' if diff > 0 else 'L'
    return spread_result, total_result


def record(rows, column):
    results = [r.get(column) for r in rows]
    wins, losses, pushes = results.count('W'), results.count('L'), results.count('P')
    rate = f'{wins / (wins + losses):.1%}' if wins + losses else '-'
    return f'{wins}-{losses}' + (f'-{pushes}' if pushes else '') + f' ({rate})'


def write_workbook(path, season, today_rows, log_rows, updated):
    wb = Workbook()
    ws = wb.active
    ws.title = 'Today'
    ws['A1'] = f'NBA games, {updated:%a %b} {updated.day}: the model\'s leans (tracked, not bet)'
    ws['A1'].font = Font(bold=True, size=14, color=NAVY)
    ws['A2'] = (f"Updated {updated:%a %b} {updated.day} {updated:%Y}, {updated:%I:%M %p}".replace(', 0', ', ')
                + ".  Lines are DraftKings' (Action Network).  These leans are not picks: on the untouched 2020-21 to "
                  "2025-26 seasons they went 52.4% (spreads) and 52.5% (overs), about break-even at -110.")
    ws['A2'].font = Font(italic=True, color=GRAY_TEXT)
    logged = [r for r in log_rows if r.get('spread_result') or r.get('total_result')]
    ws['A3'] = (f"Tracked this season: spread leans {record(logged, 'spread_result')}, over leans "
                f"{record(logged, 'total_result')}.")
    ws['A3'].font = Font(bold=True)
    headers = ('Tip-off (ET)', 'Game', 'Home Spread (DK)', 'Spread Lean', 'Lean (pts)', 'Total (DK)', 'Total Lean',
               'Lean (pts)', 'Fair Spread')
    for c, title in enumerate(headers, start=1):
        _header(ws.cell(5, c, title))
    for r, row in enumerate(today_rows, start=6):
        values = (row['tipoff'], f"{row['away']} @ {row['home']}", row['home_spread'], row['spread_lean'],
                  row['correction'], row['total'], row['total_lean'], row['total_correction'], row['fair_spread'])
        for c, value in enumerate(values, start=1):
            ws.cell(r, c, value)
        for c in (5, 8):
            ws.cell(r, c).number_format = '+0.0;-0.0;0.0'
        ws.cell(r, 9).number_format = '+0.0;-0.0;0.0'
    if not today_rows:
        ws.cell(6, 1, 'No games left to tip off today.').font = Font(italic=True, color=GRAY_TEXT)
    for c, width in enumerate((11, 30, 10, 18, 9, 9, 12, 9, 10), start=1):
        ws.column_dimensions[get_column_letter(c)].width = width

    log = wb.create_sheet('Season Log')
    for c, (key, title, width) in enumerate(LOG_LAYOUT, start=1):
        _header(log.cell(1, c, title))
        log.column_dimensions[get_column_letter(c)].width = width
    for r, row in enumerate(sorted(log_rows, key=lambda x: (str(x.get('date')), int(x.get('game_id') or 0))), start=2):
        for c, (key, _, _) in enumerate(LOG_LAYOUT, start=1):
            log.cell(r, c, row.get(key))
        for key in ('spread_result', 'total_result'):
            if row.get(key) in ('W', 'L'):
                log.cell(r, [k for k, _, _ in LOG_LAYOUT].index(key) + 1).fill = _fill(
                    WIN_FILL if row[key] == 'W' else LOSS_FILL)
    log.freeze_panes = 'E2'
    log.auto_filter.ref = f'A1:{get_column_letter(len(LOG_LAYOUT))}{max(2, len(log_rows) + 1)}'

    info = wb.create_sheet('Model Info')
    lines = [
        'How the leans are made', '',
        'Model B (the correction model) predicts where a game lands against the closing line from team ratings built '
        'from box scores (efficiency, the four factors, threes, pace and more, each corrected for the opponents faced), '
        'rest, back-to-backs, travel, altitude, home court and who is out (ESPN\'s injury list).',
        f'Spread lean: the side it favors by {SPREAD_LEAN} points or more.  Total lean: the over when it favors the over '
        f'by {OVER_LEAN} or more.',
        'The research (nba_research_validate.py, November 2, 2026): on the discovery seasons 2015-16 to 2019-20 these '
        'went 53.5% and 54.0%; on the untouched 2020-21 to 2025-26 seasons 52.4% (2,392-2,174) and 52.5% (1,033-935).  '
        'The model\'s information is real (it held up out of sample) but too small to beat the juice, so the leans are '
        'tracked here, not bet.',
        'Fair Spread is Model A, the stats-only projection (the home team\'s spread it implies); it misses the margin by '
        'about 9.9 points against 9.6 for the closing line.',
    ]
    for r, text in enumerate(lines, start=1):
        info.cell(r, 1, text).font = Font(bold=True, size=14, color=NAVY) if r == 1 else Font()
    info.column_dimensions['A'].width = 140
    temporary = path + '.saving'
    try:
        wb.save(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def main():
    now = pd.Timestamp.now(tz='America/New_York')
    season = current_season(now.date())
    print(f'NBA game lines tracker, {season_label(season)}, run {now:%a %b %d %I:%M %p} ET', flush=True)
    injuries = nba_injuries.load_injuries()
    out_ids = nba_injuries.out_tonight(injuries)
    print(f'ESPN injury list: {len(out_ids)} players out', flush=True)
    table = nba_lab.game_table(range(nba_lab.FIRST_SEASON, season + 1), refresh=True, out_ids=out_ids)
    spread_model, total_model, margin_model = fit_models(table, season)
    current = nba_model.add_sums(table[table['season'] == season])
    today = now.tz_localize(None).normalize()
    games = current[current['tipoff'].dt.normalize() == today]
    lines = nba_lines.day_lines(today, games)
    games = games.merge(lines, on='game_id', how='left')
    path = workbook_path(season)
    log = read_log(path)
    stamp = f'{now:%Y-%m-%d %H:%M}'
    today_rows = []
    for g, spread_pred, total_pred, margin_pred in zip(games.itertuples(index=False), spread_model.predict(games),
                                                       total_model.predict(games), margin_model.predict(games)):
        if g.start <= now.tz_convert('UTC'):
            continue
        row = {'date': f'{g.tipoff:%Y-%m-%d}', 'tipoff': f'{g.tipoff:%I:%M %p}'.lstrip('0'),
               'away': g.away_short_display_name, 'home': g.home_short_display_name,
               'home_spread': None if pd.isna(g.dk_home_spread) else float(g.dk_home_spread),
               'total': None if pd.isna(g.dk_total) else float(g.dk_total),
               'correction': round(float(spread_pred), 2), 'total_correction': round(float(total_pred), 2),
               'fair_spread': round(-float(margin_pred), 1), 'lines_at': stamp, 'game_id': int(g.game_id),
               'season': season}
        row['spread_lean'] = lean_text(row, row['correction'])
        row['total_lean'] = (f"Over {row['total']:g}" if row['total'] is not None
                             and row['total_correction'] >= OVER_LEAN else '')
        log[row['game_id']] = row
        today_rows.append(row)
    finals = table[(table['season'] == season) & table['margin'].notna()].set_index('game_id')
    for game_id, row in log.items():
        if game_id in finals.index:
            f = finals.loc[game_id]
            row['score'] = f"{row['away']} {int(f['away_score'])} @ {row['home']} {int(f['home_score'])}"
            row['spread_result'], row['total_result'] = grade(row)
    updated = datetime.now()
    try:
        write_workbook(path, season, today_rows, list(log.values()), updated)
    except OSError as error:
        print(f'Could not save the games workbook ({error.strerror or error}; is it open in Excel?): {path}')
        return 1
    leans = [r for r in today_rows if r['spread_lean'] or r['total_lean']]
    print(f'{len(today_rows)} games still to tip off today; {len(leans)} with a tracked lean (not bets)')
    for r in today_rows:
        print(f"  {r['tipoff']:>8} {r['away']} @ {r['home']}: spread {r['home_spread']}, lean "
              f"{r['spread_lean'] or '-'} ({r['correction']:+.2f}); total {r['total']}, "
              f"{r['total_lean'] or '-'} ({r['total_correction']:+.2f})")
    print(f'Games workbook updated: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
