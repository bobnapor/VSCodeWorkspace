"""Weekly college football spread picks: the rule frozen by the pre-registered research (cfb_research_validate.CHOSEN:
the side the correction model favors by 2+ points), on this season's FBS games.  Run it from this folder:
python cfb_prod_picks.py

Expect about 55%: the rule went 55.3% on the 2010-2019 discovery seasons and 56.0% on the untouched 2021-2025 test
(+6.9% a bet at the books' median price), but fell short of the test's z >= 2 bar, so treat it as the NFL picks are
treated: bet at an honest expectation of about 55% and judge it on its record and its line value.

Each run refreshes the data (ESPN's play-by-play for this season's ratings, CollegeFootballData's games and its
DraftKings and Bovada lines), fits Model B on every regular season from 2006 to last season (as in the test) and writes
<Desktop>\\CFB\\<season>_cfb_picks.xlsx (cfb_common.cfb_folder):
  This Week   the games not yet kicked off with a pick: the side, DraftKings' line (the books' median when DraftKings has
              none), the lean, the kickoff; and the season's record.
  Season Log  every pick this season, one row per game: the line it went out at, the closing line, the final margin and
              the result.  A pick follows the latest run until its game kicks off and is then kept as it was, so it is
              graded at the last line this run showed.  Weeks played before the first run are filled in at the closing
              line and marked 'closing line'.
The run log goes to <Desktop>\\CFB\\logs.
"""

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import cfb_cfbd
import cfb_lab
import cfb_model
import cfb_research_validate as validate
from cfb_common import cfb_folder, current_season

SHEET_WEEK, SHEET_LOG = 'This Week', 'Season Log'
LOG_COLUMNS = ['game_id', 'Week', 'Kickoff (ET)', 'Away', 'Home', 'Pick', 'Pick Line', 'Lean (pts)', 'Home Spread at Pick',
               'Closing Home Spread', 'Line vs Close', 'Final Margin (Home)', 'Result', 'Source']
NAVY, WIN_FILL, LOSS_FILL, PICK_FILL = '1F4E78', 'C6EFCE', 'FFC7CE', 'BDD7EE'


def fit_model(table, season):
    """Model B for spreads (CHOSEN's alpha), fitted on every regular season from 2006 to the season before."""
    alpha, _family, _threshold = validate.CHOSEN['spreads']
    train = table[table['regular'] & table['season'].between(cfb_model.FIRST_LINES_SEASON, season - 1)
                  & table['cover'].notna()]
    return cfb_model.RidgeModel(cfb_model.SPREAD_FEATURES, alpha).fit(train, 'cover')


def season_picks(table, model, season):
    """This season's regular-season games with a line and a lean of CHOSEN's threshold or more, with the pick."""
    _alpha, family, threshold = validate.CHOSEN['spreads']
    assert family == 'S1', 'the picks below implement S1 (either side)'
    games = table[(table['season'] == season) & table['regular'] & table['close_home_spread'].notna()].copy()
    games['lean'] = model.predict(games)
    games = games[games['lean'].abs() >= threshold].copy()
    names = {}
    for g in cfb_cfbd.games(season).itertuples():
        names[g.home_id], names[g.away_id] = g.home_team, g.away_team
    games['home_team'] = games['home_id'].map(names).fillna(games['home_team'])
    games['away_team'] = games['away_id'].map(names).fillna(games['away_team'])
    games['bet_home_spread'] = games['dk_home_spread'].where(games['dk_home_spread'].notna(), games['close_home_spread'])
    home = games['lean'] > 0
    games['pick_team'] = np.where(home, games['home_team'], games['away_team'])
    games['pick_line'] = np.where(home, games['bet_home_spread'], -games['bet_home_spread'])
    games['kickoff'] = games['start'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    return games


def grade(row):
    """'Win', 'Loss' or 'Push' at the pick's line, or '' before the game is final."""
    margin, home_spread, pick, home = row['Final Margin (Home)'], row['Home Spread at Pick'], row['Pick'], row['Home']
    if pd.isna(margin) or pd.isna(home_spread):
        return ''
    cover = margin + home_spread
    if cover == 0:
        return 'Push'
    home_picked = str(pick).startswith(f'{home} ')
    return 'Win' if (cover > 0) == home_picked else 'Loss'


def read_log(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.read_excel(path, sheet_name=SHEET_LOG)
    return log.reindex(columns=LOG_COLUMNS)


def update_log(old, picks, table, season, now):
    """The Season Log: rows for games already kicked off stay as they were (a first run fills earlier weeks at the
    closing line); games not yet kicked off follow this run's picks."""
    started = picks['kickoff'] <= now
    kept = old[old['game_id'].isin(picks.loc[started, 'game_id'])] if len(old) else old
    rows = []
    for g in picks.itertuples():
        if g.game_id in set(kept['game_id']):
            continue
        source = 'run before kickoff' if g.kickoff > now else 'closing line'
        rows.append({'game_id': g.game_id, 'Week': int(g.week), 'Kickoff (ET)': g.kickoff, 'Away': g.away_team,
                     'Home': g.home_team, 'Pick': f'{g.pick_team} {g.pick_line:+g}', 'Pick Line': g.pick_line,
                     'Lean (pts)': round(abs(g.lean), 2), 'Home Spread at Pick': g.bet_home_spread, 'Source': source})
    log = pd.concat([kept, pd.DataFrame(rows, columns=LOG_COLUMNS)], ignore_index=True)
    facts = table[table['season'] == season].set_index('game_id')
    log['Closing Home Spread'] = log['game_id'].map(facts['close_home_spread'].where(facts['margin'].notna()))
    log['Final Margin (Home)'] = log['game_id'].map(facts['margin'])
    # points of line the pick got versus the close (positive: the pick beat the closing line)
    home_picked = [str(p).startswith(f'{h} ') for p, h in zip(log['Pick'], log['Home'])]
    gap = log['Home Spread at Pick'] - log['Closing Home Spread']
    log['Line vs Close'] = np.where(home_picked, gap, -gap)
    # a pick filled in from the closing line was never offered earlier, so it has no line value
    log.loc[log['Source'] == 'closing line', 'Line vs Close'] = np.nan
    log['Result'] = log.apply(grade, axis=1)
    return log.sort_values(['Kickoff (ET)', 'game_id']).reset_index(drop=True)


def record_text(log):
    results = log['Result'].value_counts()
    wins, losses, pushes = (int(results.get(k, 0)) for k in ('Win', 'Loss', 'Push'))
    rate = f'{wins / (wins + losses):.1%}' if wins + losses else '-'
    return wins, losses, pushes, rate


def write_workbook(path, log, upcoming, season, now):
    from openpyxl import Workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Font, PatternFill

    fill = lambda color: PatternFill('solid', start_color=color, end_color=color)          # noqa: E731
    wb = Workbook()
    ws = wb.active
    ws.title = SHEET_WEEK
    wins, losses, pushes, rate = record_text(log)
    ws['A1'] = f'College Football Picks - {season}'
    ws['A1'].font = Font(size=18, bold=True, color=NAVY)
    ws['A2'] = (f'Updated {now:%A %B %d, %I:%M %p} ET.  The side the model favors by 2+ points (56.0% on the untouched '
                f'2021-2025 test; expect about 55%).  Season so far: {wins}-{losses}-{pushes} ({rate}).')
    ws['A2'].font = Font(italic=True, color='595959')
    headers = ('Kickoff (ET)', 'Game', 'Pick', 'Lean (pts)', 'Line source')
    for col, title in enumerate(headers, start=1):
        cell = ws.cell(4, col, title)
        cell.font, cell.fill = Font(bold=True, color='FFFFFF'), fill(NAVY)
    if upcoming.empty:
        ws.cell(5, 1, 'No picks among the games still to kick off.').font = Font(italic=True, color='595959')
    for r, g in enumerate(upcoming.sort_values(['kickoff', 'game_id']).itertuples(), start=5):
        values = (g.kickoff.strftime('%a %b %d %I:%M %p'), f'{g.away_team} @ {g.home_team}',
                  f'{g.pick_team} {g.pick_line:+g}', round(abs(g.lean), 1),
                  'DraftKings' if pd.notna(g.dk_home_spread) else 'books median')
        for col, value in enumerate(values, start=1):
            ws.cell(r, col, value)
        ws.cell(r, 3).font, ws.cell(r, 3).fill = Font(bold=True), fill(PICK_FILL)
    for letter, width in zip('ABCDE', (22, 44, 30, 11, 14)):
        ws.column_dimensions[letter].width = width

    log_ws = wb.create_sheet(SHEET_LOG)
    for col, title in enumerate(LOG_COLUMNS, start=1):
        cell = log_ws.cell(1, col, title)
        cell.font, cell.fill = Font(bold=True, color='FFFFFF'), fill(NAVY)
        cell.alignment = Alignment(wrap_text=True, horizontal='center')
    for r, row in enumerate(log.itertuples(index=False), start=2):
        for col, value in enumerate(row, start=1):
            log_ws.cell(r, col, None if (isinstance(value, float) and np.isnan(value)) else value)
        log_ws.cell(r, 3).number_format = 'ddd mmm dd h:mm AM/PM'
    widths = (12, 6, 20, 22, 22, 28, 9, 9, 10, 10, 9, 10, 8, 18)
    for col, width in enumerate(widths, start=1):
        log_ws.column_dimensions[log_ws.cell(1, col).column_letter].width = width
    log_ws.column_dimensions['A'].hidden = True
    log_ws.freeze_panes = 'B2'
    last = max(2, len(log) + 1)
    for outcome, color in (('Win', WIN_FILL), ('Loss', LOSS_FILL)):
        log_ws.conditional_formatting.add(f'M2:M{last}', FormulaRule(formula=[f'$M2="{outcome}"'], fill=fill(color)))
    temporary = path + '.saving'
    wb.save(temporary)
    os.replace(temporary, path)


def main():
    season = current_season()
    log_path = os.path.join(cfb_folder('logs'), f'cfb_picks_{datetime.now():%Y%m%d_%H%M%S}.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True,
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    now = pd.Timestamp.now(tz='America/New_York').tz_localize(None)
    logging.info(f'College football picks, {season}; refreshing the data (about 2 minutes)')
    table = cfb_lab.default_table(refresh=True)
    model = fit_model(table, season)
    picks = season_picks(table, model, season)
    upcoming = picks[picks['kickoff'] > now]
    path = os.path.join(cfb_folder(), f'{season}_cfb_picks.xlsx')
    log = update_log(read_log(path), picks, table, season, now)
    write_workbook(path, log, upcoming, season, now)
    wins, losses, pushes, rate = record_text(log)
    logging.info(f'{len(upcoming)} picks among the games still to kick off:')
    for g in upcoming.sort_values(['kickoff', 'game_id']).itertuples():
        logging.info(f'  {g.kickoff:%a %b %d %I:%M %p}  {g.pick_team} {g.pick_line:+g}  ({g.away_team} at {g.home_team}), '
                     f'lean {abs(g.lean):.1f}')
    logging.info(f'Season record of the picks: {wins}-{losses}-{pushes} ({rate})')
    logging.info(f'Workbook: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
