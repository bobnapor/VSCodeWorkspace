"""Weekly college football spread picks: the rule frozen by the pre-registered research (cfb_research_validate.CHOSEN:
the side the correction model favors by 2+ points), on this season's FBS games.  Run it from this folder:
python cfb_prod_picks.py  (the daily run, cfb_run_all.py, runs it first).

Expect about 55%: the rule went 55.3% on the 2010-2019 discovery seasons and 56.0% on the untouched 2021-2025 test
(+6.9% a bet at the books' median price), but fell short of the test's z >= 2 bar, so it is treated as the NFL picks are:
bet at an honest expectation of about 55% and judged on its record and its line value.

Each run refreshes the data (ESPN's play-by-play for this season's ratings, CollegeFootballData's games and its
DraftKings and Bovada lines), fits Model B on every regular season from 2006 to last season (as in the test) and writes
<Desktop>\\CFB\\<season>_cfb_picks.xlsx (cfb_common.cfb_folder):
  This Week   the picks in games not yet kicked off: the side at DraftKings' line (the books' median when DraftKings has
              none), the lean and the kickoff; and the season's record.
  Season Log  every pick this season, one row per game: the line it went out at, the closing line, line versus close,
              the final margin and the result.  A pick follows the latest run until it goes out on a share card on an
              earlier day (cfb_share_card.py; the share log) or its game kicks off, and is then kept as it was, so it is
              graded at the line that went out.  Weeks played before the first run were filled in at the closing line
              and are marked 'closing line'.
Ratings Current is No when either team's latest finished game is not in ESPN's play-by-play yet (it arrives a day or
two after the game), so the lean lacks that game; the share card holds such picks back until it is.
The run log goes to <Desktop>\\CFB\\logs.
"""

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import cfb_cfbd
import cfb_data
import cfb_lab
import cfb_model
import cfb_research_validate as validate
from cfb_common import cfb_folder, current_season, shared_games

SHEET_WEEK, SHEET_LOG = 'This Week', 'Season Log'
LOG_COLUMNS = ['Week', 'Kickoff (ET)', 'Away', 'Home', 'Pick', 'Pick Line', 'Lean (pts)', 'Book', 'Ratings Current',
               'Home Spread at Pick', 'Closing Home Spread', 'Line vs Close', 'Final Margin (Home)', 'Result', 'Source',
               'game_id', 'pick_team_id', 'home_id', 'away_id']
DETAIL = ('game_id', 'pick_team_id', 'home_id', 'away_id')        # hidden: the ESPN ids the card and the record use
NAVY, WIN_FILL, LOSS_FILL, PICK_FILL, HOLD_FILL = '1F4E78', 'C6EFCE', 'FFC7CE', 'BDD7EE', 'FFF2CC'


def fit_model(table, season):
    """Model B for spreads (CHOSEN's alpha), fitted on every regular season from 2006 to the season before."""
    alpha, _family, _threshold = validate.CHOSEN['spreads']
    train = table[table['regular'] & table['season'].between(cfb_model.FIRST_LINES_SEASON, season - 1)
                  & table['cover'].notna()]
    return cfb_model.RidgeModel(cfb_model.SPREAD_FEATURES, alpha).fit(train, 'cover')


def current_ratings(season, schedule, now):
    """{team id: whether its latest finished game is in this season's reduced play-by-play}.  `schedule` is
    cfb_cfbd.games with Eastern kickoffs in 'kickoff'."""
    sums, _ = cfb_data.load_team_games([season])
    with_stats = set(sums['game_id'])
    final = schedule[(schedule['status'] == 'STATUS_FINAL') & (schedule['kickoff'] <= now)]
    latest = {}
    for g in final.sort_values('kickoff').itertuples():
        latest[g.home_id], latest[g.away_id] = g.game_id, g.game_id
    return {team: game in with_stats for team, game in latest.items()}


def season_picks(table, model, season, schedule, now):
    """This season's regular-season games with a line and a lean of CHOSEN's threshold or more, with the pick."""
    _alpha, family, threshold = validate.CHOSEN['spreads']
    assert family == 'S1', 'the picks below implement S1 (either side)'
    games = table[(table['season'] == season) & table['regular'] & table['close_home_spread'].notna()].copy()
    games['lean'] = model.predict(games)
    games = games[games['lean'].abs() >= threshold].copy()
    names = {}
    for g in schedule.itertuples():
        names[g.home_id], names[g.away_id] = g.home_team, g.away_team
    games['home_team'] = games['home_id'].map(names).fillna(games['home_team'])
    games['away_team'] = games['away_id'].map(names).fillna(games['away_team'])
    games['book'] = np.where(games['dk_home_spread'].notna(), 'DraftKings', 'books median')
    games['bet_home_spread'] = games['dk_home_spread'].where(games['dk_home_spread'].notna(), games['close_home_spread'])
    home = games['lean'] > 0
    games['pick_team'] = np.where(home, games['home_team'], games['away_team'])
    games['pick_team_id'] = np.where(home, games['home_id'], games['away_id'])
    games['pick_line'] = np.where(home, games['bet_home_spread'], -games['bet_home_spread'])
    games['kickoff'] = games['start'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    fresh = current_ratings(season, schedule, now)
    games['current'] = [fresh.get(h, True) and fresh.get(a, True) for h, a in zip(games['home_id'], games['away_id'])]
    return games


def grade(row):
    """'Win', 'Loss' or 'Push' at the pick's line, or '' before the game is final."""
    margin, home_spread = row['Final Margin (Home)'], row['Home Spread at Pick']
    if pd.isna(margin) or pd.isna(home_spread):
        return ''
    cover = margin + home_spread
    if cover == 0:
        return 'Push'
    return 'Win' if (cover > 0) == (row['pick_team_id'] == row['home_id']) else 'Loss'


def read_log(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_excel(path, sheet_name=SHEET_LOG).reindex(columns=LOG_COLUMNS)


def update_log(old, picks, table, season, now, frozen):
    """The Season Log: a row whose game has kicked off, or whose pick went out on an earlier day's card (`frozen`, game
    ids), stays as it was; the rest follow this run's picks (a first run fills earlier weeks at the closing line)."""
    if len(old):
        kept = old[(pd.to_datetime(old['Kickoff (ET)']) <= now) | old['game_id'].isin(frozen)]
    else:
        kept = old
    done = set(kept['game_id'])
    rows = []
    for g in picks.itertuples():
        if g.game_id in done:
            continue
        rows.append({'Week': int(g.week), 'Kickoff (ET)': g.kickoff, 'Away': g.away_team, 'Home': g.home_team,
                     'Pick': f'{g.pick_team} {g.pick_line:+g}', 'Pick Line': g.pick_line,
                     'Lean (pts)': round(abs(g.lean), 2), 'Book': g.book, 'Ratings Current': 'Yes' if g.current else 'No',
                     'Home Spread at Pick': g.bet_home_spread,
                     'Source': 'run before kickoff' if g.kickoff > now else 'closing line',
                     'game_id': g.game_id, 'pick_team_id': g.pick_team_id, 'home_id': g.home_id, 'away_id': g.away_id})
    log = pd.concat([kept, pd.DataFrame(rows, columns=LOG_COLUMNS)], ignore_index=True)
    facts = table[table['season'] == season].set_index('game_id')
    log['Closing Home Spread'] = log['game_id'].map(facts['close_home_spread'].where(facts['margin'].notna()))
    log['Final Margin (Home)'] = log['game_id'].map(facts['margin'])
    # points of line the pick got versus the close (positive: the pick beat the closing line)
    gap = log['Home Spread at Pick'] - log['Closing Home Spread']
    log['Line vs Close'] = np.where(log['pick_team_id'] == log['home_id'], gap, -gap)
    # a pick filled in from the closing line was never offered earlier, so it has no line value
    log.loc[log['Source'] == 'closing line', 'Line vs Close'] = np.nan
    log['Result'] = log.apply(grade, axis=1)
    log['Kickoff (ET)'] = pd.to_datetime(log['Kickoff (ET)'])
    return log.sort_values(['Kickoff (ET)', 'game_id']).reset_index(drop=True)


def record_text(log):
    results = log['Result'].value_counts()
    wins, losses, pushes = (int(results.get(k, 0)) for k in ('Win', 'Loss', 'Push'))
    rate = f'{wins / (wins + losses):.1%}' if wins + losses else '-'
    return wins, losses, pushes, rate


def write_workbook(path, log, season, now):
    from openpyxl import Workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

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
    upcoming = log[log['Kickoff (ET)'] > now]
    headers = ('Kickoff (ET)', 'Game', 'Pick', 'Lean (pts)', 'Book', 'Ratings current')
    for col, title in enumerate(headers, start=1):
        cell = ws.cell(4, col, title)
        cell.font, cell.fill = Font(bold=True, color='FFFFFF'), fill(NAVY)
    if upcoming.empty:
        ws.cell(5, 1, 'No picks among the games still to kick off.').font = Font(italic=True, color='595959')
    for r, (_, row) in enumerate(upcoming.iterrows(), start=5):
        values = (row['Kickoff (ET)'].strftime('%a %b %d %I:%M %p'), f"{row['Away']} @ {row['Home']}", row['Pick'],
                  round(float(row['Lean (pts)']), 1), row['Book'], row['Ratings Current'])
        for col, value in enumerate(values, start=1):
            ws.cell(r, col, value)
        ws.cell(r, 3).font = Font(bold=True)
        ws.cell(r, 3).fill = fill(HOLD_FILL if row['Ratings Current'] == 'No' else PICK_FILL)
    note = 5 + max(1, len(upcoming)) + 1
    ws.cell(note, 1, "Ratings current = No: a team's latest game is not in the play-by-play yet (it comes a day or two "
                     "after the game); the card holds the pick back until it is.").font = Font(italic=True, color='595959')
    for letter, width in zip('ABCDEF', (22, 44, 30, 11, 14, 15)):
        ws.column_dimensions[letter].width = width

    log_ws = wb.create_sheet(SHEET_LOG)
    for col, title in enumerate(LOG_COLUMNS, start=1):
        cell = log_ws.cell(1, col, title)
        cell.font, cell.fill = Font(bold=True, color='FFFFFF'), fill(NAVY)
        cell.alignment = Alignment(wrap_text=True, horizontal='center')
    for r, row in enumerate(log.itertuples(index=False), start=2):
        for col, value in enumerate(row, start=1):
            log_ws.cell(r, col, None if pd.isna(value) else value)
        log_ws.cell(r, 2).number_format = 'ddd mmm dd h:mm AM/PM'
    widths = (6, 20, 22, 22, 28, 9, 9, 13, 10, 10, 10, 9, 10, 8, 18, 12, 12, 10, 10)
    for col, width in enumerate(widths, start=1):
        log_ws.column_dimensions[get_column_letter(col)].width = width
    for key in DETAIL:
        log_ws.column_dimensions[get_column_letter(LOG_COLUMNS.index(key) + 1)].hidden = True
    log_ws.freeze_panes = 'B2'
    last, result = max(2, len(log) + 1), get_column_letter(LOG_COLUMNS.index('Result') + 1)
    for outcome, color in (('Win', WIN_FILL), ('Loss', LOSS_FILL)):
        log_ws.conditional_formatting.add(f'{result}2:{result}{last}',
                                          FormulaRule(formula=[f'${result}2="{outcome}"'], fill=fill(color)))
    temporary = path + '.saving'
    try:
        wb.save(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def main():
    season = current_season()
    log_path = os.path.join(cfb_folder('logs'), f'cfb_picks_{datetime.now():%Y%m%d_%H%M%S}.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True,
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    logging.info(f'Logging to file: {log_path}')
    now = pd.Timestamp.now(tz='America/New_York').tz_localize(None)
    logging.info(f'College football picks, {season}; refreshing the data (about 2 minutes)')
    table = cfb_lab.default_table(refresh=True)
    schedule = cfb_cfbd.games(season)
    schedule['kickoff'] = schedule['start'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    model = fit_model(table, season)
    picks = season_picks(table, model, season, schedule, now)
    path = os.path.join(cfb_folder(), f'{season}_cfb_picks.xlsx')
    old = read_log(path)
    kickoffs = dict(zip(schedule['game_id'], schedule['kickoff']))
    log = update_log(old, picks, table, season, now, shared_games(season, kickoffs, now))
    try:
        write_workbook(path, log, season, now)
    except OSError as error:
        logging.error(f'Could not save the picks workbook ({error.strerror or error}; is it open in Excel?): {path}')
        return 1
    upcoming = log[log['Kickoff (ET)'] > now]
    logging.info(f'{len(upcoming)} picks among the games still to kick off:')
    for _, g in upcoming.iterrows():
        held = '  (ratings not current: held back from the card)' if g['Ratings Current'] == 'No' else ''
        logging.info(f"  {g['Kickoff (ET)']:%a %b %d %I:%M %p}  {g['Pick']}  ({g['Away']} at {g['Home']}), "
                     f"lean {float(g['Lean (pts)']):.1f}{held}")
    wins, losses, pushes, rate = record_text(log)
    logging.info(f'Season record of the picks: {wins}-{losses}-{pushes} ({rate})')
    logging.info(f'Picks workbook updated: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
