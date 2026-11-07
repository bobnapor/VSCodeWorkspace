"""The day's shareable graphic of new college football picks, and the same picks as text, in CFB\\Cards on the Desktop.

It reads what the picks run wrote (cfb_prod_picks.py: <season>_cfb_picks.xlsx in the CFB folder, cfb_common.cfb_folder)
and shares no logic with it: the picks in games kicking off within CARD_DAYS, at the line the workbook shows.  A pick
whose ratings are not current (a team's latest game is not in the play-by-play yet) waits for a later card.

One card a day, of new picks only, as the NFL card does: a pick an earlier day's card showed is not shown again, even
if its line moved (CFB\\data\\share_card_log.csv records every pick shown and the day it was).  A day's last run is the
one that counts: a later run on the same day replaces the earlier ones (their card files and log entries; picks in
games already under way stay).  From the next day on, the picks run keeps a shared pick as it went out, so it is graded
at the line on the card.  With nothing new, no card is made.

Every run then rebuilds the card record, CFB\\<season>_cfb_card_record.xlsx (cfb_card_record.py): every pick the cards
have shown, graded at the card's line, with a Bet? column to set to No for a pick you did not bet.

The graphic is 1080 px wide, 1080 x 1350 (4:5) when the picks fit, else taller.  Its look is card_design's
(shared/card_design.py, the one thing shared with the NFL suite), so the NFL and college cards look alike, with each
school's colors and abbreviation in its badge (no logos).

Run it from this folder:  python cfb_share_card.py   (the daily run, cfb_run_all.py, runs it after the picks).
"""

import os
import sys
from datetime import datetime

import pandas as pd

import cfb_card_record as card_record
import cfb_cfbd
from cfb_common import cfb_folder, current_season, pick_key, read_share_log, share_log_path

# the cards' common look, the one thing shared with the NFL suite
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
import card_design as design  # noqa: E402
from card_design import clock, date_range  # noqa: E402

CARD_FOLDER = 'Cards'
CARD_DAYS = 7                   # the card shows games kicking off within this many days; later picks wait
# the rule's record over the discovery and test seasons, 2010-2025 (cfb_research_validate): 624-500
HIT_RATE = 624 / 1124
MIN_EDGE = 0.03                 # the price guide: the worst price at which a bet still returns 3% at HIT_RATE
SPREADS_COLOR, SPREADS_ACCENT = design.SECTION_COLORS['spreads']
KIND_NOTE = 'The hit rate is for every pick of this kind since 2010, not for any one pick.'


# ---------------------------------------------------------------- small helpers

def american_from_chance(chance, edge=0.0):
    """The worst American price at which a bet winning `chance` of the time still returns `edge` a unit."""
    decimal = (1 + edge) / chance
    return round(100 * (decimal - 1)) if decimal >= 2 else round(-100 / (decimal - 1))


def bet_at(rate):
    return f'{american_from_chance(rate, MIN_EDGE):+d} or better'


def matchup(pick, at='@'):
    """'vs SMU' or '@ SMU' from the picked team's side (vs at a neutral site)."""
    home = pick['team_id'] == pick['home_id']
    other = pick['away'] if home else pick['home']
    return f"{'vs' if home or pick['neutral'] else at} {other}"


def file_time(path):
    return datetime.fromtimestamp(os.path.getmtime(path))


# ---------------------------------------------------------------- the picks, from the picks workbook

def upcoming_picks(path, season, now):
    """(picks in games not yet kicked off whose ratings are current, how many are held back, when the lines were
    taken), from the picks workbook's Season Log."""
    if not os.path.exists(path):
        print(f'No picks workbook yet ({path}); run cfb_prod_picks.py first')
        return [], 0, None
    log = pd.read_excel(path, sheet_name='Season Log')
    log['Kickoff (ET)'] = pd.to_datetime(log['Kickoff (ET)'])
    log = log[(log['Kickoff (ET)'] > now) & (log['Kickoff (ET)'] <= now + pd.Timedelta(days=CARD_DAYS))]
    teams = cfb_cfbd.teams(season)
    games = cfb_cfbd.games(season)
    neutral = dict(zip(games['game_id'], games['neutral_site']))
    picks, held = [], 0
    for _, row in log.iterrows():
        if row['Ratings Current'] == 'No':
            held += 1
            continue
        team_id = int(row['pick_team_id'])
        line = float(row['Pick Line'])
        picks.append({'game_id': int(row['game_id']), 'week': int(row['Week']), 'team_id': team_id,
                      'home_id': int(row['home_id']), 'away_id': int(row['away_id']),
                      'team': row['Home'] if team_id == int(row['home_id']) else row['Away'],
                      'home': row['Home'], 'away': row['Away'], 'line': f'{line:+g}',
                      'kickoff': row['Kickoff (ET)'], 'neutral': bool(neutral.get(int(row['game_id']), False)),
                      'abbr': teams.get(team_id, (None, None, None, None))[1] or row['Pick'][:4].upper(),
                      'colors': teams.get(team_id, (None, None) + design.DEFAULT_COLORS)[2:]})
    picks.sort(key=lambda p: (p['kickoff'], p['team'].casefold()))
    return picks, held, file_time(path)


def describe(pick):
    return f"{pick['team']} {pick['line']}"


# ---------------------------------------------------------------- drawing (the look is card_design's)

class Layout(design.Layout):
    """card_design's layout with the college card's one section: spread tiles."""

    def content(self, card):
        y = self.header('COLLEGE FOOTBALL', card['season'], card['week'], card['dates'], 64 * self.s) + 46 * self.s
        y = self.heading(y, 'SPREADS', SPREADS_COLOR, card['caption'], card['note'], SPREADS_ACCENT)
        tiles = [{'badge': (pick['abbr'], pick['colors']), 'name': pick['team'].upper(), 'line': pick['line'],
                  'sub': ' · '.join(part for part in (matchup(pick), clock(pick['kickoff'])) if part)}
                 for pick in card['picks']]
        return self.tiles(tiles, y, SPREADS_ACCENT)


def draw_card(card, note='', handle=''):
    """The card as an image (card_design.render)."""
    return design.render(Layout, lambda layout: layout.content(card), note, handle, KIND_NOTE)


def pick_text(card):
    """Every pick as plain text, ready to paste into a message or a post."""
    title = (f"College football week {card['week']}: new picks, {card['day']}, times ET" if card['update'] else
             f"College football week {card['week']} picks" + (f" ({card['dates']})" if card['dates'] else '')
             + ', times ET')
    lines = [title, '', f"Spreads ({card['caption']})", card['note']]
    lines += [f"  {p['team']} {p['line']} {matchup(p)}, {clock(p['kickoff'])}" for p in card['picks']]
    return '\n'.join(lines + ['', KIND_NOTE, ''])


# ---------------------------------------------------------------- running it

def clear_day_files(base, keep=()):
    """Remove the card and list files an earlier run made today (`base`.png/.txt and time-stamped copies), except
    `keep`."""
    folder, stem = os.path.split(base)
    for name in os.listdir(folder) if os.path.isdir(folder) else []:
        path = os.path.join(folder, name)
        root, extension = os.path.splitext(name)
        if extension in ('.png', '.txt') and (root == stem or root.startswith(stem + '_')) and path not in keep:
            try:
                os.remove(path)
            except OSError as error:
                print(f'Could not remove {name}, made by an earlier run today ({error.strerror or error}); delete it by hand')


def save(write, path):
    """Write with `write(path)`; if the file is held open by another app, save a time-stamped copy beside it."""
    try:
        write(path)
        return path
    except OSError as error:
        stem, extension = os.path.splitext(path)
        other = f'{stem}_{datetime.now():%H%M%S}{extension}'
        write(other)
        print(f'Could not replace {os.path.basename(path)} ({error.strerror or error}; is it open in an app such as '
              f'Photos?), so saved {os.path.basename(other)} beside it')
        return other


def make_files(season=None, note='', handle='', output_dir=None, log_path=None, now=None):
    """Save a card and a list of the picks no earlier day's card showed: (card path, list path), or (None, None) when
    there is nothing new.  A later run on the same day replaces the day's earlier runs."""
    season = season or current_season()
    log_path = log_path or share_log_path()
    now = now or pd.Timestamp.now(tz='America/New_York').tz_localize(None)
    picks, held, lines_at = upcoming_picks(os.path.join(cfb_folder(), f'{season}_cfb_picks.xlsx'), season, now)
    if held:
        print(f"Held back {held} pick(s): a team's latest game is not in the play-by-play yet (it comes a day or two "
              f"after the game), so these come on a later card")
    log = read_share_log(log_path)
    log = log[log['season'].notna()]
    today = now.strftime('%Y-%m-%d')
    season_log = log[log['season'] == str(season)]
    earlier = season_log[season_log['shown_on'] < today]
    shown_before = set(earlier['key'])
    fresh = [p for p in picks if pick_key(p['team_id'], p['game_id']) not in shown_before]
    if len(picks) > len(fresh):
        print(f'Left off {len(picks) - len(fresh)} pick(s) already on an earlier card')
    kickoff_of = {pick_key(p['team_id'], p['game_id']): p['kickoff'] for p in picks}
    # today's entries from earlier runs give way to this run's picks, except in games already under way (they are
    # no longer among the picks, so they keep their entry)
    replaced = (log['season'] == str(season)) & (log['shown_on'] == today) & log['key'].isin(kickoff_of)
    log = log[~replaced]
    day = f'{now:%a %b} {now.day}'
    folder = output_dir or cfb_folder(CARD_FOLDER)
    if not fresh:
        last = earlier['shown_on'].max() if len(earlier) else None
        print(f'No new picks since the last card ({pd.Timestamp(last):%a %b} {pd.Timestamp(last).day}): nothing to '
              f'share today' if last else 'No picks yet: nothing to share today')
        if replaced.any():
            log.to_csv(log_path, index=False)
        return None, None
    week = min(p['week'] for p in fresh)
    base = os.path.join(folder, f'{season} week {week} - {day}')
    update = len(earlier) > 0
    card = {'season': season, 'week': week, 'day': day, 'update': update, 'picks': fresh,
            'dates': f'New picks · {day}' if update else date_range([p['kickoff'] for p in fresh]),
            'caption': f'lines as of {clock(lines_at)}' if lines_at else '',
            'note': f'Picks hit {HIT_RATE:.0%} · bet the line shown at {bet_at(HIT_RATE)}'}
    print(f"Week {week}{', new since the last card' if update else ''}: {len(fresh)} spread pick(s)")
    os.makedirs(folder, exist_ok=True)
    image, text = draw_card(card, note, handle), pick_text(card)

    def write_text(path):
        with open(path, 'w', encoding='utf-8-sig') as f:
            f.write(text)
    paths = save(lambda path: image.save(path, optimize=True), base + '.png'), save(write_text, base + '.txt')
    clear_day_files(base, keep=paths)
    new = [{'season': str(season), 'week': str(p['week']), 'key': pick_key(p['team_id'], p['game_id']),
            'pick': describe(p), 'shown_on': today} for p in fresh]
    pd.concat([log, pd.DataFrame(new, columns=log.columns)], ignore_index=True).to_csv(log_path, index=False)
    return paths


def main(**options):
    card_path, list_path = make_files(**options)
    if card_path:
        print(f'Share card saved: {card_path}')
        print(f'Pick list saved: {list_path}')
    record = card_record.update()
    if record:
        print(record[1])
        print(f'Card record updated: {record[0]}')
    return 0


if __name__ == '__main__':
    FOOTER = 'For entertainment only. Please bet responsibly.'
    HANDLE = ''                         # e.g. '@yourname', shown at the bottom right
    OUTPUT_DIR = None                   # None: CFB\Cards on the Desktop
    sys.exit(main(note=FOOTER, handle=HANDLE, output_dir=OUTPUT_DIR))
