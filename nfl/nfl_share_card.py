"""The day's shareable graphic of new picks from the three weekly flows, and the same picks as text, in NFL\\Cards.

It reads what the flows wrote (all in the NFL folder on the Desktop, nfl_sheet.nfl_folder), so it shares no logic with
them and can run on its own at any time:
  Spreads  NFL/<season>_nfl_projections.xlsx: the week's underdog picks and favorite picks, from the
           Season Log by nfl_sheet's Python versions of the workbook's formulas (checked to agree with them), at the
           spreads the sheet holds, so a line typed into the sheet is the one shown.
  Props    NFL/<season>_nfl_props.xlsx: the picks of the last props run (its This Week tab), only the strongest
           (PROP_GROUPS: overs the model gives 80%+, receptions overs at lines of 2.5 or less, unders the model gives
           80%+; 61-79% in the backtest against 54-57% for the rest), each at the workbook's Bet At, which goes by market
           and line.  Overs are picked early in the week and unders within 24 hours of kickoff, so the midweek card has
           the overs and game-day cards the unders.
  Totals   NFL/<season>_wind_unders.csv: the UNDER and RAIN flags of the latest weather check (WATCH is a
           tracking list, not a bet), only from a check within WEATHER_WINDOW_HOURS of kickoff (game day).
The week is the earliest week of the season with an unplayed game, as in the flows; picks in games already under way
are left off (SHOW_STARTED_GAMES).  Each section says when its lines or forecast were taken, since lines move.

One card a day, of new picks only: a pick an earlier day's card showed is not shown again, even if its line moved
(NFL\\data\\share_card_log.csv, nfl_sheet.read_share_log, records every pick shown and the day it was).  A day's last
run is the one that counts: a later run on the same day replaces the earlier ones (their card files and log entries;
picks in games already under way stay).  From the next day on, the flows keep a shared pick as it was shared (the
projections workbook its spread and model inputs, the weather log its check), so it is tracked at the line that went
out.  The week's first card says the week's dates; later ones say "New picks" and the day.  With nothing new, no card
is made.  Until last week's final game is in nflverse's stats (late morning the day after Monday night's game), the
new week's spread and prop picks wait for a later card, since they were made without that game.

Every run then rebuilds the card record, NFL\\<season>_nfl_card_record.xlsx (nfl_card_record.py): every pick the cards
have shown, graded at the card's line, with a Bet? column to set to No for a pick you did not bet, which takes it out
of the record.

The graphic is 1080 px wide: 1080 x 1350 (4:5, the tallest Instagram feed post, and fine for texts, X and Facebook)
when the picks fit, shrinking the type a little first; otherwise taller.  The card shows everything shared (no "more"
line): it is saved in NFL\\Cards as '<season> week <N> - <day>.png', with the same picks as plain text, ready to paste
into a message or a post, beside it as '<season> week <N> - <day>.txt'.

So that people can choose which picks to follow, each section says how often its kind of pick has hit, and each prop
its group's rate.  Each section has its own color: spreads blue, props yellow, game totals orange.
Spread picks of one kind are equally strong: underdog picks hit about the same at every size of the model's lean (54% /
56.5% / 58% / 51% from the smallest to the largest, p = 0.41), and the weather unders of one flag likewise.  The rates
come from the projections workbook's Model Info, PROP_GROUPS and, for the weather unders, the research in
nfl_wind_unders.py (WEATHER_RATES).

Run it from this folder:  python nfl_share_card.py
The daily run (the 'NFL daily run' file on the Desktop, run_all.bat) makes both after the three flows.
"""

import os
import re
import sys
from datetime import datetime

import openpyxl
import pandas as pd

import nfl_card_record as card_record
import nfl_prop_signal as prop_signal
import nfl_props_sheet as props_sheet
import nfl_sheet as sheet
from nfl_injury_features import TEAM_FULL_NAMES, TEAM_RENAMES
from nflverse_data import CACHE_DIR, load_games

# the cards' common look, the one thing shared with the college football suite
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
import card_design as design  # noqa: E402
from card_design import FAINT, MUTED, PANEL, WHITE, Painter, clock, date_range  # noqa: E402

CARD_FOLDER = 'Cards'                      # the cards and lists go in this folder of the NFL folder

SECTIONS = tuple((key, title) + design.SECTION_COLORS[key]   # key, label, pill color, light text color
                 for key, title in (('spreads', 'SPREADS'), ('props', 'PLAYER PROPS'), ('totals', 'GAME TOTALS')))
PROP_SHORT = {'Passing Yards': 'PASS YDS', 'Pass Completions': 'COMPLETIONS', 'Passing TDs': 'PASS TDS',
              'Rush Attempts': 'RUSH ATT', 'Rushing Yards': 'RUSH YDS', 'Receiving Yards': 'REC YDS',
              'Receptions': 'RECEPTIONS', 'Interceptions Thrown': 'INTS', 'Pass Attempts': 'PASS ATT',
              'Pass + Rush Yards': 'PASS+RUSH YDS', 'Rush + Rec Yards': 'RUSH+REC YDS'}
# (primary, secondary): a team's badge is its primary color inside a ring of the secondary
TEAM_COLORS = {
    'ARI': ('#97233F', '#FFB612'), 'ATL': ('#A71930', '#A5ACAF'), 'BAL': ('#241773', '#9E7C0C'), 'BUF': ('#00338D', '#C60C30'),
    'CAR': ('#0085CA', '#BFC0BF'), 'CHI': ('#0B162A', '#C83803'), 'CIN': ('#FB4F14', '#000000'), 'CLE': ('#FF3C00', '#311D00'),
    'DAL': ('#041E42', '#869397'), 'DEN': ('#FB4F14', '#002244'), 'DET': ('#0076B6', '#B0B7BC'), 'GB': ('#203731', '#FFB612'),
    'HOU': ('#03202F', '#A71930'), 'IND': ('#002C5F', '#A2AAAD'), 'JAX': ('#006778', '#D7A22A'), 'KC': ('#E31837', '#FFB81C'),
    'LA': ('#003594', '#FFA300'), 'LAC': ('#0080C6', '#FFC20E'), 'LV': ('#000000', '#A5ACAF'), 'MIA': ('#008E97', '#FC4C02'),
    'MIN': ('#4F2683', '#FFC62F'), 'NE': ('#002244', '#C60C30'), 'NO': ('#D3BC8D', '#101820'), 'NYG': ('#0B2265', '#A71930'),
    'NYJ': ('#125740', '#FFFFFF'), 'PHI': ('#004C54', '#A5ACAF'), 'PIT': ('#FFB612', '#101820'), 'SEA': ('#002244', '#69BE28'),
    'SF': ('#AA0000', '#B3995D'), 'TB': ('#D50A0A', '#FF7900'), 'TEN': ('#0C2340', '#4B92DB'), 'WAS': ('#5A1414', '#FFB612'),
}
TEAM_LABELS = {'LA': 'LAR'}
CODE_OF = {name: code for code, name in TEAM_FULL_NAMES.items()}
# The weather flags' historical win rates (nfl_wind_unders.py, README section 12): unders with 15+ mph of recorded wind
# went 57% in every era since 1999 (499 games); in rain the wind rule does not flag, 184-135
WEATHER_RATES = {'UNDER': 0.57, 'RAIN': 184 / (184 + 135)}
# A weather flag goes on the card only from a check this close to kickoff: further out, the weather check marks its
# forecast 'recheck near kickoff' (nfl_wind_unders.py), and the edge was measured on forecasts made close to kickoff
WEATHER_WINDOW_HOURS = 36
# The props strong enough for the card, and how often their group won (the rules' picks with the returner skip, ESPN
# BET 2024-2025 and DraftKings 2026, pooled; analysis of October 22, 2026).  The model's chance ranks overs (54.5% below
# 80% outside receptions; 79.2% of 101 at 80%+; 77.7% / 69.2% at 80%+ in the discovery / untouched test seasons) and
# receptions overs at low lines win at any chance (69.7% of 313 at lines of 2.5 or less: 0.5 lines 71.2% / 68.0%, 1.5-2.5
# lines 68.1% / 71.1%), while at 3.5 and up they won 56.6% (83) and are not on the card; unders are flat at 55-57%
# except at 80%+ (61.3% / 63.2%).  Each prop's price is the props workbook's Bet At, by market and line.
STRONG_CHANCE = 0.80
PROP_GROUPS = {'over_strong': (0.792, 'overs the model gives 80%+'),
               'receptions_over': (0.697, 'receptions overs at lines of 2.5 or less'),
               'under_strong': (0.615, 'unders the model gives 80%+')}
KIND_NOTE = 'Hit rates are by kind of bet, not by pick: picks of the same kind are equally strong.'


# ---------------------------------------------------------------- small helpers

def nickname(code):
    return TEAM_FULL_NAMES.get(code, str(code)).split()[-1]


def label(code):
    return TEAM_LABELS.get(code, code)


def team_badge(code):
    """(badge text, (primary, secondary) colors) of a team, for card_design's badges."""
    return (label(code) if code else '?'), TEAM_COLORS.get(code, design.DEFAULT_COLORS)


def matchup(pick, short=False, at='@'):
    """'vs Browns' or '@ Browns' from the picked team's side ('vs CLE' with short); the game when the team is unknown."""
    team, home, away = pick.get('team'), pick['home'], pick['away']
    name = label if short else nickname
    if team not in (home, away):
        return f'{name(away)} {at} {name(home)}'
    other = away if team == home else home
    return f"{'vs' if team == home or pick.get('neutral') else at} {name(other)}"


def price(bet_at):
    """'-125 or better' for a prop's worst acceptable price, or '' without one."""
    try:
        return f'{int(bet_at):+d} or better'
    except (TypeError, ValueError):
        return ''


def file_time(path):
    return datetime.fromtimestamp(os.path.getmtime(path))


def share(value):
    """A rate as a cell holds it (0.572), or None."""
    return float(value) if sheet._number(value) else None


def span(rates):
    """'76%', or '75-76%' when the kinds of bet shown hit at different rates."""
    shown = sorted({round(100 * rate) for rate in rates if rate is not None})
    return '' if not shown else f'{shown[0]}%' if len(shown) == 1 else f'{shown[0]}-{shown[-1]}%'


def bet_at(rate):
    """'-116 or better': the worst price at which a bet winning `rate` of the time still returns MIN_EDGE (3%), the rule
    behind the props' Bet At price, so every section's price guide is set the same way."""
    price = prop_signal.american_from_chance(rate, prop_signal.MIN_EDGE)
    return f'{price:+d} or better' if price is not None else ''


def hit_notes(dog_rate, props, totals):
    """{section: how often that kind of pick has hit (in the backtests since 1999), and the line and worst price to bet
    it at}: picks of one kind are equally strong (the differences within a kind are chance), so the rates go with the
    sections, and with the group for props, whose rows each carry their own price."""
    notes = {}
    if dog_rate:
        notes['spreads'] = f'Picks hit {dog_rate:.0%} · bet the line shown at {bet_at(dog_rate)}'
    if props:
        notes['props'] = 'Bet only at price shown or better'
    kinds = [(label, WEATHER_RATES[flag]) for flag, label in (('UNDER', 'Wind'), ('RAIN', 'Rain'))
             if any(pick['flag'] == flag for pick in totals)]
    if kinds:
        (first, rate), rest = kinds[0], kinds[1:]
        # with both kinds on the card, the stricter price (the lower rate's) covers both
        notes['totals'] = (f'{first} unders hit {rate:.0%}' + ''.join(f', {label.lower()} unders {r:.0%}' for label, r in rest)
                           + f' · bet the line shown at {bet_at(min(r for _, r in kinds))}')
    return notes


# ---------------------------------------------------------------- the picks, from what the flows wrote

def week_games(season):
    """(week, {(home, away): (kickoff, neutral site)}, the season's regular-season games) for the earliest week of
    `season` with an unplayed game, as the flows choose it (the last week once every game is played)."""
    games = load_games(refresh=True)
    games = games[(games['season'] == season) & (games['game_type'] == 'REG')].copy()
    if games.empty:
        raise ValueError(f'nflverse has no {season} schedule yet')
    for column in ('home_team', 'away_team'):
        games[column] = games[column].replace(TEAM_RENAMES)
    unplayed = games[games['result'].isna()]
    week = int(unplayed['week'].min()) if len(unplayed) else int(games['week'].max())
    kickoffs = {}
    for g in games[games['week'] == week].itertuples():
        when = pd.Timestamp(f'{g.gameday} {g.gametime}') if isinstance(g.gametime, str) else pd.Timestamp(g.gameday)
        kickoffs[(g.home_team, g.away_team)] = (when, g.location == 'Neutral')
    return week, kickoffs, games


def stale_stats(games, week, season, cache_dir=CACHE_DIR):
    """Whether a game of last week is missing from the nflverse game stats the spread and prop picks were made from.
    nflverse posts a game's stats hours after it ends (late morning the day after), so on the morning after Monday
    night's game the new week's ratings and player forecasts lack that game."""
    if week <= 1:
        return False
    played = set(games[(games['week'] == week - 1) & games['result'].notna()]['game_id'])
    for name in (f'stats_team_week_{season}.csv', f'stats_player_week_{season}.csv'):
        path = os.path.join(cache_dir, name)
        if os.path.exists(path) and played - set(pd.read_csv(path, usecols=['game_id'])['game_id']):
            return True
    return False


def spread_picks(path, season, week, kickoffs):
    """(straight picks, when the lines were taken, how often the underdog picks have hit) for the week, from the
    projections workbook."""
    if not os.path.exists(path):
        print(f'Spreads: no projections workbook yet ({path})')
        return [], None, None
    workbook = openpyxl.load_workbook(path)
    log = sheet._find_log_sheet(workbook)
    if log is None:
        return [], None, None
    fields, settings = sheet._field_columns(log), sheet._settings_from(workbook)
    # how often the underdog picks have hit, from Model Info's backtest pooled over the sizes of lean (they do not
    # differ beyond chance)
    dog_rows = [(rate, games) for start, rate, games in sheet._hit_rate_table_from(workbook) if start >= 0 and games]
    dog_rate = sum(rate * games for rate, games in dog_rows) / sum(games for _, games in dog_rows) if dog_rows else None
    picks = []
    for r in range(2, log.max_row + 1):
        record = {key: None for key, _ in sheet.LOG_LAYOUT}
        record.update({field: log.cell(r, col).value for field, col in fields.items()})
        if str(record['year']) != str(season) or record['week'] != week or not sheet._number(record['spread']):
            continue
        record['kickoff'] = sheet._kickoff_text(record['kickoff'])
        home, away = CODE_OF.get(record['home']), CODE_OF.get(record['away'])
        if home is None or away is None:
            continue
        spread = float(record['spread'])
        kickoff, neutral = kickoffs.get((home, away), (None, False))
        game = {'home': home, 'away': away, 'kickoff': kickoff, 'neutral': neutral, 'spread': spread}
        if sheet._is_pick(record, settings):
            picks.append(dict(game, team=home if spread > 0 else away, line=f'+{abs(spread):g}', kind='underdog'))
        elif sheet._is_favorite_pick(record, settings):
            picks.append(dict(game, team=home if spread < 0 else away, line=f'-{abs(spread):g}', kind='favorite'))
    return picks, file_time(path), dog_rate


def prop_picks(path, season, week, kickoffs):
    """(the last props run's picks, best first by the model's chance; when the lines were taken), from the props
    workbook's This Week tab, or nothing when that tab is for another week."""
    if not os.path.exists(path):
        print(f'Props: no props workbook yet ({path})')
        return [], None
    workbook = openpyxl.load_workbook(path, read_only=True)
    rows = [list(row) for row in workbook[props_sheet.SHEET_THIS_WEEK].iter_rows(values_only=True)] \
        if props_sheet.SHEET_THIS_WEEK in workbook.sheetnames else []
    workbook.close()
    title = str(rows[0][0]) if rows and rows[0] and rows[0][0] else ''
    found = re.search(r'week (\d+), (\d{4})', title)
    if not found or (int(found.group(1)), int(found.group(2))) != (week, season):
        print(f'Props: the props workbook is not for week {week} of {season} ("{title}"); run the props flow')
        return [], None
    status = str(rows[1][0]) if len(rows) > 1 and rows[1] else ''
    updated = re.search(r'Updated (\w{3} \w{3} \d{1,2} \d{4}, \d{1,2}:\d{2} [AP]M)', status)
    updated = datetime.strptime(updated.group(1), '%a %b %d %Y, %I:%M %p') if updated else file_time(path)
    teams = {(r.get('week'), r.get('player'), r.get('prop'), r.get('side')): r.get('team') for r in props_sheet.read_log(path)}
    headers = {header: key for key, header, _ in props_sheet.THIS_WEEK_LAYOUT}
    picks, side, columns = [], None, None
    for row in rows:
        first = '' if not row or row[0] is None else str(row[0])
        heading = re.match(r'(Over|Under) picks \(', first)
        if heading:
            side, columns = heading.group(1), None
        elif side and columns is None and first in headers:
            columns = {headers[value]: i for i, value in enumerate(row) if value in headers}
        elif side and columns:
            if not row or all(value is None for value in row) or first.startswith('None'):
                side, columns = None, None
                continue
            pick = {key: row[i] if i < len(row) else None for key, i in columns.items()}
            if not pick.get('player'):
                continue
            away, _, home = str(pick.get('game') or '').partition(' at ')
            number = re.search(r'\d+(\.\d+)?', str(pick.get('pick') or ''))
            picks.append({'player': str(pick.get('player') or ''), 'prop': str(pick.get('prop') or ''), 'side': side,
                          'line': float(number.group(0)) if number else None, 'chance': float(pick.get('chance') or 0),
                          'bet_at': pick.get('bet_at'), 'home': home, 'away': away,
                          'team': teams.get((week, pick.get('player'), pick.get('prop'), side)),
                          'kickoff': kickoffs.get((home, away), (None, False))[0],
                          'late': str(pick.get('why') or '').startswith('LATE'), 'confidence': share(pick.get('hist'))})
    picks.sort(key=lambda p: -p['chance'])
    return picks, updated


def total_picks(path, season, week):
    """(the week's weather unders, when the forecast was checked) from the wind-unders season log: only flags from a
    check within WEATHER_WINDOW_HOURS of kickoff, since a check further out is a rough forecast that the weather check
    itself marks 'recheck near kickoff', and the edge was measured on forecasts made close to kickoff."""
    if not os.path.exists(path):
        print(f'Totals: no weather check yet ({path})')
        return [], None
    log = pd.read_csv(path)
    for column in ('rain_flag', 'rain_mm'):
        if column not in log.columns:
            log[column] = None
    log = log[(log['season'] == season) & (log['week'] == week)]
    flagged = log[(log['flag'] == 'UNDER') | (log['rain_flag'] == 'RAIN')]
    if 'hours_before_kickoff' in flagged.columns:
        near = pd.to_numeric(flagged['hours_before_kickoff'], errors='coerce') <= WEATHER_WINDOW_HOURS
        if (~near).any():
            print(f'Totals: left off {int((~near).sum())} weather flag(s) from a check more than {WEATHER_WINDOW_HOURS} '
                  f'hours before kickoff (the daily run on game day rechecks them)')
        flagged = flagged[near]
    picks = []
    for row in flagged.itertuples():
        wind = f'{float(row.wind_mph):.0f} mph wind' if pd.notna(row.wind_mph) else ''
        reason = wind if row.flag == 'UNDER' else 'rain' + (f', {wind}' if wind else '')
        picks.append({'home': CODE_OF.get(row.home, row.home), 'away': CODE_OF.get(row.away, row.away),
                      'kickoff': pd.Timestamp(row.kickoff_et), 'total': float(row.total) if pd.notna(row.total) else None,
                      'odds': float(row.under_odds) if pd.notna(row.under_odds) else None, 'reason': reason,
                      'flag': 'UNDER' if row.flag == 'UNDER' else 'RAIN'})
    checked = pd.to_datetime(log['checked_at']).max() if len(log) else None
    return picks, checked


# ---------------------------------------------------------------- drawing (the look is card_design's)

class Layout(design.Layout):
    """card_design's layout with the NFL card's sections: spread tiles, the props panel and the game totals panel."""

    def spreads(self, picks, y, accent, card):
        """Tiles, two to a row: the team's badge, name and line, and the game."""
        tiles = [{'badge': team_badge(pick['team']), 'name': nickname(pick['team']).upper(), 'line': pick['line'],
                  'sub': ' · '.join(part for part in (matchup(pick), clock(pick['kickoff'])) if part)} for pick in picks]
        return self.tiles(tiles, y, accent)

    def props(self, picks, y, accent, card):
        """A panel of rows in the section's order (card_order): the player's team badge, name and bet; the game, then
        how often such picks hit and the worst price to take."""
        s, painter = self.s, self.painter
        height, pad, diameter = 80 * s, 24 * s, 46 * s
        name_font, bet_font, side_font, hit_font = (self.font('SemiBold SemiCondensed', 31), self.font('Bold Condensed', 28),
                                                    self.font('Condensed', 26), self.font('SemiBold Condensed', 26))
        bottom = y + len(picks) * height
        self.under(Painter.rounded, (self.left, y, self.right, bottom), 22 * s, PANEL)
        tx = self.left + pad + diameter + 18 * s
        for i, pick in enumerate(picks):
            top = y + i * height
            if i:
                self.rule(top, pad)
            self.over(painter.badge, self.left + pad + diameter / 2, top + height / 2, diameter, *team_badge(pick['team']))
            game = ' · '.join(part for part in (matchup(pick, short=True), clock(pick['kickoff'])) if part)
            hit = f"hits {pick['confidence']:.0%}" if pick.get('confidence') is not None else ''
            bet_at = price(pick['bet_at'])
            tail = (' · ' if hit and bet_at else '') + bet_at
            side_width = max(painter.width(game, side_font), painter.width(hit, hit_font) + painter.width(tail, side_font))
            room = self.right - pad - 28 * s - side_width - tx
            line = '' if pick['line'] is None else f"{pick['line']:g} "
            bet = f"{pick['side'].upper()} {line}{PROP_SHORT.get(pick['prop'], pick['prop'].upper())}"
            self.text(tx, top + height * 0.35, painter.fit(pick['player'], name_font, room), name_font, WHITE)
            self.text(tx, top + height * 0.71, painter.fit(bet, bet_font, room), bet_font, accent)
            self.text(self.right - pad, top + height * 0.35, game, side_font, MUTED, align='right')
            self.text(self.right - pad, top + height * 0.71, tail, side_font, FAINT, align='right')
            if hit:
                self.text(self.right - pad - painter.width(tail, side_font), top + height * 0.71, hit, hit_font, accent,
                          align='right')
        return bottom

    def totals(self, picks, y, accent, card):
        """A panel of rows: both teams' badges, the game and the total, and the weather behind the pick."""
        s, painter = self.s, self.painter
        height, pad, diameter = 96 * s, 26 * s, 54 * s
        game_font, why_font = self.font('Bold Condensed', 38), self.font('Condensed', 26)
        bottom = y + len(picks) * height
        self.under(Painter.rounded, (self.left, y, self.right, bottom), 22 * s, PANEL)
        for i, pick in enumerate(picks):
            top = y + i * height
            if i:
                self.rule(top, pad)
            cx, gap = self.left + pad + diameter / 2, 10 * s
            self.over(painter.badge, cx, top + height / 2, diameter, *team_badge(pick['away']))
            self.over(painter.badge, cx + diameter + gap, top + height / 2, diameter, *team_badge(pick['home']))
            tx = self.left + pad + 2 * diameter + gap + 22 * s
            bet = 'UNDER' + ('' if pick['total'] is None else f" {pick['total']:g}")
            room = self.right - pad - painter.width(bet, game_font) - 24 * s - tx
            game = f"{nickname(pick['away']).upper()} @ {nickname(pick['home']).upper()}"
            self.text(tx, top + height * 0.37, painter.fit(game, game_font, room), game_font, WHITE)
            self.text(self.right - pad, top + height * 0.37, bet, game_font, accent, align='right')
            why = ' · '.join(part for part in (pick['reason'], clock(pick['kickoff'])) if part)
            self.text(tx, top + height * 0.7, painter.fit(why, why_font, self.right - pad - tx), why_font, MUTED)
        return bottom

    def content(self, card):
        """The header and every section with picks; returns where they end."""
        y = self.header('NFL', card['season'], card['week'], card['dates'], 64 * self.s) + 46 * self.s
        shown = [section for section in SECTIONS if card[section[0]]]
        for i, (key, title, color, accent) in enumerate(shown):
            y = self.heading(y + (34 * self.s if i else 0), title, color, card['captions'].get(key),
                             card['notes'].get(key), accent)
            y = getattr(self, key)(card[key], y, accent, card)
        return y if shown else self.empty(y)


def draw_card(card, note='', handle=''):
    """The card as an image (card_design.render)."""
    explain = KIND_NOTE if any(card[key] for key, _, _, _ in SECTIONS) else ''
    return design.render(Layout, lambda layout: layout.content(card), note, handle, explain)


# ---------------------------------------------------------------- every pick, as text

def when(pick):
    return f", {clock(pick['kickoff'])}" if clock(pick['kickoff']) else ''


def pick_lines(card):
    """[(section title, how often its kind of pick hits, [one line per pick])]: every pick, all the props included."""
    spreads = [f"{nickname(p['team'])} {p['line']} {matchup(p)}{when(p)}"
               + (' (favorite pick)' if p['kind'] == 'favorite' else '') for p in card['spreads']]
    props = []
    for p in card['props']:
        team = f" ({label(p['team'])})" if p['team'] else ''
        line = '' if p['line'] is None else f" {p['line']:g}"
        bet_at = price(p['bet_at'])
        hit = f" (hits {p['confidence']:.0%})" if p.get('confidence') is not None else ''
        props.append(f"{p['player']}{team} {p['side'].lower()}{line} {p['prop'].lower()}, {matchup(p, short=True)}{when(p)}"
                     + (f'; {bet_at}' if bet_at else '') + hit)
    totals = []
    for p in card['totals']:
        total = '' if p['total'] is None else f" {p['total']:g}"
        odds = '' if p['odds'] is None else f" ({p['odds']:+.0f})"
        totals.append(f"{nickname(p['away'])} @ {nickname(p['home'])} under{total}{odds}, {p['reason']}{when(p)}")
    captions, notes = card['captions'], card['notes']
    return [(f'{title} ({captions[key]})' if captions.get(key) else title, notes.get(key, ''), lines)
            for key, title, lines in (('spreads', 'Spreads', spreads),
                                      ('props', 'Player props', props), ('totals', 'Game totals', totals)) if lines]


def pick_text(card):
    """Every pick as plain text, all the props included, ready to paste into a message or a post."""
    title = (f"NFL week {card['week']}: new picks, {card['day']}, times ET" if card.get('update') else
             f"NFL week {card['week']} picks" + (f" ({card['dates']})" if card['dates'] else '') + ', times ET')
    lines = [title, '']
    sections = pick_lines(card)
    for title, note, picks in sections:
        lines += [title] + ([note] if note else []) + [f'  {line}' for line in picks] + ['']
    return '\n'.join(lines + [KIND_NOTE, ''] if sections else lines + ['No picks yet this week.', ''])


# ---------------------------------------------------------------- what earlier cards showed

def pick_key(kind, pick):
    """A pick's identity without its line, so a pick shown once is not shown again when its line moves."""
    game = f"{pick['away']}@{pick['home']}"
    if kind == 'spreads':
        return f"spread|{pick['kind']}|{pick['team']}|{game}"
    if kind == 'props':
        return f"prop|{pick['player']}|{pick['prop']}|{pick['side']}|{game}"
    return f"total|{game}"


def describe(kind, pick):
    if kind == 'spreads':
        return f"{nickname(pick['team'])} {pick['line']}"
    if kind == 'props':
        return f"{pick['player']} {pick['side']}" + ('' if pick['line'] is None else f" {pick['line']:g}") + f" {pick['prop']}"
    return f"{nickname(pick['away'])} @ {nickname(pick['home'])} under" + ('' if pick['total'] is None else f" {pick['total']:g}")


# ---------------------------------------------------------------- running it

def by_kickoff(pick):
    kickoff = pick['kickoff']
    return pd.Timestamp.max if kickoff is None or pd.isna(kickoff) else kickoff


def card_order(kind, pick):
    """Where a pick goes within its section: by kickoff, then alphabetically by the team's name as the card shows it
    (for game totals the away team, then the home team), then for props the player's name (as shown, first name first)
    and the prop.  Every key is the pick's own, so the same picks always come out in the same order."""
    def name(text):
        return str(text or '').casefold()
    if kind == 'totals':
        return (by_kickoff(pick), name(nickname(pick['away'])), name(nickname(pick['home'])))
    if kind == 'props':
        return (by_kickoff(pick), name(nickname(pick['team']) if pick.get('team') else ''), name(pick['player']),
                name(pick['prop']), pick['side'], -1.0 if pick['line'] is None else float(pick['line']))
    return (by_kickoff(pick), name(nickname(pick['team'])), str(pick['line']), pick['kind'])


def card_props(props):
    """The props strong enough for the card, each with its group's hit rate as its confidence: overs the model gives
    80%+, receptions overs at lines of 2.5 or less, and unders the model gives 80%+ (PROP_GROUPS).  The rest (about
    54-57%, not enough after the usual juice) stay in the props workbook, as do late overs (about 54%)."""
    chosen = []
    for p in props:
        group = prop_group(p)
        if group:
            chosen.append(dict(p, confidence=PROP_GROUPS[group][0], group=group))
    return chosen


def prop_group(pick):
    """Which of PROP_GROUPS a prop pick belongs to, or None."""
    if pick.get('late'):
        return None
    if pick['side'] == 'Over' and pick['chance'] >= STRONG_CHANCE:
        return 'over_strong'
    if (pick['side'] == 'Over' and pick['prop'] == 'Receptions' and pick['line'] is not None
            and prop_signal.line_group(pick['line']) != prop_signal.LINE_GROUPS[-1][0]):
        return 'receptions_over'
    if pick['side'] == 'Under' and pick['chance'] >= STRONG_CHANCE:
        return 'under_strong'
    return None


def make_files(season=None, show_started=False, note='', handle='', output_dir=None, log_path=None,
               now=None):
    """Read the week's picks and save a card and a list of the ones no earlier day's card showed: (the card's path, the
    list's path), or (None, None) when there is nothing new.  A later run on the same day replaces the day's earlier
    runs: their card files go, and the day's entries in the share log become this run's picks (picks in games already
    under way stay, since no later card can show them)."""
    season = season or sheet.get_current_season_year()
    source = sheet.nfl_folder()   # where the three flows write their files
    log_path = log_path or sheet.share_log_path()
    now = now or pd.Timestamp.now(tz='America/New_York').tz_localize(None)
    week, kickoffs, games = week_games(season)
    spreads, lines_at, dog_rate = spread_picks(os.path.join(source, f'{season}_nfl_projections.xlsx'), season, week,
                                               kickoffs)
    props, props_at = prop_picks(os.path.join(source, f'{season}_nfl_props.xlsx'), season, week, kickoffs)
    totals, checked_at = total_picks(os.path.join(source, f'{season}_wind_unders.csv'), season, week)
    groups = {'spreads': spreads, 'props': card_props(props), 'totals': totals}
    groups = {kind: sorted(picks, key=lambda p, kind=kind: card_order(kind, p)) for kind, picks in groups.items()}
    if not show_started:
        upcoming = {kind: [p for p in picks if by_kickoff(p) > now] for kind, picks in groups.items()}
        started = sum(map(len, groups.values())) - sum(map(len, upcoming.values()))
        if started:
            print(f'Left off {started} pick(s) in games already under way (SHOW_STARTED_GAMES)')
        groups = upcoming
    if stale_stats(games, week, season):
        held = sum(len(groups[kind]) for kind in ('spreads', 'props'))
        if held:
            print(f'Held back {held} spread and prop pick(s): last week\'s final game is not in nflverse\'s stats '
                  f'yet (they come late morning the day after), so these come on a later card')
        groups.update(spreads=[], props=[])
    log = sheet.read_share_log(log_path)
    week_log = log[(log['season'] == str(season)) & (log['week'] == str(week))]
    today = now.strftime('%Y-%m-%d')
    earlier = week_log[week_log['shown_on'] < today]
    shown_before = set(earlier['key'])
    fresh = {kind: [p for p in picks if pick_key(kind, p) not in shown_before] for kind, picks in groups.items()}
    repeats = sum(map(len, groups.values())) - sum(map(len, fresh.values()))
    if repeats:
        print(f'Left off {repeats} pick(s) already on an earlier card')
    day = f'{now:%a %b} {now.day}'
    folder = output_dir or sheet.nfl_folder(CARD_FOLDER)
    base = os.path.join(folder, f'{season} week {week} - {day}')

    def started(key):
        away, _, home = key.rsplit('|', 1)[1].partition('@')
        kickoff = kickoffs.get((home, away), (None, None))[0]
        return kickoff is not None and not pd.isna(kickoff) and kickoff <= now
    # today's entries from earlier runs give way to this run's picks, except in games already under way
    replaced = ((log['season'] == str(season)) & (log['week'] == str(week)) & (log['shown_on'] == today)
                & ~log['key'].map(started).astype(bool))
    log = log[~replaced]
    if not any(fresh.values()):
        last = earlier['shown_on'].max() if len(earlier) else None
        print(f'No new picks since the last card ({pd.Timestamp(last):%a %b} {pd.Timestamp(last).day}): nothing to share today'
              if last else f'No picks yet for week {week}: nothing to share today')
        clear_day_files(base)
        if replaced.any():
            log.to_csv(log_path, index=False)
        return None, None
    spreads, props, totals = (fresh[kind] for kind in ('spreads', 'props', 'totals'))
    days = [kickoff for kickoff, _ in kickoffs.values() if kickoff is not None and not pd.isna(kickoff)]
    checked = checked_at is not None and not pd.isna(checked_at)
    captions = {'spreads': f'lines as of {clock(lines_at)}' if lines_at else '',
                'props': f'lines as of {clock(props_at)}' if props_at else '',
                'totals': f'weather forecast as of {clock(checked_at)}' if checked else ''}
    update = len(earlier) > 0
    card = {'season': season, 'week': week, 'dates': f'New picks · {day}' if update else (date_range(days) if days else ''),
            'update': update, 'day': day, 'spreads': spreads, 'props': props, 'totals': totals,
            'captions': captions, 'notes': hit_notes(dog_rate, props, totals)}
    print(f"Week {week}{', new since the last card' if update else ''}: {len(spreads)} spread pick(s), "
          f"{len(props)} player prop(s), {len(totals)} game total(s)")
    os.makedirs(folder, exist_ok=True)
    image, text = draw_card(card, note, handle), pick_text(card)

    def write_text(path):
        with open(path, 'w', encoding='utf-8-sig') as f:   # the BOM lets every Windows app read the dashes
            f.write(text)
    paths = save(lambda path: image.save(path, optimize=True), base + '.png'), save(write_text, base + '.txt')
    clear_day_files(base, keep=paths)
    kept = set(log.loc[(log['season'] == str(season)) & (log['week'] == str(week)), 'key'])
    shared = {'spreads': spreads, 'props': props, 'totals': totals}
    new = [{'season': str(season), 'week': str(week), 'key': pick_key(kind, p), 'pick': describe(kind, p), 'shown_on': today}
           for kind, picks in shared.items() for p in picks if pick_key(kind, p) not in kept]
    pd.concat([log, pd.DataFrame(new, columns=log.columns)], ignore_index=True).to_csv(log_path, index=False)
    return paths


def clear_day_files(base, keep=()):
    """Remove the card and list files an earlier run made today (`base`.png/.txt and the time-stamped copies made while
    an app held the file), except `keep`."""
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
    """`write(path)`, or, when Windows will not replace the file because an app has it open (Photos keeps the image it
    shows mapped into memory), `write` to a name with the time in it beside it.  Returns the path written."""
    try:
        write(path)
        return path
    except OSError as error:
        stem, extension = os.path.splitext(path)
        other = f'{stem}_{datetime.now():%H%M%S}{extension}'
        write(other)
        print(f'Could not replace {os.path.basename(path)} ({error.strerror or error}; is it open in an app such as '
              f'Photos?), so saved {os.path.basename(other)} beside it; close the app to keep the usual name next time')
        return other


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
    SHOW_STARTED_GAMES = False          # True: keep picks in games already under way
    FOOTER = 'For entertainment only. Please bet responsibly.'
    HANDLE = ''                         # e.g. '@yourname', shown at the bottom right
    OUTPUT_DIR = None                   # None: NFL\Cards on the Desktop
    sys.exit(main(show_started=SHOW_STARTED_GAMES, note=FOOTER, handle=HANDLE, output_dir=OUTPUT_DIR))
