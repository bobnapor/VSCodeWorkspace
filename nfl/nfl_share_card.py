"""The day's shareable graphic of new picks from the three weekly flows, and the same picks as text, on the Desktop.

It reads what the flows wrote, so it shares no logic with them and can run on its own at any time:
  Spreads  <Desktop>/<season>_nfl_projections.xlsx: the week's underdog picks, favorite picks and teaser legs, from the
           Season Log by nfl_sheet's Python versions of the workbook's formulas (checked to agree with them), at the
           spreads the sheet holds, so a line typed into the sheet is the one shown.
  Props    <Desktop>/<season>_nfl_props.xlsx: the picks of the last props run (its This Week tab), only the strongest
           (PROP_GROUPS: overs the model gives 80%+, receptions overs, unders the model gives 80%+; 61-77% in the
           backtest against 54-57% for the rest).  Overs are picked early in the week and unders within 24 hours of
           kickoff, so the midweek card has the overs and game-day cards the unders.
  Totals   <Desktop>/<season>_wind_unders.csv: the UNDER and RAIN flags of the latest weather check (WATCH is a
           tracking list, not a bet), only from a check within WEATHER_WINDOW_HOURS of kickoff (game day).
The week is the earliest week of the season with an unplayed game, as in the flows; picks in games already under way
are left off (SHOW_STARTED_GAMES).  Each section says when its lines or forecast were taken, since lines move.

One card a day, of new picks only: a pick an earlier day's card showed is not shown again, even if its line moved
(data_cache/share_card_log.csv, nfl_sheet.read_share_log, records every pick shown and the day it was).  A day's last
run is the one that counts: a later run on the same day replaces the earlier ones (their card files and log entries;
picks in games already under way stay).  From the next day on, the flows keep a shared pick as it was shared (the
projections workbook its spread and model inputs, the weather log its check), so it is tracked at the line that went
out.  The week's first card says the week's dates; later ones say "New picks" and the day.  With nothing new, no card
is made.  Until last week's final game is in nflverse's stats (late morning the day after Monday night's game), the
new week's spread, teaser and prop picks wait for a later card, since they were made without that game.

The graphic is 1080 px wide: 1080 x 1350 (4:5, the tallest Instagram feed post, and fine for texts, X and Facebook)
when the picks fit, shrinking the type a little first; otherwise taller.  The card shows everything shared (no "more"
line): it is saved in the 'NFL picks' folder on the Desktop as '<season> week <N> - <day>.png', with the same picks as
plain text, ready to paste into a message or a post, beside it as '<season> week <N> - <day>.txt'.

So that people can choose which picks to follow, each section says how often its kind of pick has hit, and each prop
its group's rate.  Each section has its own color: spreads blue, teaser legs purple, props yellow, game totals orange.
Spread picks of one kind are equally strong: underdog picks hit about the same at every size of the model's lean (54% /
56.5% / 58% / 51% from the smallest to the largest, p = 0.41), and the weather unders of one flag likewise.  The rates
come from the projections workbook's Model Info, PROP_GROUPS and, for the weather unders, the research in
nfl_wind_unders.py (WEATHER_RATES).

Run it from this folder:  python nfl_share_card.py
The daily run (run_all.bat, the 'NFL daily run' shortcut) makes both after the three flows.
"""

import math
import os
import re
import sys
from datetime import datetime

import openpyxl
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import nfl_prop_signal as prop_signal
import nfl_props_sheet as props_sheet
import nfl_sheet as sheet
from nfl_injury_features import TEAM_FULL_NAMES, TEAM_RENAMES
from nflverse_data import CACHE_DIR, load_games

CARD_FOLDER = 'NFL picks'                  # the cards and lists go in this folder on the Desktop

WIDTH, HEIGHT_MIN, HEIGHT_FEED, HEIGHT_TALL = 1080, 1080, 1350, 1920
# (type scale, tallest card) to try in turn: the feed shape, shrinking the type a little, then taller, up to 9:16
SHAPES = [(scale, HEIGHT_FEED) for scale in (1.0, 0.94, 0.88)] + \
         [(scale, HEIGHT_TALL) for scale in (1.0, 0.94, 0.88, 0.82, 0.76, 0.7)]
SUPERSAMPLE = 2                            # drawn at twice the size and scaled down, for smooth edges
MARGIN = 64                                # left and right margins, in pixels
FOOTER_HEIGHT, FOOTER_LINE, FOOTER_GAP = 92, 34, 36   # the small print (one line, each extra line) and the room above it
FONT_DIR = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
BAHNSCHRIFT = os.path.join(FONT_DIR, 'bahnschrift.ttf')
FALLBACK_FONTS = {'Bold': 'segoeuib.ttf', 'SemiBold': 'seguisb.ttf', 'Regular': 'segoeui.ttf', 'Light': 'segoeuil.ttf'}

BG_TOP, BG_BOTTOM = (7, 17, 38), (14, 34, 70)
GOLD, WHITE, MUTED, FAINT = (255, 196, 37), (255, 255, 255), (156, 174, 203), (112, 131, 163)
BADGE_EDGE = (64, 84, 120)
TILE, PANEL, RULE, WATERMARK = 22, 16, 26, 18   # white overlays, by opacity (0-255)
SECTIONS = (  # key, label, pill color, light text color
    ('spreads', 'SPREADS', (47, 112, 235), (133, 180, 255)),
    ('teasers', 'TEASER LEGS', (130, 84, 236), (198, 175, 255)),
    ('props', 'PLAYER PROPS', (234, 179, 8), (253, 224, 120)),
    ('totals', 'GAME TOTALS', (234, 88, 12), (253, 186, 116)),
)
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
# The props strong enough for the card, and how often their group won (the rules' picks, ESPN BET 2024-2025 and
# DraftKings 2026, pooled; analysis of October 22, 2026).  The model's chance ranks overs (54.5% below 80% outside
# receptions, 74.6% at 80%+; 77.7% / 69.2% at 80%+ in the discovery / untouched test seasons) and receptions overs win
# at any chance (65.3% / 68.8%); unders are flat at 55-57% except at 80%+ (61.3% / 63.2%).
STRONG_CHANCE = 0.80
PROP_GROUPS = {'over_strong': (0.767, 'overs the model gives 80%+'), 'receptions_over': (0.643, 'receptions overs'),
               'under_strong': (0.615, 'unders the model gives 80%+')}
KIND_NOTE = 'Hit rates are by kind of bet, not by pick: picks of the same kind are equally strong.'


# ---------------------------------------------------------------- small helpers

def nickname(code):
    return TEAM_FULL_NAMES.get(code, str(code)).split()[-1]


def label(code):
    return TEAM_LABELS.get(code, code)


def clock(moment):
    """'Sun 1:00 PM' (Eastern), or '' when unknown."""
    if moment is None or pd.isna(moment):
        return ''
    return f'{moment:%a %I:%M %p}'.replace(' 0', ' ', 1)


def date_range(days):
    """'Oct 22–26' or 'Oct 29 – Nov 2'."""
    first, last = min(days), max(days)
    if first.month != last.month:
        return f'{first:%b} {first.day} – {last:%b} {last.day}'
    return f'{first:%b} {first.day}' + (f'–{last.day}' if last.day != first.day else '')


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


def hit_notes(dog_rate, legs, props, totals):
    """{section: how often that kind of pick has hit (in the backtests since 1999), and the line and worst price to bet
    it at}: picks of one kind are equally strong (the differences within a kind are chance), so the rates go with the
    sections, and with the group for props, whose rows each carry their own price."""
    notes = {}
    if dog_rate:
        notes['spreads'] = f'Picks hit {dog_rate:.0%} · bet the line shown at {bet_at(dog_rate)}'
    rates = [leg['confidence'] for leg in legs if leg.get('confidence') is not None]
    if rates:
        # both legs must win, so a 2-leg teaser hits about the legs' rate squared (every 2-leg teaser from the same
        # week's legs since 1999 won 58.0%, 542 of 935; underdog legs 57.6%, against 57.1% / 57.3% squared)
        both = (sum(rates) / len(rates)) ** 2
        notes['teasers'] = f'2-leg teaser hits {both:.0%} · bet the lines shown, teaser at {bet_at(both)}'
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
    """(straight picks, teaser legs, when the lines were taken, how often the underdog picks have hit) for the week, from
    the projections workbook."""
    if not os.path.exists(path):
        print(f'Spreads: no projections workbook yet ({path})')
        return [], [], None, None
    workbook = openpyxl.load_workbook(path)
    log = sheet._find_log_sheet(workbook)
    if log is None:
        return [], [], None, None
    fields, settings = sheet._field_columns(log), sheet._settings_from(workbook)
    # how often each kind of pick has hit, from Model Info's tables: the underdog picks' backtest pooled over the sizes of
    # lean (they do not differ beyond chance), and the teaser legs since 1999
    dog_rows = [(rate, games) for start, rate, games in sheet._hit_rate_table_from(workbook) if start >= 0 and games]
    dog_rate = sum(rate * games for rate, games in dog_rows) / sum(games for _, games in dog_rows) if dog_rows else None
    teaser_rates = {('Underdog' if str(label).startswith('Underdog') else 'Favorite'): rate
                    for label, rate, _since_2018 in sheet._teaser_table_from(workbook)}
    picks, legs = [], []
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
        leg = sheet._teaser_leg(record, settings)
        if leg:
            underdog = home if spread > 0 else away
            favorite = away if underdog == home else home
            legs.append(dict(game, team=underdog if leg[2] == 'Underdog' else favorite, line=leg[1], kind=leg[2].lower(),
                             confidence=teaser_rates.get(leg[2])))
    return picks, legs, file_time(path), dog_rate


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


# ---------------------------------------------------------------- drawing

def hex_rgb(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def luminance(rgb):
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255


class Painter:
    """Fonts (Bahnschrift's styles, or Segoe UI where it is missing) and the shapes the card is made of.  Every drawing
    method takes the ImageDraw first, so the same calls draw white overlays on a mask (an opacity as the fill) and
    colors on the card."""

    def __init__(self):
        self.fonts, self.caps = {}, {}

    def font(self, style, size):
        key = (style, max(1, int(round(size))))
        if key not in self.fonts:
            if os.path.exists(BAHNSCHRIFT):
                font = ImageFont.truetype(BAHNSCHRIFT, key[1])
                try:
                    font.set_variation_by_name(style)
                except (AttributeError, OSError, ValueError):
                    pass
            else:
                weight = next((w for w in ('SemiBold', 'Bold', 'Light') if w in style), 'Regular')
                font = ImageFont.truetype(os.path.join(FONT_DIR, FALLBACK_FONTS[weight]), key[1])
            self.fonts[key] = font
        return self.fonts[key]

    def cap(self, font):
        """(top, bottom) of a capital letter drawn at y = 0, so text can be centered on its capitals."""
        if id(font) not in self.caps:
            size = int(font.size)
            canvas = Image.new('L', (size * 2, size * 2), 0)
            ImageDraw.Draw(canvas).text((0, 0), 'H', fill=255, font=font)
            box = canvas.getbbox()
            self.caps[id(font)] = (box[1], box[3]) if box else (0, size)
        return self.caps[id(font)]

    @staticmethod
    def width(text, font, tracking=0):
        if not tracking:
            return font.getsize(text)[0]
        return sum(font.getsize(ch)[0] for ch in text) + tracking * (len(text) - 1)

    def fit(self, text, font, room):
        """`text`, cut short with an ellipsis if it is wider than `room`."""
        if self.width(text, font) <= room:
            return text
        while text and self.width(text + '…', font) > room:
            text = text[:-1]
        return text.rstrip() + '…'

    def text(self, draw, x, cy, text, font, fill, align='left', tracking=0):
        """Draw `text` with its capitals centered on `cy`; x is its left, right or center edge by `align`."""
        top, bottom = self.cap(font)
        y = cy - (top + bottom) / 2
        width = self.width(text, font, tracking)
        x = x - width if align == 'right' else x - width / 2 if align == 'center' else x
        if not tracking:
            draw.text((x, y), text, fill=fill, font=font)
            return
        for ch in text:
            draw.text((x, y), ch, fill=fill, font=font)
            x += font.getsize(ch)[0] + tracking

    @staticmethod
    def rounded(draw, box, radius, fill):
        x0, y0, x1, y1 = box
        r = max(0.0, min(radius, (x1 - x0) / 2, (y1 - y0) / 2))
        draw.rectangle((x0 + r, y0, x1 - r, y1), fill=fill)
        draw.rectangle((x0, y0 + r, x1, y1 - r), fill=fill)
        for cx, cy, start in ((x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90)):
            draw.pieslice((cx - r, cy - r, cx + r, cy + r), start, start + 90, fill=fill)

    @staticmethod
    def line(draw, box, fill, width):
        draw.line(box, fill=fill, width=width)

    def badge(self, draw, cx, cy, diameter, team):
        """A team's colors as a round badge with its abbreviation (no logos)."""
        primary, secondary = (hex_rgb(c) for c in TEAM_COLORS.get(team, ('#475569', '#94A3B8')))
        r, edge, ring = diameter / 2, max(2.0, diameter * 0.04), diameter * 0.085
        draw.ellipse((cx - r - edge, cy - r - edge, cx + r + edge, cy + r + edge), fill=BADGE_EDGE)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=secondary)
        draw.ellipse((cx - r + ring, cy - r + ring, cx + r - ring, cy + r - ring), fill=primary)
        text = label(team) if team else '?'
        font = self.font('Bold SemiCondensed', diameter * (0.34 if len(text) > 2 else 0.4))
        self.text(draw, cx, cy, text, font, (17, 24, 39) if luminance(primary) > 0.55 else WHITE, align='center')


def background(width, height):
    """The card's vertical gradient."""
    column = Image.new('RGB', (1, 256))
    column.putdata([tuple(int(round(a + (b - a) * i / 255)) for a, b in zip(BG_TOP, BG_BOTTOM)) for i in range(256)])
    return column.resize((width, height), Image.BILINEAR)


class Layout:
    """The card at one type scale, recorded rather than drawn so its height is known before the canvas is made:
    `overlay` holds the see-through white shapes (drawn on a mask, an opacity as the fill) and `paint` what goes on
    top.  Lengths are in supersampled pixels; `s` turns a full-size length into them."""

    def __init__(self, painter, scale):
        self.painter, self.s = painter, SUPERSAMPLE * scale
        self.left, self.right = MARGIN * SUPERSAMPLE, (WIDTH - MARGIN) * SUPERSAMPLE
        self.overlay, self.paint = [], []

    def font(self, style, size):
        return self.painter.font(style, size * self.s)

    def caps(self, font):
        top, bottom = self.painter.cap(font)
        return bottom - top

    def under(self, method, *args, **kwargs):
        self.overlay.append((method, args, kwargs))

    def over(self, method, *args, **kwargs):
        self.paint.append((method, args, kwargs))

    def text(self, *args, **kwargs):
        self.over(self.painter.text, *args, **kwargs)

    def rule(self, y, inset=0):
        self.under(Painter.line, (self.left + inset, y, self.right - inset, y), RULE, SUPERSAMPLE)

    # each part is laid out from its top `y` and returns its bottom

    def header(self, card, y):
        s = self.s
        kicker, title, dates = (self.font('SemiBold SemiCondensed', 28), self.font('Bold Condensed', 112),
                                self.font('SemiCondensed', 30))
        top, cy = y, y + self.caps(kicker) / 2
        self.over(Painter.rounded, (self.left, cy - 3 * s, self.left + 40 * s, cy + 3 * s), 3 * s, GOLD)
        self.text(self.left + 54 * s, cy, f"NFL · {card['season']} SEASON", kicker, GOLD, tracking=4 * s)
        y += self.caps(kicker) + 26 * s
        self.text(self.left, y + self.caps(title) / 2, f"WEEK {card['week']} PICKS", title, WHITE)
        y += self.caps(title) + 28 * s
        when = '  ·  '.join(part for part in (card['dates'].upper(), 'TIMES ET') if part)
        self.text(self.left, y + self.caps(dates) / 2, when, dates, MUTED, tracking=s)
        y += self.caps(dates)
        number = self.font('Bold Condensed', 300)
        self.under(self.painter.text, self.right + 6 * s, (top + y) / 2, str(card['week']), number, WATERMARK, align='right')
        return y

    def heading(self, y, title, color, caption, note='', accent=WHITE):
        """The section's pill and caption (when its lines were taken), then how often its kind of pick hits."""
        s = self.s
        font, small = self.font('Bold SemiCondensed', 24), self.font('SemiCondensed', 24)
        height, pad, tracking = 46 * s, 20 * s, 3 * s
        width = self.painter.width(title, font, tracking) + 2 * pad
        self.over(Painter.rounded, (self.left, y, self.left + width, y + height), height / 2, color)
        self.text(self.left + pad, y + height / 2, title, font, BG_TOP if luminance(color) > 0.6 else WHITE, tracking=tracking)
        if caption:
            room = self.right - self.left - width - 24 * s
            self.text(self.right, y + height / 2, self.painter.fit(caption, small, room), small, MUTED, align='right')
        y += height + 18 * s
        if note:
            hit = self.font('SemiBold SemiCondensed', 27)
            self.text(self.left + 2 * s, y + self.caps(hit) / 2, self.painter.fit(note, hit, self.right - self.left), hit, accent)
            y += self.caps(hit) + 20 * s
        return y

    def spreads(self, picks, y, accent, card):
        """Tiles, two to a row: the team's badge, name and line, and the game."""
        s, painter = self.s, self.painter
        gap, height, diameter, pad = 16 * s, 116 * s, 68 * s, 22 * s
        column = (self.right - self.left - gap) / 2
        name_font, game_font = self.font('Bold Condensed', 46), self.font('Condensed', 27)
        for i, pick in enumerate(picks):
            x, top = self.left + (i % 2) * (column + gap), y + (i // 2) * (height + gap)
            self.under(Painter.rounded, (x, top, x + column, top + height), 22 * s, TILE)
            self.over(painter.badge, x + pad + diameter / 2, top + height / 2, diameter, pick['team'])
            tx = x + pad + diameter + 20 * s
            room = x + column - pad - tx
            name = painter.fit(nickname(pick['team']).upper(), name_font, room - painter.width(pick['line'], name_font) - 12 * s)
            self.text(tx, top + height * 0.36, name, name_font, WHITE)
            self.text(tx + painter.width(name, name_font) + 12 * s, top + height * 0.36, pick['line'], name_font, accent)
            game = ' · '.join(part for part in (matchup(pick), clock(pick['kickoff'])) if part)
            self.text(tx, top + height * 0.7, painter.fit(game, game_font, room), game_font, MUTED)
        rows = (len(picks) + 1) // 2
        return y + rows * height + (rows - 1) * gap

    def teasers(self, legs, y, accent, card):
        """Chips that wrap: the team's badge, name and teased line, and the kickoff."""
        s, painter = self.s, self.painter
        height, gap, diameter = 62 * s, 14 * s, 40 * s
        font, time_font = self.font('Bold Condensed', 34), self.font('Condensed', 27)
        x = self.left
        for leg in legs:
            name, when = nickname(leg['team']).upper(), clock(leg['kickoff'])
            width = (11 * s + diameter + 14 * s + painter.width(name, font) + 10 * s + painter.width(leg['line'], font)
                     + (14 * s + painter.width(when, time_font) if when else 0) + 24 * s)
            if x > self.left and x + width > self.right:
                x, y = self.left, y + height + gap
            self.under(Painter.rounded, (x, y, x + width, y + height), height / 2, TILE)
            self.over(painter.badge, x + 11 * s + diameter / 2, y + height / 2, diameter, leg['team'])
            tx = x + 11 * s + diameter + 14 * s
            self.text(tx, y + height / 2, name, font, WHITE)
            tx += painter.width(name, font) + 10 * s
            self.text(tx, y + height / 2, leg['line'], font, accent)
            if when:
                self.text(tx + painter.width(leg['line'], font) + 14 * s, y + height / 2, when, time_font, MUTED)
            x += width + gap
        return y + height

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
            self.over(painter.badge, self.left + pad + diameter / 2, top + height / 2, diameter, pick['team'])
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
            self.over(painter.badge, cx, top + height / 2, diameter, pick['away'])
            self.over(painter.badge, cx + diameter + gap, top + height / 2, diameter, pick['home'])
            tx = self.left + pad + 2 * diameter + gap + 22 * s
            bet = 'UNDER' + ('' if pick['total'] is None else f" {pick['total']:g}")
            room = self.right - pad - painter.width(bet, game_font) - 24 * s - tx
            game = f"{nickname(pick['away']).upper()} @ {nickname(pick['home']).upper()}"
            self.text(tx, top + height * 0.37, painter.fit(game, game_font, room), game_font, WHITE)
            self.text(self.right - pad, top + height * 0.37, bet, game_font, accent, align='right')
            why = ' · '.join(part for part in (pick['reason'], clock(pick['kickoff'])) if part)
            self.text(tx, top + height * 0.7, painter.fit(why, why_font, self.right - pad - tx), why_font, MUTED)
        return bottom

    def empty(self, y):
        s, middle, height = self.s, (self.left + self.right) / 2, 230 * self.s
        self.under(Painter.rounded, (self.left, y, self.right, y + height), 22 * s, PANEL)
        self.text(middle, y + height * 0.4, 'NO PICKS YET', self.font('Bold Condensed', 64), WHITE, align='center')
        self.text(middle, y + height * 0.66, 'Nothing qualifies so far this week. Check back closer to kickoff.',
                  self.font('SemiCondensed', 28), MUTED, align='center')
        return y + height

    def content(self, card):
        """The header and every section with picks; returns where they end."""
        y = self.header(card, 64 * self.s) + 46 * self.s
        shown = [section for section in SECTIONS if card[section[0]]]
        for i, (key, title, color, accent) in enumerate(shown):
            y = self.heading(y + (34 * self.s if i else 0), title, color, card['captions'].get(key),
                             card['notes'].get(key), accent)
            y = getattr(self, key)(card[key], y, accent, card)
        return y if shown else self.empty(y)

    def footer(self, bottom, note, handle, explain=''):
        """The small print under a rule: what the hit rates mean (when the card has any), the note and the handle."""
        s, painter = self.s, self.painter
        small, bold = self.font('SemiCondensed', 23), self.font('SemiBold SemiCondensed', 25)
        self.rule(bottom - footer_height(explain) * s)
        cy, middle, width = bottom - FOOTER_HEIGHT * s / 2, (self.left + self.right) / 2, self.right - self.left
        if explain:
            x, align = (self.left, 'left') if handle else (middle, 'center')
            self.text(x, cy - FOOTER_LINE * s, painter.fit(explain, small, width), small, MUTED, align=align)
        if handle:
            self.text(self.right, cy, handle, bold, GOLD, align='right')
            room = width - painter.width(handle, bold) - 30 * s
            self.text(self.left, cy, painter.fit(note, small, room), small, FAINT)
        elif note:
            self.text(middle, cy, painter.fit(note, small, width), small, FAINT, align='center')


def footer_height(explain):
    return FOOTER_HEIGHT + (FOOTER_LINE if explain else 0)


def layout_card(card, painter, note='', handle=''):
    """(the card's layout, its height in pixels) at the first of SHAPES the picks fit, else the last."""
    explain = KIND_NOTE if any(card[key] for key, _, _, _ in SECTIONS) else ''
    for scale, tallest in SHAPES:
        layout = Layout(painter, scale)
        bottom = layout.content(card) + (FOOTER_GAP + footer_height(explain)) * layout.s
        height = max(HEIGHT_MIN, int(math.ceil(bottom / SUPERSAMPLE)))
        if height <= tallest:
            break
    layout.footer(height * SUPERSAMPLE, note, handle, explain)
    return layout, height


def draw_card(card, note='', handle=''):
    """The card as an image, WIDTH pixels wide."""
    layout, height = layout_card(card, Painter(), note, handle)
    size = (WIDTH * SUPERSAMPLE, height * SUPERSAMPLE)
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    for method, args, kwargs in layout.overlay:
        method(draw, *args, **kwargs)
    image = background(*size)
    image.paste(WHITE, (0, 0), mask)
    draw = ImageDraw.Draw(image)
    for method, args, kwargs in layout.paint:
        method(draw, *args, **kwargs)
    return image.resize((WIDTH, height), Image.LANCZOS)


# ---------------------------------------------------------------- every pick, as text

def when(pick):
    return f", {clock(pick['kickoff'])}" if clock(pick['kickoff']) else ''


def pick_lines(card):
    """[(section title, how often its kind of pick hits, [one line per pick])]: every pick, all the props included."""
    spreads = [f"{nickname(p['team'])} {p['line']} {matchup(p)}{when(p)}"
               + (' (favorite pick)' if p['kind'] == 'favorite' else '') for p in card['spreads']]
    teasers = [f"{nickname(p['team'])} {p['line']} (from {(p['spread'] if p['team'] == p['home'] else -p['spread']):+g}) "
               f"{matchup(p)}{when(p)}" for p in card['teasers']]
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
            for key, title, lines in (('spreads', 'Spreads', spreads), ('teasers', 'Teaser legs', teasers),
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
    if kind == 'teasers':
        return f"teaser|{pick['team']}|{game}"
    if kind == 'props':
        return f"prop|{pick['player']}|{pick['prop']}|{pick['side']}|{game}"
    return f"total|{game}"


def describe(kind, pick):
    if kind in ('spreads', 'teasers'):
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
    80%+, receptions overs, and unders the model gives 80%+ (PROP_GROUPS).  The rest (about 54-57%, not enough after
    the usual juice) stay in the props workbook, as do late overs (about 54%)."""
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
    if pick['side'] == 'Over' and pick['prop'] == 'Receptions':
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
    desktop = sheet.get_desktop_path()
    log_path = log_path or sheet.share_log_path()
    now = now or pd.Timestamp.now(tz='America/New_York').tz_localize(None)
    week, kickoffs, games = week_games(season)
    spreads, legs, lines_at, dog_rate = spread_picks(os.path.join(desktop, f'{season}_nfl_projections.xlsx'), season, week,
                                                     kickoffs)
    props, props_at = prop_picks(os.path.join(desktop, f'{season}_nfl_props.xlsx'), season, week, kickoffs)
    totals, checked_at = total_picks(os.path.join(desktop, f'{season}_wind_unders.csv'), season, week)
    groups = {'spreads': spreads, 'teasers': legs, 'props': card_props(props), 'totals': totals}
    groups = {kind: sorted(picks, key=lambda p, kind=kind: card_order(kind, p)) for kind, picks in groups.items()}
    if not show_started:
        upcoming = {kind: [p for p in picks if by_kickoff(p) > now] for kind, picks in groups.items()}
        started = sum(map(len, groups.values())) - sum(map(len, upcoming.values()))
        if started:
            print(f'Left off {started} pick(s) in games already under way (SHOW_STARTED_GAMES)')
        groups = upcoming
    if stale_stats(games, week, season):
        held = sum(len(groups[kind]) for kind in ('spreads', 'teasers', 'props'))
        if held:
            print(f'Held back {held} spread, teaser and prop pick(s): last week\'s final game is not in nflverse\'s stats '
                  f'yet (they come late morning the day after), so these come on a later card')
        groups.update(spreads=[], teasers=[], props=[])
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
    folder = output_dir or os.path.join(desktop, CARD_FOLDER)
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
    spreads, legs, props, totals = (fresh[kind] for kind in ('spreads', 'teasers', 'props', 'totals'))
    days = [kickoff for kickoff, _ in kickoffs.values() if kickoff is not None and not pd.isna(kickoff)]
    checked = checked_at is not None and not pd.isna(checked_at)
    captions = {'spreads': f'lines as of {clock(lines_at)}' if lines_at else '',
                'teasers': f'{sheet.TEASER_POINTS:g}-point · 2 legs per teaser',
                'props': f'lines as of {clock(props_at)}' if props_at else '',
                'totals': f'weather forecast as of {clock(checked_at)}' if checked else ''}
    update = len(earlier) > 0
    card = {'season': season, 'week': week, 'dates': f'New picks · {day}' if update else (date_range(days) if days else ''),
            'update': update, 'day': day, 'spreads': spreads, 'teasers': legs, 'props': props, 'totals': totals,
            'captions': captions, 'notes': hit_notes(dog_rate, legs, props, totals)}
    print(f"Week {week}{', new since the last card' if update else ''}: {len(spreads)} spread pick(s), {len(legs)} teaser "
          f"leg(s), {len(props)} player prop(s), {len(totals)} game total(s)")
    os.makedirs(folder, exist_ok=True)
    image, text = draw_card(card, note, handle), pick_text(card)

    def write_text(path):
        with open(path, 'w', encoding='utf-8-sig') as f:   # the BOM lets every Windows app read the dashes
            f.write(text)
    paths = save(lambda path: image.save(path, optimize=True), base + '.png'), save(write_text, base + '.txt')
    clear_day_files(base, keep=paths)
    kept = set(log.loc[(log['season'] == str(season)) & (log['week'] == str(week)), 'key'])
    shared = {'spreads': spreads, 'teasers': legs, 'props': props, 'totals': totals}
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
    return 0


if __name__ == '__main__':
    SHOW_STARTED_GAMES = False          # True: keep picks in games already under way
    FOOTER = 'For entertainment only. Please bet responsibly.'
    HANDLE = ''                         # e.g. '@yourname', shown at the bottom right
    OUTPUT_DIR = None                   # None: the 'NFL picks' folder on the Desktop
    sys.exit(main(show_started=SHOW_STARTED_GAMES, note=FOOTER, handle=HANDLE, output_dir=OUTPUT_DIR))
