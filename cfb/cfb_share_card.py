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

The graphic is 1080 px wide, 1080 x 1350 (4:5) when the picks fit, else taller; the drawing follows the NFL card's
(nfl_share_card.py), so the two look alike, with each school's colors and abbreviation in its badge (no logos).

Run it from this folder:  python cfb_share_card.py   (the daily run, cfb_run_all.py, runs it after the picks).
"""

import math
import os
import sys
from datetime import datetime

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import cfb_card_record as card_record
import cfb_cfbd
from cfb_common import cfb_folder, current_season, pick_key, read_share_log, share_log_path

CARD_FOLDER = 'Cards'
CARD_DAYS = 7                   # the card shows games kicking off within this many days; later picks wait
# the rule's record over the discovery and test seasons, 2010-2025 (cfb_research_validate): 624-500
HIT_RATE = 624 / 1124
MIN_EDGE = 0.03                 # the price guide: the worst price at which a bet still returns 3% at HIT_RATE

WIDTH, HEIGHT_MIN, HEIGHT_FEED, HEIGHT_TALL = 1080, 1080, 1350, 1920
SHAPES = [(scale, HEIGHT_FEED) for scale in (1.0, 0.94, 0.88)] + \
         [(scale, HEIGHT_TALL) for scale in (1.0, 0.94, 0.88, 0.82, 0.76, 0.7)]
SUPERSAMPLE = 2
MARGIN = 64
FOOTER_HEIGHT, FOOTER_LINE, FOOTER_GAP = 92, 34, 36
FONT_DIR = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
BAHNSCHRIFT = os.path.join(FONT_DIR, 'bahnschrift.ttf')
FALLBACK_FONTS = {'Bold': 'segoeuib.ttf', 'SemiBold': 'seguisb.ttf', 'Regular': 'segoeui.ttf', 'Light': 'segoeuil.ttf'}
BG_TOP, BG_BOTTOM = (7, 17, 38), (14, 34, 70)
GOLD, WHITE, MUTED, FAINT = (255, 196, 37), (255, 255, 255), (156, 174, 203), (112, 131, 163)
BADGE_EDGE = (64, 84, 120)
TILE, RULE, WATERMARK = 22, 26, 18
SPREADS_COLOR, SPREADS_ACCENT = (47, 112, 235), (133, 180, 255)
KIND_NOTE = 'The hit rate is for every pick of this kind since 2010, not for any one pick.'


# ---------------------------------------------------------------- small helpers

def clock(moment):
    """'Sat 3:30 PM' (Eastern), or '' when unknown."""
    if moment is None or pd.isna(moment):
        return ''
    return f'{moment:%a %I:%M %p}'.replace(' 0', ' ', 1)


def date_range(days):
    """'Nov 7–14' or 'Oct 31 – Nov 7'."""
    first, last = min(days), max(days)
    if first.month != last.month:
        return f'{first:%b} {first.day} – {last:%b} {last.day}'
    return f'{first:%b} {first.day}' + (f'–{last.day}' if last.day != first.day else '')


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
                      'colors': teams.get(team_id, (None, None, '#475569', '#94A3B8'))[2:]})
    picks.sort(key=lambda p: (p['kickoff'], p['team'].casefold()))
    return picks, held, file_time(path)


def describe(pick):
    return f"{pick['team']} {pick['line']}"


# ---------------------------------------------------------------- drawing (as nfl_share_card.py)

def hex_rgb(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def luminance(rgb):
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255


class Painter:
    """Fonts (Bahnschrift's styles, or Segoe UI where it is missing) and the shapes the card is made of."""

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
        if self.width(text, font) <= room:
            return text
        while text and self.width(text + '…', font) > room:
            text = text[:-1]
        return text.rstrip() + '…'

    def text(self, draw, x, cy, text, font, fill, align='left', tracking=0):
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

    def badge(self, draw, cx, cy, diameter, abbr, colors):
        """A school's colors as a round badge with its abbreviation (no logos)."""
        primary, secondary = (hex_rgb(c) for c in colors)
        r, edge, ring = diameter / 2, max(2.0, diameter * 0.04), diameter * 0.085
        draw.ellipse((cx - r - edge, cy - r - edge, cx + r + edge, cy + r + edge), fill=BADGE_EDGE)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=secondary)
        draw.ellipse((cx - r + ring, cy - r + ring, cx + r - ring, cy + r - ring), fill=primary)
        size = 0.4 if len(abbr) <= 2 else 0.34 if len(abbr) == 3 else 0.27
        font = self.font('Bold SemiCondensed', diameter * size)
        self.text(draw, cx, cy, abbr, font, (17, 24, 39) if luminance(primary) > 0.55 else WHITE, align='center')


def background(width, height):
    column = Image.new('RGB', (1, 256))
    column.putdata([tuple(int(round(a + (b - a) * i / 255)) for a, b in zip(BG_TOP, BG_BOTTOM)) for i in range(256)])
    return column.resize((width, height), Image.BILINEAR)


class Layout:
    """The card at one type scale, recorded rather than drawn so its height is known before the canvas is made."""

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

    def header(self, card, y):
        s = self.s
        kicker, title, dates = (self.font('SemiBold SemiCondensed', 28), self.font('Bold Condensed', 112),
                                self.font('SemiCondensed', 30))
        top, cy = y, y + self.caps(kicker) / 2
        self.over(Painter.rounded, (self.left, cy - 3 * s, self.left + 40 * s, cy + 3 * s), 3 * s, GOLD)
        self.text(self.left + 54 * s, cy, f"COLLEGE FOOTBALL · {card['season']} SEASON", kicker, GOLD, tracking=4 * s)
        y += self.caps(kicker) + 26 * s
        self.text(self.left, y + self.caps(title) / 2, f"WEEK {card['week']} PICKS", title, WHITE)
        y += self.caps(title) + 28 * s
        when = '  ·  '.join(part for part in (card['dates'].upper(), 'TIMES ET') if part)
        self.text(self.left, y + self.caps(dates) / 2, when, dates, MUTED, tracking=s)
        y += self.caps(dates)
        number = self.font('Bold Condensed', 300)
        self.under(self.painter.text, self.right + 6 * s, (top + y) / 2, str(card['week']), number, WATERMARK,
                   align='right')
        return y

    def heading(self, y, title, caption, note):
        s = self.s
        font, small = self.font('Bold SemiCondensed', 24), self.font('SemiCondensed', 24)
        height, pad, tracking = 46 * s, 20 * s, 3 * s
        width = self.painter.width(title, font, tracking) + 2 * pad
        self.over(Painter.rounded, (self.left, y, self.left + width, y + height), height / 2, SPREADS_COLOR)
        self.text(self.left + pad, y + height / 2, title, font, WHITE, tracking=tracking)
        if caption:
            room = self.right - self.left - width - 24 * s
            self.text(self.right, y + height / 2, self.painter.fit(caption, small, room), small, MUTED, align='right')
        y += height + 18 * s
        if note:
            hit = self.font('SemiBold SemiCondensed', 27)
            self.text(self.left + 2 * s, y + self.caps(hit) / 2, self.painter.fit(note, hit, self.right - self.left), hit,
                      SPREADS_ACCENT)
            y += self.caps(hit) + 20 * s
        return y

    def spreads(self, picks, y):
        """Tiles, two to a row: the school's badge, name and line, and the game."""
        s, painter = self.s, self.painter
        gap, height, diameter, pad = 16 * s, 116 * s, 68 * s, 22 * s
        column = (self.right - self.left - gap) / 2
        name_font, game_font = self.font('Bold Condensed', 46), self.font('Condensed', 27)
        for i, pick in enumerate(picks):
            x, top = self.left + (i % 2) * (column + gap), y + (i // 2) * (height + gap)
            self.under(Painter.rounded, (x, top, x + column, top + height), 22 * s, TILE)
            self.over(painter.badge, x + pad + diameter / 2, top + height / 2, diameter, pick['abbr'], pick['colors'])
            tx = x + pad + diameter + 20 * s
            room = x + column - pad - tx
            name = painter.fit(pick['team'].upper(), name_font, room - painter.width(pick['line'], name_font) - 12 * s)
            self.text(tx, top + height * 0.36, name, name_font, WHITE)
            self.text(tx + painter.width(name, name_font) + 12 * s, top + height * 0.36, pick['line'], name_font,
                      SPREADS_ACCENT)
            game = ' · '.join(part for part in (matchup(pick), clock(pick['kickoff'])) if part)
            self.text(tx, top + height * 0.7, painter.fit(game, game_font, room), game_font, MUTED)
        rows = (len(picks) + 1) // 2
        return y + rows * height + (rows - 1) * gap

    def content(self, card):
        y = self.header(card, 64 * self.s) + 46 * self.s
        y = self.heading(y, 'SPREADS', card['caption'], card['note'])
        return self.spreads(card['picks'], y)

    def footer(self, bottom, note, handle):
        s, painter = self.s, self.painter
        small, bold = self.font('SemiCondensed', 23), self.font('SemiBold SemiCondensed', 25)
        self.rule(bottom - (FOOTER_HEIGHT + FOOTER_LINE) * s)
        cy, middle, width = bottom - FOOTER_HEIGHT * s / 2, (self.left + self.right) / 2, self.right - self.left
        x, align = (self.left, 'left') if handle else (middle, 'center')
        self.text(x, cy - FOOTER_LINE * s, painter.fit(KIND_NOTE, small, width), small, MUTED, align=align)
        if handle:
            self.text(self.right, cy, handle, bold, GOLD, align='right')
            room = width - painter.width(handle, bold) - 30 * s
            self.text(self.left, cy, painter.fit(note, small, room), small, FAINT)
        elif note:
            self.text(middle, cy, painter.fit(note, small, width), small, FAINT, align='center')


def draw_card(card, note='', handle=''):
    """The card as an image, WIDTH pixels wide, at the first of SHAPES the picks fit."""
    painter = Painter()
    for scale, tallest in SHAPES:
        layout = Layout(painter, scale)
        bottom = layout.content(card) + (FOOTER_GAP + FOOTER_HEIGHT + FOOTER_LINE) * layout.s
        height = max(HEIGHT_MIN, int(math.ceil(bottom / SUPERSAMPLE)))
        if height <= tallest:
            break
    layout.footer(height * SUPERSAMPLE, note, handle)
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
