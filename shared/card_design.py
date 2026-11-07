"""The share cards' common design: the only code the NFL and college football suites share (nfl_share_card.py and
cfb_share_card.py import it; nothing else crosses between them).  It is the look and nothing else: the card's size and
shapes, fonts, colors, the gradient, team badges, the header, section headings, pick tiles, the empty card, the small
print, and rendering at the first type scale the picks fit.  What goes on a card, and every sport's data, stays in its
own suite.

The card is 1080 px wide: 1080 x 1350 (4:5, the tallest Instagram feed post) when the picks fit, shrinking the type a
little first; otherwise taller (up to 9:16).  It is drawn at twice the size and scaled down, for smooth edges.  A layout
is recorded rather than drawn, so its height is known before the canvas is made: `overlay` holds the see-through white
shapes (drawn on a mask, an opacity as the fill) and `paint` what goes on top.

The suites import it by adding this folder to sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
"""

import math
import os

from PIL import Image, ImageDraw, ImageFont

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
DEFAULT_COLORS = ('#475569', '#94A3B8')    # a badge for a team without colors
TILE, PANEL, RULE, WATERMARK = 22, 16, 26, 18   # white overlays, by opacity (0-255)
# each kind of section's (pill color, light text color): spreads blue, props yellow, game totals orange
SECTION_COLORS = {'spreads': ((47, 112, 235), (133, 180, 255)), 'props': ((234, 179, 8), (253, 224, 120)),
                  'totals': ((234, 88, 12), (253, 186, 116))}


# ---------------------------------------------------------------- text as the cards show it

def clock(moment):
    """'Sun 1:00 PM' (Eastern), or '' when unknown."""
    if moment is None or moment != moment:
        return ''
    return f'{moment:%a %I:%M %p}'.replace(' 0', ' ', 1)


def date_range(days):
    """'Oct 22–26' or 'Oct 29 – Nov 2'."""
    first, last = min(days), max(days)
    if first.month != last.month:
        return f'{first:%b} {first.day} – {last:%b} {last.day}'
    return f'{first:%b} {first.day}' + (f'–{last.day}' if last.day != first.day else '')


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

    def badge(self, draw, cx, cy, diameter, text, colors):
        """A team's colors, (primary, secondary) as hex, as a round badge with its abbreviation (no logos)."""
        primary, secondary = (hex_rgb(c) for c in colors)
        r, edge, ring = diameter / 2, max(2.0, diameter * 0.04), diameter * 0.085
        draw.ellipse((cx - r - edge, cy - r - edge, cx + r + edge, cy + r + edge), fill=BADGE_EDGE)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=secondary)
        draw.ellipse((cx - r + ring, cy - r + ring, cx + r - ring, cy + r - ring), fill=primary)
        size = 0.4 if len(text) <= 2 else 0.34 if len(text) == 3 else 0.27
        font = self.font('Bold SemiCondensed', diameter * size)
        self.text(draw, cx, cy, text, font, (17, 24, 39) if luminance(primary) > 0.55 else WHITE, align='center')


def background(width, height):
    """The card's vertical gradient."""
    column = Image.new('RGB', (1, 256))
    column.putdata([tuple(int(round(a + (b - a) * i / 255)) for a, b in zip(BG_TOP, BG_BOTTOM)) for i in range(256)])
    return column.resize((width, height), Image.BILINEAR)


class Layout:
    """The card at one type scale.  Lengths are in supersampled pixels; `s` turns a full-size length into them.  Each
    part is laid out from its top `y` and returns its bottom.  A suite subclasses it for its own sections."""

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

    def header(self, league, season, week, dates, y):
        """'<league> · <season> SEASON', 'WEEK <week> PICKS', the dates, and the week's number as a watermark."""
        s = self.s
        kicker, title, small = (self.font('SemiBold SemiCondensed', 28), self.font('Bold Condensed', 112),
                                self.font('SemiCondensed', 30))
        top, cy = y, y + self.caps(kicker) / 2
        self.over(Painter.rounded, (self.left, cy - 3 * s, self.left + 40 * s, cy + 3 * s), 3 * s, GOLD)
        self.text(self.left + 54 * s, cy, f'{league} · {season} SEASON', kicker, GOLD, tracking=4 * s)
        y += self.caps(kicker) + 26 * s
        self.text(self.left, y + self.caps(title) / 2, f'WEEK {week} PICKS', title, WHITE)
        y += self.caps(title) + 28 * s
        when = '  ·  '.join(part for part in (dates.upper(), 'TIMES ET') if part)
        self.text(self.left, y + self.caps(small) / 2, when, small, MUTED, tracking=s)
        y += self.caps(small)
        number = self.font('Bold Condensed', 300)
        self.under(self.painter.text, self.right + 6 * s, (top + y) / 2, str(week), number, WATERMARK, align='right')
        return y

    def heading(self, y, title, color, caption, note='', accent=WHITE):
        """The section's pill and caption (when its lines were taken), then how often its kind of pick hits."""
        s = self.s
        font, small = self.font('Bold SemiCondensed', 24), self.font('SemiCondensed', 24)
        height, pad, tracking = 46 * s, 20 * s, 3 * s
        width = self.painter.width(title, font, tracking) + 2 * pad
        self.over(Painter.rounded, (self.left, y, self.left + width, y + height), height / 2, color)
        self.text(self.left + pad, y + height / 2, title, font, BG_TOP if luminance(color) > 0.6 else WHITE,
                  tracking=tracking)
        if caption:
            room = self.right - self.left - width - 24 * s
            self.text(self.right, y + height / 2, self.painter.fit(caption, small, room), small, MUTED, align='right')
        y += height + 18 * s
        if note:
            hit = self.font('SemiBold SemiCondensed', 27)
            self.text(self.left + 2 * s, y + self.caps(hit) / 2, self.painter.fit(note, hit, self.right - self.left), hit,
                      accent)
            y += self.caps(hit) + 20 * s
        return y

    def tiles(self, tiles, y, accent):
        """Tiles, two to a row: a badge, a name and its line, and a line below (the game).  Each tile is a dict of
        badge ((text, colors)), name, line and sub."""
        s, painter = self.s, self.painter
        gap, height, diameter, pad = 16 * s, 116 * s, 68 * s, 22 * s
        column = (self.right - self.left - gap) / 2
        name_font, sub_font = self.font('Bold Condensed', 46), self.font('Condensed', 27)
        for i, tile in enumerate(tiles):
            x, top = self.left + (i % 2) * (column + gap), y + (i // 2) * (height + gap)
            self.under(Painter.rounded, (x, top, x + column, top + height), 22 * s, TILE)
            self.over(painter.badge, x + pad + diameter / 2, top + height / 2, diameter, *tile['badge'])
            tx = x + pad + diameter + 20 * s
            room = x + column - pad - tx
            name = painter.fit(tile['name'], name_font, room - painter.width(tile['line'], name_font) - 12 * s)
            self.text(tx, top + height * 0.36, name, name_font, WHITE)
            self.text(tx + painter.width(name, name_font) + 12 * s, top + height * 0.36, tile['line'], name_font, accent)
            self.text(tx, top + height * 0.7, painter.fit(tile['sub'], sub_font, room), sub_font, MUTED)
        rows = (len(tiles) + 1) // 2
        return y + rows * height + (rows - 1) * gap

    def empty(self, y):
        s, middle, height = self.s, (self.left + self.right) / 2, 230 * self.s
        self.under(Painter.rounded, (self.left, y, self.right, y + height), 22 * s, PANEL)
        self.text(middle, y + height * 0.4, 'NO PICKS YET', self.font('Bold Condensed', 64), WHITE, align='center')
        self.text(middle, y + height * 0.66, 'Nothing qualifies so far this week. Check back closer to kickoff.',
                  self.font('SemiCondensed', 28), MUTED, align='center')
        return y + height

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


def render(layout_class, content, note='', handle='', explain=''):
    """The card as an image, WIDTH pixels wide: `content(layout)` lays out everything above the small print on a
    `layout_class` (a Layout) and returns its bottom; the first of SHAPES it fits in is used, else the last."""
    painter = Painter()
    for scale, tallest in SHAPES:
        layout = layout_class(painter, scale)
        bottom = content(layout) + (FOOTER_GAP + footer_height(explain)) * layout.s
        height = max(HEIGHT_MIN, int(math.ceil(bottom / SUPERSAMPLE)))
        if height <= tallest:
            break
    layout.footer(height * SUPERSAMPLE, note, handle, explain)
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
