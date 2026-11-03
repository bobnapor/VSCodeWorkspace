"""Paths and season numbering shared by the NBA scripts (nothing here is shared with the NFL code).

What the tracker (nba_prod_games.py) writes goes in the NBA folder on the Desktop (nba_folder), which OneDrive syncs;
the NBA_FOLDER environment variable points it elsewhere (tests use a temporary folder).  The download cache
(data_cache, beside this file) rebuilds itself.

Seasons are numbered as ESPN numbers them, by the year they end: 2027 is the 2026-27 season (season_label).
"""

import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'data_cache')
REGULAR_SEASON = 2          # ESPN's season_type: 1 preseason, 2 regular season, 3 playoffs, 5 play-in


def get_desktop_path():
    """The Desktop as Windows knows it (OneDrive's Desktop when it is backed up there)."""
    try:
        import ctypes
        from ctypes import wintypes
        buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
        ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buf)   # CSIDL_DESKTOPDIRECTORY
        if buf.value:
            return buf.value
    except (AttributeError, OSError):
        pass
    return os.path.join(os.path.expanduser('~'), 'Desktop')


def nba_folder(*sub):
    """<Desktop>\\NBA (made if missing), or NBA_FOLDER, and its subfolders."""
    path = os.path.join(os.environ.get('NBA_FOLDER') or os.path.join(get_desktop_path(), 'NBA'), *sub)
    os.makedirs(path, exist_ok=True)
    return path


def current_season(today=None):
    """The season being played or next up: 2027 from July 2026 to June 2027."""
    today = today or date.today()
    return today.year + 1 if today.month >= 7 else today.year


def season_label(season):
    """'2026-27' for 2027."""
    return f'{int(season) - 1}-{str(int(season))[-2:]}'
