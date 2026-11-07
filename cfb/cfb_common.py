"""Paths and season numbering shared by the college football scripts (nothing here is shared with the NFL or NBA code).

The download cache (data_cache, beside this file, gitignored) rebuilds itself.  What a tracker would write goes in the
CFB folder on the Desktop (cfb_folder), which OneDrive syncs; the CFB_FOLDER environment variable points it elsewhere.

Seasons are numbered by the year they start, as ESPN and sportsdataverse number college football: 2025 is the fall 2025
season, its bowls and playoff included (they are played in January 2026).
"""

import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'data_cache')
CFBD_KEY_FILE = os.path.join(HERE, 'cfbd_key.txt')     # gitignored; holds only the CollegeFootballData API key
REGULAR_SEASON, POSTSEASON = 2, 3       # ESPN's season_type
FBS = 'cfb:fbs'                         # the subdivision id in sportsdataverse's cfb_groups


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


def cfb_folder(*sub):
    """<Desktop>\\CFB (made if missing), or CFB_FOLDER, and its subfolders."""
    path = os.path.join(os.environ.get('CFB_FOLDER') or os.path.join(get_desktop_path(), 'CFB'), *sub)
    os.makedirs(path, exist_ok=True)
    return path


def cfbd_key():
    """The CollegeFootballData API key: the text of cfbd_key.txt beside this file, else CFBD_API_KEY."""
    if os.path.exists(CFBD_KEY_FILE):
        with open(CFBD_KEY_FILE, encoding='utf-8') as f:
            key = f.read().strip()
        if key:
            return key
    key = os.environ.get('CFBD_API_KEY', '').strip()
    if not key:
        raise RuntimeError(f'No CollegeFootballData API key: put it in {CFBD_KEY_FILE}')
    return key


def current_season(today=None):
    """The season being played or next up: 2026 from July 2026 to June 2027."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1
