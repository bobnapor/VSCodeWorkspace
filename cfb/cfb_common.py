"""Paths and season numbering shared by the college football scripts (nothing here is shared with the NFL or NBA code).

The download cache (data_cache, beside this file, gitignored) rebuilds itself.  What a tracker would write goes in the
CFB folder on the Desktop (cfb_folder), which OneDrive syncs; the CFB_FOLDER environment variable points it elsewhere.

Seasons are numbered by the year they start, as ESPN and sportsdataverse number college football: 2025 is the fall 2025
season, its bowls and playoff included (they are played in January 2026).
"""

import os
from datetime import date

import pandas as pd

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


# ---------------------------------------------------------------- what the share cards showed

SHARE_LOG_COLUMNS = ['season', 'week', 'key', 'pick', 'shown_on']


def share_log_path():
    """CFB\\data\\share_card_log.csv: every pick a share card has shown, and the day it did."""
    return os.path.join(cfb_folder('data'), 'share_card_log.csv')


def read_share_log(path=None):
    path = path or share_log_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=SHARE_LOG_COLUMNS)
    return pd.read_csv(path, dtype=str).reindex(columns=SHARE_LOG_COLUMNS)


def pick_key(team_id, game_id):
    """A spread pick's identity in the share log, without its line: 'spread|<team id>|<game id>' (ESPN ids)."""
    return f'spread|{int(team_id)}|{int(game_id)}'


def key_game(key):
    """The ESPN game id of a share-log key."""
    return int(str(key).rsplit('|', 1)[-1])


def shared_games(season, kickoffs, now, path=None):
    """Game ids of `season` with a pick a share card showed on an earlier day than today, or today in a game that has
    kicked off (`kickoffs`: {game id: kickoff, Eastern}).  A day's last run sets what its card shared, and from then on
    the picks run keeps those picks as they were shared, so they are graded at the line that went out."""
    log = read_share_log(path)
    log = log[log['season'] == str(season)]
    today = now.strftime('%Y-%m-%d')
    games = set()
    for key, day in zip(log['key'], log['shown_on']):
        game = key_game(key)
        kickoff = kickoffs.get(game)
        if day < today or (kickoff is not None and kickoff <= now):
            games.add(game)
    return games
