"""The NBA injury list as ESPN publishes it (site.api.espn.com), for the tracker: who is out tonight.

ESPN lists every team's injured and suspended players with a status ('Out', 'Day-To-Day', 'Suspension'), the expected
return date and a news line, updated through the day as teams report.  The player's ESPN id (the one the box scores
use) is in his profile link.  out_tonight() is the set the models treat as not playing: Out and suspended players, and
anyone ESPN's fantasy status marks OUT; Day-To-Day players are treated as playing (most do), and their status is
shown beside any pick on their team.
"""

import re

import pandas as pd
import requests

INJURIES_URL = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries'
COLUMNS = ['team_id', 'athlete_id', 'player', 'status', 'fantasy', 'return_date', 'updated', 'comment']


def _athlete_id(injury):
    athlete = injury.get('athlete') or {}
    if athlete.get('id'):
        return int(athlete['id'])
    for link in athlete.get('links', []):
        found = re.search(r'/id/(\d+)', link.get('href', ''))
        if found:
            return int(found.group(1))
    return None


def load_injuries():
    """One row per listed player (COLUMNS); empty when ESPN cannot be reached."""
    try:
        response = requests.get(INJURIES_URL, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        response.raise_for_status()
        teams = response.json().get('injuries', [])
    except (requests.RequestException, ValueError) as error:
        print(f'Could not load ESPN\'s NBA injury list ({error}); every player is treated as available')
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    for team in teams:
        for injury in team.get('injuries', []):
            details = injury.get('details') or {}
            rows.append({'team_id': int(team['id']), 'athlete_id': _athlete_id(injury),
                         'player': (injury.get('athlete') or {}).get('displayName'), 'status': injury.get('status'),
                         'fantasy': ((details.get('fantasyStatus') or {}).get('abbreviation')),
                         'return_date': details.get('returnDate'), 'updated': injury.get('date'),
                         'comment': injury.get('shortComment')})
    return pd.DataFrame(rows, columns=COLUMNS)


def out_tonight(injuries):
    """The ESPN ids of players out: listed Out or suspended, or marked OUT by ESPN's fantasy status."""
    out = injuries['status'].isin(['Out', 'Suspension']) | injuries['fantasy'].eq('OUT')
    return set(injuries.loc[out & injuries['athlete_id'].notna(), 'athlete_id'].astype(int))
