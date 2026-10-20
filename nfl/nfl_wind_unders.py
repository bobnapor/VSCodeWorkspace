"""Game-day wind check for NFL totals: flag the under in windy outdoor games.

Evidence (nfl_research_totals.py, October 2026): outdoor games with 15+ mph of wind went under 57% of the time at the
closing total in every era since 1999 (499 games, weeks 5-16), because the market lowers the total for wind by only
about half of what scoring actually falls.  On archived weather-model forecasts since 2018 the under went 56% at 15+ mph
(50 games) and 60% at 10-15 mph (169 games).  On recorded wind, weeks 17-18 went 65% and the playoffs 71%; weeks 1-4 did
not hold (48%), so they get no flag.

For one week (default: the next week with unplayed games):
1. Every home game at an open-air stadium (the schedule's roof is 'outdoors'; domes, retractable roofs, which are
   usually closed in bad weather, and neutral-site games are skipped).
2. The hourly wind forecast at the stadium from Open-Meteo (free, no key): wind at 10 m in mph averaged over the
   kickoff hour and the three after, the measure the research tested.
3. DraftKings' current total and prices from Action Network (other books if DraftKings has none).
4. Flags, week 5 on: UNDER at 15+ mph; WATCH at 10-15 mph (60% on forecasts since 2018, spotted rather than
   pre-registered, so it is tracked separately).
5. A season log, <Desktop>/<season>_wind_unders.csv: one row per outdoor game with its latest check before kickoff
   (forecast, total, flag), and the final total and the under's result once it is played, so the live record builds up.

Forecasts more than a day or two out are rough: run it again close to kickoff (Thursday for Thursday games, Saturday
night or Sunday morning for Sunday games) and bet the line you can get.  The edge was measured at the closing total,
which already includes the forecast, so betting near kickoff is enough.

Run it from this folder:  python nfl_wind_unders.py            (the next week with unplayed games)
                          python nfl_wind_unders.py 2026 8     (a given season and week)
"""

import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

import nfl_model as nm
import nfl_sheet as sheet
from nfl_injury_features import TEAM_FULL_NAMES
from nfl_lab_extra import TEAM_SITES
from nflverse_data import load_games

FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'
GAME_HOURS = 4                 # the kickoff hour and the three after
UNDER_MPH, WATCH_MPH = 15.0, 10.0
FIRST_FLAG_WEEK = 5
FORECAST_DAYS = 16             # how far ahead Open-Meteo forecasts
LOG_COLUMNS = ['season', 'week', 'game_id', 'kickoff_et', 'away', 'home', 'stadium', 'wind_mph', 'checked_at',
               'hours_before_kickoff', 'total', 'under_odds', 'over_odds', 'book', 'flag', 'final_total', 'result']
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}


def now_eastern():
    return pd.Timestamp.now(tz='America/New_York').tz_localize(None)


def get_json(url, params, attempts=3):
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def game_wind(latitude, longitude, kickoff, url=FORECAST_URL):
    """Forecast wind (mph, 10 m) averaged over the kickoff hour and the three after; kickoff is naive Eastern time."""
    start = kickoff.floor('H')
    hourly = get_json(url, {'latitude': latitude, 'longitude': longitude, 'hourly': 'wind_speed_10m', 'wind_speed_unit': 'mph',
                            'timezone': 'America/New_York', 'start_date': start.date().isoformat(),
                            'end_date': (start + timedelta(days=1)).date().isoformat()})['hourly']
    series = pd.Series(hourly['wind_speed_10m'], index=pd.to_datetime(hourly['time']), dtype=float)
    return float(series.reindex(pd.date_range(start, periods=GAME_HOURS, freq='H')).mean())


def fetch_totals(season, week):
    """{(home full name, away full name): (total, under odds, over odds, book)} from Action Network's pre-game lines."""
    books = sheet.SPREAD_BOOK_PREFERENCE
    try:
        games = get_json(sheet.ACTION_NETWORK_SCOREBOARD_URL, {'bookIds': ','.join(str(b) for b, _ in books), 'season': season,
                                                               'week': week}).get('games', [])
    except (requests.RequestException, ValueError) as e:
        print(f'Could not fetch totals for {season} week {week}: {e}')
        return {}
    totals = {}
    for game in games:
        names = {team['id']: team['full_name'] for team in game.get('teams', [])}
        home, away = names.get(game.get('home_team_id')), names.get(game.get('away_team_id'))
        for book_id, book_name in books:
            lines = [line for line in game.get('markets', {}).get(str(book_id), {}).get('event', {}).get('total', []) if not line.get('is_live')]
            under = next((line for line in lines if line.get('side') == 'under'), None)
            over = next((line for line in lines if line.get('side') == 'over'), None)
            if under is not None:
                totals[(home, away)] = (float(under['value']), under.get('odds'), over.get('odds') if over else None, book_name)
                break
    return totals


def flag(wind, week, game_type):
    if np.isnan(wind) or (game_type == 'REG' and week < FIRST_FLAG_WEEK):
        return ''
    return 'UNDER' if wind >= UNDER_MPH else 'WATCH' if wind >= WATCH_MPH else ''


def next_week(schedule, season):
    """The earliest week of `season` with an unplayed game, else its last week."""
    games = schedule[schedule['season'] == season]
    unplayed = games[games['result'].isna()]
    return int(unplayed['week'].min()) if not unplayed.empty else int(games['week'].max())


def update_results(log, schedule):
    """Fill the final total and the under's result for logged games that have been played."""
    finals = schedule.set_index('game_id')['total']
    for index, row in log.iterrows():
        final = finals.get(row['game_id'])
        if pd.isna(final) or pd.isna(row['total']):
            continue
        log.at[index, 'final_total'] = final
        log.at[index, 'result'] = 'Win' if final < row['total'] else 'Loss' if final > row['total'] else 'Push'
    return log


def record(log, label):
    graded = log[(log['flag'] == label) & log['result'].isin(['Win', 'Loss', 'Push'])]
    wins, losses, pushes = (int((graded['result'] == r).sum()) for r in ('Win', 'Loss', 'Push'))
    rate = '%.1f%%' % (100 * wins / (wins + losses)) if wins + losses else '-'
    return '%d-%d-%d (%s)' % (wins, losses, pushes, rate)


def main(season=None, week=None, output_dir=None):
    season = season or sheet.get_current_season_year()
    schedule = nm.normalize_teams(load_games(refresh=True))
    week = week or next_week(schedule, season)
    now = now_eastern()
    games = schedule[(schedule['season'] == season) & (schedule['week'] == week)].sort_values(['gameday', 'gametime'])
    if games.empty:
        print(f'No games in {season} week {week}')
        return
    totals = fetch_totals(season, week)
    path = os.path.join(output_dir or sheet.get_desktop_path(), f'{season}_wind_unders.csv')
    log = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=LOG_COLUMNS)

    print(f'Wind check, {season} week {week} (run {now:%a %b %d %I:%M %p} ET). UNDER at {UNDER_MPH:g}+ mph, WATCH at '
          f'{WATCH_MPH:g}-{UNDER_MPH:g} mph, from week {FIRST_FLAG_WEEK}; wind = forecast at 10 m averaged over the game.\n')
    print('%-19s %-30s %8s %8s %9s %-6s %s' % ('Kickoff (ET)', 'Game', 'Wind', 'Total', 'Under', 'Flag', 'Notes'))
    skipped = []
    for g in games.itertuples():
        home, away = TEAM_FULL_NAMES.get(g.home_team, g.home_team), TEAM_FULL_NAMES.get(g.away_team, g.away_team)
        label = f'{away.split()[-1]} at {home.split()[-1]}'
        if g.location != 'Home' or g.roof != 'outdoors':
            skipped.append(f'{label} ({"neutral site" if g.location != "Home" else g.roof if isinstance(g.roof, str) else "retractable roof"})')
            continue
        if not isinstance(g.gametime, str):
            skipped.append(f'{label} (kickoff time not set yet)')
            continue
        kickoff = pd.Timestamp(f'{g.gameday} {g.gametime}')
        hours = (kickoff - now).total_seconds() / 3600
        line = totals.get((home, away))
        if hours <= 0:
            note = 'kicked off'
            logged = log[log['game_id'] == g.game_id]
            wind = float(logged['wind_mph'].iloc[0]) if len(logged) else float('nan')
            game_flag = logged['flag'].iloc[0] if len(logged) and isinstance(logged['flag'].iloc[0], str) else ''
        elif hours > 24 * FORECAST_DAYS:
            wind, game_flag, note = float('nan'), '', 'too far out to forecast'
        else:
            try:
                wind = game_wind(*TEAM_SITES[g.home_team][:2], kickoff)
            except (requests.RequestException, ValueError, KeyError) as e:
                wind, note = float('nan'), f'no forecast ({e})'
            else:
                note = f'{hours / 24:.1f} days out, recheck near kickoff' if hours > 36 else ''
            game_flag = flag(wind, week, g.game_type)
            if g.game_type == 'REG' and week < FIRST_FLAG_WEEK:
                note = (note + '; ' if note else '') + 'weeks 1-4 untested-to-negative, no flag'
            row = {'season': season, 'week': week, 'game_id': g.game_id, 'kickoff_et': kickoff.strftime('%Y-%m-%d %H:%M'), 'away': away,
                   'home': home, 'stadium': g.stadium, 'wind_mph': round(wind, 1) if not np.isnan(wind) else None,
                   'checked_at': now.strftime('%Y-%m-%d %H:%M'), 'hours_before_kickoff': round(hours, 1),
                   'total': line[0] if line else None, 'under_odds': line[1] if line else None, 'over_odds': line[2] if line else None,
                   'book': line[3] if line else None, 'flag': game_flag, 'final_total': None, 'result': None}
            others = log[log['game_id'] != g.game_id]
            log = pd.concat([others, pd.DataFrame([row])], ignore_index=True) if len(others) else pd.DataFrame([row], columns=LOG_COLUMNS)
        wind_text = '%.1f mph' % wind if not np.isnan(wind) else '-'
        total_text = '%g' % line[0] if line else '-'
        under_text = '%+d' % line[1] if line and line[1] is not None else '-'
        print('%-19s %-30s %8s %8s %9s %-6s %s' % (kickoff.strftime('%a %b %d %I:%M%p'), label, wind_text, total_text, under_text,
                                                  game_flag, note))
    if skipped:
        print('\nnot checked (no wind exposure or no stadium location): ' + '; '.join(skipped))

    log = update_results(log, schedule)
    log = log[LOG_COLUMNS].sort_values(['season', 'week', 'kickoff_et'])
    try:
        log.to_csv(path, index=False)
        print(f'\nSeason log: {path}')
    except PermissionError:
        print(f'\nCould not save {path} (is it open in Excel?); close it and run again')
    print(f'Season record: UNDER flags {record(log, "UNDER")}, WATCH flags {record(log, "WATCH")} '
          f'(the latest check before kickoff counts; pushes refunded; break-even 52.4% at -110)')


if __name__ == '__main__':
    arguments = [int(a) for a in sys.argv[1:3]]
    main(*arguments)
