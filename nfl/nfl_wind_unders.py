"""Game-day weather check for NFL totals: flag the under in windy or rainy outdoor games.

Evidence (nfl_research_totals.py, October 2026): outdoor games with 15+ mph of wind went under 57% of the time at the
closing total in every era since 1999 (499 games, weeks 5-16), because the market lowers the total for wind by only
about half of what scoring actually falls.  On archived weather-model forecasts since 2018 the under went 53% at 15+ mph
(25-22) and 60% at 10-15 mph (96-63).  On recorded wind, weeks 17-18 went 65% and the playoffs 71%; weeks 1-4 did
not hold (48%; 1-4 at 15+ mph and 26-29 at 10-15 on forecasts), so they get no flag.
Rain (nfl_research_totals2.py, pre-registered): outdoor games with 0.5+ mm of rain over the game that the wind rule does
not flag went under 57.7% (184-135; 57.8% / 54.7% / 61.7% across 1999-2005 / 2006-2017 / 2018-2026), and 61% on
archived forecasts since 2018 (35-22); the market lowers the total by about a fifth of what scoring falls.  Found after
that test (so tracked rather than trusted): the edge is rain with wind of 9+ mph on the forecast's scale (about 66%,
137-70 since 1999), while calm rain went about 51%.

For one week (default: the next week with unplayed games):
1. Every home game at an open-air stadium (the schedule's roof is 'outdoors'; domes, retractable roofs, which are
   usually closed in bad weather, and neutral-site games are skipped).
2. The hourly forecast at the stadium from Open-Meteo (free, no key): wind at 10 m in mph averaged over the kickoff hour
   and the three after, and rain (mm) and snow (cm) over the four hours from kickoff, the measures the research tested.
3. DraftKings' current total and prices from Action Network (other books if DraftKings has none).
4. Flags, week 5 on: UNDER at 15+ mph; WATCH at 10-15 mph (60% on forecasts since 2018, spotted rather than
   pre-registered, so it is tracked separately).  RAIN is a separate flag with its own record: 0.5+ mm of rain over the
   game, no snow, in a game the wind rule does not flag UNDER (a game can be WATCH and RAIN); RAIN games with 9+ mph of
   wind are noted as the strongest spot and get a record line of their own.
5. A season log, <Desktop>/NFL/<season>_wind_unders.csv: one row per outdoor game with its latest check before kickoff
   (forecast wind and rain, total, flags), and the final total and the under's result once it is played, so the live
   record builds up.  A game whose under went out on an earlier day's share card (nfl_share_card.py) keeps the check it
   was shared with, so it is graded at the total that went out.

Forecasts more than a day or two out are rough, rain timing most of all, so the daily run checks every day and the
share card shows a flag only from a check within 36 hours of kickoff; bet the line you can get.  The edge was measured
at the closing total, which already includes the forecast, so betting near kickoff is enough; in windy games the total
fell about a point from open to close (2007-2021), so once the forecast is settled, earlier is better.

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
RAIN_MM, SNOW_CM = 0.5, 0.1    # rain over the game window (mm) and the snow (cm) that makes it a snow game instead
RAIN_WIND_MPH = 9.0            # rain with at least this much wind: the strongest spot (found after the test)
FIRST_FLAG_WEEK = 5
FORECAST_DAYS = 16             # how far ahead Open-Meteo forecasts
LOG_COLUMNS = ['season', 'week', 'game_id', 'kickoff_et', 'away', 'home', 'stadium', 'wind_mph', 'rain_mm', 'snow_cm',
               'checked_at', 'hours_before_kickoff', 'total', 'under_odds', 'over_odds', 'book', 'flag', 'rain_flag',
               'final_total', 'result']
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


def game_weather(latitude, longitude, kickoff, url=FORECAST_URL):
    """Forecast (wind mph, rain mm, snow cm) for a game, the measures the research tested: wind at 10 m averaged over the
    kickoff hour and the three after; rain and snow summed over the four hourly totals stamped one to four hours after
    the kickoff hour (each covers the hour before its stamp).  Kickoff is naive Eastern time.  The request is in UTC:
    Open-Meteo labels local times with today's UTC offset for every date, so a game on the other side of a
    daylight-saving change would otherwise be read an hour off."""
    start = kickoff.floor('H').tz_localize('America/New_York').tz_convert('UTC').tz_localize(None)
    hourly = get_json(url, {'latitude': latitude, 'longitude': longitude, 'hourly': 'wind_speed_10m,precipitation,snowfall',
                            'wind_speed_unit': 'mph', 'timezone': 'GMT', 'start_date': start.date().isoformat(),
                            'end_date': (start + timedelta(days=1)).date().isoformat()})['hourly']
    frame = pd.DataFrame({k: hourly[k] for k in ('wind_speed_10m', 'precipitation', 'snowfall')},
                         index=pd.to_datetime(hourly['time'])).astype(float)
    during = frame.reindex(pd.date_range(start, periods=GAME_HOURS, freq='H'))
    after = frame.reindex(pd.date_range(start + timedelta(hours=1), periods=GAME_HOURS, freq='H'))
    return (float(during['wind_speed_10m'].mean()), float(after['precipitation'].sum(min_count=GAME_HOURS)),
            float(after['snowfall'].sum(min_count=GAME_HOURS)))


def game_wind(latitude, longitude, kickoff, url=FORECAST_URL):
    """Forecast wind (mph, 10 m) averaged over the kickoff hour and the three after; kickoff is naive Eastern time."""
    return game_weather(latitude, longitude, kickoff, url)[0]


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


def rain_flag(wind, rain, snow, week, game_type):
    """RAIN: rain over the game window and no snow, in a game the wind rule does not flag UNDER (the tested rule)."""
    if np.isnan(wind) or np.isnan(rain) or np.isnan(snow) or (game_type == 'REG' and week < FIRST_FLAG_WEEK):
        return ''
    return 'RAIN' if rain >= RAIN_MM and snow < SNOW_CM and wind < UNDER_MPH else ''


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


def record(log, label, column='flag', min_wind=None):
    graded = log[(log[column] == label) & log['result'].isin(['Win', 'Loss', 'Push'])]
    if min_wind is not None:
        graded = graded[pd.to_numeric(graded['wind_mph'], errors='coerce') >= min_wind]
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
    path = os.path.join(output_dir or sheet.nfl_folder(), f'{season}_wind_unders.csv')
    log = pd.read_csv(path).reindex(columns=LOG_COLUMNS) if os.path.exists(path) else pd.DataFrame(columns=LOG_COLUMNS)
    # games whose under went out on an earlier day's share card keep the check it was shared with (and its total)
    shared = sheet.shared_games(season, ('total',))

    print(f'Weather check, {season} week {week} (run {now:%a %b %d %I:%M %p} ET), from week {FIRST_FLAG_WEEK}. UNDER at '
          f'{UNDER_MPH:g}+ mph, WATCH at {WATCH_MPH:g}-{UNDER_MPH:g} mph, RAIN at {RAIN_MM:g}+ mm of rain without an UNDER; '
          f'forecast wind at 10 m averaged over the game, rain summed over it.\n')
    print('%-19s %-30s %8s %7s %8s %9s %-11s %s' % ('Kickoff (ET)', 'Game', 'Wind', 'Rain', 'Total', 'Under', 'Flag', 'Notes'))
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
        logged = log[log['game_id'] == g.game_id]
        kept = (week, g.away_team, g.home_team) in shared and len(logged) > 0
        if hours <= 0 or kept:
            note = 'kicked off' if hours <= 0 else (f'kept as shared on an earlier day\'s card (total '
                                                     f'{pd.to_numeric(logged["total"], errors="coerce").iloc[0]:g})')
            wind, rain = (float(pd.to_numeric(logged[c], errors='coerce').iloc[0]) if len(logged) else float('nan') for c in ('wind_mph', 'rain_mm'))
            game_flag = logged['flag'].iloc[0] if len(logged) and isinstance(logged['flag'].iloc[0], str) else ''
            wet_flag = logged['rain_flag'].iloc[0] if len(logged) and isinstance(logged['rain_flag'].iloc[0], str) else ''
        elif hours > 24 * FORECAST_DAYS:
            wind, rain, game_flag, wet_flag, note = float('nan'), float('nan'), '', '', 'too far out to forecast'
        else:
            try:
                wind, rain, snow = game_weather(*TEAM_SITES[g.home_team][:2], kickoff)
            except (requests.RequestException, ValueError, KeyError) as e:
                wind, rain, snow, note = float('nan'), float('nan'), float('nan'), f'no forecast ({e})'
            else:
                note = f'{hours / 24:.1f} days out, recheck near kickoff' if hours > 36 else ''
            game_flag = flag(wind, week, g.game_type)
            wet_flag = rain_flag(wind, rain, snow, week, g.game_type)
            if g.game_type == 'REG' and week < FIRST_FLAG_WEEK:
                note = (note + '; ' if note else '') + 'weeks 1-4 untested-to-negative, no flag'
            if not np.isnan(snow) and snow >= SNOW_CM:
                note = (note + '; ' if note else '') + f'snow forecast ({snow:.1f} cm), no rain flag'
            row = {'season': season, 'week': week, 'game_id': g.game_id, 'kickoff_et': kickoff.strftime('%Y-%m-%d %H:%M'), 'away': away,
                   'home': home, 'stadium': g.stadium, 'wind_mph': round(wind, 1) if not np.isnan(wind) else None,
                   'rain_mm': round(rain, 1) if not np.isnan(rain) else None, 'snow_cm': round(snow, 1) if not np.isnan(snow) else None,
                   'checked_at': now.strftime('%Y-%m-%d %H:%M'), 'hours_before_kickoff': round(hours, 1),
                   'total': line[0] if line else None, 'under_odds': line[1] if line else None, 'over_odds': line[2] if line else None,
                   'book': line[3] if line else None, 'flag': game_flag, 'rain_flag': wet_flag, 'final_total': None, 'result': None}
            others = log[log['game_id'] != g.game_id]
            log = pd.concat([others, pd.DataFrame([row])], ignore_index=True) if len(others) else pd.DataFrame([row], columns=LOG_COLUMNS)
        if wet_flag and not np.isnan(wind) and wind >= RAIN_WIND_MPH:
            note = (note + '; ' if note else '') + f'rain with {RAIN_WIND_MPH:g}+ mph of wind, the strongest spot'
        wind_text = '%.1f mph' % wind if not np.isnan(wind) else '-'
        rain_text = '%.1f mm' % rain if not np.isnan(rain) else '-'
        total_text = '%g' % line[0] if line else '-'
        under_text = '%+d' % line[1] if line and line[1] is not None else '-'
        print('%-19s %-30s %8s %7s %8s %9s %-11s %s' % (kickoff.strftime('%a %b %d %I:%M%p'), label, wind_text, rain_text, total_text,
                                                       under_text, ' '.join(f for f in (game_flag, wet_flag) if f), note))
    if skipped:
        print('\nnot checked (no weather exposure or no stadium location): ' + '; '.join(skipped))

    log = update_results(log, schedule)
    log = log[LOG_COLUMNS].sort_values(['season', 'week', 'kickoff_et'])
    try:
        log.to_csv(path, index=False)
        print(f'\nSeason log: {path}')
    except PermissionError:
        print(f'\nCould not save {path} (is it open in Excel?); close it and run again')
    print(f'Season record: UNDER flags {record(log, "UNDER")}, WATCH flags {record(log, "WATCH")}, RAIN flags '
          f'{record(log, "RAIN", "rain_flag")} (of which with {RAIN_WIND_MPH:g}+ mph of wind '
          f'{record(log, "RAIN", "rain_flag", RAIN_WIND_MPH)}) (the latest check before kickoff counts; pushes refunded; '
          f'break-even 52.4% at -110)')


if __name__ == '__main__':
    arguments = [int(a) for a in sys.argv[1:3]]
    main(*arguments)
