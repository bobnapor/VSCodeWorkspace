"""Totals, round 2: rain, the market's memory, referees, backup quarterbacks, single stats, the opening total, September.

Round 1 (nfl_research_totals.py) found one edge, wind: outdoor games with 15+ mph of recorded wind go under 57% in every
era because the market prices about half the wind's effect; a totals correction built from team ratings had nothing.
This round tests seven new ideas.  Each idea's reason, frame and pass bar were fixed here on October 20, 2026, before
any over/under result in this file was computed; the only data looked at first were weather distributions (to set the
rain threshold), match rates and coverage.  Nothing in production changes either way; a pass makes an idea a candidate
for forward tracking, not a bet.

Data: the nflverse schedule (closing totals and prices, kickoff times, referees, starting quarterbacks), the research
table's team ratings, Sportsbook Reviews Online's opening and closing totals (2007-2021, nfl_market_history.py), and
the weather at every outdoor home stadium from Open-Meteo (free): ERA5 reanalysis (hourly, every season, the same model
throughout) and the archived weather-model forecasts (2018 on), both requested in UTC.  Weather window: wind, gusts and
temperature over the kickoff hour and the three after; precipitation and snowfall over the four hourly totals stamped
one to four hours after the kickoff hour (each covers the hour before its stamp).  1999 has no kickoff times: Monday
games 9 pm Eastern, western stadiums 4:05 pm, the rest 1 pm.

Frame, unless a test says otherwise: regular-season games, weeks 5-16, with a closing total and a final score; eras
1999-2005 / 2006-2017 / 2018-2026.  Over margin = total points - closing total; pushes drop out of hit rates.

Bars.  A RULE passes when its bet hits >= 52.4% (break-even at -110) in every era AND its pooled record beats 50% at a
one-sided binomial p <= 0.01; seven ideas are tested, so an era-by-era bar alone is too easy to clear by luck.  An
INFORMATION test passes when the signal's correlation with the over margin has the predicted sign in every era AND the
pooled heteroskedasticity-robust slope t is >= 2.5 in the predicted direction.

H1 Rain.  A wet ball and poor footing hurt passing and kicking the way wind does, and the market may price rain the same
   half-way.  Rain game: ERA5 precipitation >= 0.5 mm over the game window (about 10% of outdoor games, roughly
   "measurable rain during the game") and snowfall < 0.1 cm.  RULE: the under in rain games the wind rule does not
   already flag (recorded wind below 15 mph or not recorded), so a pass means bets the wind rule does not make.
   V2, run either way but deciding only if H1 passes: on the archived forecasts (2018 on, weeks 5-16), forecast
   precipitation >= 0.5 mm, forecast snowfall < 0.1 cm and forecast wind below 15 mph, the under hits >= 52.4%.
H2 The market's memory.  Bettors and books react to the last game they watched: a shootout inflates the next total, a
   rock fight deflates it.  Signal: the two teams' over margins in their previous game this season, added.  INFORMATION
   test, predicted sign negative; RULE: the under when the sum is +20 or more, the over when it is -20 or less (about
   one standard deviation).  Both must pass.
H3 Referees.  "This referee's games go over" is a staple of betting previews, and crews do differ in how many flags they
   throw.  Signal: the referee's mean over margin in his previous three regular seasons, shrunk toward zero
   (sum / (games + 100)); 0 for a referee with no history.  INFORMATION test, predicted sign positive.
H4 Backup quarterbacks.  With a starter out the total drops, nobody wants an over with a backup, and backups throw the
   interceptions that hand the other side short fields.  Signal: the number of teams (0-2) starting a quarterback other
   than their regular starter (most starts in the team's previous four games this season, the most recent on ties, so
   an injury fill-in counts and a permanent change stops counting within two starts).
   INFORMATION test, predicted sign positive; RULE: the over when at least one team starts a backup.  Both must pass.
H5 Single stats.  Each of the 74 rating sums (both offenses' or both defenses' ratings of a stat) against the over
   margin, as nfl_research_winning.py M1 did for spreads: pass = the same sign in every era and pooled |t| >= 3.4
   (Bonferroni for 74 tests).  Round 1's ridge on all of them had nothing on 1999-2005, but one stat could matter while
   the rest add noise.
H6 The opening total (Sportsbook Reviews Online, 2007-2021; two eras, 2007-2017 and 2018-2021; regular season, weeks
   5-16; outcomes against the archive's own totals; totals outside 30-66 or moves over 7 points dropped as archive
   errors).  H6a: totals drift down during the week (sharp money is said to like unders), so the under at the opening
   total beats 50%.  RULE: every under at the opener; rule bar on the two eras.  H6b: the market under-reacts to news
   (as it does to wind), so a total that moved is still wrong the same way at the close.  INFORMATION test of the move
   (close - open), predicted sign positive; RULE: when the total moved 1.5 points or more, bet the way it moved, at the
   close.  Both must pass.
H7 September.  Offenses are rusty early (starters barely play in the preseason) while totals carry last season's
   scoring.  Frame: weeks 1-2 instead of 5-16.  RULE: every under.

Diagnostics (no decisions): snow games (ERA5 snowfall >= 0.1 cm); the under by ERA5 rain band; how far the market lowers
the total for rain against how far scoring falls; ERA5 wind with its threshold calibrated to flag the same share of
games as recorded 15 mph (no outcomes used), on all games and on the 55 outdoor games round 1 never saw because their
wind was not recorded (too few to test, which is why this was planned as a test and dropped before any result); gusts
beyond the sustained wind; the forecast wind rule outside weeks 5-16 (weeks 1-4, 17-18, playoffs) and at the forecast
threshold calibrated the same way; in windy games, how far the total moves from open to close and the under at each.

Run it from this folder:  python nfl_research_totals2.py fetch   (the weather, about an hour the first time; cached as
                                                                   data_cache/game_weather_era5.csv and _forecast.csv)
                          python nfl_research_totals2.py         (the tests)
                          python nfl_research_totals2.py after   (diagnostics added after the verdict, no decisions)

Results (October 20, 2026; eras 1999-2005 / 2006-2017 / 2018-2026, weeks 5-16; 4,791 games, 3,469 outdoor with weather):
H1 rain PASSES: in rain games the wind rule does not flag (325) the under went 57.8% (37-27) / 54.7% (81-67) / 61.7%
   (66-41), pooled 184-135 (57.7%, p = 0.004, +12.8% ROI at the real closing odds since 2006); all 405 rain games 58.9%.
   It rises with the rain (dry 49.1%, 0.1-0.5 mm 52.0%, 0.5-2.5 mm 56.9%, 2.5+ mm 59.0%), and the market lowers the total
   by about a fifth of what scoring falls (against dry games: closing total -0.44, actual total -2.10).  V2 passes:
   forecast rain with forecast wind under 15 mph went 35-22 (61.4%) since 2018.
H2 the market's memory fails: no information (+0.021 / -0.025 / +0.015, t -0.07); fading a combined +/-20 went 50.4%.
H3 referees fail: +0.001 / +0.003 / -0.037 (t -0.50).
H4 backup quarterbacks fail, the wrong way: backup games lean under in every era (-0.035 / -0.028 / -0.030, t -2.12);
   the over with a backup went 49.1%.
H5 single stats PASSES with three of 74: both defenses' points allowed (-0.033 / -0.063 / -0.070, t -4.10), touchdowns
   allowed per drive (-0.023 / -0.055 / -0.080, t -3.92) and sack rate forced (+0.092 / +0.044 / +0.036, t +3.85):
   totals run too high when both defenses have allowed a lot and too low when both have been good, so the market
   over-trusts defensive results (as nfl_research_winning.py found for spreads).  The first two are nearly the same
   stat (r = 0.85); as one signed composite the three carry +0.061 / +0.066 / +0.073, about the size of the spread
   correction's information, but that is in-sample (the scan used every era), so it is a forward-tracking candidate only.
H6 the opening total fails: every under at the opener 50.1% / 53.6% (p = 0.15); the move carries nothing (-0.019 /
   +0.010); following a 1.5-point move went 48.6%.
H7 September fails: weeks 1-2 unders 56.3% / 51.2% / 53.1% (pooled 53.1%, p = 0.04).
Diagnostics: snow games went over 51-37 (58.0%, p = 0.08); ERA5 wind calibrated to recorded 15 mph (12.1 mph) gives the
   same wind rule (56.7% vs 56.8%), and the 45 outdoor games without recorded wind went 3-1 at it; gusts add nothing
   beyond the sustained wind; on forecasts since 2018 the wind rule fails in weeks 1-4 (1-4 at 15+ mph, 26-29 at 10-15),
   holds in weeks 5-16 (10-15 mph 96-61, the calibrated 12.2+ mph 70-51) and is too thin to judge in weeks 17-18 and
   the playoffs (playoffs 12.2+ mph 9-2); in windy games the total falls 0.94 points from open to close (other outdoor
   games 0.23) and the under hit 59.3% at the opener against 56.4% at the close (2007-2021).
After the verdict (`after`, no decisions): the rain edge is rain with wind.  Rain games with recorded wind under 10 mph
   went 104-99 (51.2%), with 10-15 mph 76-35 (68.5%, 66-71% in every era); on ERA5's (the forecast's) scale, rain with
   9-12 mph of wind went 73-38 and with 12+ mph 64-32 (137-70, 66.2%), rain under 9 mph 97-93; dry games 47-54% at every
   wind.  On forecasts since 2018, rain with 10-15 mph of wind went 13-3 and with less 22-19.  It held in warm and cold
   rain (58.0%, 56.5%) and at low and high totals (55.6%, 61.2%), in 18 of 27 seasons, mostly by day (night games 20-23).
   The forecast archive marks about half the games ERA5 calls rainy (correlation 0.56), so forecast flags are fewer and
   surer.  Production: nfl_wind_unders.py adds a separate RAIN flag (the tested rule) and notes the rain-with-9+-mph
   subset, which is tracked forward before it is trusted.

U1-U2, the two non-weather leads on games round 2 never used (pre-registered October 21, 2026, after the results above
and before either signal was computed on these games).  Every test above used regular-season weeks 5-16 only, so weeks
1-4, weeks 17-18 and the playoffs are out of sample for both.
U1 Defense over-trust.  The composite fixed after H5: - both defenses' points allowed - their touchdowns allowed per
   drive + their sack rate forced, each standardized with its weeks 5-16 mean and SD.  Games: every game outside
   regular-season weeks 5-16 with a closing total and a final score, 1999-2026 (2,550: weeks 1-4, 17-18, playoffs).
   Pass: in a regression of the over margin on the composite, the closing total and the season-to-date league average
   total, the composite's coefficient is positive with heteroskedasticity-robust t >= 2.0 (one confirmatory test of a
   fixed signal, so a one-sided 2.3% bar rather than the 2.5 used when searching).  The controls matter: the composite
   tracks the total itself (good defenses get low totals), so without them a market-wide "extreme totals are too
   extreme" effect would pass as defense over-trust (a dry run on shuffled scores, where that link is mechanical,
   passed the raw correlation).  Shown beside it: the raw correlation, the total's own coefficient, by segment and era,
   the same regression on weeks 5-16 where the stats were found, and the hit rate betting its side at 1 and 1.5 SD.
U2 Backup quarterbacks lean under (H4's signal, which ran the opposite way to its prediction: t = -2.12).  Games:
   regular-season weeks 2-4 and 17-18 (week 1 has no earlier starts, and the signal is built from regular-season games
   only); 1,792 games, 388 with a backup.  Pass: the same regression with the backup count in place of the composite,
   its coefficient negative with robust t <= -2.0.  Underpowered: if the weeks 5-16 effect holds exactly, the expected t
   is about -1.3, so a fail says little; a pass would say a lot.
Run it:  python nfl_research_totals2.py unseen

U1-U2 results (October 21, 2026): both fail.  U1: given the closing and league totals the composite went the wrong way
out of sample (-0.111 points per unit, t -0.92; raw correlation -0.011): weeks 1-4 -0.17 (t -1.1), weeks 17-18 -0.25
(t -0.9), playoffs +0.16 (t +0.4); 1999-2005 +0.02, 2006-2017 +0.07, 2018-2026 -0.39 (t -2.0).  In weeks 5-16, where the
stats were found, the same controlled regression gives +0.341 (t +3.80), so the lead there was not the total's level in
disguise, but it did not carry to any other part of the season; betting its side went 51.4% at 1 SD and 48.2% at 1.5
SD.  Read: a scan artifact, or an effect too fragile to bet.  U2: backups leaned under again but weakly (-0.565 points
per backup, t -0.82; weeks 17-18 -1.25, weeks 2-4 -0.34; the under with a backup 199-186, 51.7%), inconclusive as
expected from its power.  So the totals edges that survive every test are the weather ones.
"""

import math
import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

from nflverse_data import CACHE_DIR

RAIN_MM, SNOW_CM = 0.5, 0.1
MEMORY_LIMIT = 20.0
REF_SEASONS, REF_SHRINK = 3, 100.0
BACKUP_WINDOW = 4
MOVE_LIMIT, MOVE_MAX, SBR_RANGE = 1.5, 7.0, (30.0, 66.0)
SBR_ERAS = (('2007-2017', 2007, 2017), ('2018-2021', 2018, 2021))
EARLY_WEEKS = (1, 2)
P_LIMIT, T_LIMIT, SCAN_T = 0.01, 2.5, 3.4

ERA5_URL = 'https://archive-api.open-meteo.com/v1/archive'
FORECAST_URL = 'https://historical-forecast-api.open-meteo.com/v1/forecast'
FORECAST_FIRST = 2018
HOURLY = ('temperature_2m', 'precipitation', 'snowfall', 'wind_speed_10m', 'wind_gusts_10m')
GAME_HOURS = 4
CALLS_PER_MINUTE = 80   # Open-Meteo's free tier allows 600 a minute, 5,000 an hour and 10,000 a day; each location is a call
WEATHER_FILES = {'era5': os.path.join(CACHE_DIR, 'game_weather_era5.csv'),
                 'forecast': os.path.join(CACHE_DIR, 'game_weather_forecast.csv')}

# Every outdoor stadium that hosted a home (non-neutral) game since 1999, by nflverse stadium_id.
STADIUMS = {
    'BAL00': (39.278, -76.623), 'BOS00': (42.091, -71.264), 'BOS99': (42.091, -71.264), 'BRG00': (30.412, -91.184),
    'BUF00': (42.774, -78.787), 'CAR00': (35.226, -80.853), 'CHI98': (41.862, -87.617), 'CHI99': (40.099, -88.236),
    'CIN00': (39.095, -84.516), 'CIN99': (39.097, -84.507), 'CLE00': (41.506, -81.700), 'DAL99': (32.840, -96.911),
    'DEN00': (39.744, -105.020), 'DEN99': (39.746, -105.021), 'GNB00': (44.501, -88.062), 'JAX00': (30.324, -81.637),
    'KAN00': (39.049, -94.484), 'LAX97': (33.864, -118.261), 'LAX99': (34.014, -118.288), 'MIA00': (25.958, -80.239),
    'MIN98': (44.976, -93.225), 'NAS00': (36.166, -86.771), 'NYC00': (40.812, -74.076), 'NYC01': (40.814, -74.074),
    'OAK00': (37.752, -122.201), 'PHI00': (39.901, -75.168), 'PHI99': (39.907, -75.171), 'PHO99': (33.426, -111.933),
    'PIT00': (40.447, -80.016), 'PIT99': (40.447, -80.012), 'SDG00': (32.783, -117.120), 'SEA00': (47.595, -122.332),
    'SEA99': (47.650, -122.302), 'SFO00': (37.714, -122.386), 'SFO01': (37.403, -121.970), 'TAM00': (27.976, -82.503),
    'WAS00': (38.908, -76.865),
}


def kickoff_utc(games):
    """Kickoff hour in UTC (naive).  nflverse kickoff times are Eastern; 1999 has none, so Monday games get 9 pm, games
    at western stadiums 4:05 pm and the rest 1 pm (the usual slots that season)."""
    clock = games['gametime'].copy()
    missing = clock.isna()
    west = games['stadium_id'].map(lambda s: STADIUMS[s][1] < -100)
    clock[missing & (games['weekday'] == 'Monday')] = '21:00'
    clock[missing & (games['weekday'] != 'Monday') & west] = '16:05'
    clock = clock.fillna('13:00')
    local = pd.to_datetime(games['gameday'] + ' ' + clock)
    return local.dt.tz_localize('America/New_York').dt.tz_convert('UTC').dt.tz_localize(None).dt.floor('H')


def outdoor_games(games):
    """Played outdoor home games (no neutral sites), with their stadium coordinates and kickoff hour in UTC."""
    g = games[(games['roof'] == 'outdoors') & (games['location'] == 'Home') & games['result'].notna()].copy()
    unknown = sorted(set(g['stadium_id']) - set(STADIUMS))
    if unknown:
        raise ValueError(f'No coordinates for outdoor stadiums {unknown}')
    g['kick_utc'] = kickoff_utc(g)
    return g


def summarize(hourly, kick):
    """The game's weather from one location's hourly series: wind, gusts and temperature over the kickoff hour and the
    three after; precipitation and snowfall over the four hourly totals that end one to four hours after the kickoff hour
    (each hourly total covers the hour before its time stamp)."""
    frame = pd.DataFrame({k: hourly[k] for k in HOURLY}, index=pd.to_datetime(hourly['time'])).astype(float)
    during = frame.reindex(pd.date_range(kick, periods=GAME_HOURS, freq='H'))
    after = frame.reindex(pd.date_range(kick + timedelta(hours=1), periods=GAME_HOURS, freq='H'))
    return {'wind': during['wind_speed_10m'].mean(), 'gust': during['wind_gusts_10m'].mean(),
            'gust_max': during['wind_gusts_10m'].max(), 'temp': during['temperature_2m'].mean(),
            'precip': after['precipitation'].sum(min_count=GAME_HOURS), 'snow': after['snowfall'].sum(min_count=GAME_HOURS)}


def _request(url, params, tries=6):
    for attempt in range(tries):
        try:
            response = requests.get(url, params=params, timeout=120)
            if response.status_code == 200:
                return response.json()
            message = f'HTTP {response.status_code}: {response.text[:200]}'
        except requests.RequestException as e:
            message = str(e)
        wait = 30 * 2 ** attempt
        print(f'  retry in {wait}s ({message})', flush=True)
        time.sleep(wait)
    raise RuntimeError(f'Open-Meteo request failed {tries} times')


def fetch_weather(source, games):
    """Game weather from ERA5 reanalysis ('era5') or the archived weather-model forecasts ('forecast'), requested in UTC
    and cached per game in data_cache/game_weather_<source>.csv (resumable)."""
    path = WEATHER_FILES[source]
    done = set(pd.read_csv(path)['game_id']) if os.path.exists(path) else set()
    todo = games[~games['game_id'].isin(done)]
    url = ERA5_URL if source == 'era5' else FORECAST_URL
    days = sorted(todo['kick_utc'].dt.normalize().unique())
    print(f'{source}: {len(done)} games cached, {len(todo)} to fetch on {len(days)} dates', flush=True)
    for i, day in enumerate(days, 1):
        group = todo[todo['kick_utc'].dt.normalize() == day]
        started = time.time()
        day = pd.Timestamp(day)
        params = {'latitude': ','.join('%.3f' % STADIUMS[s][0] for s in group['stadium_id']),
                  'longitude': ','.join('%.3f' % STADIUMS[s][1] for s in group['stadium_id']),
                  'start_date': day.date().isoformat(), 'end_date': (day + timedelta(days=1)).date().isoformat(),
                  'hourly': ','.join(HOURLY), 'wind_speed_unit': 'mph', 'temperature_unit': 'fahrenheit', 'timezone': 'GMT'}
        if source == 'era5':
            params['models'] = 'era5'
        data = _request(url, params)
        data = data if isinstance(data, list) else [data]
        rows = [dict(game_id=g.game_id, **summarize(d['hourly'], g.kick_utc)) for g, d in zip(group.itertuples(), data)]
        pd.DataFrame(rows).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        if i % 25 == 0 or i == len(days):
            print(f'  {source}: {i}/{len(days)} dates', flush=True)
        time.sleep(max(0.0, len(group) * 60.0 / CALLS_PER_MINUTE - (time.time() - started)))


def fetch():
    from nflverse_data import load_games
    games = outdoor_games(load_games())
    games = games[games['kick_utc'] < pd.Timestamp.utcnow().tz_localize(None).normalize() - timedelta(days=2)]   # archives lag
    fetch_weather('era5', games[(games['game_type'] == 'REG') & games['week'].between(5, 16)])
    fetch_weather('forecast', games[games['season'] >= FORECAST_FIRST])


# ---------------------------------------------------------------------------------------------------------- the tests

def one_sided_p(wins, losses):
    from scipy.stats import binom
    n = wins + losses
    return float(binom.sf(wins - 1, n, 0.5)) if n else float('nan')


def record_text(g, bet_over):
    from nfl_research_totals import bets
    w, l, rate, roi = bets(g, bet_over)
    if not w + l:
        return 'none', float('nan'), 0, 0
    return '%d-%d %.3f%s' % (w, l, rate, '' if roi is None else ' (%+.1f%%)' % (100 * roi)), rate, w, l


def rule(label, f, mask, bet_over, era_list=None, show_only=False):
    """A rule's record by era and pooled (bet_over is a Series: True = the over); True if it clears the rule bar."""
    from nfl_research_totals import BREAK_EVEN, ERAS
    mask = mask.fillna(False).astype(bool)
    cells, rates = [], []
    for name, lo, hi in era_list or ERAS:
        text, rate, _, _ = record_text(f[mask & f['season'].between(lo, hi)], bet_over)
        cells.append('%s %s' % (name, text))
        rates.append(rate)
    text, _, w, l = record_text(f[mask], bet_over)
    p = one_sided_p(w, l)
    ok = all(r >= BREAK_EVEN for r in rates) and p <= P_LIMIT
    print('    %-44s %s | pooled %s p=%.3f%s' % (label, ' | '.join(cells), text, p, '' if show_only else ('  PASS' if ok else '  fail')))
    return ok


def pooled(label, g, mask, bet_over):
    """One pooled record, no bar (diagnostics)."""
    text, _, w, l = record_text(g[mask.fillna(False).astype(bool)], bet_over)
    print('    %-44s %s' % (label, text))


def information(label, f, signal, sign, era_list=None):
    """Correlation of a signal with the over margin by era and the pooled robust t; True with the predicted sign in every
    era and sign * t >= 2.5."""
    from nfl_research_totals import ERAS, slope_t
    use = signal.notna() & f['over_margin'].notna()
    cors = []
    for name, lo, hi in era_list or ERAS:
        m = use & f['season'].between(lo, hi)
        cors.append(float(np.corrcoef(signal[m], f.loc[m, 'over_margin'])[0, 1]) if m.sum() > 2 else float('nan'))
    _, t = slope_t(signal[use], f.loc[use, 'over_margin'])
    ok = all(sign * c > 0 for c in cors) and sign * t >= T_LIMIT
    print('    %-44s %s | pooled t %+.2f (n=%d)  %s' % (label, ' | '.join('%s %+.3f' % (e[0], c) for e, c in zip(era_list or ERAS, cors)),
                                                        t, int(use.sum()), 'PASS' if ok else 'fail'))
    return ok


def regular(t, weeks=None):
    from nfl_research_totals import WEEKS
    lo, hi = weeks or WEEKS
    return t[(t['game_type'] == 'REG') & t['week'].between(lo, hi) & t['total'].notna() & t['total_line'].notna()].copy()


def with_weather(f, source):
    """Inner join of a game frame with a weather file; weather columns get a w_ prefix (the frame keeps the recorded
    wind and temp)."""
    w = pd.read_csv(WEATHER_FILES[source])
    w = w.rename(columns={c: f'w_{c}' for c in w.columns if c != 'game_id'})
    return f.merge(w, on='game_id', how='inner')


def calibrated(measured, recorded, limit):
    """Threshold on `measured` that flags the same share of games as `recorded` >= limit does (games with both)."""
    both = measured.notna() & recorded.notna()
    share = float((recorded[both] >= limit).mean())
    return float(measured[both].quantile(1 - share)), share, int(both.sum())


def team_games(t):
    """One row per team per regular-season game, in order: season, team, game, over margin, starting quarterback."""
    reg = t[t['game_type'] == 'REG']
    sides = [reg[['game_id', 'season', 'week', 'gameday', f'{side}_team', f'{side}_qb_id', 'over_margin']]
             .rename(columns={f'{side}_team': 'team', f'{side}_qb_id': 'qb'}) for side in ('home', 'away')]
    return pd.concat(sides, ignore_index=True).sort_values(['season', 'team', 'gameday', 'week']).reset_index(drop=True)


def memory_and_backups(t):
    """Per game: the two teams' previous-game over margins added (H2) and how many start a backup quarterback (H4)."""
    tg = team_games(t)
    tg['prev_over'] = tg.groupby(['season', 'team'])['over_margin'].shift(1)
    backup = pd.Series(np.nan, index=tg.index)
    for _, g in tg.groupby(['season', 'team'], sort=False):
        history = []
        for index, qb in zip(g.index, g['qb']):
            recent = [q for q in history[-BACKUP_WINDOW:] if isinstance(q, str)]
            if recent and isinstance(qb, str):
                last = {q: i for i, q in enumerate(recent)}
                starter = max(last, key=lambda q: (recent.count(q), last[q]))
                backup[index] = float(qb != starter)
            history.append(qb)
    tg['backup'] = backup
    return tg.groupby('game_id').agg(memory=('prev_over', lambda s: s.sum(min_count=2)),
                                     backups=('backup', lambda s: s.sum(min_count=2)))


def referee_signal(t):
    """H3: each referee's over margin in his previous three regular seasons, sum / (games + 100); 0 with no history."""
    reg = t[(t['game_type'] == 'REG') & t['over_margin'].notna() & t['referee'].notna()]
    by = reg.groupby(['referee', 'season'])['over_margin'].agg(['sum', 'count']).reset_index()
    signal = {}
    for season in range(int(t['season'].min()), int(t['season'].max()) + 1):
        prior = by[by['season'].between(season - REF_SEASONS, season - 1)].groupby('referee')[['sum', 'count']].sum()
        signal.update({(ref, season): row['sum'] / (row['count'] + REF_SHRINK) for ref, row in prior.iterrows()})
    return pd.Series([signal.get((r, s), 0.0) for r, s in zip(t['referee'], t['season'])], index=t.index)


def sbr_frame(f):
    """Weeks 5-16, 2007-2021, with the archive's opening and closing totals; outcomes against its own closing total."""
    import nfl_market_history as mh
    m = mh.attach_open_lines(f[f['season'].between(2007, 2021)])
    m = m[m['open_total'].between(*SBR_RANGE) & m['close_total'].between(*SBR_RANGE)].copy()
    m['move'] = m['close_total'] - m['open_total']
    m = m[m['move'].abs() <= MOVE_MAX].reset_index(drop=True)
    return m.assign(over_margin=m['total'] - m['close_total'], over_odds=np.nan, under_odds=np.nan)


def main():
    import nfl_lab as lab
    from nfl_research_totals import BREAK_EVEN, ERAS, SUM_FEATURES, WIND_LIMIT, bets, slope_t, totals_table
    t = totals_table(lab.default_table())
    f = regular(t)
    under, over = pd.Series(False, index=f.index), pd.Series(True, index=f.index)
    verdict = {}
    print('frame: %d games, weeks 5-16, %d-%d' % (len(f), f['season'].min(), f['season'].max()))

    # H1 rain
    e = with_weather(f, 'era5')
    e_under = pd.Series(False, index=e.index)
    rain = (e['w_precip'] >= RAIN_MM) & (e['w_snow'] < SNOW_CM)
    windy = e['wind_out'] >= WIND_LIMIT
    print('\n=== H1: rain (ERA5; %d outdoor games with weather, %d rain games, %d of them not flagged by the wind rule) ===' % (
        len(e), int(rain.sum()), int((rain & ~windy).sum())))
    verdict['H1 rain under (wind rule not flagging)'] = rule('RULE: under, rain, no wind flag', e, rain & ~windy, e_under)
    rule('all rain games', e, rain, e_under, show_only=True)
    for lo, hi in ((0.0, 0.1), (0.1, 0.5), (0.5, 2.5), (2.5, 999.0)):
        band = e['w_precip'].between(lo, hi, inclusive='left') & (e['w_snow'] < SNOW_CM) & ~windy
        rule('  precipitation %.1f-%s mm, no wind flag' % (lo, '' if hi > 900 else '%.1f' % hi), e, band, e_under, show_only=True)
    print('    how far the market lowers the total and how far scoring falls, against the season-to-date league average:')
    for label, m in (('dry (< 0.1 mm), no wind flag', (e['w_precip'] < 0.1) & ~windy), ('rain, no wind flag', rain & ~windy),
                     ('rain 2.5+ mm, no wind flag', (e['w_precip'] >= 2.5) & (e['w_snow'] < SNOW_CM) & ~windy)):
        g = e[m]
        print('      %-32s closing total %+.2f, actual total %+.2f, over margin %+.2f (n=%d)' % (
            label, (g['total_line'] - g['league_total']).mean(), (g['total'] - g['league_total']).mean(), g['over_margin'].mean(), len(g)))
    fc = with_weather(f[f['season'] >= FORECAST_FIRST], 'forecast')
    fc_rain = (fc['w_precip'] >= RAIN_MM) & (fc['w_snow'] < SNOW_CM) & (fc['w_wind'] < WIND_LIMIT)
    w, l, rate, _ = bets(fc[fc_rain], pd.Series(False, index=fc.index))
    verdict['V2 rain on forecasts (decides only if H1 passes)'] = rate >= BREAK_EVEN
    print('    V2: forecast rain, forecast wind under 15, 2018 on: %d-%d %.3f  %s' % (w, l, rate, 'PASS' if rate >= BREAK_EVEN else 'fail'))

    # H2 memory, H4 backups
    f = f.join(memory_and_backups(t), on='game_id')
    print('\n=== H2: the market\'s memory (both teams\' previous-game over margins, added; SD %.1f) ===' % f['memory'].std())
    h2_info = information('INFO: memory (predicted negative)', f, f['memory'], -1)
    extreme = (f['memory'] >= MEMORY_LIMIT) | (f['memory'] <= -MEMORY_LIMIT)
    h2_rule = rule('RULE: fade a combined +/-%g' % MEMORY_LIMIT, f, extreme, f['memory'] <= -MEMORY_LIMIT)
    rule('  under after +%g' % MEMORY_LIMIT, f, f['memory'] >= MEMORY_LIMIT, under, show_only=True)
    rule('  over after -%g' % MEMORY_LIMIT, f, f['memory'] <= -MEMORY_LIMIT, over, show_only=True)
    verdict['H2 memory (information and rule)'] = h2_info and h2_rule

    print('\n=== H3: referees ===')
    f['referee_signal'] = referee_signal(t).loc[f.index]
    print('    referees with history: %.0f%% of games' % (100 * (f['referee_signal'] != 0).mean()))
    verdict['H3 referee'] = information('INFO: referee history (predicted positive)', f, f['referee_signal'], +1)

    print('\n=== H4: backup quarterbacks (%.1f%% of games have at least one) ===' % (100 * (f['backups'] >= 1).mean()))
    h4_info = information('INFO: backups starting (predicted positive)', f, f['backups'], +1)
    h4_rule = rule('RULE: over with a backup', f, f['backups'] >= 1, over)
    verdict['H4 backups (information and rule)'] = h4_info and h4_rule

    print('\n=== H5: single stats, the 74 rating sums against the over margin (pass: same sign every era, |t| >= %.1f) ===' % SCAN_T)
    rows = []
    for col in SUM_FEATURES:
        use = f[col].notna()
        cors = [float(np.corrcoef(f.loc[use & f['season'].between(lo, hi), col], f.loc[use & f['season'].between(lo, hi), 'over_margin'])[0, 1])
                for _, lo, hi in ERAS]
        _, t_stat = slope_t(f.loc[use, col], f.loc[use, 'over_margin'])
        rows.append((col, cors, t_stat, (all(c > 0 for c in cors) or all(c < 0 for c in cors)) and abs(t_stat) >= SCAN_T))
    rows.sort(key=lambda r: -abs(r[2]))
    for col, cors, t_stat, ok in rows[:12]:
        print('    %-36s %s | t %+.2f%s' % (col, ' / '.join('%+.3f' % c for c in cors), t_stat, '  PASS' if ok else ''))
    passed = [r[0] for r in rows if r[3]]
    print('    passing: %s' % (', '.join(passed) or 'none'))
    verdict['H5 single stats'] = bool(passed)

    s = sbr_frame(f)
    s_under = pd.Series(False, index=s.index)
    print('\n=== H6: the opening total (SBR archive, %d games; mean move %+.2f, SD %.2f) ===' % (len(s), s['move'].mean(), s['move'].std()))
    opener = s.assign(over_margin=s['total'] - s['open_total'])
    verdict['H6a under at the opener'] = rule('RULE: every under at the opener', opener, pd.Series(True, index=s.index), s_under, SBR_ERAS)
    rule('every under at the close (same games)', s, pd.Series(True, index=s.index), s_under, SBR_ERAS, show_only=True)
    h6_info = information('INFO: move (predicted positive)', s, s['move'], +1, SBR_ERAS)
    h6_rule = rule('RULE: follow a move of %g+ at the close' % MOVE_LIMIT, s, s['move'].abs() >= MOVE_LIMIT, s['move'] > 0, SBR_ERAS)
    verdict['H6b move (information and rule)'] = h6_info and h6_rule
    rule('  moved down %g+: under' % MOVE_LIMIT, s, s['move'] <= -MOVE_LIMIT, s_under, SBR_ERAS, show_only=True)
    rule('  moved up %g+: over' % MOVE_LIMIT, s, s['move'] >= MOVE_LIMIT, pd.Series(True, index=s.index), SBR_ERAS, show_only=True)

    early = regular(t, EARLY_WEEKS)
    print('\n=== H7: September (weeks 1-2, %d games) ===' % len(early))
    verdict['H7 weeks 1-2 unders'] = rule('RULE: every under, weeks 1-2', early, pd.Series(True, index=early.index), pd.Series(False, index=early.index))

    threshold, share, n_both = calibrated(e['w_wind'], e['wind'], WIND_LIMIT)
    unseen = e['wind'].isna()

    print('\n=== diagnostics (no decisions) ===')
    rule('snow games (ERA5 snowfall >= %.1f cm): the over' % SNOW_CM, e, e['w_snow'] >= SNOW_CM, pd.Series(True, index=e.index), show_only=True)
    print('    ERA5 wind %.1f mph flags the same %.1f%% of games as recorded 15 mph (%d games with both):' % (threshold, 100 * share, n_both))
    rule('ERA5 wind >= %.1f (calibrated), all games' % threshold, e, e['w_wind'] >= threshold, e_under, show_only=True)
    rule('recorded wind >= 15, same games', e, windy, e_under, show_only=True)
    pooled('no recorded wind (%d games), ERA5 >= %.1f' % (int(unseen.sum()), threshold), e, unseen & (e['w_wind'] >= threshold), e_under)
    print('    gusts beyond the sustained wind (partial correlation with the over margin, by era):')
    cells = []
    for name, lo, hi in ERAS:
        g = e[e['season'].between(lo, hi) & e['w_gust'].notna() & e['w_wind'].notna()]
        x = np.column_stack([np.ones(len(g)), g['w_wind']])
        resid = lambda y: y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        cells.append('%s wind %+.3f, gust %+.3f, gust given wind %+.3f' % (
            name, np.corrcoef(g['w_wind'], g['over_margin'])[0, 1], np.corrcoef(g['w_gust'], g['over_margin'])[0, 1],
            np.corrcoef(resid(g['w_gust'].to_numpy()), resid(g['over_margin'].to_numpy()))[0, 1]))
    print('      ' + '\n      '.join(cells))

    print('    forecast wind (2018 on) by part of the season:')
    allg = t[t['total'].notna() & t['total_line'].notna() & (t['season'] >= FORECAST_FIRST)]
    fa = with_weather(allg, 'forecast')
    fa_under = pd.Series(False, index=fa.index)
    f_threshold, f_share, f_both = calibrated(fc['w_wind'], fc['wind'], WIND_LIMIT)
    print('      calibrated forecast threshold: %.1f mph flags the same %.1f%% of weeks 5-16 games as recorded 15 mph (%d games)' % (
        f_threshold, 100 * f_share, f_both))
    parts = (('weeks 1-4', (fa['game_type'] == 'REG') & (fa['week'] <= 4)), ('weeks 5-16', (fa['game_type'] == 'REG') & fa['week'].between(5, 16)),
             ('weeks 17-18', (fa['game_type'] == 'REG') & (fa['week'] >= 17)), ('playoffs', fa['game_type'] != 'REG'))
    for name, part in parts:
        for label, m in (('15+', fa['w_wind'] >= 15), ('10-15', fa['w_wind'].between(10, 15, inclusive='left')),
                         ('%.1f+ (calibrated)' % f_threshold, fa['w_wind'] >= f_threshold)):
            pooled('      %s, forecast %s' % (name, label), fa, part & m, fa_under)

    print('    windy games (recorded 15+ mph) and the opening total:')
    s_windy = s['wind_out'] >= WIND_LIMIT
    print('      mean move: windy %+.2f (n=%d), other outdoor %+.2f, domes %+.2f' % (
        s.loc[s_windy, 'move'].mean(), int(s_windy.sum()), s.loc[~s_windy & (s['dome'] == 0), 'move'].mean(), s.loc[s['dome'] == 1, 'move'].mean()))
    rule('      windy: under at the opener', opener, s_windy, s_under, SBR_ERAS, show_only=True)
    rule('      windy: under at the close', s, s_windy, s_under, SBR_ERAS, show_only=True)

    print('\n=== verdict (pre-registered) ===')
    for name, ok in verdict.items():
        print('    %-52s %s' % (name, 'PASS' if ok else 'fail'))


def after_checks():
    """After H1 and H5 passed (diagnostics run after the verdict; no decisions): is the rain effect rain itself, and how
    steady is it; what do the three passing stats add up to?"""
    import nfl_lab as lab
    from nfl_research_totals import ERAS, WIND_LIMIT, bets, totals_table
    t = totals_table(lab.default_table())
    e = with_weather(regular(t), 'era5')
    e_under = pd.Series(False, index=e.index)
    rain = (e['w_precip'] >= RAIN_MM) & (e['w_snow'] < SNOW_CM) & ~(e['wind_out'] >= WIND_LIMIT)
    print('rain games the wind rule does not flag (%d), split after the fact:' % int(rain.sum()))
    for label, m in (('recorded wind under 10 mph (calm rain)', e['wind'] < 10), ('recorded wind 10-15 mph', e['wind'].between(10, 15, inclusive='left')),
                     ('ERA5 wind under 8 mph', e['w_wind'] < 8), ('below 45 degrees', e['w_temp'] < 45), ('45 degrees or warmer', e['w_temp'] >= 45),
                     ('total under the league average', e['total_line'] < e['league_total']), ('total at or above the league average', e['total_line'] >= e['league_total']),
                     ('day games', e['night'] == 0), ('night games', e['night'] == 1)):
        rule(label, e, rain & m, e_under, show_only=True)
    d = e[rain & (e['over_margin'] != 0)].assign(win=lambda x: x['over_margin'] < 0)
    by = d.groupby('season')['win'].agg(['sum', 'size'])
    print('    by season: ' + '  '.join('%d: %d-%d' % (s, w, n - w) for s, (w, n) in by.iterrows()))
    print('    seasons above 52.4%%: %d of %d; per season about %.0f games' % (int(((by['sum'] / by['size']) > 0.524).sum()), len(by), by['size'].mean()))
    fc = with_weather(regular(t)[lambda x: x['season'] >= FORECAST_FIRST], 'forecast')
    both = fc.merge(e[['game_id', 'w_precip']].rename(columns={'w_precip': 'era5_precip'}), on='game_id')
    print('    forecast vs ERA5 game-window precipitation since 2018: correlation %.2f; rain by both %d, forecast only %d, ERA5 only %d' % (
        np.corrcoef(both['w_precip'], both['era5_precip'])[0, 1], int(((both['w_precip'] >= RAIN_MM) & (both['era5_precip'] >= RAIN_MM)).sum()),
        int(((both['w_precip'] >= RAIN_MM) & (both['era5_precip'] < RAIN_MM)).sum()), int(((both['w_precip'] < RAIN_MM) & (both['era5_precip'] >= RAIN_MM)).sum())))
    fc_rain = (fc['w_precip'] >= RAIN_MM) & (fc['w_snow'] < SNOW_CM) & (fc['w_wind'] < WIND_LIMIT)
    by = fc[fc_rain & (fc['over_margin'] != 0)].assign(win=lambda x: x['over_margin'] < 0).groupby('season')['win'].agg(['sum', 'size'])
    print('    V2 by season: ' + '  '.join('%d: %d-%d' % (s, w, n - w) for s, (w, n) in by.iterrows()))
    for label, m in (('forecast rain and forecast wind 10-15', fc_rain & (fc['w_wind'] >= 10)), ('forecast rain and forecast wind under 10', fc_rain & (fc['w_wind'] < 10))):
        w, l, rate, roi = bets(fc[m], pd.Series(False, index=fc.index))
        print('    %-44s %d-%d %.3f' % (label, w, l, rate))
    print('    the under by ERA5 wind (the forecast\'s scale) and rain, all outdoor games, pooled:')
    wet = (e['w_precip'] >= RAIN_MM) & (e['w_snow'] < SNOW_CM)
    dry = e['w_precip'] < 0.1
    for lo, hi in ((0, 6), (6, 9), (9, 12), (12, 99)):
        band = e['w_wind'].between(lo, hi, inclusive='left')
        cells = []
        for label, m in (('dry', dry), ('rain', wet)):
            w, l, rate, _ = bets(e[band & m], e_under)
            cells.append('%s %d-%d %.3f' % (label, w, l, rate))
        print('      ERA5 wind %2d-%-2s mph: %s' % (lo, '' if hi > 90 else hi, ' | '.join(cells)))

    f = regular(t)
    stats = {'tot_def__pts': -1, 'tot_def__o_td_per_drive': -1, 'tot_def__o_sack_rate': +1}
    z = sum(sign * (f[c] - f[c].mean()) / f[c].std() for c, sign in stats.items())
    print('\nthe three passing stats as one signed z-score composite (seen on all eras, so in-sample):')
    print('    correlation with the over margin: ' + ' | '.join('%s %+.3f' % (name, np.corrcoef(z[f['season'].between(lo, hi)], f.loc[f['season'].between(lo, hi), 'over_margin'])[0, 1])
                                                          for name, lo, hi in ERAS))
    print('    pairwise correlations of the three: %s' % f[list(stats)].corr().round(2).values.tolist())


COMPOSITE = {'tot_def__pts': -1, 'tot_def__o_td_per_drive': -1, 'tot_def__o_sack_rate': +1}
UNSEEN_T = 2.0


def defense_composite(frame, reference):
    """U1's signal: the three stats that passed H5, signed toward the over, standardized with `reference` (weeks 5-16)."""
    return sum(sign * (frame[c] - reference[c].mean()) / reference[c].std() for c, sign in COMPOSITE.items())


def corr_t(signal, y):
    from nfl_research_totals import slope_t
    return float(np.corrcoef(signal, y)[0, 1]), slope_t(signal, y)[1]


def controlled(signal, g):
    """Robust (HC0) regression of the over margin on a signal, the closing total and the league average total:
    (signal coefficient, its t, closing-total coefficient, its t)."""
    x = np.column_stack([np.ones(len(g)), np.asarray(signal, float), g['total_line'].to_numpy(float),
                         np.asarray(g['league_total'], float)])
    y = g['over_margin'].to_numpy(float)
    inverse = np.linalg.inv(x.T @ x)
    beta = inverse @ x.T @ y
    e = y - x @ beta
    se = np.sqrt(np.diag(inverse @ (x.T * e ** 2) @ x @ inverse))
    return beta[1], beta[1] / se[1], beta[2], beta[2] / se[2]


def unseen():
    """U1-U2 (pre-registered in the docstring): round 2's non-weather leads on weeks 1-4, 17-18 and the playoffs."""
    import nfl_lab as lab
    from nfl_research_totals import ERAS, totals_table
    t = totals_table(lab.default_table())
    seen = regular(t)
    u = t[t['total'].notna() & t['total_line'].notna() & ~((t['game_type'] == 'REG') & t['week'].between(5, 16))].copy()
    u['segment'] = np.where(u['game_type'] != 'REG', 'playoffs', np.where(u['week'] <= 4, 'weeks 1-4', 'weeks 17-18'))
    u['composite'] = defense_composite(u, seen)
    sd = float(defense_composite(seen, seen).std())

    print('=== U1: defense over-trust, games outside weeks 5-16 (%d) ===' % len(u))
    coef, t_stat, total_coef, total_t = controlled(u['composite'], u)
    u1 = coef > 0 and t_stat >= UNSEEN_T
    print('    TEST: composite given the closing and league totals: %+.3f points per unit, robust t %+.2f  %s' % (
        coef, t_stat, 'PASS' if u1 else 'fail'))
    print('      raw correlation %+.3f (t %+.2f); the closing total\'s own coefficient %+.3f (t %+.2f)' % (
        *corr_t(u['composite'], u['over_margin']), total_coef, total_t))
    for name, g in u.groupby('segment'):
        c, ct, _, _ = controlled(g['composite'], g)
        print('      %-12s %+.3f, t %+.2f (n=%d)' % (name, c, ct, len(g)))
    for name, lo, hi in ERAS:
        g = u[u['season'].between(lo, hi)]
        c, ct, _, _ = controlled(g['composite'], g)
        print('      %-12s %+.3f, t %+.2f (n=%d)' % (name, c, ct, len(g)))
    c, ct, tc, tt = controlled(defense_composite(seen, seen), seen)
    print('      for reference, weeks 5-16 where the stats were found: %+.3f, t %+.2f (closing total %+.3f, t %+.2f)' % (c, ct, tc, tt))
    for k in (1.0, 1.5):
        pooled('    bet its side, |composite| >= %.1f SD' % k, u, u['composite'].abs() >= k * sd, u['composite'] > 0)
        pooled('      the under side only (bad defenses)', u, u['composite'] <= -k * sd, pd.Series(False, index=u.index))
        pooled('      the over side only (good defenses)', u, u['composite'] >= k * sd, pd.Series(True, index=u.index))

    b = t[(t['game_type'] == 'REG') & (t['week'].between(2, 4) | (t['week'] >= 17)) & t['total'].notna() & t['total_line'].notna()]
    b = b.join(memory_and_backups(t), on='game_id')
    b = b[b['backups'].notna()]
    print('\n=== U2: backup quarterbacks lean under, regular-season weeks 2-4 and 17-18 (%d games, %d with a backup) ===' % (
        len(b), int((b['backups'] >= 1).sum())))
    coef, t_stat, _, _ = controlled(b['backups'], b)
    u2 = coef < 0 and t_stat <= -UNSEEN_T
    print('    TEST: backups given the closing and league totals: %+.3f points per backup, robust t %+.2f  %s' % (
        coef, t_stat, 'PASS' if u2 else 'fail'))
    print('      raw correlation %+.3f (t %+.2f)' % corr_t(b['backups'], b['over_margin']))
    for name, g in b.groupby(np.where(b['week'] <= 4, 'weeks 2-4', 'weeks 17-18')):
        c, ct, _, _ = controlled(g['backups'], g)
        print('      %-12s %+.3f, t %+.2f (n=%d)' % (name, c, ct, len(g)))
    bs = seen.join(memory_and_backups(t), on='game_id')
    bs = bs[bs['backups'].notna()]
    c, ct, _, _ = controlled(bs['backups'], bs)
    print('      for reference, weeks 5-16 where it was seen: %+.3f, t %+.2f' % (c, ct))
    pooled('    the under with a backup', b, b['backups'] >= 1, pd.Series(False, index=b.index))
    pooled('    the under with no backup', b, b['backups'] == 0, pd.Series(False, index=b.index))

    print('\n=== verdict (pre-registered) ===')
    print('    U1 defense over-trust out of sample: %s' % ('PASS' if u1 else 'fail'))
    print('    U2 backup quarterbacks under out of sample: %s' % ('PASS' if u2 else 'fail'))


if __name__ == '__main__':
    if 'fetch' in sys.argv[1:]:
        fetch()
    elif 'after' in sys.argv[1:]:
        after_checks()
    elif 'unseen' in sys.argv[1:]:
        unseen()
    else:
        main()
