"""A totals (over/under) model: does what the ratings say about scoring beat the closing total?

The spread work found the market prices team quality well but misprices pace and leans on noisy stats.  Totals are a
separate market this project had never examined, so every era is clean for testing.  Pre-registered before any totals
result below was computed (October 20, 2026); nothing in production changes either way.

Frame: regular-season games, weeks 5-16, with a closing total.  The over margin is total points minus the closing total
(positive = over; a push counts as 0 in correlations and is dropped from hit rates).  Eras 1999-2005 / 2006-2017 /
2018-2026: 2006 on walk-forward (the 14 seasons before), 1999-2005 scored by a fit on 2006-2019, as the spread model.

The totals correction: a ridge regression (alpha 10000, standardized features, as the spread correction) of the over
margin on what predicts scoring.  For each of the 37 rated stats, both offenses' ratings added together and both
defenses' allowed ratings added together (74 features: sums matter for a total the way differences matter for a
margin), plus a dome, the wind and temperature outdoors (none and 70 degrees under a roof), a night game, a divisional
game, a neutral site and the two teams' rest days added together.  Weather is the recorded game-time weather, which a
bet at the close would see in the forecast.

T1 (information): the correction's correlation with the over margin > 0 in every era, and a pooled robust t >= 2.5.
T2 (the rule, primary): every under the correction favors, the analog of the underdog rule (if casual money leans to
   overs as it leans to favorites, unders carry the tailwind).  Pass: >= 53.5% pooled and >= 52.4% in every era.  Every
   over the correction favors is reported beside it and judged by the same bar.
T3 (classic angles, no model; each passes if the under hits >= 52.4% in every era): night games (kickoff 7 pm Eastern
   or later; 1999 Monday games count as night), wind of 15 mph or more outdoors, divisional games, and closing totals at
   least 10% above the season-to-date league average.  Every under, no model, is the baseline.
A pass makes a rule a candidate for forward tracking only.

Run it from this folder:  python nfl_research_totals.py         (T1-T3)
                          python nfl_research_totals.py wind    (diagnostics after T3 passed, no decisions)

Results (October 20, 2026; eras 1999-2005 / 2006-2017 / 2018-2026, weeks 5-16, 4,791 games):
The closing total is close to unbiased: every under hit 49.7% / 50.0% / 51.5% (the mean over margin is positive,
+1.18 / +0.60 / +0.41, from high-scoring games).
T1 fails: the correction's correlation with the over margin is -0.004 / +0.050 / +0.084 (pooled t +3.20).  Its team-
rating sums carry nothing on 1999-2005 (-0.031 / +0.028 / +0.045); its weather and context features carry what it has
(+0.063 / +0.061 / +0.053, a diagnostic, not a pre-registered test).
T2 fails: every under the correction favors went 47.2% / 50.7% / 58.1% (72, 75 and 501 games: before 2018 the model
leaned over almost everywhere); every over it favors 50.1% / 50.0% / 52.1%.
T3: wind of 15 mph or more outdoors PASSES: the under hit 57.1% (92-69) / 55.9% (123-97) / 58.5% (69-49), 56.9% pooled
over 499 games (about 18 a season), +9.4% and +13.3% ROI at the real closing odds since 2006.  Night games (52.3% /
49.5% / 54.8%), divisional games (50.4% / 54.0% / 50.4%) and totals 10% above the league average (51.1% / 53.8% /
55.1%) fail.
Wind diagnostics: the under rises with the wind in every era (0-5 mph 47.1%, 5-10 50.0%, 10-15 54.8%, 15-20 56.7%,
20+ 55.8%).  Against the season's league average the market lowers the total by 0.4 / 0.5 / 1.0 / 1.5 / 2.3 points
across those bands while actual totals run +1.0 / +0.2 / -1.2 / -2.9 / -4.7, so it prices about half the wind's effect.
It holds in cold and mild weather (57.8% below 40 degrees, 56.7% above), at low and high totals (56.1%, 58.6%) and in
18 of 27 seasons; cold but calm games go over (46.5% under).
Caveat: the wind is the recorded kickoff wind, not the forecast a bet would use.  Games that turned windier than
forecast flatter the result, so it has to hold on forecasts before it is bet.

V1, the wind under on forecasts (pre-registered after the diagnostics above and before any forecast was fetched).
Open-Meteo's historical forecast archive (free; weather-model forecasts stitched from their first hours, so about what a
forecast checked shortly before kickoff said; available from 2018) gives the wind at 10 m in mph at the home stadium
(nfl_lab_extra.TEAM_SITES; the Oakland Coliseum for the 2018-2019 Raiders), averaged over the kickoff hour and the
three hours after.  Games: outdoor regular-season games, weeks 5-16, 2018 on, not at neutral sites.  Pass: the under
hits >= 52.4% at the closing total when that forecast wind is 15 mph or more.  Also shown: the under by forecast wind
band, how forecast and recorded wind compare, and the recorded-wind rule on the same games.
Run it:  python nfl_research_totals.py forecast   (forecasts are cached in data_cache as wind_forecast_utc_<team>_<season>.csv)

V1 result: PASS, narrowly.  961 outdoor games since 2018 have a forecast; forecast and recorded wind correlate 0.73 (the
forecast's 10 m wind averages 7.3 mph against 8.6 recorded, so 15 mph on the forecast flags fewer games).  Forecast wind
of 15 mph or more: the under went 25-22 (53.2%, +3.0% ROI at the real closing odds; 2018 1-1, 2019 4-3, 2020 4-7, 2021
6-3, 2022 4-2, 2023 1-2, 2024 1-1, 2025 4-3); the recorded-wind rule on the same games went 69-48 (59.0%).  By forecast
band (diagnostic): 0-5 mph 45.7% (calm games go over), 5-10 mph 55.3%, 10-15 mph 60.4% (159 games, +16.9%), 15-20 mph
50.0% (40 games), 20+ 5-2.  The forecast sorts games the way the recorded wind does (the under rises from calm to 10-15
mph), so the edge is not an artifact of weather surprises, but 47 games is a small sample: the rule is a forward-
tracking candidate, with 10-15 mph forecasts (seen here, not pre-registered) tracked beside it.
Correction (October 20, 2026, same day): the first run read every November-January game's forecast an hour early
(Open-Meteo labels local times with the time zone's current UTC offset for every date, so EST dates were read at EDT)
and reported 28-22 (56.0%) on 50 games.  Forecasts are now requested in UTC and kickoffs converted with their own date's
daylight-saving rule; the cache was converted (+4 hours) and checked against a fresh UTC download.  The numbers above
are the corrected run.
"""

import math

import numpy as np
import pandas as pd

import nfl_lab as lab
import nfl_signal
from nfl_research_favorites import league_total

WEEKS = nfl_signal.PICK_WEEKS
ERAS = (('1999-2005', 1999, 2005), ('2006-2017', 2006, 2017), ('2018-2026', 2018, 2026))
BACKWARD_TRAIN = nfl_signal.BACKWARD_TRAIN
ALPHA = nfl_signal.SIGNAL_ALPHA
SUM_FEATURES = [f'tot_off__{own}' for own, _ in lab.ALL_PAIRS] + [f'tot_def__{own}' for own, _ in lab.ALL_PAIRS]
CONTEXT = ['dome', 'wind_out', 'temp_out', 'night', 'div_game', 'neutral', 'rest_sum']
FEATURES = SUM_FEATURES + CONTEXT
BREAK_EVEN = 0.5238
WIND_LIMIT = 15.0
HIGH_TOTAL = 1.10
SIZE_BINS = (0.0, 0.25, 0.5, 1.0, 99.0)


def totals_table(table):
    """The research table plus the totals features and the over margin."""
    t = table.copy()
    for own, allowed in lab.ALL_PAIRS:
        t[f'tot_off__{own}'] = t[f'h_r_{own}'] + t[f'a_r_{own}']
        t[f'tot_def__{own}'] = t[f'h_r_{allowed}'] + t[f'a_r_{allowed}']
    outdoors = t['dome'] == 0
    t['wind_out'] = np.where(outdoors, t['wind'], 0.0)
    t['temp_out'] = np.where(outdoors, t['temp'], 70.0)
    t['night'] = [int(nfl_signal.is_night_game(g, d)) for g, d in zip(t['gametime'], t['weekday'])]
    t['rest_sum'] = t['home_rest'].fillna(7).clip(4, 14) + t['away_rest'].fillna(7).clip(4, 14)
    t['over_margin'] = t['total'] - t['total_line']
    t['league_total'] = league_total(t)
    return t


def scored(t, columns, alpha=ALPHA):
    """Out-of-sample totals correction, weeks 5-16: walk-forward from 2006, 1999-2005 from a 2006-2019 fit."""
    fit = lab.ridge_margin(columns, alpha=alpha, target='over_margin')
    forward = lab.walk_forward(t, fit)
    played = t[t['result'].notna() & (t['game_type'] == 'REG')]
    train = played[played['season'].between(*BACKWARD_TRAIN)]
    backward = [lab.test_games(t, s).assign(pred=lambda g: np.asarray(fit(train, g), float)) for s in range(1999, 2006)]
    f = pd.concat(backward + [forward], ignore_index=True)
    return f[f['week'].between(*WEEKS) & f['total_line'].notna() & f['total'].notna()]


def eras(f):
    return [(name, f[f['season'].between(lo, hi)]) for name, lo, hi in ERAS]


def slope_t(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    slope = (x * y).sum() / (x ** 2).sum()
    e = y - slope * x
    return slope, slope / (math.sqrt(((x ** 2) * (e ** 2)).sum()) / (x ** 2).sum())


def bets(g, bet_over):
    """(wins, losses, hit rate, ROI at the real closing odds or None) of betting over where bet_over, else under."""
    d = g[g['over_margin'] != 0]
    over = bet_over.loc[d.index].to_numpy(bool)
    win = np.where(over, d['over_margin'] > 0, d['over_margin'] < 0)
    w, n = int(win.sum()), len(win)
    odds = pd.to_numeric(pd.Series(np.where(over, d['over_odds'], d['under_odds']), index=d.index), errors='coerce')
    priced = odds.notna().to_numpy()
    roi = None
    if priced.any():
        o = odds.to_numpy(float)[priced]
        roi = float(np.where(win[priced], np.where(o < 0, 100.0 / -o, o / 100.0), -1.0).mean())
    return w, n - w, (w / n if n else float('nan')), roi


def line(label, f, mask, bet_over):
    """One rule's record by era and pooled; returns ([era rates], pooled rate)."""
    cells, rates = [], []
    for name, g in eras(f) + [('pooled', f)]:
        w, l, rate, roi = bets(g[mask.loc[g.index]], bet_over)
        cells.append('%s %s' % (name, '%d-%d %.3f%s' % (w, l, rate, '' if roi is None else ' (%+.1f%%)' % (100 * roi)) if w + l else 'none'))
        rates.append(rate)
    print('    %-44s %s' % (label, ' | '.join(cells)))
    return rates[:-1], rates[-1]


def main():
    t = totals_table(lab.default_table())
    f = scored(t, FEATURES)
    under = pd.Series(False, index=f.index)
    every = pd.Series(True, index=f.index)
    print('totals frame: %d games, weeks %d-%d, %d-%d; mean over margin by era %s' % (
        len(f), *WEEKS, f['season'].min(), f['season'].max(), ' / '.join('%+.2f' % g['over_margin'].mean() for _, g in eras(f))))

    print('\n=== T1: information, correlation of the totals correction with the over margin ===')
    infos = [float(np.corrcoef(g['pred'], g['over_margin'])[0, 1]) for _, g in eras(f)]
    slope, t_stat = slope_t(f['pred'], f['over_margin'])
    print('    %s | pooled slope %+.2f, robust t %+.2f' % (' | '.join('%s %+.3f' % (name, v) for (name, _, _), v in zip(ERAS, infos)), slope, t_stat))
    for label, columns in (('diagnostic: the 74 rating sums alone', SUM_FEATURES), ('diagnostic: weather and context alone', CONTEXT)):
        g = scored(t, columns)
        print('    %-40s %s' % (label, ' | '.join('%s %+.3f' % (name, np.corrcoef(e['pred'], e['over_margin'])[0, 1]) for name, e in eras(g))))
    t1 = all(v > 0 for v in infos) and t_stat >= 2.5

    print('\n=== T2: the rule, weeks %d-%d (hit rate, ROI at the real closing odds where known; break-even 52.4%%) ===' % WEEKS)
    lean_under, lean_over = f['pred'] < 0, f['pred'] > 0
    u_eras, u_pooled = line('RULE: every under the correction favors', f, lean_under, under)
    o_eras, o_pooled = line('every over the correction favors', f, lean_over, every)
    line('the correction\'s side, every game', f, every, f['pred'] > 0)
    print('    by size of the lean:')
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        size = f['pred'].abs().between(lo, hi, inclusive='left')
        line('  under, lean %.2f-%s' % (lo, '' if hi > 90 else '%.2f' % hi), f, lean_under & size, under)
        line('  over, lean %.2f-%s' % (lo, '' if hi > 90 else '%.2f' % hi), f, lean_over & size, every)
    t2_under = u_pooled >= 0.535 and all(r >= BREAK_EVEN for r in u_eras)
    t2_over = o_pooled >= 0.535 and all(r >= BREAK_EVEN for r in o_eras)

    print('\n=== T3: classic angles, no model (the under) ===')
    angles = {
        'every under (baseline)': every,
        'night games': f['night'] == 1,
        'wind %g mph or more, outdoors' % WIND_LIMIT: f['wind_out'] >= WIND_LIMIT,
        'divisional games': f['div_game'] == 1,
        'total >= %.0f%% of the league average' % (100 * HIGH_TOTAL): f['total_line'] >= HIGH_TOTAL * f['league_total'],
    }
    t3 = {}
    for label, mask in angles.items():
        rates, pooled = line(label, f, mask, under)
        if label != 'every under (baseline)':
            t3[label] = all(r >= BREAK_EVEN for r in rates)

    print('\n=== verdict (pre-registered) ===')
    print('    T1 information: %s' % ('PASS' if t1 else 'fail'))
    print('    T2 every under the correction favors: %s (pooled %.3f) | every over it favors: %s (pooled %.3f)' % (
        'PASS' if t2_under else 'fail', u_pooled, 'PASS' if t2_over else 'fail', o_pooled))
    print('    T3 ' + ' | '.join('%s: %s' % (label, 'PASS' if ok else 'fail') for label, ok in t3.items()))


def wind_checks():
    """After T3 passed (diagnostics, no decisions): is the wind under a market under-adjustment, and how steady is it?"""
    t = totals_table(lab.default_table())
    f = t[(t['game_type'] == 'REG') & t['week'].between(*WEEKS) & t['total'].notna() & t['total_line'].notna() & (t['dome'] == 0)].copy()
    f['wind_band'] = pd.cut(f['wind_out'], [-0.1, 5, 10, 15, 20, 99], labels=['0-5', '5-10', '10-15', '15-20', '20+'])
    under = pd.Series(False, index=f.index)
    known = f['wind_out'].notna()
    print('outdoor games, weeks %d-%d: %d, wind recorded for %d; the under by wind (mph):' % (*WEEKS, len(f), int(known.sum())))
    for band, g in f[known].groupby('wind_band', observed=True):
        line('wind %s' % band, g, pd.Series(True, index=g.index), under)
    print('\nhow far the market moves the total and how far scoring moves, against the season-to-date league average:')
    for band, g in f[known].groupby('wind_band', observed=True):
        print('    wind %-6s closing total %+.2f, actual total %+.2f, over margin %+.2f (n=%d)' % (
            band, (g['total_line'] - g['league_total']).mean(), (g['total'] - g['league_total']).mean(), g['over_margin'].mean(), len(g)))
    windy = f[f['wind_out'] >= WIND_LIMIT]
    print('\nwind %g mph or more, by temperature and by total:' % WIND_LIMIT)
    for label, mask in (('below 40 degrees', windy['temp_out'] < 40), ('40 degrees or warmer', windy['temp_out'] >= 40),
                        ('total under the league average', windy['total_line'] < windy['league_total']),
                        ('total at or above the league average', windy['total_line'] >= windy['league_total'])):
        line(label, windy, mask, pd.Series(False, index=windy.index))
    cold = f[(f['temp_out'] < 40) & (f['wind_out'] < 10)]
    line('cold (below 40) but calm (under 10 mph)', cold, pd.Series(True, index=cold.index), pd.Series(False, index=cold.index))
    print('\nwind %g mph or more, the under by season:' % WIND_LIMIT)
    d = windy[windy['over_margin'] != 0].assign(win=lambda x: x['over_margin'] < 0)
    by = d.groupby('season')['win'].agg(['sum', 'size'])
    print('    ' + '  '.join('%d: %d-%d' % (s, w, n - w) for s, (w, n) in by.iterrows()))
    print('    seasons with the under above 52.4%%: %d of %d' % (int(((by['sum'] / by['size']) > BREAK_EVEN).sum()), len(by)))


FORECAST_URL = 'https://historical-forecast-api.open-meteo.com/v1/forecast'
FORECAST_FIRST = 2018
OAKLAND_COLISEUM = (37.75, -122.20)
GAME_HOURS = 4   # the kickoff hour and the three after


def stadium(team, season):
    """(latitude, longitude) of a team's home stadium in a season."""
    from nfl_lab_extra import TEAM_SITES
    if team == 'LV' and season <= 2019:
        return OAKLAND_COLISEUM
    return TEAM_SITES[team][:2]


def season_forecast(team, season):
    """Hourly forecast wind (mph, UTC) at a team's home stadium for a season, cached in data_cache.  Requested in UTC
    because Open-Meteo labels local times with the time zone's current UTC offset for every date, ignoring the
    daylight-saving change of the date itself."""
    import os
    from datetime import date, timedelta

    import requests
    from nflverse_data import CACHE_DIR
    path = os.path.join(CACHE_DIR, f'wind_forecast_utc_{team}_{season}.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    lat, lon = stadium(team, season)
    end = min(date(season + 1, 1, 15), date.today() - timedelta(days=1))
    response = requests.get(FORECAST_URL, timeout=60, params={
        'latitude': lat, 'longitude': lon, 'start_date': f'{season}-09-01', 'end_date': end.isoformat(),
        'hourly': 'wind_speed_10m', 'wind_speed_unit': 'mph', 'timezone': 'GMT'})
    response.raise_for_status()
    hourly = response.json()['hourly']
    frame = pd.DataFrame({'time': hourly['time'], 'wind': hourly['wind_speed_10m']})
    frame.to_csv(path, index=False)
    return frame


def forecast_check():
    """V1: the wind under on archived forecasts, outdoor games 2018 on (pre-registered in the docstring)."""
    t = totals_table(lab.default_table())
    f = t[(t['game_type'] == 'REG') & t['week'].between(*WEEKS) & t['total'].notna() & t['total_line'].notna()
          & (t['dome'] == 0) & (t['neutral'] == 0) & (t['season'] >= FORECAST_FIRST) & t['gametime'].notna()].copy()
    kickoff = (pd.to_datetime(f['gameday'] + ' ' + f['gametime']).dt.tz_localize('America/New_York')
               .dt.tz_convert('UTC').dt.tz_localize(None).dt.floor('H'))
    winds = []
    for (team, season), games in f.groupby(['home_team', 'season']):
        hourly = season_forecast(team, int(season)).assign(time=lambda h: pd.to_datetime(h['time'])).set_index('time')['wind']
        for index in games.index:
            window = pd.date_range(kickoff[index], periods=GAME_HOURS, freq='H')
            winds.append((index, hourly.reindex(window).mean()))
    f['forecast_wind'] = pd.Series(dict(winds))
    f = f[f['forecast_wind'].notna()]
    both = f[f['wind_out'].notna()]
    print('outdoor games, weeks %d-%d, %d on: %d with a forecast; forecast vs recorded wind: correlation %.2f, mean %.1f vs %.1f mph' % (
        *WEEKS, FORECAST_FIRST, len(f), np.corrcoef(both['forecast_wind'], both['wind_out'])[0, 1], both['forecast_wind'].mean(), both['wind_out'].mean()))
    under = pd.Series(False, index=f.index)
    every = pd.Series(True, index=f.index)
    print('\nthe under by forecast wind (mph), at the closing total:')
    band = pd.cut(f['forecast_wind'], [-0.1, 5, 10, 15, 20, 99], labels=['0-5', '5-10', '10-15', '15-20', '20+'])
    for name, g in f.groupby(band, observed=True):
        w, l, rate, roi = bets(g, under)
        print('    forecast %-6s %d-%d %.3f%s' % (name, w, l, rate, '' if roi is None else ' (ROI %+.1f%%)' % (100 * roi)))
    windy = f['forecast_wind'] >= WIND_LIMIT
    w, l, rate, roi = bets(f[windy], under)
    print('\nV1 RULE, forecast wind %g mph or more: %d-%d %.3f%s' % (WIND_LIMIT, w, l, rate, '' if roi is None else ' (ROI %+.1f%%)' % (100 * roi)))
    by = f[windy & (f['over_margin'] != 0)].assign(win=lambda d: d['over_margin'] < 0).groupby('season')['win'].agg(['sum', 'size'])
    print('    by season: ' + '  '.join('%d: %d-%d' % (s, x, n - x) for s, (x, n) in by.iterrows()))
    rw, rl, rrate, _ = bets(f[f['wind_out'] >= WIND_LIMIT], under)
    print('    the recorded-wind rule on the same games: %d-%d %.3f; games flagged by both %d, forecast only %d, recorded only %d' % (
        rw, rl, rrate, int((windy & (f['wind_out'] >= WIND_LIMIT)).sum()), int((windy & ~(f['wind_out'] >= WIND_LIMIT)).sum()),
        int((~windy & (f['wind_out'] >= WIND_LIMIT)).sum())))
    print('V1 verdict (pre-registered): %s' % ('PASS' if rate >= BREAK_EVEN else 'fail'))


if __name__ == '__main__':
    import sys

    if 'wind' in sys.argv[1:]:
        wind_checks()
    elif 'forecast' in sys.argv[1:]:
        forecast_check()
    else:
        main()
