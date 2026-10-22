"""Do the prop picks overrate players back from a long absence?  (Pre-registered October 21, 2026, before any outcome
was looked at.)

Why: the prop models (nfl_prop_model.py) build a player's recent and season-long features from his last games played,
counted in games, with no measure of time missed.  A player back from a long absence is forecast from his role before
it: Tank Dell (HOU), back in 2026 week 7 for his first game since December 2024, gets over picks from "6 targets a game
lately (18% of the team's)", his 2024 role.  Players coming back are often eased in, so the forecasts may run high; or
the books' lines may already be low enough, or too low (the owner's point: the rust may be priced in, or overdone).

Definitions
  missed     the number of games his current team played (regular season and playoffs, games.csv, by game day)
             strictly between the player's previous game with stats (any team; nflverse weekly player stats, offense)
             and this one.  A player's first game in the data (2006 on) has none and is left out.
  RETURNER   a regular-season game in week 2 or later in which the player had missed 3 or more of his team's games
             right before it.  Week 1 is left out: every player's features come from last season then, and an offseason
             of recovery is a different situation.
  CONTROL    every other regular-season game in week 2 or later (missed 0-2), same markets and players.
  Prop-sized players only: a forecast of at least 2 receptions (rec) or 20 receiving yards (rec_yds), about the players
  books post these props for.  Results below 0 count as 0, as in the models' training.
  Note: the weekly stats list a player only when he recorded a stat, so a game played without one counts as missed; the
  3-game bar keeps that noise small for prop-sized players.

H1 (the forecasts; the decisive test).  The early-week variant's out-of-sample forecasts (data_cache/
prop_forecasts_early_v1.csv, 2018-2025, each season forecast by models fitted on the 10 seasons before it; the variant
the over rule uses) in the receiving markets (rec, rec_yds: the over rule's biggest markets, and Tank Dell's).
  Calibration ratio = sum of results / sum of forecasts, per market, for returners and controls; gap = returners' ratio /
  controls' ratio.  Per-game error e = ln((result + 1) / (forecast + 1)).
  PASS (the forecasts overrate returners): gap <= 0.90 in both markets in both 2018-2021 and 2022-2025, and returners'
  mean e below the controls' with Welch t <= -3 (2018-2025, the two markets pooled).
H2 (the picks; descriptive).  The over rule's picks (early-week model, opening lines: ESPN BET 2024-2025 and DraftKings
2026 weeks 1-6, the rows nfl_research_props.overs_test scores), week 2 on, on returners against the rule's other picks:
win rates, counts, Fisher exact p.  Too few to decide alone.

Decision (fixed now).
  H1 PASS and returners' over picks win less often than the others: propose an automatic returner skip for over picks
  (no over pick for a player who missed 3+ of his team's games right before, week 2 on), for the owner's approval, like
  the QB skip.  With no graded returner picks at all, the decision follows H1 alone.
  H1 PASS but returners' picks win at least as often: the lines already price the absence; no change.
  H1 fails: the forecasts do not overrate returners; no change.
Reported for context, not decisive: every over market, missed 1-2 / 3-7 / 8+, week 1 returners, and this week's over
picks.

Results (run October 21, 2026, after the pre-registration above was hashed):
  H1 PASS.  Calibration ratio, returners vs controls: rec 0.814 vs 1.010 (2018-21, gap 0.806) and 0.745 vs 1.004
  (2022-25, gap 0.742); rec_yds 0.705 vs 0.987 (gap 0.715) and 0.728 vs 0.998 (gap 0.729).  Per-game error -0.730 (862)
  vs -0.285 (35,640), Welch t = -11.56.  By games missed (rec / rec_yds): none 1.013 / 1.001, 1-2 0.866 / 0.832, 3-7
  0.803 / 0.750, 8+ 0.701 / 0.597.  Week 1 returners 0.937 / 0.850.  Other over markets (returners / controls): passing
  0.96-0.99, rush_att 0.874, rush_yds 0.834, rush_rec_yds 0.760.
  H2: returners' over picks 28-35 (44.4%) vs the rule's other picks 757-513 (59.6%), Fisher p = 0.018 (discovery 24-32,
  test 4-3).
  This week (2026 week 7): Tank Dell's two overs (his team played 29 games since his last); every other over pick 0.
  DECISION: propose the returner skip; the owner approved it, and it was built into nfl_prop_signal / nfl_prod_props
  (README section 11.8; the production measure, nfl_prop_model.games_missed, equals this study's on 119,738 games).

Run from nfl/:  python nfl_research_props_returners.py
"""

import logging
import os

import numpy as np
import pandas as pd
from scipy import stats

import nfl_prop_model as pm
import nfl_prop_signal as ps
import nfl_props_sheet as props_sheet
import nfl_sheet as sheet
import nflverse_data as nd
from nfl_injury_features import TEAM_RENAMES

MIN_MISSED, FIRST_WEEK = 3, 2
PRIMARY = ('rec', 'rec_yds')
PROP_SIZED = {'rec': 2.0, 'rec_yds': 20.0}
ERAS = ((2018, 2021), (2022, 2025))
GAP_BAR, T_BAR = 0.90, -3.0
BUCKETS = ((0, 0, 'played his last team game'), (1, 2, 'missed 1-2'), (3, 7, 'missed 3-7'), (8, 10 ** 6, 'missed 8+'))
KEYS = ['season', 'week', 'game_id', 'player_id']


def missed_games(player_weeks, games, upcoming=None):
    """Player games (and `upcoming` ones: player_id, season, week, game_id, team) with `missed`, the games his current
    team played strictly between his previous game with stats and this one (NaN for his first game in the data)."""
    g = games.copy()
    for column in ('home_team', 'away_team'):
        g[column] = g[column].replace(TEAM_RENAMES)
    g['day'] = pd.to_datetime(g['gameday'])
    schedule = pd.concat([g[['game_id', 'day']].assign(team=g['home_team']),
                          g[['game_id', 'day']].assign(team=g['away_team'])])
    days = {team: np.sort(rows['day'].to_numpy()) for team, rows in schedule.groupby('team')}
    pw = player_weeks[player_weeks['position'].isin(pm.OFFENSE_POSITIONS)][
        ['player_id', 'season', 'week', 'season_type', 'game_id', 'team']].assign(upcoming=False)
    if upcoming is not None and len(upcoming):
        pw = pd.concat([pw, upcoming.assign(season_type='REG', upcoming=True)[list(pw.columns)]], ignore_index=True)
    pw['team'] = pw['team'].replace(TEAM_RENAMES)
    pw['day'] = pw['game_id'].map(g.set_index('game_id')['day'])
    pw = pw.dropna(subset=['day']).sort_values(['player_id', 'day']).reset_index(drop=True)
    pw['previous'] = pw.groupby('player_id')['day'].shift(1)
    missed = np.full(len(pw), np.nan)
    for team, rows in pw.groupby('team'):
        if team not in days:
            continue
        known = rows['previous'].notna().to_numpy()
        now = np.searchsorted(days[team], rows['day'].to_numpy(), 'left')
        before = np.searchsorted(days[team], rows['previous'].fillna(pd.Timestamp('1900-01-01')).to_numpy(), 'right')
        values = (now - before).astype(float)
        values[~known] = np.nan
        missed[rows.index.to_numpy()] = values
    pw['missed'] = missed
    return pw


def calibration(rows):
    forecast = rows['forecast'].sum()
    return rows['result'].sum() / forecast if forecast else np.nan


def h1(missed):
    """The decisive test: do the early-week forecasts overrate returners in the receiving markets?"""
    path = os.path.join(nd.CACHE_DIR, f'prop_forecasts_early_v{pm.MODEL_VERSION}.csv')
    f = pd.read_csv(path).merge(missed[KEYS + ['season_type', 'missed']], on=KEYS, how='inner')
    f = f[(f['season_type'] == 'REG') & f['missed'].notna() & f['result'].notna()].copy()
    f['result'] = f['result'].clip(lower=0)
    f['e'] = np.log((f['result'] + 1) / (f['forecast'] + 1))
    print(f"\n=== H1: early-week forecasts vs results, {f['season'].min()}-{f['season'].max()} (out of sample), "
          f"regular season week {FIRST_WEEK} on ===")
    print(f"{'market':10} {'era':9} {'returners':>9} {'ratio':>6} {'controls':>9} {'ratio':>6} {'gap':>6}")
    gaps, pooled = {}, []
    for market in PRIMARY:
        sub = f[(f['market'] == market) & (f['week'] >= FIRST_WEEK) & (f['forecast'] >= PROP_SIZED[market])]
        pooled.append(sub)
        for first, last in ERAS:
            era = sub[sub['season'].between(first, last)]
            ret, ctl = era[era['missed'] >= MIN_MISSED], era[era['missed'] < MIN_MISSED]
            gap = calibration(ret) / calibration(ctl)
            gaps[(market, first)] = gap
            print(f"{market:10} {first}-{str(last)[2:]} {len(ret):>9} {calibration(ret):>6.3f} {len(ctl):>9} "
                  f"{calibration(ctl):>6.3f} {gap:>6.3f}")
    both = pd.concat(pooled)
    ret, ctl = both[both['missed'] >= MIN_MISSED]['e'], both[both['missed'] < MIN_MISSED]['e']
    t = stats.ttest_ind(ret, ctl, equal_var=False).statistic
    print(f"per-game error ln((result+1)/(forecast+1)): returners {ret.mean():+.3f} ({len(ret)}), controls "
          f"{ctl.mean():+.3f} ({len(ctl)}), Welch t = {t:+.2f}")
    passed = all(gap <= GAP_BAR for gap in gaps.values()) and t <= T_BAR
    print(f"H1 {'PASS' if passed else 'FAIL'} (needs every gap <= {GAP_BAR} and t <= {T_BAR})")

    print("\nContext (not decisive): calibration ratio by games missed, 2018-2025, week 2 on, prop-sized players")
    for market in PRIMARY:
        sub = f[(f['market'] == market) & (f['week'] >= FIRST_WEEK) & (f['forecast'] >= PROP_SIZED[market])]
        cells = []
        for low, high, label in BUCKETS:
            rows = sub[sub['missed'].between(low, high)]
            cells.append(f"{label} {calibration(rows):.3f} ({len(rows)})")
        print(f"  {market:8} " + '; '.join(cells))
    print("Week 1 (left out of the test): returners vs the rest")
    for market in PRIMARY:
        sub = f[(f['market'] == market) & (f['week'] == 1) & (f['forecast'] >= PROP_SIZED[market])]
        ret, ctl = sub[sub['missed'] >= MIN_MISSED], sub[sub['missed'] < MIN_MISSED]
        print(f"  {market:8} returners {calibration(ret):.3f} ({len(ret)}), the rest {calibration(ctl):.3f} ({len(ctl)})")
    print("Other over markets (all forecasts, week 2 on): returners' ratio / controls' ratio")
    cells = []
    for market in ps.OVER_MARKETS:
        if market in PRIMARY:
            continue
        sub = f[(f['market'] == market) & (f['week'] >= FIRST_WEEK)]
        ret, ctl = sub[sub['missed'] >= MIN_MISSED], sub[sub['missed'] < MIN_MISSED]
        if len(ret):
            cells.append(f"{market} {calibration(ret) / calibration(ctl):.3f} ({len(ret)})")
    print('  ' + '; '.join(cells))
    return passed


def h2(missed):
    """Descriptive: the over rule's picks on returners against its other picks (week 2 on)."""
    import nfl_research_props as rp
    _, early = rp.load_early()
    _, _, lines = rp.load()
    rows = ps.score_history(lines, early, 'open')
    picks = ps.over_picks(rows)
    picks = picks.merge(missed[KEYS + ['missed']].rename(columns={'player_id': 'gsis_id'}),
                        on=['season', 'week', 'game_id', 'gsis_id'], how='left')
    picks = picks[(picks['week'] >= FIRST_WEEK) & picks['outcome'].isin(['W', 'L'])]
    returners = picks['missed'] >= MIN_MISSED
    print(f"\n=== H2: the over rule's picks, opening lines, week {FIRST_WEEK} on (descriptive) ===")
    table = []
    for label, mask in (('returners', returners), ('other picks', ~returners)):
        wins, losses = int((picks[mask]['outcome'] == 'W').sum()), int((picks[mask]['outcome'] == 'L').sum())
        table.append([wins, losses])
        rate = wins / (wins + losses) if wins + losses else float('nan')
        print(f"  {label:12} {wins}-{losses} ({rate:.1%})")
    for label, mask in (('discovery 2024-25', picks['season'] < 2026), ('test 2026', picks['season'] == 2026)):
        sub = picks[mask & returners]
        print(f"    returners, {label}: {int((sub['outcome'] == 'W').sum())}-{int((sub['outcome'] == 'L').sum())}")
    print(f"  Fisher exact p = {stats.fisher_exact(table)[1]:.3f}")
    if not sum(table[0]):
        return None
    return table[0][0] / sum(table[0]) < table[1][0] / max(1, sum(table[1]))


def this_week(weeks, games):
    """This week's over picks in the props workbook, with the games each player's team played since his last game."""
    log = props_sheet.read_log(props_sheet.workbook_path(sheet.get_current_season_year()))
    if not log:
        return
    week = max(r['week'] for r in log if isinstance(r.get('week'), int))
    rows = [r for r in log if r.get('week') == week and r.get('side') == 'Over']
    upcoming = pd.DataFrame({'player_id': [r['gsis_id'] for r in rows], 'season': [r['season'] for r in rows],
                             'week': week, 'game_id': [r['game_id'] for r in rows], 'team': [r['team'] for r in rows]})
    found = missed_games(weeks, games, upcoming.drop_duplicates())
    found = found[found['upcoming']].set_index(['player_id', 'game_id'])['missed']
    print(f"\nThis week's over picks (week {week}): games each player's team played since his last game")
    for r in rows:
        print(f"  {r['player']:22} {r['prop']:18} {r['pick']:12} missed {found.get((r['gsis_id'], r['game_id']), np.nan):g}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    weeks, games, _ = pm.load_inputs(sheet.get_current_season_year())
    missed = missed_games(weeks, games)
    passed = h1(missed)
    worse = h2(missed)
    this_week(weeks, games)
    if worse is None:
        verdict = ('propose a returner skip for over picks, for the owner\'s approval (no returner picks: H1 alone)'
                   if passed else 'no change: the forecasts do not overrate returners')
        picks_text = 'no graded returner picks'
    else:
        verdict = ('propose a returner skip for over picks, for the owner\'s approval' if passed and worse else
                   'no change: the lines already price the absence' if passed else
                   'no change: the forecasts do not overrate returners')
        picks_text = 'win less' if worse else 'win at least as often'
    print(f"\nDECISION (pre-registered): H1 {'PASS' if passed else 'FAIL'}, returners' picks {picks_text} -> {verdict}")
