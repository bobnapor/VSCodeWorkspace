"""New data, tested: does information the model does not use today improve the picks, for favorites as well as underdogs?

Stage 'open' (Sportsbook Reviews Online opening and closing lines, 2007-2021, nfl_market_history.py).  Pre-registered:
  1. Does the correction (trained against the close) predict how the line moves from open to close?  A positive
     correlation means the market later moves toward the model's side, so betting earlier is worth more than the
     closing-line backtests show.
  2. The same picks graded at the opening line instead of the close, by side (underdog / favorite at the open) and by size.
  3. A correction model trained against the opening line (target: result minus the opening spread), walk-forward on
     2007+ data, graded at the open.
  4. Is the close efficient with respect to its own movement (does open-to-close movement predict the cover at the close)?
Games with an implausible opening total (outside 30-70) or a move of more than 7 points are dropped as parse errors.

Stage 'qb' (nflverse player stats, 1999 on): a pre-game rating of each game's actual starting quarterback (EPA per play
over his career, decayed and shrunk toward replacement level), so a team's rating follows its quarterback rather than
lagging after a change.  Stage 'luck' (schedules, 1999 on): season-to-date win %, record luck (win % minus the Pythagorean
win % from points) and one-score-game record, the things the betting public reacts to.  Each block is added to the
production correction model and judged, before running, by one rule: it must raise the correlation of the correction with
the cover margin in all three eras (1999-2005 untouched, scored by a model fitted on 2006-2019; 2006-2017; 2018-2026), and
the underdog-side and favorite-side hit rates are reported separately.

Run it from this folder:  python nfl_research_data.py open qb luck
"""

import sys

import numpy as np
import pandas as pd

import nfl_lab as lab
import nfl_signal
from nfl_market_history import attach_open_lines

WEEKS = (5, 16)


def rate(win):
    return '%.3f (n=%d)' % (win.mean(), len(win)) if len(win) else '   -        '


def picks_table(frame, pred_col, line_col, label):
    """Hit rate of the side `pred_col` favors, graded against `line_col` (spread, positive = home favored), by side and size."""
    f = frame.copy()
    f['res'] = f['result'] - f[line_col]
    f = f[(f['res'] != 0) & (f[line_col] != 0)]
    home_dog = f[line_col] < 0
    toward_dog = np.where(home_dog, f[pred_col], -f[pred_col])
    win = np.where(f[pred_col] > 0, f['res'] > 0, f['res'] < 0)
    size = np.abs(f[pred_col])
    print('  %s' % label)
    for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 99)):
        b = (size >= lo) & (size < hi)
        print('    |pred| %.1f-%-4.1f underdog side %s | favorite side %s' % (
            lo, hi, rate(win[b & (toward_dog > 0)]), rate(win[b & (toward_dog < 0)])))
    print('    all: underdog side %s | favorite side %s | both %s' % (rate(win[toward_dog > 0]), rate(win[toward_dog < 0]), rate(win)))


def stage_open(table):
    t = attach_open_lines(table)
    t['sbr_move'] = t['sbr_close'] - t['sbr_open']          # > 0: the market moved toward the home team
    valid = t['sbr_open'].notna() & t['open_total'].between(30, 70) & (t['sbr_move'].abs() <= 7)
    print('games with SBR lines %d, kept %d after dropping parse errors' % (t['sbr_open'].notna().sum(), valid.sum()))

    fit = lab.ridge_margin(nfl_signal.SIGNAL_FEATURES, alpha=nfl_signal.SIGNAL_ALPHA, target='resid')
    fwd = lab.walk_forward(t, fit)
    f = fwd[fwd['game_id'].isin(t.loc[valid, 'game_id']) & fwd['week'].between(*WEEKS)].copy()
    f = f.merge(t[['game_id', 'sbr_open', 'sbr_close', 'sbr_move']], on='game_id', suffixes=('', '_x'))
    print('test games (2007-2021, weeks 5-16, walk-forward production correction): %d' % len(f))

    print('\n=== 1. does the correction predict the open-to-close move? ===')
    for name, g in (('2007-2014', f[f['season'] <= 2014]), ('2015-2021', f[f['season'] >= 2015]), ('all', f)):
        print('  %-10s corr(correction, move) %+.3f   mean move toward the model side %+.3f pts (n=%d)' % (
            name, np.corrcoef(g['pred'], g['sbr_move'])[0, 1], (np.sign(g['pred']) * g['sbr_move']).mean(), len(g)))

    print('\n=== 2. the production correction graded at the close vs at the open (sides defined at that line) ===')
    picks_table(f, 'pred', 'spread_line', 'graded at the close (nflverse)')
    picks_table(f, 'pred', 'sbr_open', 'graded at the open (SBR)')

    print('\n=== 3. a correction model trained against the opening line, graded at the open ===')
    o = t[valid].copy()
    o['resid_open'] = o['result'] - o['sbr_open']
    fit_open = lab.ridge_margin(nfl_signal.SIGNAL_FEATURES, alpha=nfl_signal.SIGNAL_ALPHA, target='resid_open')
    outs = []
    for season in range(2011, 2022):
        train = o[(o['season'] < season) & o['result'].notna() & (o['game_type'] == 'REG')]
        test = o[(o['season'] == season) & o['result'].notna() & (o['game_type'] == 'REG') & o['week'].between(*WEEKS)]
        if len(test):
            out = test.copy()
            out['pred_open'] = np.asarray(fit_open(train, test), dtype=float)
            outs.append(out)
    oo = pd.concat(outs)
    print('  corr(pred_open, result - open) %.3f vs production correction with result - open %.3f (2011-2021, n=%d)' % (
        np.corrcoef(oo['pred_open'], oo['resid_open'])[0, 1],
        np.corrcoef(f[f['season'] >= 2011]['pred'], f[f['season'] >= 2011]['result'] - f[f['season'] >= 2011]['sbr_open'])[0, 1], len(oo)))
    picks_table(oo, 'pred_open', 'sbr_open', 'open-trained model, graded at the open (2011-2021)')

    print('\n=== 4. is the close efficient with respect to its own movement? ===')
    g = f.assign(resid_close=f['result'] - f['spread_line'])
    g = g[g['resid_close'] != 0]
    print('  corr(move, result - close) %+.3f (n=%d)' % (np.corrcoef(g['sbr_move'], g['resid_close'])[0, 1], len(g)))
    for lo, hi, name in ((1.5, 99, 'moved 1.5+ toward home'), (-99, -1.5, 'moved 1.5+ toward away')):
        b = g[(g['sbr_move'] >= lo) & (g['sbr_move'] <= hi)]
        print('  %-24s home covers the close %.3f (n=%d)' % (name, (b['resid_close'] > 0).mean(), len(b)))
    for name, side in (('the team the line moved toward', 1), ('the team the line moved away from', -1)):
        b = g[g['sbr_move'] != 0]
        win = np.where(np.sign(b['sbr_move']) * side > 0, b['resid_close'] > 0, b['resid_close'] < 0)
        print('  betting %-34s at the close: %s' % (name, rate(win)))


# ---------------------------------------------------------------- shared evaluation of a feature block

def evaluate(table, columns, label, alpha=nfl_signal.SIGNAL_ALPHA, split=None):
    """Correlation with the cover margin and hit rates by side for a correction model on `columns`, in all three eras.

    `split` replaces the eras with walk-forward seasons only, as [(name, first season, last season)].
    """
    fit = lab.ridge_margin(columns, alpha=alpha, target='resid')
    fwd = lab.walk_forward(table, fit)
    if split:
        eras = tuple((name, fwd[fwd['season'].between(lo, hi)]) for name, lo, hi in split)
    else:
        played = table[table['result'].notna() & (table['game_type'] == 'REG')]
        train = played[played['season'].between(2006, 2019)]
        back = []
        for season in range(1999, 2006):
            test = lab.test_games(table, season)
            out = test.copy()
            out['pred'] = np.asarray(fit(train, test), dtype=float)
            back.append(out)
        eras = (('1999-2005', pd.concat(back)), ('2006-2017', fwd[fwd['season'] <= 2017]), ('2018-2026', fwd[fwd['season'] > 2017]))
    cells = []
    for name, f in eras:
        f = f[f['week'].between(*WEEKS) & (f['resid'] != 0) & (f['spread_line'] != 0)]
        win = np.where(f['pred'] > 0, f['resid'] > 0, f['resid'] < 0)
        toward_dog = np.where(f['spread_line'] < 0, f['pred'], -f['pred'])
        big = np.abs(f['pred']) >= 0.5
        cells.append('%s r=%.3f | dog %.3f fav %.3f | |c|>=.5 dog %.3f (%d) fav %.3f (%d)' % (
            name, np.corrcoef(f['pred'], f['resid'])[0, 1], win[toward_dog > 0].mean(), win[toward_dog < 0].mean(),
            win[big & (toward_dog > 0)].mean(), (big & (toward_dog > 0)).sum(), win[big & (toward_dog < 0)].mean(), (big & (toward_dog < 0)).sum()))
    print('%-34s %s' % (label, '\n%34s %s' % ('', '\n%34s ' % '').join(cells) if False else ' || '.join(cells)))


# ---------------------------------------------------------------- quarterback ratings

QB_DECAY = 0.95          # per game the quarterback plays; about a season's half-life
QB_SEASON_CARRY = 0.9    # quarterback skill carries over between seasons
QB_PRIOR_PLAYS = 200     # pseudo-plays at replacement level that shrink short careers
QB_REPLACEMENT_EPA = -0.10


def qb_ratings(first=1999, last=2026):
    """Pre-game EPA-per-play rating for every (quarterback id, season, week) he played, from nflverse player stats."""
    from nflverse_data import PLAYER_WEEK_URL, _cached_download

    columns = ['player_id', 'position', 'season', 'week', 'attempts', 'sacks_suffered', 'carries', 'passing_epa', 'rushing_epa']
    frames = []
    for season in range(first, last + 1):   # only the quarterback columns, so 28 seasons fit in a 32-bit process
        path = _cached_download(PLAYER_WEEK_URL.format(year=season), f'stats_player_week_{season}.csv')
        frame = pd.read_csv(path, usecols=lambda c: c in columns, low_memory=False)
        frames.append(frame[frame['position'] == 'QB'])
    qb = pd.concat(frames, ignore_index=True)
    qb['plays'] = qb['attempts'].fillna(0) + qb['sacks_suffered'].fillna(0) + qb['carries'].fillna(0)
    qb['epa'] = qb['passing_epa'].fillna(0) + qb['rushing_epa'].fillna(0)
    qb = qb[qb['plays'] > 0].sort_values(['player_id', 'season', 'week'])
    history = {}
    for pid, g in qb.groupby('player_id', sort=False):
        keys, ratings = [], []
        s_epa = s_plays = 0.0
        last_season = None
        for row in g.itertuples():
            if last_season is not None and row.season != last_season:
                s_epa, s_plays = s_epa * QB_SEASON_CARRY, s_plays * QB_SEASON_CARRY
            last_season = row.season
            s_epa = s_epa * QB_DECAY + row.epa
            s_plays = s_plays * QB_DECAY + row.plays
            keys.append(row.season * 100 + row.week)
            ratings.append((s_epa + QB_PRIOR_PLAYS * QB_REPLACEMENT_EPA) / (s_plays + QB_PRIOR_PLAYS))
        history[pid] = (np.array(keys), np.array(ratings))
    return history


def add_qb_features(table):
    """qb_home / qb_away: the starting quarterback's rating from his games before this one; qb_diff = home minus away."""
    history = qb_ratings(1999, int(table['season'].max()))
    t = table.copy()

    def lookup(pid, key):
        if not isinstance(pid, str) or pid not in history:
            return QB_REPLACEMENT_EPA
        keys, ratings = history[pid]
        i = np.searchsorted(keys, key) - 1          # last game strictly before this one
        return float(ratings[i]) if i >= 0 else QB_REPLACEMENT_EPA

    key = t['season'] * 100 + t['week']
    t['qb_home'] = [lookup(p, k) for p, k in zip(t['home_qb_id'], key)]
    t['qb_away'] = [lookup(p, k) for p, k in zip(t['away_qb_id'], key)]
    t['qb_diff'] = t['qb_home'] - t['qb_away']
    return t


def stage_qb(table):
    t = add_qb_features(table)
    reg = t[(t['game_type'] == 'REG') & t['result'].notna()]
    print('\n=== quarterback block: starter rating coverage %.3f; corr(qb_diff, result) %.3f, corr(qb_diff, cover margin) %.3f ===' % (
        reg['home_qb_id'].notna().mean(), np.corrcoef(reg['qb_diff'], reg['result'])[0, 1], np.corrcoef(reg['qb_diff'], reg['resid'])[0, 1]))
    base = nfl_signal.SIGNAL_FEATURES
    evaluate(t, base, 'production model')
    evaluate(t, base + ['qb_diff'], '+ starting-QB rating')
    evaluate(t, base + ['qb_home', 'qb_away'], '+ starting-QB ratings, each side')


# ---------------------------------------------------------------- record and luck (what the public sees)

def add_luck_features(table):
    """Season-to-date, before each game: shrunk win %, record luck (win % minus Pythagorean win %), one-score net wins."""
    t = table.copy()
    long = pd.concat([
        t[['game_id', 'season', 'gameday', 'week', 'game_type', 'home_team', 'home_score', 'away_score']]
        .rename(columns={'home_team': 'team', 'home_score': 'pf', 'away_score': 'pa'}),
        t[['game_id', 'season', 'gameday', 'week', 'game_type', 'away_team', 'away_score', 'home_score']]
        .rename(columns={'away_team': 'team', 'away_score': 'pf', 'home_score': 'pa'}),
    ]).sort_values(['team', 'season', 'gameday'])
    long = long[long['game_type'] == 'REG']
    played = long['pf'].notna()
    long['w'] = np.where(played, (long['pf'] > long['pa']) + 0.5 * (long['pf'] == long['pa']), 0.0)
    long['g'] = played.astype(float)
    long['close_net'] = np.where(played & ((long['pf'] - long['pa']).abs() <= 8), np.sign(long['pf'] - long['pa']), 0.0)
    long['pf0'], long['pa0'] = long['pf'].fillna(0), long['pa'].fillna(0)
    grp = long.groupby(['team', 'season'])
    for col in ('w', 'g', 'close_net', 'pf0', 'pa0'):
        long['pre_' + col] = grp[col].cumsum() - long[col]       # everything before this game
    long['win_pct'] = (long['pre_w'] + 2) / (long['pre_g'] + 4)
    pf, pa = long['pre_pf0'] + 22 * 2, long['pre_pa0'] + 22 * 2
    pyth = pf ** 2.37 / (pf ** 2.37 + pa ** 2.37)
    long['luck'] = long['win_pct'] - pyth
    feats = long[['game_id', 'team', 'win_pct', 'luck', 'pre_close_net']]
    for side in ('home', 'away'):
        t = t.merge(feats.rename(columns={'team': f'{side}_team', 'win_pct': f'{side}_win_pct', 'luck': f'{side}_luck',
                                          'pre_close_net': f'{side}_close_net'}), on=['game_id', f'{side}_team'], how='left')
    t['win_pct_diff'] = t['home_win_pct'] - t['away_win_pct']
    t['luck_diff'] = t['home_luck'] - t['away_luck']
    t['close_net_diff'] = t['home_close_net'] - t['away_close_net']
    return t


def stage_luck(table):
    t = add_luck_features(table)
    reg = t[(t['game_type'] == 'REG') & t['result'].notna() & t['week'].between(*WEEKS)]
    print('\n=== record and luck block: corr with the cover margin: win%% %.3f, luck %.3f, one-score net %.3f ===' % tuple(
        np.corrcoef(reg[c].fillna(0), reg['resid'])[0, 1] for c in ('win_pct_diff', 'luck_diff', 'close_net_diff')))
    base = nfl_signal.SIGNAL_FEATURES
    evaluate(t, base, 'production model')
    evaluate(t, base + ['win_pct_diff', 'luck_diff', 'close_net_diff'], '+ record, luck, one-score games')


# ---------------------------------------------------------------- pass rush and coverage (QB hits from 2006, passes defended from 1999)

def add_pressure_features(table):
    """Opponent-adjusted, recency-weighted ratings of QB hits per dropback and passes defended per attempt, both directions.

    o_qbh: hits the team's quarterbacks took per dropback (protection); d_qbh: the same for the opponents it faced (pass
    rush).  o_pd / d_pd: passes defended against the offense per attempt / by the defense.  QB hits are only recorded
    consistently from 2006, so earlier seasons are left missing (rated at the league average).
    """
    from nfl_injury_features import TEAM_RENAMES
    from nflverse_data import PLAYER_WEEK_URL, _cached_download

    cols = ['game_id', 'team', 'season', 'attempts', 'sacks_suffered', 'def_qb_hits', 'def_pass_defended']
    frames = []
    for season in range(1999, int(table['season'].max()) + 1):
        path = _cached_download(PLAYER_WEEK_URL.format(year=season), f'stats_player_week_{season}.csv')
        f = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
        f = f.groupby(['game_id', 'team', 'season'], as_index=False)[cols[3:]].sum(min_count=1)
        if season < 2006:
            f['def_qb_hits'] = np.nan
        frames.append(f)
    tg = pd.concat(frames, ignore_index=True)
    tg['team'] = tg['team'].replace(TEAM_RENAMES)
    opp = tg[['game_id', 'team', 'def_qb_hits', 'def_pass_defended']].rename(columns={'team': 'opp', 'def_qb_hits': 'hits_taken', 'def_pass_defended': 'pd_against'})
    tg = tg.merge(opp, on='game_id')
    tg = tg[tg['team'] != tg['opp']]
    dropbacks = tg['attempts'].fillna(0) + tg['sacks_suffered'].fillna(0)
    tg['o_qbh'] = np.where(dropbacks > 0, tg['hits_taken'] / dropbacks.where(dropbacks > 0), np.nan)
    tg['o_pd'] = np.where(tg['attempts'] > 0, tg['pd_against'] / tg['attempts'].where(tg['attempts'] > 0), np.nan)

    games = table[['game_id', 'season', 'week', 'gameday', 'home_team', 'away_team', 'result']]
    rows = pd.concat([games.rename(columns={'home_team': 'team', 'away_team': 'opp'}),
                      games.rename(columns={'away_team': 'team', 'home_team': 'opp'})], ignore_index=True)
    rows['played'] = rows['result'].notna()
    rows = rows.merge(tg[['game_id', 'team', 'o_qbh', 'o_pd']], on=['game_id', 'team'], how='left')
    allowed = rows[['game_id', 'team', 'o_qbh', 'o_pd']].rename(columns={'team': 'opp', 'o_qbh': 'd_qbh', 'o_pd': 'd_pd'})
    rows = rows.merge(allowed, on=['game_id', 'opp'], how='left')
    rows['played'] = rows['played'] & rows['o_pd'].notna()
    rated = lab.rate_teams(rows, [('o_qbh', 'd_qbh'), ('o_pd', 'd_pd')])
    t = table.copy()
    for side, prefix in (('home_team', 'h_'), ('away_team', 'a_')):
        t = t.merge(rated[['game_id', 'team', 'r_o_qbh', 'r_d_qbh', 'r_o_pd', 'r_d_pd']]
                    .rename(columns={'team': side, **{c: prefix + c for c in ('r_o_qbh', 'r_d_qbh', 'r_o_pd', 'r_d_pd')}}),
                    on=['game_id', side], how='left')
    for stat in ('qbh', 'pd'):
        t[f'off__o_{stat}'] = t[f'h_r_o_{stat}'] - t[f'a_r_o_{stat}']
        t[f'def__o_{stat}'] = t[f'h_r_d_{stat}'] - t[f'a_r_d_{stat}']
    return t


def stage_pressure(table):
    t = add_pressure_features(table)
    reg = t[(t['game_type'] == 'REG') & t['result'].notna() & t['week'].between(*WEEKS) & (t['season'] >= 2010)]
    print('\n=== pass rush and coverage block (2010 on): corr with the cover margin: protection %+.3f, pass rush %+.3f, '
          'passes defended against %+.3f, by %+.3f ===' % tuple(np.corrcoef(reg[c].fillna(0), reg['resid'])[0, 1]
                                                             for c in ('off__o_qbh', 'def__o_qbh', 'off__o_pd', 'def__o_pd')))
    base = nfl_signal.SIGNAL_FEATURES
    block = ['off__o_qbh', 'def__o_qbh', 'off__o_pd', 'def__o_pd']
    split = (('2010-2017', 2010, 2017), ('2018-2026', 2018, 2026))
    evaluate(t, base, 'production model', split=split)
    evaluate(t, base + block, '+ pass rush and coverage', split=split)


# ---------------------------------------------------------------- popularity (who the public bets), from the schedule

def add_popularity_features(table):
    """pop_home / pop_away: the team's primetime games (Thursday, Sunday night, Monday) in its season schedule, which the
    league sets before the season around the biggest draws; pop_diff = home minus away.  Known before every game."""
    t = table.copy()
    prime = (t['weekday'].isin(['Thursday', 'Monday'])
             | (pd.to_datetime(t['gametime'], format='%H:%M', errors='coerce').dt.hour >= 19)) & (t['game_type'] == 'REG')
    long = pd.concat([t.loc[prime, ['season', 'home_team']].rename(columns={'home_team': 'team'}),
                      t.loc[prime, ['season', 'away_team']].rename(columns={'away_team': 'team'})])
    counts = long.groupby(['season', 'team']).size()
    t['pop_home'] = [counts.get((s, h), 0) for s, h in zip(t['season'], t['home_team'])]
    t['pop_away'] = [counts.get((s, a), 0) for s, a in zip(t['season'], t['away_team'])]
    t['pop_diff'] = t['pop_home'] - t['pop_away']
    return t


def stage_popularity(table):
    """Pre-registered direction: books shade lines against popular teams, so the more popular team covers less."""
    t = add_popularity_features(table)
    f = t[(t['game_type'] == 'REG') & t['result'].notna() & t['week'].between(*WEEKS) & (t['resid'] != 0) & (t['spread_line'] != 0)].copy()
    print('\n=== popularity (primetime games scheduled) ===')
    eras = (('1999-2005', f['season'] <= 2005), ('2006-2017', f['season'].between(2006, 2017)), ('2018-2026', f['season'] >= 2018))
    for name, mask in eras:
        g = f[mask]
        print('  %s corr(pop_diff, cover margin) %+.3f (n=%d)' % (name, np.corrcoef(g['pop_diff'], g['resid'])[0, 1], len(g)))
    print('  betting against the more popular team, by gap and by which side that is:')
    home_more_popular = f['pop_diff'] > 0
    against_win = np.where(home_more_popular, f['resid'] < 0, f['resid'] > 0)     # the less popular side covered
    less_popular_is_dog = np.where(home_more_popular, f['spread_line'] > 0, f['spread_line'] < 0)
    for gap in (2, 3, 4):
        m = f['pop_diff'].abs() >= gap
        cells = []
        for name, emask in eras:
            for side, smask in (('fav', ~less_popular_is_dog), ('dog', less_popular_is_dog)):
                sel = (m & emask & smask).to_numpy()
                cells.append('%s %s %s' % (name[2:], side, rate(pd.Series(against_win[sel]))))
        print('    gap >= %d: %s' % (gap, ' | '.join(cells)))
    base = nfl_signal.SIGNAL_FEATURES
    evaluate(t, base, 'production model')
    evaluate(t, base + ['pop_diff'], '+ popularity')


# ---------------------------------------------------------------- rating memory, tuned for the correction (not the margin)

RATING_GRID = [(d, c) for d in (0.80, 0.90, 0.95) for c in (0.4, 0.6, 0.8)]


def stage_ratings(table):
    """Pre-registered: of the 9 (decay, season carry) settings, pick the best correlation on 2006-2017 only; adopt it only if
    it also beats production (decay 0.90, carry 0.6) on 1999-2005 and on 2018-2026."""
    print('\n=== rating memory grid (decay per game, season carry): correlation with the cover margin, weeks 5-16 ===')
    fit = lab.ridge_margin(nfl_signal.SIGNAL_FEATURES, alpha=nfl_signal.SIGNAL_ALPHA, target='resid')
    results = {}
    for decay, carry in RATING_GRID:
        t = table if (decay, carry) == (0.90, 0.6) else lab.build_table(lab.ALL_PAIRS, {'decay': decay, 'carry': carry})
        fwd = lab.walk_forward(t, fit)
        played = t[t['result'].notna() & (t['game_type'] == 'REG')]
        train = played[played['season'].between(2006, 2019)]
        back = []
        for season in range(1999, 2006):
            test = lab.test_games(t, season)
            out = test.copy()
            out['pred'] = np.asarray(fit(train, test), dtype=float)
            back.append(out)
        r = {}
        for name, f in (('1999-2005', pd.concat(back)), ('2006-2017', fwd[fwd['season'] <= 2017]), ('2018-2026', fwd[fwd['season'] > 2017])):
            f = f[f['week'].between(*WEEKS)]
            r[name] = np.corrcoef(f['pred'], f['resid'])[0, 1]
        results[(decay, carry)] = r
        print('  decay %.2f carry %.1f: 1999-2005 r=%.3f | 2006-2017 r=%.3f | 2018-2026 r=%.3f%s' % (
            decay, carry, r['1999-2005'], r['2006-2017'], r['2018-2026'], '   <- production' if (decay, carry) == (0.90, 0.6) else ''))
        del t
    best = max(results, key=lambda k: results[k]['2006-2017'])
    prod = results[(0.90, 0.6)]
    ok = all(results[best][e] > prod[e] for e in ('1999-2005', '2018-2026'))
    print('  chosen on 2006-2017: decay %.2f carry %.1f -> %s' % (best[0], best[1], 'ADOPT (beats production in both other eras)'
                                                                 if ok and best != (0.90, 0.6) else 'keep production'))


def stage_early(table):
    """The production pick rule decided and graded at the opening line versus at the close, 2007-2021 (what betting
    Monday instead of at kickoff would have done), for the underdog picks and for the favorites the correction likes."""
    t = attach_open_lines(table)
    t['sbr_move'] = t['sbr_close'] - t['sbr_open']
    valid = t['sbr_open'].notna() & t['open_total'].between(30, 70) & (t['sbr_move'].abs() <= 7)
    fit = lab.ridge_margin(nfl_signal.SIGNAL_FEATURES, alpha=nfl_signal.SIGNAL_ALPHA, target='resid')
    fwd = lab.walk_forward(t, fit)
    f = fwd[fwd['game_id'].isin(t.loc[valid, 'game_id']) & fwd['week'].between(*WEEKS)].copy()
    print('\n=== betting at the open vs at the close (rule decided at that line, graded at that line) ===')
    for era, mask in (('2007-2014', f['season'] <= 2014), ('2015-2021', f['season'] >= 2015)):
        g = f[mask]
        cells = []
        for name, line in (('close', 'spread_line'), ('open', 'sbr_open')):
            res = g['result'] - g[line]
            keep = (res != 0) & (g[line] != 0)
            dog_home = g[line] < 0
            toward_dog = np.where(dog_home, g['pred'], -g['pred'])
            win = np.where(g['pred'] > 0, res > 0, res < 0)
            dog_pick = keep & (toward_dog > 0)
            fav_pick = keep & (toward_dog < -0.5)
            cells.append('at %s: underdog picks %s, favorites (correction >= 0.5 toward them) %s' % (
                name, rate(pd.Series(win[dog_pick])), rate(pd.Series(win[fav_pick]))))
        print('  %s %s\n            %s' % (era, cells[0], cells[1]))


if __name__ == '__main__':
    table = lab.default_table()
    stages = sys.argv[1:] or ['open']
    for stage in stages:
        {'open': stage_open, 'qb': stage_qb, 'luck': stage_luck, 'pressure': stage_pressure, 'popularity': stage_popularity,
         'ratings': stage_ratings, 'early': stage_early}[stage](table)
