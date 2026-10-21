"""Quarterback swings: back the injured starter out of a team's ratings and put the actual starter in (pre-registered).

The concern (the owner, October 21, 2026): the pick model rates teams from their recent games, so when the usual
starter is hurt it still sees the team he played for.  The line moves at once (Ravens -3 to -1.5 when Lamar Jackson
missed practice), so the gap between stale ratings and the new line can look like an edge, and the model's view of
that game may be worthless without a manual check.

The QB swing.  Every quarterback gets the rating the projection model already uses (nfl_injury_features.build_qb_quality:
passing EPA per dropback, older games decayed 0.97 each, shrunk toward replacement level, -0.08, by 150 dropbacks; a
quarterback with no history is at replacement level).  A team's ratings (nfl_lab.rate_teams) are a weighted average of
its earlier games: each game's weight decays 0.90 per later game and everything is multiplied by 0.6 at a new season
(plus 3 games of league average, left out here: that shrinkage applies to every team whoever starts, and the
quarterback's own level was tested before, adding nothing).  With exactly those weights over the starters of those
games, every team carries an embedded quarterback: the weighted average of its starters' ratings (all ratings as of
the day of the game being predicted).  The swing is today's starter's rating minus the embedded one: about 0 for a
team whose starter played the games behind its ratings, about -0.15 to -0.25 EPA per dropback the week a star is
replaced by a backup, and nonzero after an offseason change of starter too.  Backing the starter's stats out and putting the backup's
estimate in would shift every quarterback-driven rating by a multiple of the swing, and because the pick model is
linear in its ratings, the whole adjustment comes down to a multiple of the swing; adding the swing (home minus away)
as a feature lets the cover margin choose that multiple.

Data: the research table (lab.default_table()), nflverse player stats 1999 on (quarterback ratings) and the schedule's
starter for every game, which is the actual starter (known by kickoff in nearly every game, so this is the best case;
in production the starter is a projection).  Frame: regular season, weeks 5-16, closing spread, eras 1999-2005 /
2006-2017 / 2018-2026, scored exactly as production (nfl_signal.scored_history: 2006 on walk-forward on the 14 seasons
before, 1999-2005 by a fit on 2006-2019).  Pushes drop out of hit rates.  A downgrade is a swing of -0.05 or worse
(a backup in for a starter, or a clearly worse starter).

Q1 (the fix, primary).  The production pick model plus one feature, qb_swing_diff (home swing minus away swing),
same ridge, same alpha, same scoring.  Pass, the bar every new feature in this project has had to meet: its
correction's correlation with the cover margin is higher than production's in every era; and its underdog picks
(the production rule: any lean to the underdog) hit at least as often as production's in at least two of the three
eras.  A pass makes it a candidate for production (which would then need the projected starter at bet time).
Q2 (the concern).  Production's underdog picks on a team with a downgrade (the picked underdog's swing <= -0.05),
against one (the favorite's), and on neither.  The concern is confirmed when picks on a downgraded team hit below
52.4% in at least two eras AND below the picks with no downgrade on either side in every era.  If confirmed and Q1
fails, the objective fix is to skip such picks; if not confirmed, the picks stand as made, the injury already being
in the line.
Diagnostics (no decisions): how often each situation comes up; whether production's correction leans toward the
downgraded team (the stale-ratings bias, in points); whether the market over- or under-adjusts the spread for a
downgrade (the cover margin against the swing); production's favorite picks by situation; Q1's coefficient sign.

Run it from this folder:  python nfl_research_qb.py

Results (October 21, 2026; 4,769 games, weeks 5-16; 18.3% have a downgrade on one side or both):
Q1 fails.  The swing adds little: the correction's correlation with the cover margin goes from 0.083 / 0.055 / 0.063 to
0.086 / 0.060 / 0.062 (lower in 2018-2026), its underdog picks hit 53.7% / 54.7% / 54.8% against production's 53.7% /
55.4% / 55.0%, and only 3.4% of games change side (fitted on 2012-2025, a 0.25 swing moves the correction 0.32
points).  Backing the starter out does not make the model better.
Q2 is CONFIRMED.  Production's underdog picks on a downgraded team went 48.9% (23-24) / 53.5% (53-46) / 50.6% (39-38),
pooled 115-108 (51.6%), against 54.6% / 55.6% / 56.2% (771-619, 55.5%) with no downgrade on either side; against a
downgraded team 38-32 (54.3%); both sides 10-11.  In downgrade games the correction's information collapses (+0.142 /
+0.016 / -0.061).  The pre-registered response: skip underdog picks on a team whose swing is -0.05 or worse.  On these
seasons that leaves 54.0% / 55.7% / 56.4% (809-651, 55.4%) instead of 53.7% / 55.4% / 55.0% (934-770, 54.8%) and drops
about 9 picks a season that were near break-even.  The direction holds in every era but the gap is about one
standard error, so it is a cheap, sensible rule rather than a proven edge.
Diagnostics: production's correction does not lean toward the downgraded team (mean -0.13 points toward it), so the
problem is not a bias but a signal that stops working once the quarterback changes.  The market's adjustment has no
stable error (the downgraded team's cover margin -1.88 / -0.56 / +0.96 points by era).  Favorite picks against a
downgraded underdog went 12-21 (against 133-79 with neither side downgraded), seen here, 33 games: a caution only.
"""

import numpy as np
import pandas as pd

import nfl_lab as lab
import nfl_signal
from nfl_injury_features import REPLACEMENT_QB_EPA, build_qb_quality, qb_rating_before, week_key
from nflverse_data import load_player_weeks

ERAS = (('1999-2005', 1999, 2005), ('2006-2017', 2006, 2017), ('2018-2026', 2018, 2026))
DECAY, CARRY = 0.90, 0.6   # nfl_lab.rate_teams defaults, as in production
DOWNGRADE = -0.05
BREAK_EVEN = 0.5238
QB_COLUMNS = ['player_id', 'position', 'season', 'week', 'season_type', 'attempts', 'sacks_suffered', 'passing_epa']


def add_swings(table):
    """qb_swing_home / qb_swing_away / qb_swing_diff for every game: today's starter's rating minus the rating of the
    quarterbacks embedded in the team's ratings, weighted as nfl_lab.rate_teams weights the games."""
    history = build_qb_quality(load_player_weeks(1999, int(table['season'].max()), columns=QB_COLUMNS))
    games = table.sort_values(['gameday', 'week']).reset_index(drop=True)
    keys = [week_key(s, w) for s, w in zip(games['season'], games['week'])]

    state = {}   # team -> [{qb: weight}, season]
    swings = {'home': np.full(len(games), np.nan), 'away': np.full(len(games), np.nan)}
    for i, g in enumerate(games.itertuples()):
        key = keys[i]
        for side, team, qb in (('home', g.home_team, g.home_qb_id), ('away', g.away_team, g.away_qb_id)):
            weights, season = state.setdefault(team, [{}, None])
            if season is not None and season != g.season:
                weights = {q: w * CARRY for q, w in weights.items()}
                state[team][0] = weights
            state[team][1] = g.season
            total = sum(weights.values())
            if total <= 0:
                swings[side][i] = 0.0
                continue
            embedded = sum(w * qb_rating_before(history, q, key) for q, w in weights.items()) / total
            today = qb_rating_before(history, qb, key) if isinstance(qb, str) else REPLACEMENT_QB_EPA
            swings[side][i] = today - embedded
        if pd.notna(g.result):
            for team, qb in ((g.home_team, g.home_qb_id), (g.away_team, g.away_qb_id)):
                weights = {q: w * DECAY for q, w in state[team][0].items()}
                if isinstance(qb, str):
                    weights[qb] = weights.get(qb, 0.0) + 1.0
                state[team][0] = weights
    out = games[['game_id']].assign(qb_swing_home=swings['home'], qb_swing_away=swings['away'])
    out['qb_swing_diff'] = out['qb_swing_home'] - out['qb_swing_away']
    return table.merge(out, on='game_id', how='left')


def scored(table, features, season):
    """nfl_signal.scored_history with a feature list: 2006 on walk-forward, 1999-2005 fitted on 2006-2019."""
    fit = lab.ridge_margin(features, alpha=nfl_signal.SIGNAL_ALPHA, target='resid')
    forward = lab.walk_forward(table, fit, last_test=season)
    played = table[table['result'].notna() & (table['game_type'] == 'REG')]
    train = played[played['season'].between(*nfl_signal.BACKWARD_TRAIN)]
    backward = [lab.test_games(table, s).assign(pred=lambda t: np.asarray(fit(train, t), float))
                for s in range(nfl_signal.BACKWARD_TEST[0], nfl_signal.BACKWARD_TEST[1] + 1)]
    f = pd.concat(backward + [forward], ignore_index=True)
    f = f[f['week'].between(*nfl_signal.PICK_WEEKS) & (f['spread_line'] != 0)].copy()
    f['toward_dog'] = np.where(f['spread_line'] > 0, -f['pred'], f['pred'])   # spread_line > 0: home favored
    f['dog_swing'] = np.where(f['spread_line'] > 0, f['qb_swing_away'], f['qb_swing_home'])
    f['fav_swing'] = np.where(f['spread_line'] > 0, f['qb_swing_home'], f['qb_swing_away'])
    f['dog_covered'] = np.where(f['spread_line'] > 0, f['resid'] < 0, f['resid'] > 0)
    return f


def record(f, mask):
    d = f[mask & (f['resid'] != 0)]
    w = int(d['dog_covered'].sum())
    return w, len(d) - w, (w / len(d) if len(d) else float('nan'))


def eras(f):
    return [(name, f[f['season'].between(lo, hi)]) for name, lo, hi in ERAS]


def info(f, column='pred'):
    return [float(np.corrcoef(g[column], g['resid'])[0, 1]) for _, g in eras(f)]


def main():
    season = int(lab.default_table()['season'].max())
    table = add_swings(lab.default_table())
    prod = scored(table, nfl_signal.SIGNAL_FEATURES, season)
    swing = scored(table, nfl_signal.SIGNAL_FEATURES + ['qb_swing_diff'], season)
    print('frame: %d games, weeks %d-%d, %d-%d' % (len(prod), *nfl_signal.PICK_WEEKS, prod['season'].min(), prod['season'].max()))
    down = prod['qb_swing_home'].le(DOWNGRADE) | prod['qb_swing_away'].le(DOWNGRADE)
    print('games with a downgrade (swing <= %.2f) on either side: %d (%.1f%%); by era %s' % (
        DOWNGRADE, int(down.sum()), 100 * down.mean(), ' / '.join('%.1f%%' % (100 * (g['qb_swing_home'].le(DOWNGRADE) | g['qb_swing_away'].le(DOWNGRADE)).mean()) for _, g in eras(prod))))
    print('swing quantiles:', prod['qb_swing_home'].quantile([.01, .05, .1, .5, .9, .99]).round(3).to_dict())

    print('\n=== Q1: the pick model plus the QB swing ===')
    p_info, s_info = info(prod), info(swing)
    print('    correlation with the cover margin  production %s' % ' / '.join('%+.3f' % v for v in p_info))
    print('                                       with swing %s' % ' / '.join('%+.3f' % v for v in s_info))
    p_rates, s_rates = [], []
    for (name, pg), (_, sg) in zip(eras(prod), eras(swing)):
        pw, pl, pr = record(pg, pg['toward_dog'] > 0)
        sw, sl, sr = record(sg, sg['toward_dog'] > 0)
        p_rates.append(pr)
        s_rates.append(sr)
        print('    underdog picks %s: production %d-%d %.3f | with swing %d-%d %.3f' % (name, pw, pl, pr, sw, sl, sr))
    changed = (np.sign(prod['toward_dog'].to_numpy()) != np.sign(swing['toward_dog'].to_numpy()))
    print('    picks that change side: %d of %d games (%.1f%%)' % (int(changed.sum()), len(prod), 100 * changed.mean()))
    q1 = all(s > p for s, p in zip(s_info, p_info)) and sum(s >= p for s, p in zip(s_rates, p_rates)) >= 2
    print('    in downgrade games only, correlation: production %s | with swing %s' % (
        ' / '.join('%+.3f' % v for v in info(prod[down])), ' / '.join('%+.3f' % v for v in info(swing[down.to_numpy()]))))

    print('\n=== Q2: production underdog picks by the quarterback situation ===')
    picks = prod['toward_dog'] > 0
    on = prod['dog_swing'] <= DOWNGRADE
    against = prod['fav_swing'] <= DOWNGRADE
    groups = (('on a downgraded team', picks & on & ~against), ('against a downgraded team', picks & against & ~on),
              ('both sides downgraded', picks & on & against), ('no downgrade on either side', picks & ~on & ~against))
    rates = {}
    for label, mask in groups:
        cells = []
        for name, g in eras(prod):
            w, l, r = record(g, mask.loc[g.index])
            cells.append('%s %d-%d %.3f' % (name, w, l, r))
            rates.setdefault(label, []).append(r)
        w, l, r = record(prod, mask)
        print('    %-28s %s | pooled %d-%d %.3f' % (label, ' | '.join(cells), w, l, r))
    on_rates, none_rates = rates['on a downgraded team'], rates['no downgrade on either side']
    q2 = sum(r < BREAK_EVEN for r in on_rates) >= 2 and all(a < b for a, b in zip(on_rates, none_rates))

    print('\n=== diagnostics (no decisions) ===')
    one = prod['qb_swing_home'].le(DOWNGRADE) ^ prod['qb_swing_away'].le(DOWNGRADE)
    toward_down = np.where(prod['qb_swing_home'].le(DOWNGRADE), prod['pred'], -prod['pred'])
    print('    production correction toward the downgraded team, games with one: mean %+.3f pts (n=%d); all games |correction| mean %.3f' % (
        float(toward_down[one.to_numpy()].mean()), int(one.sum()), float(prod['pred'].abs().mean())))
    cover_down = np.where(prod['qb_swing_home'].le(DOWNGRADE), prod['resid'], -prod['resid'])
    print('    cover margin of the downgraded team (market adjustment): ' + ' | '.join(
        '%s %+.2f pts (n=%d)' % (name, float(cover_down[(one & prod['season'].between(lo, hi)).to_numpy()].mean()), int((one & prod['season'].between(lo, hi)).sum()))
        for name, lo, hi in ERAS))
    print('    correlation of the cover margin with qb_swing_diff: ' + ' / '.join('%+.3f' % v for v in info(prod, 'qb_swing_diff')))
    night = np.array([nfl_signal.is_night_game(t, d) for t, d in zip(prod['gametime'], prod['weekday'])])
    fav = np.array([nfl_signal.is_favorite_pick(t, w, n) for t, w, n in zip(prod['toward_dog'], prod['week'], night)])
    for label, mask in (('favorite picks against a downgraded underdog', fav & (prod['dog_swing'] <= DOWNGRADE).to_numpy()),
                        ('favorite picks on a downgraded favorite', fav & (prod['fav_swing'] <= DOWNGRADE).to_numpy()),
                        ('favorite picks, neither', fav & ~((prod['dog_swing'] <= DOWNGRADE) | (prod['fav_swing'] <= DOWNGRADE)).to_numpy())):
        d = prod[mask & (prod['resid'] != 0).to_numpy()]
        w = int((~d['dog_covered']).sum())
        print('    %-46s %d-%d' % (label, w, len(d) - w))
    fit = lab.ridge_margin(nfl_signal.SIGNAL_FEATURES + ['qb_swing_diff'], alpha=nfl_signal.SIGNAL_ALPHA, target='resid')
    played = table[table['result'].notna() & (table['game_type'] == 'REG') & table['season'].between(season - 14, season - 1)]
    probe = played.head(2).copy()
    probe['qb_swing_diff'] = [0.0, 0.25]
    base = probe.copy()
    for column in nfl_signal.SIGNAL_FEATURES:
        base[column] = played[column].median()
    base['qb_swing_diff'] = [0.0, 0.25]
    a, b = fit(played, base)
    print('    Q1 model fitted on %d-%d: a home swing 0.25 better than the away one moves the correction %+.2f pts toward home' % (
        season - 14, season - 1, b - a))

    print('\n=== verdict (pre-registered) ===')
    print('    Q1 the QB swing in the pick model: %s' % ('PASS' if q1 else 'fail'))
    print('    Q2 picks on a downgraded team underperform (the concern): %s' % ('CONFIRMED' if q2 else 'not confirmed'))


if __name__ == '__main__':
    main()
