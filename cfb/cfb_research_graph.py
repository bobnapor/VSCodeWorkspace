"""College football: do ratings solved over the whole schedule graph find an edge the game-by-game ratings miss?
Pre-registered November 7, 2026, before any of these ratings had been computed.

The idea.  College teams play 12 of about 130 FBS teams, so this week's opponents rarely share an opponent, and the
opponents' opponents rarely do either: A beat C and D, who played E and F; B beat G and H, who played I and J.  The
production ratings (cfb_lab.rate_teams) are built game by game: each result is corrected by the opponent's rating on
that day and never revisited, so A's week-2 win over C keeps C's week-2 rating even after C proves strong, and nothing
learned about E and F reaches A.  The graph ratings here (rate_graph) solve every team at once, before each game day,
from all of the season's results so far (the way SP+, Massey and Sagarin do): for every stat of cfb_lab.STATS,
    stat of offense i against defense j = league mean + home field + offense_i + defense_j,
a ridge solve in which each team's offense and defense are pulled toward a preseason prior (last season's final
values carried at CARRY toward its tier's mean, LAM pseudo-games strong).  Every result then moves every team it is
connected to, both ways and back in time.  The same solve gives each game's uncertainty (u): the standard error of the
two teams' margin ratings' difference, large when they are linked by few or long paths of games (early season,
non-conference games) and small when they share opponents.

Protocol, fixed before the first run.  The seasons and bar are the original validation's (cfb_research_validate):
discovery 2010-2019, 2020 for training only, regular season, FBS against FBS, break-even 52.38% at -110.
  1. Tune (no lines involved): of LAMS x CARRIES, the graph ratings whose Model A (cfb_model, alpha 3000, trained from
     2004) misses the final margin by the least on 2010-2019.  Reported beside the production ratings' miss.
  2. Discovery, Model B (the correction against the closing spread) on 2010-2019 with three feature sets: SEQ (the
     production ratings), GRAPH (the graph ratings in their place), BOTH (both).  Each set's alpha is the one of
     cfb_research_validate.ALPHAS correlating best with the cover margin.  The best of the three by r is the BASE, and
     each context family of cfb_research_context (added November 7, 2026, after step 1 and before step 2 had run) is
     scored added to it: BASE+PRIOR (recruiting talent, returning production), BASE+TRAVEL (distance, time zones,
     body clock, altitude), BASE+SPOT (lookahead, letdown, off an upset), BASE+ALL.  A set other than SEQ goes on
     only if its r beats SEQ's; the best such set is chosen.  Then its rules are scored, the original families S1-S3
     plus
        S1L / S1H: S1 (the side the correction favors by >= t) in games with u at or below / above U_SPLIT, the
                   median u of the discovery games (fixed there and used unchanged later),
     at each of cfb_research_validate.THRESHOLDS, and the rule with the largest z against break-even and at least 60
     picks a season is chosen (none if no rule beats break-even).  Also reported, not used to choose: SEQ's r by u
     half, to see whether the market (or the production model) is weakest where the graph is thin.
  3. Test (run once, after the choice is written into CHOSEN): the chosen rule on 2021-2025, models trained on every
     earlier season.  These seasons have been looked at once before, for the production rule (S1 at 2 with SEQ: 191-150,
     56.0%, r +0.059), but no graph rating, feature set or u rule has been scored on them.  Passes only with the
     original bar: (a) z >= 2 against break-even, (b) the correction's t >= 2, (c) above break-even in 4 of 5 seasons;
     and, to replace the production rule, (d) more profit at -110 than S1 at 2 with SEQ on the same seasons.
  4. Forward: the 2026 games played so far, for whatever passes, as cfb_research_validate's forward step.

    python cfb_research_graph.py tune         step 1 (builds and caches one table per setting)
    python cfb_research_graph.py discovery    step 2
    python cfb_research_graph.py test         step 3
    python cfb_research_graph.py forward      step 4

Results.  Step 1 (November 7, 2026, run once): Model A's miss of the final margin, 2010-2019 (6,640 games; closing
line 12.330): production ratings 13.086; graph lam 2 / 5 / 10 with carry 0.4: 13.333 / 13.361 / 13.459, with carry
0.7: 13.018 / 13.051 / 13.221.  TUNED = (2, 0.7), the only settings beating the production ratings: last season
matters more than the in-season prior.  The graph ratings lose most in weeks 1-2 (15.1 against 14.1 at carry 0.4).
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

import cfb_lab
import cfb_model
import cfb_research_context as context
import cfb_research_validate as val
from cfb_common import CACHE_DIR

LAMS = (2.0, 5.0, 10.0)
CARRIES = (0.4, 0.7)
FIXED = 1e-3                    # the league mean and home field are all but unpenalized
GRAPH_VERSION = 1
# filled in after step 1 and step 2, before the test
TUNED = (2.0, 0.7)              # step 1, November 7, 2026
U_SPLIT = None                  # median u of the discovery games
CHOSEN = None                   # (feature set, alpha, family, threshold)


# ---------------------------------------------------------------- the graph ratings

class GraphRater:
    """cfb_lab.rate_teams' replacement (pass it as game_table's rater); also keeps each game's u in self.unc."""

    def __init__(self, lam, carry):
        self.lam, self.carry, self.unc = lam, carry, {}

    def __call__(self, schedule, stats, tiers):
        names = [s for s, _ in cfb_lab.STATS]
        allowed = [a for _, a in cfb_lab.STATS]
        margin = names.index('margin')
        k = len(names)
        obs = {(row[0], row[1]): np.array(row[2:], dtype=float)
               for row in stats[['game_id', 'team_id'] + names].itertuples(index=False)}
        games = schedule.sort_values(['start', 'game_id'])
        games = games.assign(day=games['start'].dt.tz_convert('America/New_York').dt.date)
        neutral = games['neutral_site'].fillna(False).astype(bool) if 'neutral_site' in games else False
        games = games.assign(neutral=neutral)
        final, final_tier, league = {}, {}, np.zeros((2, k))         # last season: team -> (offense, defense)
        out = []
        started = time.time()
        for season, g in games.groupby('season', sort=True):
            teams = sorted(set(g['home_id']) | set(g['away_id']))
            T = len(teams)
            idx = {t: i for i, t in enumerate(teams)}
            n, MU, HF = 2 * T + 2, 2 * T, 2 * T + 1
            prior = np.zeros((n, k))
            means = {}
            for tier in (True, False):
                members = [t for t in final if final_tier[t] == tier]
                means[tier] = (np.mean([final[t][0] for t in members], axis=0) if members else np.zeros(k),
                               np.mean([final[t][1] for t in members], axis=0) if members else np.zeros(k))
            for t, i in idx.items():
                tier = tiers.get((season, t), False)
                mo, md = means[final_tier.get(t, tier)] if t in final else means[tier]
                if t in final:
                    prior[i] = mo + self.carry * (final[t][0] - mo)
                    prior[T + i] = md + self.carry * (final[t][1] - md)
                else:
                    prior[i], prior[T + i] = means[tier]
            prior[MU], prior[HF] = league
            penalty = np.full(n, self.lam)
            penalty[MU] = penalty[HF] = FIXED
            M = np.diag(penalty)
            B = penalty[:, None] * prior
            gp = {}
            X = prior
            for _, day in g.groupby('day', sort=True):
                factor = cho_factor(M)
                X = cho_solve(factor, B)
                for game in day.itertuples(index=False):
                    h, a = idx[game.home_id], idx[game.away_id]
                    for team, i in ((game.home_id, h), (game.away_id, a)):
                        own, alw = X[MU] + X[i], X[MU] + X[T + i]
                        own = own.copy()
                        own[margin] = (X[i, margin] - X[T + i, margin]) / 2
                        out.append((game.game_id, team, gp.get(team, 0), own, alw))
                    c = np.zeros(n)
                    c[h], c[T + h], c[a], c[T + a] = 0.5, -0.5, -0.5, 0.5
                    self.unc[game.game_id] = float(np.sqrt(c @ cho_solve(factor, c)))
                for game in day.itertuples(index=False):
                    home_obs, away_obs = obs.get((game.game_id, game.home_id)), obs.get((game.game_id, game.away_id))
                    if home_obs is None or away_obs is None:
                        continue
                    h, a = idx[game.home_id], idx[game.away_id]
                    side = 0.0 if game.neutral else 1.0
                    for off, dfn, sign, y in ((h, a, side, home_obs), (a, h, -side, away_obs)):
                        cols = [off, T + dfn, MU, HF]
                        vec = np.array([1.0, 1.0, 1.0, sign])
                        fitted = vec @ X[cols]
                        y = np.where(np.isnan(y), fitted, y)       # a missing stat says nothing
                        M[np.ix_(cols, cols)] += np.outer(vec, vec)
                        B[cols] += np.outer(vec, y)
                    for team in (game.home_id, game.away_id):
                        gp[team] = gp.get(team, 0) + 1
            X = cho_solve(cho_factor(M), B)
            final = {t: (X[i].copy(), X[T + i].copy()) for t, i in idx.items()}
            final_tier = {t: tiers.get((season, t), False) for t in idx}
            league = X[[MU, HF]].copy()
            print(f'    rated {season} ({T} teams, {time.time() - started:.0f}s)', flush=True)
        rows = []
        for game_id, team, gp_, own, alw in out:
            row = {'game_id': game_id, 'team_id': team, 'gp': gp_}
            for j, name in enumerate(names):
                row[f'r_{name}_own'] = own[j]
                if allowed[j]:
                    row[f'r_{name}_allowed'] = alw[j]
            rows.append(row)
        return pd.DataFrame(rows)


def graph_table(lam, carry):
    """The research table with graph ratings (cached per setting), with each game's u."""
    path = os.path.join(CACHE_DIR, f'cfb_graph_v{GRAPH_VERSION}_t{cfb_lab.TABLE_VERSION}_l{lam:g}_c{carry:g}.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    rater = GraphRater(lam, carry)
    table = cfb_lab.game_table(rater=rater)
    table['u'] = table['game_id'].map(rater.unc)
    table.to_pickle(path)
    return table


# ---------------------------------------------------------------- feature sets

RATING_FEATURES = [f for f in cfb_model.SPREAD_FEATURES if f.endswith(('_own', '_allowed'))]
OTHER_FEATURES = [f for f in cfb_model.SPREAD_FEATURES if f not in RATING_FEATURES]
FEATURE_SETS = {'SEQ': cfb_model.SPREAD_FEATURES,
                'GRAPH': ['g' + f for f in RATING_FEATURES] + OTHER_FEATURES,
                'BOTH': cfb_model.SPREAD_FEATURES + ['g' + f for f in RATING_FEATURES]}
FAMILIES = dict(context.FAMILIES, ALL=[c for f in context.FAMILIES.values() for c in f])


def features_of(name):
    """'BOTH+PRIOR' -> BOTH's features and the PRIOR family's."""
    base, _, family = name.partition('+')
    return FEATURE_SETS[base] + (FAMILIES[family] if family else [])


def combined(lam, carry):
    """The production table with the graph ratings beside it (gd_*), u and the context features."""
    table = cfb_lab.default_table()
    graph = graph_table(lam, carry)[['game_id', 'u'] + RATING_FEATURES]
    graph = graph.rename(columns={f: 'g' + f for f in RATING_FEATURES})
    return context.add_features(table.merge(graph, on='game_id', how='left'))


def picks(scored, family, threshold):
    if family in ('S1L', 'S1H'):
        low = scored['u'] <= U_SPLIT
        scored = scored[low if family == 'S1L' else ~low]
        family = 'S1'
    return val.picks(scored, 'spreads', family, threshold)


# ---------------------------------------------------------------- the steps

def tune():
    print('== Step 1: Model A (alpha 3000) miss of the final margin, 2010-2019 regular season ==')
    seq = cfb_lab.default_table()
    rows = val.regular(seq)
    s = cfb_model.scored(seq, 'margin', 3000, *val.DISCOVERY, train_from=cfb_lab.FIRST_SEASON, rows=rows)
    s = s[s['margin'].notna() & s['close_home_spread'].notna()]
    print(f'  production ratings: {np.abs(s["pred"] - s["margin"]).mean():.3f} '
          f'(closing line {np.abs(-s["close_home_spread"] - s["margin"]).mean():.3f}, {len(s)} games)')
    best = None
    for lam in LAMS:
        for carry in CARRIES:
            table = graph_table(lam, carry)
            s = cfb_model.scored(table, 'margin', 3000, *val.DISCOVERY, train_from=cfb_lab.FIRST_SEASON,
                                 rows=val.regular(table))
            s = s[s['margin'].notna() & s['close_home_spread'].notna()]
            miss = np.abs(s['pred'] - s['margin']).mean()
            print(f'  graph lam {lam:g}, carry {carry:g}: {miss:.3f} ({len(s)} games)', flush=True)
            if best is None or miss < best[0]:
                best = (miss, lam, carry)
    print(f'  -> TUNED = ({best[1]:g}, {best[2]:g})')
    return 0


def discovery():
    if TUNED is None:
        print('TUNED is empty: run tune first.')
        return 2
    table = combined(*TUNED)
    rows = val.regular(table) & (table['season'] != 2020)
    print(f'== Step 2: Model B on {val.DISCOVERY[0]}-{val.DISCOVERY[1]} (graph lam {TUNED[0]:g}, carry {TUNED[1]:g}) ==')
    found = context.coverage(table[table['season'].between(*val.DISCOVERY)]).mean()
    print('  context coverage, discovery games: ' + ', '.join(f'{c[2:]} {v:.0%}' for c, v in found.items()))
    runs = {}

    def run(name):
        best = None
        for alpha in val.ALPHAS:
            s = cfb_model.scored(table, 'cover', alpha, *val.DISCOVERY, features=features_of(name), rows=rows)
            s = s[s['cover'].notna()]
            r, t = val.correlation(s['pred'], s['cover'])
            if best is None or r > best[0]:
                best = (r, t, alpha, s)
        r, t, alpha, s = best
        by_season = ' '.join(f'{val.correlation(x["pred"], x["cover"])[0]:+.3f}' for _, x in s.groupby('season'))
        print(f'  {name:12}: r {r:+.4f} (t {t:+.2f}) at alpha {alpha}; by season {by_season}', flush=True)
        runs[name] = best
    for name in FEATURE_SETS:
        run(name)
    base = max(FEATURE_SETS, key=lambda n: runs[n][0])
    print(f'  BASE = {base}')
    for family in FAMILIES:
        run(f'{base}+{family}')
    split = float(runs['SEQ'][3]['u'].median())
    print(f'\n  u (the margin ratings\' difference, standard error units): median {split:.4f} -> U_SPLIT')
    seq = runs['SEQ'][3]
    for label, part in (('u low (well connected)', seq[seq['u'] <= split]), ('u high (thin graph)', seq[seq['u'] > split])):
        r, t = val.correlation(part['pred'], part['cover'])
        won, _ = val.picks(part, 'spreads', 'S1', 2.0)
        w, l, rate, _ = val.record(won)
        print(f'  SEQ, {label}: r {r:+.4f} (t {t:+.2f}); S1 at 2 {w}-{l} {rate:.1%}; weeks {part["week"].median():.0f} '
              f'median; conference games {part["conference"].mean():.0%}')
    candidates = [n for n in runs if n != 'SEQ' and runs[n][0] > runs['SEQ'][0]]
    if not candidates:
        print('\n  No set beats SEQ on discovery: nothing goes to the test.')
        return 0
    name = max(candidates, key=lambda n: runs[n][0])
    r, t, alpha, s = runs[name]
    global U_SPLIT
    U_SPLIT = split
    print(f'\n  chosen set {name} (r {r:+.4f}, alpha {alpha}); its rules:')
    seasons = val.DISCOVERY[1] - val.DISCOVERY[0] + 1
    best = None
    for family in ('S1', 'S2', 'S3', 'S1L', 'S1H'):
        line = []
        for threshold in val.THRESHOLDS:
            won, _ = picks(s, family, threshold)
            wins, losses, rate, z = val.record(won)
            line.append(f't{threshold:g}: {wins}-{losses} {rate:.1%}')
            if wins + losses >= val.MIN_PICKS_PER_SEASON * seasons and z > 0 and (best is None or z > best[0]):
                best = (z, family, threshold, wins, losses, rate)
        print(f'  {family:3}: ' + ' | '.join(line))
    if best:
        z, family, threshold, wins, losses, rate = best
        print(f'  -> best: {family} at {threshold:g} ({wins}-{losses}, {rate:.1%}, z {z:+.2f})')
        print(f'     to test it, set U_SPLIT = {split:.4f} and CHOSEN = ({name!r}, {alpha}, {family!r}, {threshold})')
    return 0


def evaluate(first, last, label, rows_fn):
    name, alpha, family, threshold = CHOSEN
    table = combined(*TUNED)
    rows = rows_fn(table)
    s = cfb_model.scored(table, 'cover', alpha, first, last, features=features_of(name), rows=rows)
    s = s[s['cover'].notna()]
    r, t = val.correlation(s['pred'], s['cover'])
    won, taken = picks(s, family, threshold)
    wins, losses, rate, z = val.record(won)
    print(f'\n== {label}: {name}, {family} at {threshold:g}, alpha {alpha}, {first}-{last} ==')
    print(f'  correction vs cover: r {r:+.4f} (t {t:+.2f})')
    print(f'  picks: {wins}-{losses}, {rate:.1%} (z {z:+.2f}); return at the median price '
          f'{val.roi_at_price(won, taken, "spreads"):+.1%} a bet; at -110 {wins * 100 / 110 - losses:+.1f} units')
    above = 0
    for season, g in won.groupby(taken['season']):
        sw, sl, srate, _ = val.record(g)
        above += srate > val.BREAK_EVEN
        print(f'    {season}: {sw}-{sl} {srate:.1%}')
    base = cfb_model.scored(table, 'cover', val.CHOSEN['spreads'][0], first, last, rows=rows)
    base = base[base['cover'].notna()]
    bwon, _ = val.picks(base, 'spreads', 'S1', 2.0)
    bw, bl, brate, _ = val.record(bwon)
    print(f'  production rule (SEQ, S1 at 2) on the same games: {bw}-{bl} {brate:.1%}, at -110 '
          f'{bw * 100 / 110 - bl:+.1f} units')
    return z, t, above, wins * 100 / 110 - losses, bw * 100 / 110 - bl


def test():
    if CHOSEN is None:
        print('CHOSEN is empty: run discovery and write its choice in first.')
        return 2
    z, t, above, units, base_units = evaluate(*val.TEST, 'Step 3, test', val.regular)
    passed = z >= 2 and t >= 2 and above >= val.MIN_TEST_SEASONS and units > base_units
    print(f'  {"PASSES" if passed else "does not pass"} (z >= 2, t >= 2, 4+ of 5 seasons above break-even: {above}, '
          f'more profit than the production rule)')
    return 0


def forward():
    if CHOSEN is None:
        print('CHOSEN is empty.')
        return 2
    evaluate(val.FORWARD, val.FORWARD, 'Step 4, forward',
             lambda t: t['regular'] & t['season'].between(cfb_model.FIRST_LINES_SEASON, val.FORWARD))
    return 0


def main(argv):
    step = argv[0] if argv else 'tune'
    return {'tune': tune, 'discovery': discovery, 'test': test, 'forward': forward}[step]()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
