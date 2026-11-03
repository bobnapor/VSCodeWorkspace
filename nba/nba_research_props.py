"""NBA player props: is there an edge?  The pre-registered validation (written November 2, 2026, before any NBA prop
result had been looked at; the spread and total rules had just failed to clear the juice, nba_research_validate.py).

Data: ESPN's prop lines (nba_prop_lines): ESPN BET from late December 2023 through 2024-25 (DISCOVERY), and ESPN BET
in October-November 2025 plus DraftKings in 2026-27 so far (the untouched TEST).  Each line is scored with
nba_prop_model's out-of-sample forecasts (models fitted on earlier seasons only) and the chance from the nearest past
forecasts of the seasons before.  Win rates leave out pushes and voids (a player who does not play); break-even at -110
is 52.38%, at -115 53.49%.

Protocol, fixed before the first run:
  1. Markets: points, rebounds, assists, threes and the combos (RULE_MARKETS).  Steals, blocks and turnovers are left
     out in advance: their lines sit at 0.5-2.5, where books move the price rather than the line, so a win rate at the
     line says little without prices.
  2. Four rule families (FAMILIES): the full model (who plays tonight, closing game lines) at the closing prop line,
     unders and overs; the early model (absences known at the team's last game, opening game lines) at the opening prop
     line, overs and unders.  A closing line is one ESPN last updated before tip-off.
  3. On DISCOVERY only, each family's threshold is the one of THRESHOLDS with the best z-score against -110 with at least
     MIN_DISCOVERY_PICKS picks.
  4. TEST (python nba_research_props.py test, once the choices are written into CHOSEN): each family at its threshold.
     A family is proposed for betting only if on the test it wins at least 54.0% and beats -110 with z >= 1.645 (one
     pick per player-game too, as several props of one player rise and fall together).  The owner decides.

Amendment (November 2, 2026, after the first discovery run and before any test result): (a) ESPN BET's lines were last
updated after the games ended (settlement), so the 'pregame' stamp kept only 1,062 closing lines; as in the NFL suite,
an ESPN BET line counts as its closing line (DraftKings' only when stamped before tip-off).  (b) ESPN BET's feed has the
over's and the under's prices, now kept (nba_prop_lines, history version 2), so each family is also scored at the
price of its side (return per unit staked), which is what decides whether a win rate at the line is worth anything:
books often move the price, not the line.  (c) For the opening-line families, how often the line then moved toward
the pick (the edge there is catching a stale opener).  The bar in step 4 is unchanged; the return at prices is
reported beside it.

    python nba_research_props.py            discovery
    python nba_research_props.py value      discovery: value bets against the prices
    python nba_research_props.py test       the untouched test of CHOSEN

Results (November 2, 2026; discovery only, the test seasons are untouched).  At the line, every family looked strong:
unders at the close won 56.8% at 0.60 (34,183 picks), overs at the close 55.2%, and at the opening lines overs 72.8%
and unders 70.7% at 0.70.  At ESPN BET's own prices every one of them lost money: -1.1%, -4.4%, -4.2% and -2.3% a
bet.  The book moved the price instead of the line, and the openers the model 'beat' were stale numbers (Craig Porter
Jr., a bench player starting for an injured Cleveland, opened at 20.5 points, moved to 13.5 and scored 7) already
priced against it; the line later moved toward 79% of the opening-line overs, the edge priced in at the open.  Betting
only where the model's chance beats the price's implied chance (value): overs lost at every edge (-0.6% to -5.6%),
unders went from -1.2% to +1.7% a bet, the best of 20 cells scanned, not rising with the edge (the model's chances are
overconfident).  So the model's information is real (its picks lose less than the 4-5% of a random bet) but does not
beat the vig at real prices, the same finding as for spreads and totals.  Not proposed for betting.
"""

import math
import sys

import numpy as np

import nba_prop_lines
import nba_prop_model as pm

DISCOVERY_SEASONS = (2024, 2025)
TEST_SEASONS = (2026, 2027)
RULE_MARKETS = ('pts', 'reb', 'ast', 'fg3m', 'pra', 'pr', 'pa', 'ra')
FAMILIES = {'full-under@close': ('full', 'Under', 'close'), 'full-over@close': ('full', 'Over', 'close'),
            'early-over@open': ('early', 'Over', 'open'), 'early-under@open': ('early', 'Under', 'open')}
THRESHOLDS = (0.55, 0.60, 0.65, 0.70)
MIN_DISCOVERY_PICKS = 1000
BREAK_EVEN = 110 / 210
# filled in after discovery, before the test: {family: threshold}
CHOSEN = {}


def record(won):
    wins, n = int(won.sum()), len(won)
    rate = wins / n if n else float('nan')
    z = (rate - BREAK_EVEN) / math.sqrt(BREAK_EVEN * (1 - BREAK_EVEN) / n) if n else float('nan')
    return wins, n - wins, rate, z


def profit(won, push, odds):
    """Units won per unit staked at American `odds` (NaN without a price)."""
    odds = np.asarray(odds, dtype=float)
    win = np.where(odds < 0, 100 / np.abs(odds), odds / 100)
    return np.where(np.isnan(odds), np.nan, np.where(push, 0.0, np.where(won, win, -1.0)))


def score(lines, history, which):
    """The lines with the model's chances at the closing ('close') or opening ('open') line, graded, with the price of
    the side picked where the feed has it (ESPN BET)."""
    rows = lines[lines['market'].isin(RULE_MARKETS)].copy()
    if which == 'close':
        espn_bet = rows['provider'].astype(int) == 58
        rows = rows[espn_bet | (rows['pregame'].astype(str).str.lower() == 'true')]
        rows['bet_line'] = rows['line'].astype(float)
        rows['bet_over_odds'], rows['bet_under_odds'] = rows['over_odds'], rows['under_odds']
    else:
        rows = rows[rows['open_line'].notna()]
        rows['bet_line'] = rows['open_line'].astype(float)
        rows['bet_over_odds'], rows['bet_under_odds'] = rows['open_over_odds'], rows['open_under_odds']
    rows = rows.merge(history, on=['season', 'game_id', 'athlete_id', 'market'], how='inner')
    over, under = [], []
    dists = {season: pm.distributions(history, season) for season in rows['season'].unique()}
    for season, market, mu, line in zip(rows['season'], rows['market'], rows['forecast'], rows['bet_line']):
        o, u = dists[season][market].chances(mu, line)
        over.append(o)
        under.append(u)
    rows['p_over'], rows['p_under'] = over, under
    decided = (rows['p_over'] + rows['p_under']).replace(0, np.nan)
    rows['side'] = np.where(rows['p_over'] >= rows['p_under'], 'Over', 'Under')
    rows['chance'] = np.maximum(rows['p_over'], rows['p_under']) / decided
    rows['won'] = np.where(rows['side'] == 'Over', rows['result'] > rows['bet_line'], rows['result'] < rows['bet_line'])
    rows['push'] = rows['result'] == rows['bet_line']
    rows['odds'] = np.where(rows['side'] == 'Over', rows['bet_over_odds'], rows['bet_under_odds']).astype(float)
    rows['profit'] = profit(rows['won'], rows['push'], rows['odds'])
    # the line's later move toward the pick (opening-line families): positive when the market came to the pick's side
    toward = np.where(rows['side'] == 'Over', rows['line'] - rows['open_line'], rows['open_line'] - rows['line'])
    rows['moved_toward'] = toward if which == 'open' else np.nan
    return rows


def family_picks(scored, side, threshold):
    picks = scored[(scored['side'] == side) & (scored['chance'] >= threshold) & ~scored['push']]
    return picks


def one_per_player_game(picks):
    return picks.sort_values('chance', ascending=False).drop_duplicates(['game_id', 'athlete_id'])


def load(seasons):
    """(lines, forecasts by variant) for `seasons`, with the forecast seasons the distributions need."""
    lines = nba_prop_lines.load_line_history(sorted(set(DISCOVERY_SEASONS + TEST_SEASONS)), progress=False)
    lines = lines[lines['season'].isin(seasons)]
    needed = list(range(min(seasons) - pm.DIST_SEASONS, max(seasons) + 1))
    frame = pm.load_frame(max(seasons))
    histories = {variant: pm.forecast_history(frame, needed, variant) for variant in ('full', 'early')}
    return lines, histories


def describe(name, picks):
    wins, losses, rate, z = record(picks['won'])
    single = one_per_player_game(picks)
    sw, sl, srate, sz = record(single['won'])
    priced = picks['profit'].dropna()
    roi = f'; at its prices {priced.mean():+.1%} a bet ({len(priced)} priced)' if len(priced) else ''
    moved = picks['moved_toward'].dropna()
    clv = (f'; line later moved toward it {np.mean(moved > 0):.0%}, away {np.mean(moved < 0):.0%}' if len(moved) else '')
    return (f'{wins}-{losses} {rate:.1%} (z {z:+.2f}); one per player-game {sw}-{sl} {srate:.1%} (z {sz:+.2f})'
            + roi + clv, z, wins + losses)


def discovery():
    lines, histories = load(DISCOVERY_SEASONS)
    for name, (variant, side, which) in FAMILIES.items():
        scored = score(lines, histories[variant], which)
        print(f'\n== {name}: {len(scored)} lines scored ({", ".join(str(s) for s in DISCOVERY_SEASONS)}) ==')
        best = None
        for threshold in THRESHOLDS:
            picks = family_picks(scored, side, threshold)
            text, z, n = describe(name, picks)
            print(f'  chance >= {threshold:.2f}: {text}')
            if n >= MIN_DISCOVERY_PICKS and (best is None or z > best[0]):
                best = (z, threshold)
        picks = family_picks(scored, side, best[1] if best else THRESHOLDS[-1])
        by_market = picks.groupby('market')['won'].agg(['sum', 'count'])
        print('  by market at the best threshold: ' + ', '.join(
            f'{m} {int(r["sum"])}-{int(r["count"] - r["sum"])} {r["sum"] / r["count"]:.0%}' for m, r in by_market.iterrows()))
        if best:
            print(f'  -> CHOSEN[{name!r}] = {best[1]}')


def test():
    if not CHOSEN:
        print('CHOSEN is empty: run the discovery first and write its choices into CHOSEN.')
        return 2
    lines, histories = load(TEST_SEASONS)
    for name, threshold in CHOSEN.items():
        variant, side, which = FAMILIES[name]
        scored = score(lines, histories[variant], which)
        picks = family_picks(scored, side, threshold)
        text, _, _ = describe(name, picks)
        wins, losses, rate, z = record(picks['won'])
        sw, sl, srate, sz = record(one_per_player_game(picks)['won'])
        print(f'\n== {name} at {threshold:.2f}, untouched test ({", ".join(str(s) for s in TEST_SEASONS)}) ==')
        print(f'  {text}')
        for season, g in picks.groupby('season'):
            w, l_, r, _ = record(g['won'])
            print(f'    {season}: {w}-{l_} {r:.1%}')
        passed = rate >= 0.54 and z >= 1.645 and srate >= 0.54 and sz >= 1.645
        print(f'  {"PASSES" if passed else "does not pass"} the pre-registered bar (54.0%+ and z >= 1.645, also one per '
              f'player-game)')
    return 0


def implied(odds):
    """The chance an American price implies (vig included); NaN for an unreadable price of 0."""
    odds = np.asarray(odds, dtype=float)
    odds = np.where(odds == 0, np.nan, odds)
    return np.where(odds < 0, -odds / (-odds + 100), 100 / (np.abs(odds) + 100))


def value(edges=(0.0, 0.03, 0.06, 0.10, 0.15)):
    """Discovery only: bet a side when the model's chance beats the chance its price implies by at least an edge.  A
    model that knows more than the price should make money at large edges; one whose chances only echo the line's
    should not."""
    lines, histories = load(DISCOVERY_SEASONS)
    for name, (variant, _, which) in FAMILIES.items():
        if name.startswith('early-under') or name.startswith('full-over'):
            continue            # each (model, line) pair once: both sides are scored below
        scored = score(lines, histories[variant], which)
        scored = scored[scored['odds'].notna() & ~scored['push']]
        scored['edge'] = scored['chance'] - implied(scored['odds'])
        print(f'\n== value bets, {variant} model at the {which} line and price ({len(scored)} priced lines) ==')
        for edge in edges:
            picks = scored[scored['edge'] >= edge]
            for side in ('Over', 'Under'):
                s = picks[picks['side'] == side]
                if len(s):
                    print(f'  edge >= {edge:.2f} {side:>5}: {len(s)} bets, won {s["won"].mean():.1%}, '
                          f'return {s["profit"].mean():+.1%} a bet')


def main(argv):
    if argv and argv[0] == 'test':
        return test()
    if argv and argv[0] == 'value':
        value()
        return 0
    discovery()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
