"""Prop picks: the model's chance on each side of a prop line, the pick rule, the price to bet at, and grading.

The rule (MIN_CHANCE, PICK_MARKETS, PICK_SIDES) was fixed on the DISCOVERY lines (ESPN BET, 2024-2025, closing lines)
and then scored once on the untouched TEST lines (DraftKings, 2026); nfl_research_props.py reproduces both.
"""

import numpy as np
import pandas as pd

DISCOVERY_SEASONS = (2024, 2025)
TEST_SEASONS = (2026,)

# The pick rule, fixed on the discovery closing lines (README section 11): unders only, because the model's overs carried
# no information against closing lines (50% at every threshold); every market except the touchdown and interception
# props (books move their price rather than the line, so a win rate at the line says little without prices) and pass
# attempts and completions (negative information against closing lines); and a chance of at least 0.65, where the
# unders' win rate reached its plateau (56.3% of 2,986).
MIN_CHANCE = 0.65
PICK_MARKETS = ('pass_yds', 'pass_rush_yds', 'rush_att', 'rush_yds', 'rec', 'rec_yds', 'rush_rec_yds')
PICK_SIDES = ('Under',)
PICK_WEEKS = (1, 22)    # every week, playoffs included

# The over rule, fixed on the discovery OPENING lines with the early-week model variant (no injury report, opening game
# lines; README section 11.7): bet the Over at DraftKings' opening line when the early-week model gives it a chance of
# at least 0.65, in every market except the touchdown and interception props.  Its edge is closing-line value: the
# market moves toward these overs during the week (in discovery 58.9% of 1,174 won at the opener, and the line rose
# for 53% of them and fell for 11%), so an over pick holds only while the line is still where it opened.
OVER_MIN_CHANCE = 0.65
OVER_MARKETS = ('pass_yds', 'pass_cmp', 'pass_att', 'pass_rush_yds', 'rush_att', 'rush_yds', 'rec', 'rec_yds',
                'rush_rec_yds')
# An over still at its opening line this close to kickoff is almost surely one the market never moves, and those won
# only about 52% in discovery (the edge sits in the lines that later rise), so a late over is priced at that rate
OVER_LATE_HOURS = 48

MIN_EDGE = 0.03         # the price shown is the worst one at which a bet still returns 3% on average
PRIOR_GAMES = 200       # shrinkage of each market's backtest win rate toward the rate of all picks (market_win_rates)
CHANCE_BINS = (0.50, 0.55, 0.60, 0.65, 0.70)


def score_lines(rows, dists, line_column='line'):
    """Adds the model's chances to rows with season, market, forecast and a line (in `line_column`).

    p_over / p_under: share of similar past forecasts that ended over / under the line; side: the likelier one;
    chance: the side's chance of winning with pushes (stake returned) left out; median: the middle past result.
    """
    p_over, p_under, median = [], [], []
    for season, market, mu, line in zip(rows['season'], rows['market'], rows['forecast'], rows[line_column]):
        dist = dists[season][market]
        over, under = dist.chances(mu, line)
        p_over.append(over)
        p_under.append(under)
        median.append(dist.median(mu))
    out = rows.copy()
    out['p_over'], out['p_under'], out['median'] = p_over, p_under, median
    decided = (out['p_over'] + out['p_under']).replace(0, np.nan)
    out['side'] = np.where(out['p_over'] >= out['p_under'], 'Over', 'Under')
    out['chance'] = np.maximum(out['p_over'], out['p_under']) / decided
    return out


def is_pick(market, side, chance, week, min_chance=None):
    """Whether a scored prop line is a pick under the (under) rule."""
    threshold = MIN_CHANCE if min_chance is None else min_chance
    if threshold is None or chance is None or np.isnan(chance):
        return False
    return (PICK_WEEKS[0] <= week <= PICK_WEEKS[1] and chance >= threshold
            and (PICK_MARKETS is None or market in PICK_MARKETS)
            and (PICK_SIDES is None or side in PICK_SIDES))


def is_over_pick(market, side, chance, week, line=None, open_line=None):
    """Whether a line scored by the early-week model is an over pick: an over the model gives OVER_MIN_CHANCE or more,
    in OVER_MARKETS, while the line is still at its opening number (pass line/open_line to check that)."""
    if chance is None or np.isnan(chance) or side != 'Over' or market not in OVER_MARKETS:
        return False
    if line is not None and open_line is not None and not pd.isna(open_line) and float(line) != float(open_line):
        return False
    return PICK_WEEKS[0] <= week <= PICK_WEEKS[1] and chance >= OVER_MIN_CHANCE


def over_picks(scored):
    """The rows of lines scored by the early-week model (at their opening line) that the over rule takes."""
    keep = [is_over_pick(m, s, c, w) for m, s, c, w in zip(scored['market'], scored['side'], scored['chance'], scored['week'])]
    return scored[keep]


def close_known(rows):
    """Rows whose stored line is a pre-game closing line (ESPN BET's, or DraftKings' last moved before kickoff)."""
    return (rows['provider'].astype(int) == 58) | (rows['pregame'].astype(str).str.lower() == 'true')


def never_moved_rate(picks):
    """Win rate of over picks whose line never moved from its opener to the close: what a late over is worth."""
    still = picks[close_known(picks) & (picks['line'] == picks['open_line']) & picks['outcome'].isin(['W', 'L'])]
    return (still['outcome'] == 'W').mean() if len(still) else None, len(still)


def grade(side, line, result):
    """'W', 'L' or 'P' for a bet on `side` of `line` given the player's result; None if there is no result."""
    if result is None or pd.isna(result) or line is None or pd.isna(line):
        return None
    if result == line:
        return 'P'
    return 'W' if (result > line) == (side == 'Over') else 'L'


def american_from_chance(chance, edge=0.0):
    """Worst American price at which a bet winning `chance` of the time (pushes aside) still returns `edge` per unit."""
    if chance is None or not 0 < chance < 1:
        return None
    decimal = (1 + edge) / chance
    if decimal <= 1:
        return None
    return round(100 * (decimal - 1)) if decimal >= 2 else round(-100 / (decimal - 1))


def profit(result, price=None):
    """Units won on a 1-unit bet: price in American odds (default -110); pushes and voids return 0."""
    price = -110 if price is None or pd.isna(price) else float(price)
    if result == 'W':
        return 100 / -price if price < 0 else price / 100
    return -1.0 if result == 'L' else 0.0


def hit_table(scored, bins=CHANCE_BINS):
    """[(bin start, wins, losses, win %)] for picks by the model's chance (pushes and voids left out)."""
    graded = scored.dropna(subset=['outcome'])
    graded = graded[graded['outcome'] != 'P']
    table = []
    for i, start in enumerate(bins):
        end = bins[i + 1] if i + 1 < len(bins) else 1.01
        rows = graded[(graded['chance'] >= start) & (graded['chance'] < end)]
        wins, losses = int((rows['outcome'] == 'W').sum()), int((rows['outcome'] == 'L').sum())
        table.append((start, wins, losses, wins / (wins + losses) if wins + losses else None))
    return table


def win_rate_for(chance, table):
    """Backtest win % of the chance bin `chance` falls in (None if the bin has no graded bets)."""
    rate = None
    for start, _, _, bin_rate in table:
        if chance >= start:
            rate = bin_rate
    return rate


def score_history(lines, forecasts, which='close'):
    """Past prop lines joined with the out-of-sample forecasts and results, scored by the model and graded.

    which='open' grades every opening line; 'close' grades closing lines: ESPN BET's stored lines (its feed kept the
    pre-game line: where it moved, it lands only slightly nearer the result than the opener, as a pre-game close
    does) and DraftKings lines last updated before kickoff (DraftKings' feed carries in-game lines after kickoff).
    Lines for players with no result (inactive, or outside the market's population, e.g. a backup quarterback) drop.
    """
    import nfl_prop_model as pm
    rows = lines.copy()
    if which == 'open':
        rows['bet_line'] = rows['open_line']
    else:
        rows = rows[close_known(rows)].copy()
        rows['bet_line'] = rows['line']
    rows = rows.dropna(subset=['bet_line'])
    merged = rows.merge(forecasts.rename(columns={'player_id': 'gsis_id'}),
                        on=['season', 'week', 'game_id', 'gsis_id', 'market'], how='inner')
    merged = merged.dropna(subset=['result'])
    dists = {season: pm.distributions(forecasts, season) for season in merged['season'].unique()}
    out = score_lines(merged, dists, line_column='bet_line')
    out['outcome'] = [grade(side, line, result) for side, line, result in zip(out['side'], out['bet_line'], out['result'])]
    out['under'] = [grade('Under', line, result) for line, result in zip(out['bet_line'], out['result'])]
    return out


def rule_picks(scored):
    """The rows of scored lines the pick rule takes."""
    keep = [is_pick(m, s, c, w) for m, s, c, w in zip(scored['market'], scored['side'], scored['chance'], scored['week'])]
    return scored[keep]


def market_win_rates(picks, prior_games=PRIOR_GAMES, markets=None):
    """Backtest win rate of a rule's picks per market, shrunk toward the rate of all its picks.

    A market with few graded picks borrows `prior_games` picks' worth of the overall rate (empirical Bayes), so a
    37% from 27 picks does not set a price on its own.  `markets` defaults to the under rule's.
    Returns ({market: (rate, wins, losses)}, overall rate).
    """
    graded = picks[picks['outcome'].isin(['W', 'L'])]
    wins_all = int((graded['outcome'] == 'W').sum())
    overall = wins_all / len(graded) if len(graded) else None
    rates = {}
    for market in markets or PICK_MARKETS or graded['market'].unique():
        sub = graded[graded['market'] == market]
        wins, losses = int((sub['outcome'] == 'W').sum()), int((sub['outcome'] == 'L').sum())
        rate = None if overall is None else (wins + prior_games * overall) / (wins + losses + prior_games)
        rates[market] = (rate, wins, losses)
    return rates, overall
