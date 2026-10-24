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
# The under rule holds at closing lines, with the final injury report in the model, so an under is a pick only within
# this many hours of its game's kickoff (adopted October 21, 2026): by then every game day's final report is out
# (Wednesday for Thursday games, Friday for Sunday games, Saturday for Monday games) and the line is near its close.
# Earlier in the week the report has no game designations, and even scored with later information the rule's unders
# hit only 54.3% at the opening lines of the test, against 57.4% at the close.
UNDER_WINDOW_HOURS = 24

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
# The returner skip (README section 11.8, nfl_research_props_returners.py, adopted October 21, 2026): no over pick for a
# player whose team played RETURNER_MISSED or more games since his last game, from week RETURNER_FIRST_WEEK on.  The
# models build a player's features from his last games played, however long ago, so they forecast a player back from
# an absence by his old role: receiving forecasts for such players ran 20-30% high (more after longer absences), and
# the over rule's picks on them went 28-35 against 757-513 for its other picks.
RETURNER_MISSED, RETURNER_FIRST_WEEK = 3, 2

MIN_EDGE = 0.03         # the price shown is the worst one at which a bet still returns 3% on average
PRIOR_GAMES = 200       # shrinkage of a market's backtest win rate toward its line group's (line_win_rates)
CHANCE_BINS = (0.50, 0.55, 0.60, 0.65, 0.70)
# Hist. Win % and Bet At go by market and line (adopted October 22, 2026): picks on low lines win far more often, as
# books set low lines where the over is likely and price it heavily (receptions overs at the opening line, discovery
# 2024-2025 / untouched test 2026: 0.5 lines 71.2% / 68.0%, 1.5-2.5 lines 68.1% / 71.1%, 3.5+ lines 54.7% / 57.1%),
# so one rate per market set a wrong price both ways (receptions overs had -167 at every line, where the history
# supports about -200 at 1.5 and about -114 at 3.5+).
LINE_GROUPS = (('0.5', 0.5), ('1.5-2.5', 2.5), ('3.5+', float('inf')))
MIN_LINE_GROUP_PICKS = 30   # a line group with fewer graded picks takes the rule's overall rate as its own


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


def is_over_pick(market, side, chance, week, line=None, open_line=None, missed=None):
    """Whether a line scored by the early-week model is an over pick: an over the model gives OVER_MIN_CHANCE or more,
    in OVER_MARKETS, while the line is still at its opening number (pass line/open_line to check that), unless the
    player is back from an absence (pass `missed`, the games his team played since his last game: returner_skip)."""
    if chance is None or np.isnan(chance) or side != 'Over' or market not in OVER_MARKETS:
        return False
    if line is not None and open_line is not None and not pd.isna(open_line) and float(line) != float(open_line):
        return False
    if returner_skip(missed, week):
        return False
    return PICK_WEEKS[0] <= week <= PICK_WEEKS[1] and chance >= OVER_MIN_CHANCE


def returner_skip(missed, week):
    """Whether a player is back from an absence the models do not see: his team played RETURNER_MISSED or more games
    since his last game, in week RETURNER_FIRST_WEEK or later (unknown, e.g. his first game in the data: no skip)."""
    return (missed is not None and not pd.isna(missed) and missed >= RETURNER_MISSED
            and week is not None and week >= RETURNER_FIRST_WEEK)


def over_picks(scored):
    """The rows of lines scored by the early-week model (at their opening line) that the over rule takes; rows with a
    `missed` column (nfl_prop_model.games_missed) also get the returner skip."""
    missed = scored['missed'] if 'missed' in scored.columns else [None] * len(scored)
    keep = [is_over_pick(m, s, c, w, missed=g)
            for m, s, c, w, g in zip(scored['market'], scored['side'], scored['chance'], scored['week'], missed)]
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
    37% from 27 picks does not set a price on its own.  `markets` defaults to the under rule's.  The daily run prices
    picks by market and line instead (line_win_rates).
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


def line_group(line):
    """The LINE_GROUPS label a prop line falls in: '0.5', '1.5-2.5' or '3.5+'."""
    for label, top in LINE_GROUPS:
        if float(line) <= top:
            return label
    return LINE_GROUPS[-1][0]


def line_win_rates(picks, prior_games=PRIOR_GAMES, markets=None):
    """Backtest win rate of a rule's picks per market and line group (LINE_GROUPS), behind Hist. Win % and Bet At.

    A line group's rate is its picks' rate over all the rule's markets (the rule's overall rate when it has fewer than
    MIN_LINE_GROUP_PICKS graded picks), and a market's rate at a line group borrows `prior_games` picks' worth of it
    (empirical Bayes, as market_win_rates borrows the overall rate), so a market with few picks at a line does not set
    a price on its own.  Picks go by the line they were graded at (`bet_line`).  `markets` defaults to the under rule's.
    Returns ({(market, group): (rate, wins, losses)}, {group: rate}, overall rate); a market with no picks at a line
    group is left out and takes the group's rate (win_rate_at).
    """
    graded = picks[picks['outcome'].isin(['W', 'L'])]
    graded = graded.assign(group=[line_group(line) for line in graded['bet_line']])
    wins_all = int((graded['outcome'] == 'W').sum())
    overall = wins_all / len(graded) if len(graded) else None
    group_rates = {}
    for label, _ in LINE_GROUPS:
        sub = graded[graded['group'] == label]
        group_rates[label] = float((sub['outcome'] == 'W').mean()) if len(sub) >= MIN_LINE_GROUP_PICKS else overall
    rates = {}
    for market in markets or PICK_MARKETS or graded['market'].unique():
        for label, _ in LINE_GROUPS:
            sub = graded[(graded['market'] == market) & (graded['group'] == label)]
            if sub.empty:
                continue
            wins, losses = int((sub['outcome'] == 'W').sum()), int((sub['outcome'] == 'L').sum())
            prior = group_rates[label]
            rate = None if prior is None else (wins + prior_games * prior) / (wins + losses + prior_games)
            rates[(market, label)] = (rate, wins, losses)
    return rates, group_rates, overall


def win_rate_at(rates, group_rates, overall, market, line):
    """Hist. Win % of a pick in `market` at `line`, from line_win_rates' tables."""
    group = line_group(line)
    found = rates.get((market, group))
    return found[0] if found else group_rates.get(group, overall)
