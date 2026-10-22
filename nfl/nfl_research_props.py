"""Evidence for the player prop picks (nfl_prod_props.py): forecast quality and the record against real prop lines.

Protocol, as in nfl_research_validate.py: the pick rule in nfl_prop_signal.py was chosen on the DISCOVERY lines
(ESPN BET 2024-2025, opening lines) and only then scored on the untouched TEST lines (DraftKings 2026).  Opening lines
are the benchmark because they are always pre-game; a stored closing line is used only when the book last moved it
before kickoff (after kickoff the feed carries in-game lines).

Run from nfl/:
    python nfl_research_props.py quality     forecast accuracy and bias by market and season (out of sample)
    python nfl_research_props.py discovery   information and hit rates on the discovery lines, and the rule grid
    python nfl_research_props.py test        the frozen under rule on the test lines (opening and clean closing lines)
    python nfl_research_props.py overs       the over side: the early-week variant at opening lines with closing-line
                                             value, and closing-line checks of the full model's overs (discovery only)
    python nfl_research_props.py overs_test  the frozen over rule on the discovery lines and the untouched test lines
    python nfl_research_props.py             quality, discovery and test
"""

import logging
import sys

import numpy as np
import pandas as pd

import nfl_prop_lines as pl
import nfl_prop_model as pm
import nfl_prop_signal as ps
import nflverse_data as nd

LAST_SEASON = 2026
# Markets where books mostly move the price rather than the line (0.5 and 1.5 touchdowns or interceptions), so a win
# rate at the line, without the price, says little about profit
PRICE_MARKETS = ('pass_td', 'pass_int')
NAIVE = {'pass_yds': 'passing_yards', 'pass_cmp': 'completions', 'pass_td': 'passing_tds',
         'pass_int': 'passing_interceptions', 'pass_att': 'attempts', 'rush_att': 'carries', 'rush_yds': 'rushing_yards',
         'rec': 'receptions', 'rec_yds': 'receiving_yards'}
_cache = {}
_early = {}


def load():
    """Feature frame, out-of-sample forecasts for every season from 2018, and the prop lines with nflverse ids."""
    if not _cache:
        weeks, games, injuries = pm.load_inputs(LAST_SEASON)
        frame = pm.build_frame(weeks, games, injuries)
        history = pm.forecast_history(frame, list(range(pm.FIRST_FORECAST_SEASON, LAST_SEASON)))
        logging.info(f"Forecasting {LAST_SEASON} to date (models fitted on {LAST_SEASON - pm.TRAIN_SEASONS}-{LAST_SEASON - 1})")
        forecasts = pd.concat([history, pm.season_forecasts(frame, LAST_SEASON)], ignore_index=True)
        seasons = ps.DISCOVERY_SEASONS + ps.TEST_SEASONS
        lines = pl.attach_players(pl.load_line_history(games, seasons), nd.load_players())
        _cache.update(frame=frame, forecasts=forecasts, lines=lines, inputs=(weeks, games, injuries))
    return _cache['frame'], _cache['forecasts'], _cache['lines']


def load_early():
    """The early-week variant (no injury report, opening game lines): its frame and out-of-sample forecasts."""
    if not _early:
        load()
        weeks, games, injuries = _cache['inputs']
        openers = pl.load_game_openers(games, list(ps.DISCOVERY_SEASONS + ps.TEST_SEASONS))
        frame = pm.build_frame(weeks, pm.with_openers(games, openers), injuries, early=True)
        history = pm.forecast_history(frame, list(range(pm.FIRST_FORECAST_SEASON, LAST_SEASON)), variant='early')
        logging.info(f"Forecasting {LAST_SEASON} to date with the early-week variant")
        forecasts = pd.concat([history, pm.season_forecasts(frame, LAST_SEASON)], ignore_index=True)
        _early.update(frame=frame, forecasts=forecasts)
    return _early['frame'], _early['forecasts']


def scored_lines(seasons, which='open'):
    """Prop lines of `seasons` scored by the model and graded (nfl_prop_signal.score_history)."""
    _, forecasts, lines = load()
    return ps.score_history(lines[lines['season'].isin(seasons)], forecasts, which)


def _rate(outcomes):
    wins, losses = int((outcomes == 'W').sum()), int((outcomes == 'L').sum())
    return (wins / (wins + losses) if wins + losses else np.nan), wins + losses


def _information(rows):
    """Correlation of the model's lean (chance over minus chance under) with the result's side of the line, and its t."""
    sign = np.sign(rows['result'] - rows['bet_line'])
    keep = sign != 0
    lean = (rows['p_over'] - rows['p_under'])[keep]
    if keep.sum() < 10 or lean.std() == 0:
        return np.nan, np.nan
    r = np.corrcoef(lean, sign[keep])[0, 1]
    return r, r * np.sqrt((keep.sum() - 2) / max(1e-9, 1 - r * r))


def market_table(rows, title):
    print(f"\n{title}")
    print(f"{'market':14} {'lines':>6} {'info r':>7} {'t':>6} {'under%':>7} {'model%':>7} | win % (n) by the model's chance: "
          + '  '.join(f"{b:.2f}+" for b in ps.CHANCE_BINS))
    for market in list(pl.MARKETS) + ['ALL']:
        sub = rows if market == 'ALL' else rows[rows['market'] == market]
        if sub.empty:
            continue
        r, t = _information(sub)
        under, _ = _rate(sub['under'])
        model, n = _rate(sub['outcome'])
        bins = '  '.join(f"{rate:.1%} ({w + l})" if rate is not None else '   -   '
                         for _, w, l, rate in ps.hit_table(sub))
        print(f"{market:14} {n:>6} {r:>7.3f} {t:>6.1f} {under:>7.1%} {model:>7.1%} | {bins}")


def quality():
    frame, forecasts, _ = load()
    print("\n=== FORECAST QUALITY (out of sample): mean absolute error, correlation and bias (forecast - result) ===")
    played = frame[frame['played']]
    forecasts = forecasts[forecasts['season'] < min(ps.TEST_SEASONS)]
    for market in pl.MARKETS:
        print(f"\n{market}")
        for season, rows in forecasts[forecasts['market'] == market].groupby('season'):
            rows = rows.dropna(subset=['result'])
            line = (f"  {season}: n {len(rows):>5}  MAE {np.mean(np.abs(rows['forecast'] - rows['result'])):7.2f}  "
                    f"r {np.corrcoef(rows['forecast'], rows['result'])[0, 1]:.3f}  bias {np.mean(rows['forecast'] - rows['result']):+6.2f}")
            if market in NAIVE:
                sub = played[(played['season'] == season)]
                sub = sub[pm.population(sub, pm.MARKET_GROUPS[market])]
                naive = sub[pm._p('slow', NAIVE[market])]
                result = pm.outcome(sub, market)
                ok = naive.notna() & result.notna()
                line += (f"   | recency average alone: MAE {np.mean(np.abs(naive[ok] - result[ok])):7.2f} "
                         f"r {np.corrcoef(naive[ok], result[ok])[0, 1]:.3f}")
            print(line)


def discovery():
    for which, label in (('close', 'closing lines'), ('open', 'opening lines')):
        rows = scored_lines(ps.DISCOVERY_SEASONS, which)
        title = f"ESPN BET {label} {ps.DISCOVERY_SEASONS[0]}-{ps.DISCOVERY_SEASONS[-1]}"
        market_table(rows, f"=== DISCOVERY: {title} (model side of every line) ===")
        for side in ('Over', 'Under'):
            market_table(rows[rows['side'] == side], f"--- {title}, model side = {side} ---")
        for season in ps.DISCOVERY_SEASONS:
            market_table(rows[rows['season'] == season], f"--- {label}, {season} only ---")
        rule_grid(rows, title)


def rule_grid(rows, title):
    print(f"\nRule grid ({title}): win % (n) of the model's side with chance >= threshold")
    thresholds = (0.55, 0.58, 0.60, 0.62, 0.65, 0.70)
    print(f"{'market':14} {'side':6} " + ' '.join(f"{t:>14.2f}" for t in thresholds))
    priced = [m for m in pl.MARKETS if m not in PRICE_MARKETS]
    for market in list(pl.MARKETS) + ['ALL', 'LINE-MOVING']:
        for side in ('both', 'Over', 'Under'):
            sub = rows if market == 'ALL' else rows[rows['market'].isin(priced)] if market == 'LINE-MOVING' \
                else rows[rows['market'] == market]
            if side != 'both':
                sub = sub[sub['side'] == side]
            cells = []
            for t in thresholds:
                rate, n = _rate(sub[sub['chance'] >= t]['outcome'])
                cells.append(f"{rate:6.1%} ({n:>5})" if n else f"{'-':>14}")
            print(f"{market:14} {side:6} " + ' '.join(cells))
    eligible = rows[rows['market'].isin(priced)]
    print("\nTop picks a week by the model's chance (line-moving markets): win % (n)")
    for top in (5, 10, 20, 40):
        best = eligible.sort_values('chance', ascending=False).groupby(['season', 'week']).head(top)
        rate, n = _rate(best['outcome'])
        print(f"  top {top:>2} a week: {rate:.1%} ({n}), lowest chance taken on average {best.groupby(['season', 'week'])['chance'].min().mean():.3f}")


def _picks(rows):
    return ps.rule_picks(rows)


def _pick_record(picks, label):
    rate, n = _rate(picks['outcome'])
    units = sum(ps.profit(o) for o in picks['outcome'].dropna())
    weeks = picks.groupby(['season', 'week']).ngroups or 1
    print(f"PICKS on {label}: {rate:.1%} of {n} graded, about {n / weeks:.0f} a week ({units:+.1f} units at -110)")
    for market, sub in picks.groupby('market'):
        r, k = _rate(sub['outcome'])
        print(f"    {market:14} {r:.1%} of {k}")
    print('    by chance: ' + ', '.join(f"{start:.2f}+ {rate:.1%} ({w + l})" for start, w, l, rate in ps.hit_table(picks)
                                       if rate is not None))


def _close_known(rows):
    return ps.close_known(rows)


def _clv(rows):
    """For over bets at the opening line: the line's move to the close as a share of the opener (+ = toward the over)."""
    known = rows[_close_known(rows) & (rows['open_line'] > 0)]
    return (known['line'] - known['open_line']) / known['open_line'], known


def clv_table(rows, title):
    print(f"\n{title}")
    print(f"{'chance':>8} {'bets':>6} {'win@open':>9} | {'with close':>10} {'avg move':>9} {'t':>6} {'rose':>6} {'fell':>6} {'win@close':>10}")
    for i, start in enumerate(ps.CHANCE_BINS + (None,)):
        if start is None:
            sub, label = rows, 'all'
        else:
            end = ps.CHANCE_BINS[i + 1] if i + 1 < len(ps.CHANCE_BINS) else 1.01
            sub, label = rows[(rows['chance'] >= start) & (rows['chance'] < end)], f"{start:.2f}+"
        rate, n = _rate(sub['outcome'])
        move, known = _clv(sub)
        late, _ = _rate(pd.Series([ps.grade('Over', l, r) for l, r in zip(known['line'], known['result'])], dtype=object))
        t = move.mean() / move.std() * np.sqrt(len(move)) if len(move) > 2 and move.std() > 0 else np.nan
        print(f"{label:>8} {n:>6} {rate:>9.1%} | {len(move):>10} {move.mean():>+9.2%} {t:>6.1f} {(move > 0).mean():>6.1%} "
              f"{(move < 0).mean():>6.1%} {late:>10.1%}")


def over_grid(rows, title):
    print(f"\nOver grid ({title}): win % at the opening line (n) [average line move to the close]")
    thresholds = (0.55, 0.60, 0.65, 0.70, 0.75)
    print(f"{'market':14} " + ' '.join(f"{t:>22.2f}" for t in thresholds))
    priced = [m for m in pl.MARKETS if m not in PRICE_MARKETS]
    for market in list(pl.MARKETS) + ['ALL', 'LINE-MOVING']:
        sub = rows if market == 'ALL' else rows[rows['market'].isin(priced)] if market == 'LINE-MOVING' \
            else rows[rows['market'] == market]
        cells = []
        for t in thresholds:
            picked = sub[sub['chance'] >= t]
            rate, n = _rate(picked['outcome'])
            move, _ = _clv(picked)
            cells.append(f"{rate:6.1%} ({n:>4}) [{move.mean():+5.1%}]" if n else f"{'-':>22}")
        print(f"{market:14} " + ' '.join(cells))


def overs():
    """The over side: the early-week variant at opening lines (with closing-line value), and closing-line checks."""
    _, early = load_early()
    _, full, lines = load()
    disc = lines[lines['season'].isin(ps.DISCOVERY_SEASONS)]
    title = f"ESPN BET {ps.DISCOVERY_SEASONS[0]}-{ps.DISCOVERY_SEASONS[-1]}"
    over = ps.score_history(disc, early, 'open')
    over = over[over['side'] == 'Over']
    market_table(over, f"=== EARLY-WEEK VARIANT (no injury report, opening game lines), {title} OPENING lines, "
                       f"model side = Over ===")
    clv_table(over, "Closing-line value of those overs, by the model's chance")
    for season in ps.DISCOVERY_SEASONS:
        market_table(over[over['season'] == season], f"--- {season} only ---")
    over_grid(over, f"early-week variant, {title}")
    full_open = ps.score_history(disc, full, 'open')
    full_open = full_open[full_open['side'] == 'Over']
    clv_table(full_open, "For comparison, the full model (final injury report, closing game lines) at the opening lines")
    closes = ps.score_history(disc, full, 'close')
    over_close = closes[closes['side'] == 'Over']
    move = np.sign(over_close['line'] - over_close['open_line'])
    for label, mask in (('the line rose since it opened', move > 0), ('the line never moved', move == 0),
                        ('the line fell since it opened', move < 0)):
        market_table(over_close[mask], f"--- full model at the CLOSING lines, model side = Over, {label} ---")


def overs_test():
    """The frozen over rule: its discovery record, then the untouched DraftKings 2026 opening lines."""
    _, early = load_early()
    _, _, lines = load()
    print(f"\n=== THE FROZEN OVER RULE: Over at the opening line, early-week chance >= {ps.OVER_MIN_CHANCE}, "
          f"markets {', '.join(ps.OVER_MARKETS)} ===")
    for label, seasons in ((f"discovery, ESPN BET {ps.DISCOVERY_SEASONS[0]}-{ps.DISCOVERY_SEASONS[-1]} opening lines",
                            ps.DISCOVERY_SEASONS),
                           (f"TEST, DraftKings {ps.TEST_SEASONS[0]} opening lines (untouched until the rule was frozen)",
                            ps.TEST_SEASONS)):
        rows = ps.score_history(lines[lines['season'].isin(seasons)], early, 'open')
        picks = ps.over_picks(rows)
        print()
        _pick_record(picks, label)
        clv_table(picks, f"Closing-line value of the picks ({label}; only lines whose close is known)")
        market_table(rows[rows['side'] == 'Over'], f"--- {label}: every line the early-week model leans Over ---")


def test():
    if ps.MIN_CHANCE is None:
        print("\nThe pick rule is not frozen yet (nfl_prop_signal.MIN_CHANCE is None); the test lines stay untouched.")
        return
    rule = f"{'/'.join(ps.PICK_SIDES or ('both sides',))}, chance >= {ps.MIN_CHANCE}, markets {', '.join(ps.PICK_MARKETS or ('all',))}"
    print(f"\n=== THE FROZEN RULE ({rule}) ON THE DISCOVERY LINES ===")
    for which, label in (('close', 'ESPN BET closing lines 2024-2025'), ('open', 'ESPN BET opening lines 2024-2025')):
        _pick_record(_picks(scored_lines(ps.DISCOVERY_SEASONS, which)), label)
    print(f"\n=== TEST: DraftKings {ps.TEST_SEASONS[0]} lines, untouched until the rule was frozen ===")
    for which in ('close', 'open'):
        rows = scored_lines(ps.TEST_SEASONS, which)
        label = 'opening lines' if which == 'open' else 'closing lines last moved before kickoff'
        market_table(rows, f"--- every DraftKings {label} (model side) ---")
        market_table(rows[rows['side'] == 'Under'], f"--- every DraftKings {label}, model side = Under ---")
        _pick_record(_picks(rows), f"DraftKings {label}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    pd.set_option('display.width', 200)
    commands = sys.argv[1:] or ['quality', 'discovery', 'test']
    for command in commands:
        {'quality': quality, 'discovery': discovery, 'test': test, 'overs': overs, 'overs_test': overs_test}[command]()
