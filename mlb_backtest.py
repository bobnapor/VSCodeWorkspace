"""
mlb_backtest.py
---------------
Backtest the mlbpicks.py model against historical odds data
(2010-2021) from sportsbookreviewsonline.com.

For every game in the odds files:
  - Train on all completed schedule games BEFORE that game's date
    using that year's team stats
  - Predict ML winner, run-line cover, O/U result
  - Compare against actual scores and book closing lines
  - Track accuracy and flat-bet ROI (risking $100 per game)

Usage:
  python mlb_backtest.py                        # all years 2010-2021
  python mlb_backtest.py --years 2018 2019 2020
  python mlb_backtest.py --years 2021 --runs 200
  python mlb_backtest.py --save                 # save full results CSV
"""

import os
import sys
import argparse
import warnings
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Import all shared helpers from mlbpicks
from mlbpicks import (
    MLB_STATS_DIR,
    normalize_name,
    load_team_stats,
    load_schedule,
    load_pitcher_stats,
    load_starters_csv,
    get_completed_before,
    build_training_data,
    build_training_data_multiyear,
    predict_games,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
ODDS_FILE_TPL = 'mlb-odds-{year}.xlsx'
DEFAULT_YEARS = list(range(2010, 2022))   # 2010 through 2021
DEFAULT_RUNS = 200     # lower for speed; use 500 for final runs
MIN_TRAINING_GAMES = 20  # skip games early in season

# ROI constants — flat $100 bet
FLAT_BET = 100

# ---------------------------------------------------------------------------
# ODDS FILE ABBREVIATION -> canonical name
# (handled by normalize_name() via NAME_MAP in mlbpicks.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ODDS FILE LOADER
# ---------------------------------------------------------------------------

def _odds_path(year):
    return os.path.join(MLB_STATS_DIR, ODDS_FILE_TPL.format(year=year))


def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def load_odds_file(year):
    """
    Load mlb-odds-yyyy.xlsx and return a list of game dicts:

    Each game dict:
        date        : date object
        year        : int
        away_team   : str  (normalized)
        home_team   : str  (normalized)
        away_score  : int
        home_score  : int
        away_ml     : float  (American ML, visitor closing line)
        home_ml     : float  (American ML, home closing line)
        run_line    : float  (always 1.5, away perspective)
        rl_away_odds: float  (run line juice on away +1.5)
        rl_home_odds: float  (run line juice on home -1.5)
        ou_line     : float  (closing O/U)
        over_odds   : float
        under_odds  : float
    """
    path = _odds_path(year)
    if not os.path.exists(path):
        print(f'  [WARN] Odds file not found: {path}')
        return []

    df = pd.read_excel(path)

    # Normalise column names — 2010 uses 'Open OU'/'Close OU',
    # 2011+ uses 'OpenOU'/'CloseOU' and adds RunLine column
    df.columns = [str(c).strip() for c in df.columns]

    # Rename ambiguous unnamed columns by position
    # Columns: Date Rot VH Team Pitcher 1-9 Final Open Close
    #          [RunLine RL_odds]  OpenOU OU_odds CloseOU OU_odds2
    cols = list(df.columns)

    # Detect whether RunLine column exists (2011+)
    has_rl = 'RunLine' in cols

    # Standardise O/U column names
    for old, new in [
        ('Open OU', 'OpenOU'),
        ('Close OU', 'CloseOU'),
    ]:
        if old in cols:
            df.rename(columns={old: new}, inplace=True)
    cols = list(df.columns)

    # The unnamed columns after RunLine / OpenOU / CloseOU are the juice cols
    # Layout (2011+): ..Close RunLine Unnamed:18 OpenOU Unnamed:20 CloseOU Unnamed:22
    # Layout (2010):  ..Close OpenOU  Unnamed:18 CloseOU Unnamed:20
    # Rename unnamed cols positionally
    unnamed = [c for c in cols if str(c).startswith('Unnamed')]
    if has_rl and len(unnamed) >= 3:
        df.rename(columns={
            unnamed[0]: 'rl_away_odds',
            unnamed[1]: 'ou_open_odds',
            unnamed[2]: 'ou_close_odds',
        }, inplace=True)
    elif not has_rl and len(unnamed) >= 2:
        df.rename(columns={
            unnamed[0]: 'ou_open_odds',
            unnamed[1]: 'ou_close_odds',
        }, inplace=True)

    # Games come in V/H pairs on the same Date+Rot block
    # Separate visitor and home rows
    v_rows = df[df['VH'] == 'V'].reset_index(drop=True)
    h_rows = df[df['VH'] == 'H'].reset_index(drop=True)

    if len(v_rows) != len(h_rows):
        print(
            f'  [WARN] {year}: V/H row mismatch '
            f'({len(v_rows)} vs {len(h_rows)}), truncating to shorter.'
        )
        n = min(len(v_rows), len(h_rows))
        v_rows = v_rows.iloc[:n]
        h_rows = h_rows.iloc[:n]

    games = []
    for v, h in zip(v_rows.itertuples(), h_rows.itertuples()):
        # Date is stored as MMDD integer (e.g. 401 = April 1)
        raw_date = int(getattr(v, 'Date', 0))
        month = raw_date // 100
        day = raw_date % 100
        try:
            game_date = date(year, month, day)
        except ValueError:
            continue   # skip malformed dates

        away = normalize_name(str(getattr(v, 'Team', '')).strip())
        home = normalize_name(str(getattr(h, 'Team', '')).strip())

        try:
            away_score = int(getattr(v, 'Final'))
            home_score = int(getattr(h, 'Final'))
        except (ValueError, TypeError):
            continue   # skip games without final score

        # Moneyline — Close column: visitor odds on V row, home on H row
        away_ml = _to_float(getattr(v, 'Close', None))
        home_ml = _to_float(getattr(h, 'Close', None))

        # Run line (always ±1.5; juice in rl_away_odds / rl_home_odds)
        run_line = 1.5
        if has_rl:
            rl_away_odds = _to_float(
                getattr(v, 'rl_away_odds', None)
            )
            rl_home_odds = _to_float(
                getattr(h, 'rl_away_odds', None)
            )
        else:
            rl_away_odds = None
            rl_home_odds = None

        # O/U — use CloseOU from visitor row (same for both rows)
        ou_line = _to_float(getattr(v, 'CloseOU', None))
        over_odds = _to_float(getattr(v, 'ou_close_odds', None))
        under_odds = _to_float(getattr(h, 'ou_close_odds', None))

        games.append({
            'date': game_date,
            'year': year,
            'away_team': away,
            'home_team': home,
            'away_score': away_score,
            'home_score': home_score,
            'away_ml': away_ml,
            'home_ml': home_ml,
            'run_line': run_line,
            'rl_away_odds': rl_away_odds,
            'rl_home_odds': rl_home_odds,
            'ou_line': ou_line,
            'over_odds': over_odds,
            'under_odds': under_odds,
        })

    return games


# ---------------------------------------------------------------------------
# ROI CALCULATION
# ---------------------------------------------------------------------------

def _ml_payout(bet, odds):
    """
    Flat-bet $bet on a winner at American moneyline odds.
    Returns net profit/loss.
    """
    if odds is None:
        return 0.0
    if odds > 0:
        return bet * odds / 100
    else:
        return bet * 100 / abs(odds)


def _rl_payout(bet, odds):
    """Run line / O/U bet payout (same formula as ML)."""
    return _ml_payout(bet, odds)


# ---------------------------------------------------------------------------
# BACKTEST CORE
# ---------------------------------------------------------------------------

def backtest_year(year, schedule, team_stats, num_runs=DEFAULT_RUNS,
                  prior_games_by_year=None, prior_stats_by_year=None):
    """
    Backtest one year.  Returns list of result dicts (one per game).

    prior_games_by_year : {yr: [completed game dicts]}  — full prior seasons
    prior_stats_by_year : {yr: team_stats dict}          — stats for those years
    When provided, each game's training set = current-season games before
    that date PLUS all games from each prior season.
    """
    odds_games = load_odds_file(year)
    if not odds_games:
        return []

    print(f'  {year}: {len(odds_games)} odds games loaded.')

    # Load pitcher stats and starters lookup for this year (and prior years)
    pitcher_stats_cur = load_pitcher_stats(year)
    starters_cur = load_starters_csv(year)
    # Pre-load prior-year pitcher stats and starters once (not per-game)
    if prior_games_by_year:
        _ps_by_year = {year: pitcher_stats_cur,
                       **{py: load_pitcher_stats(py)
                          for py in prior_games_by_year}}
        _sl_by_year = {year: starters_cur,
                       **{py: load_starters_csv(py)
                          for py in prior_games_by_year}}
    else:
        _ps_by_year = {}
        _sl_by_year = {}

    results = []
    # Cache training data by date to avoid recomputing for same date
    _cache = {}

    for g in odds_games:
        game_date = g['date']

        if game_date not in _cache:
            current_completed = get_completed_before(schedule, game_date)

            if prior_games_by_year and prior_stats_by_year:
                # Combine: current-season games + all prior-season games
                games_by_year = dict(prior_games_by_year)
                games_by_year[year] = current_completed
                stats_by_year = dict(prior_stats_by_year)
                stats_by_year[year] = team_stats
                # Update current-year entry in pre-loaded dicts
                _ps_by_year[year] = pitcher_stats_cur
                _sl_by_year[year] = starters_cur
                total_games = (
                    sum(len(v) for v in games_by_year.values())
                )
                if total_games < MIN_TRAINING_GAMES:
                    _cache[game_date] = None
                else:
                    X, y_diff, y_total = build_training_data_multiyear(
                        games_by_year, stats_by_year,
                        pitcher_stats_by_year=_ps_by_year,
                        starters_by_year=_sl_by_year,
                    )
                    _cache[game_date] = (
                        X, y_diff, y_total, total_games
                    )
            else:
                # Current season only
                if len(current_completed) < MIN_TRAINING_GAMES:
                    _cache[game_date] = None
                else:
                    X, y_diff, y_total = build_training_data(
                        current_completed, team_stats,
                        pitcher_stats=pitcher_stats_cur,
                        starters_lookup=starters_cur,
                    )
                    _cache[game_date] = (
                        X, y_diff, y_total, len(current_completed)
                    )

        cached = _cache[game_date]
        if cached is None:
            continue   # not enough training data yet

        X, y_diff, y_total, n_train = cached

        # Build a single-game prediction
        away, home = g['away_team'], g['home_team']
        if away not in team_stats or home not in team_stats:
            continue

        preds = predict_games(
            [{'away_team': away, 'home_team': home,
              'completed': False, 'date': game_date,
              'away_score': None, 'home_score': None, 'winner': None}],
            team_stats, X, y_diff, y_total,
            num_runs=num_runs,
            run_lines_map={(away, home): g['run_line']},
        )
        if not preds:
            continue

        p = preds[0]

        # ---- Actual outcomes ----
        actual_diff = g['home_score'] - g['away_score']
        actual_total = g['home_score'] + g['away_score']
        actual_winner = home if actual_diff > 0 else away

        # ---- ML evaluation ----
        ml_correct = (p['ml_pick'] == actual_winner)
        # ROI: bet on model's ML pick using book odds
        if p['ml_pick'] == home:
            ml_odds_used = g['home_ml']
        else:
            ml_odds_used = g['away_ml']
        if ml_correct and ml_odds_used is not None:
            ml_profit = _ml_payout(FLAT_BET, ml_odds_used)
        else:
            ml_profit = -FLAT_BET if ml_odds_used is not None else 0.0
        ml_wagered = FLAT_BET if ml_odds_used is not None else 0.0

        # ---- Run line evaluation ----
        rl = g['run_line']   # 1.5 away perspective
        # Model's RL pick is embedded in p['rl_pick']
        # home covers if actual_diff > rl
        # away covers if actual_diff < -rl
        # push if |actual_diff| <= rl (shouldn't happen with 1.5)
        home_covered = actual_diff > rl
        away_covered = actual_diff < -rl
        model_pick_home_rl = (p['home_cover_pct'] >= p['away_cover_pct'])

        if model_pick_home_rl:
            rl_correct = home_covered
            rl_odds_used = g['rl_home_odds'] if g['rl_home_odds'] else -110
        else:
            rl_correct = away_covered
            rl_odds_used = g['rl_away_odds'] if g['rl_away_odds'] else -110

        if home_covered or away_covered:   # no push
            if rl_correct:
                rl_profit = _rl_payout(FLAT_BET, rl_odds_used)
            else:
                rl_profit = -FLAT_BET
            rl_wagered = FLAT_BET
        else:
            rl_profit = 0.0
            rl_wagered = 0.0

        # ---- O/U evaluation ----
        ou_line = g['ou_line']
        ou_correct = None
        ou_profit = 0.0
        ou_wagered = 0.0
        if ou_line is not None:
            totals = p['_totals']
            model_over_pct = float(np.mean(totals > ou_line)) * 100
            model_picks_over = model_over_pct >= 50

            actual_over = actual_total > ou_line
            actual_under = actual_total < ou_line

            if actual_over or actual_under:   # no push
                ou_correct = (
                    (model_picks_over and actual_over) or
                    (not model_picks_over and actual_under)
                )
                if model_picks_over:
                    juice = g['over_odds'] if g['over_odds'] else -110
                else:
                    juice = g['under_odds'] if g['under_odds'] else -110

                if ou_correct:
                    ou_profit = _rl_payout(FLAT_BET, juice)
                else:
                    ou_profit = -FLAT_BET
                ou_wagered = FLAT_BET

        results.append({
            'year': year,
            'date': game_date,
            'away_team': away,
            'home_team': home,
            'away_score': g['away_score'],
            'home_score': g['home_score'],
            'n_train': n_train,
            # ML
            'ml_pick': p['ml_pick'],
            'ml_conf': p['ml_conf'],
            'ml_correct': ml_correct,
            'ml_profit': ml_profit,
            'ml_wagered': ml_wagered,
            # Run line
            'rl_pick': p['rl_pick'],
            'rl_conf': p['rl_conf'],
            'rl_correct': rl_correct,
            'rl_profit': rl_profit,
            'rl_wagered': rl_wagered,
            # O/U
            'ou_line': ou_line,
            'ou_correct': ou_correct,
            'ou_profit': ou_profit,
            'ou_wagered': ou_wagered,
            # raw model outputs
            'proj_home': p['proj_home'],
            'proj_away': p['proj_away'],
            'home_win_pct': p['home_win_pct'],
        })

    return results


# ---------------------------------------------------------------------------
# SUMMARY REPORTING
# ---------------------------------------------------------------------------

def _roi(profit, wagered):
    if wagered == 0:
        return 0.0
    return profit / wagered * 100


def print_summary(all_results):
    df = pd.DataFrame(all_results)
    if df.empty:
        print('No results to summarise.')
        return

    print(f'\n{"=" * 72}')
    print('  MLB BACKTEST RESULTS')
    print(f'{"=" * 72}')
    print(
        f'  {"Year":<6} {"Games":>6} '
        f'{"ML%":>6} {"ML ROI":>8} '
        f'{"RL%":>6} {"RL ROI":>8} '
        f'{"OU%":>6} {"OU ROI":>8}'
    )
    print(f'  {"-" * 66}')

    for year in sorted(df['year'].unique()):
        yr = df[df['year'] == year]
        n = len(yr)

        ml_acc = yr['ml_correct'].mean() * 100
        ml_roi = _roi(yr['ml_profit'].sum(), yr['ml_wagered'].sum())

        rl_acc = yr['rl_correct'].mean() * 100
        rl_roi = _roi(yr['rl_profit'].sum(), yr['rl_wagered'].sum())

        ou = yr[yr['ou_correct'].notna()]
        if len(ou):
            ou_acc = ou['ou_correct'].mean() * 100
            ou_roi = _roi(ou['ou_profit'].sum(), ou['ou_wagered'].sum())
        else:
            ou_acc = 0.0
            ou_roi = 0.0

        print(
            f'  {year:<6} {n:>6} '
            f'{ml_acc:>5.1f}% {ml_roi:>+7.1f}% '
            f'{rl_acc:>5.1f}% {rl_roi:>+7.1f}% '
            f'{ou_acc:>5.1f}% {ou_roi:>+7.1f}%'
        )

    print(f'  {"-" * 66}')

    # Totals
    n = len(df)
    ml_acc = df['ml_correct'].mean() * 100
    ml_roi = _roi(df['ml_profit'].sum(), df['ml_wagered'].sum())
    rl_acc = df['rl_correct'].mean() * 100
    rl_roi = _roi(df['rl_profit'].sum(), df['rl_wagered'].sum())
    ou = df[df['ou_correct'].notna()]
    ou_acc = ou['ou_correct'].mean() * 100 if len(ou) else 0.0
    ou_roi = _roi(ou['ou_profit'].sum(), ou['ou_wagered'].sum()) if len(ou) else 0.0

    print(
        f'  {"TOTAL":<6} {n:>6} '
        f'{ml_acc:>5.1f}% {ml_roi:>+7.1f}% '
        f'{rl_acc:>5.1f}% {rl_roi:>+7.1f}% '
        f'{ou_acc:>5.1f}% {ou_roi:>+7.1f}%'
    )
    print(f'{"=" * 72}')
    print(
        '  ML%/RL%/OU% = pick accuracy  |  ROI = return on $100 flat bets'
    )
    print(f'{"=" * 72}\n')

    # High-confidence breakdown
    _print_confidence_breakdown(df)


def _print_confidence_breakdown(df):
    """Show accuracy/ROI split by model confidence tier."""
    print(f'\n{"=" * 72}')
    print('  MONEYLINE — ACCURACY BY CONFIDENCE TIER')
    print(f'{"=" * 72}')
    print(
        f'  {"Tier":<12} {"Games":>6} {"Acc%":>7} '
        f'{"ROI":>8} {"Profit":>10} {"Wagered":>10}'
    )
    print(f'  {"-" * 60}')

    tiers = [
        ('HIGH (>=70%)',   df['ml_conf'] >= 70),
        ('MEDIUM (60-69%)', (df['ml_conf'] >= 60) & (df['ml_conf'] < 70)),
        ('LOW (<60%)',      df['ml_conf'] < 60),
    ]
    for label, mask in tiers:
        sub = df[mask]
        if len(sub) == 0:
            continue
        acc = sub['ml_correct'].mean() * 100
        roi = _roi(sub['ml_profit'].sum(), sub['ml_wagered'].sum())
        profit = sub['ml_profit'].sum()
        wagered = sub['ml_wagered'].sum()
        print(
            f'  {label:<12} {len(sub):>6} {acc:>6.1f}% '
            f'{roi:>+7.1f}% {profit:>+10.0f} {wagered:>10.0f}'
        )

    print(f'{"=" * 72}\n')


# ---------------------------------------------------------------------------
# TRAINING-WINDOW SWEEP
# ---------------------------------------------------------------------------

def sweep_training_windows(test_years, all_schedules, all_stats,
                           max_prior_years=6, num_runs=DEFAULT_RUNS):
    """
    For each test_year, backtest at every training window size from 0
    (current season only) up to max_prior_years additional seasons.

    test_years    : list of years to evaluate (need prior data available)
    all_schedules : {year: schedule list}
    all_stats     : {year: team_stats dict}
    max_prior_years : largest window to test

    Returns: dict  {window_size: [result dicts across all test years]}
    """
    window_results = {w: [] for w in range(max_prior_years + 1)}

    for test_year in test_years:
        if test_year not in all_schedules or test_year not in all_stats:
            print(f'  [SKIP] {test_year}: missing schedule or stats.')
            continue

        schedule = all_schedules[test_year]
        team_stats = all_stats[test_year]

        for window in range(max_prior_years + 1):
            label = (
                f'window={window}'
                f' ({test_year}{"+" if window else ""}'
                f'{window if window else ""})'
            )

            if window == 0:
                # Current season only
                results = backtest_year(
                    test_year, schedule, team_stats,
                    num_runs=num_runs,
                )
            else:
                # Collect `window` full prior seasons
                prior_games = {}
                prior_stats = {}
                for offset in range(1, window + 1):
                    py = test_year - offset
                    if py in all_schedules and py in all_stats:
                        prior_games[py] = [
                            g for g in all_schedules[py]
                            if g['completed']
                            and g['away_score'] is not None
                        ]
                        prior_stats[py] = all_stats[py]

                if not prior_games:
                    # No prior data available, skip
                    continue

                results = backtest_year(
                    test_year, schedule, team_stats,
                    num_runs=num_runs,
                    prior_games_by_year=prior_games,
                    prior_stats_by_year=prior_stats,
                )

            # Tag each result with the window size
            for r in results:
                r['window'] = window
            window_results[window].extend(results)

            n = len(results)
            if n > 0:
                df = pd.DataFrame(results)
                ml_acc = df['ml_correct'].mean() * 100
                ml_roi = _roi(
                    df['ml_profit'].sum(), df['ml_wagered'].sum()
                )
                print(
                    f'    {label:<30} '
                    f'n={n:>4}  ML={ml_acc:.1f}%  ROI={ml_roi:+.1f}%'
                )

    return window_results


def print_window_sweep_summary(window_results):
    """Print a table comparing accuracy/ROI across training window sizes."""
    print(f'\n{"=" * 72}')
    print('  TRAINING WINDOW SWEEP — RESULTS')
    print(f'{"=" * 72}')
    print(
        f'  {"Window":<22} {"Games":>6} '
        f'{"ML%":>6} {"ML ROI":>8} '
        f'{"RL%":>6} {"RL ROI":>8} '
        f'{"OU%":>6} {"OU ROI":>8}'
    )
    print(f'  {"-" * 68}')

    best_ml_roi = -999
    best_window = 0

    for window in sorted(window_results):
        results = window_results[window]
        if not results:
            continue
        df = pd.DataFrame(results)

        n = len(df)
        ml_acc = df['ml_correct'].mean() * 100
        ml_roi = _roi(df['ml_profit'].sum(), df['ml_wagered'].sum())
        rl_acc = df['rl_correct'].mean() * 100
        rl_roi = _roi(df['rl_profit'].sum(), df['rl_wagered'].sum())
        ou = df[df['ou_correct'].notna()]
        ou_acc = ou['ou_correct'].mean() * 100 if len(ou) else 0.0
        ou_roi = (
            _roi(ou['ou_profit'].sum(), ou['ou_wagered'].sum())
            if len(ou) else 0.0
        )

        if window == 0:
            label = 'Current season only'
        else:
            label = f'+{window} prior season{"s" if window > 1 else ""}'

        print(
            f'  {label:<22} {n:>6} '
            f'{ml_acc:>5.1f}% {ml_roi:>+7.1f}% '
            f'{rl_acc:>5.1f}% {rl_roi:>+7.1f}% '
            f'{ou_acc:>5.1f}% {ou_roi:>+7.1f}%'
        )

        if ml_roi > best_ml_roi:
            best_ml_roi = ml_roi
            best_window = window

    print(f'{"=" * 72}')
    if window == 0:
        best_label = 'current season only'
    else:
        best_label = f'+{best_window} prior season(s)'
    print(
        f'  >> Best ML ROI: window={best_window} ({best_label})'
        f'  ROI={best_ml_roi:+.1f}%'
    )
    print(f'{"=" * 72}\n')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    global MIN_TRAINING_GAMES

    parser = argparse.ArgumentParser(
        description='MLB backtest: model vs historical odds 2010-2021'
    )
    parser.add_argument(
        '--years', nargs='+', type=int, default=DEFAULT_YEARS,
        help='Years to backtest (default: 2010-2021)'
    )
    parser.add_argument(
        '--runs', type=int, default=DEFAULT_RUNS,
        help=f'Monte Carlo iterations per game (default: {DEFAULT_RUNS})'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save full game-by-game results to CSV'
    )
    parser.add_argument(
        '--min_games', type=int, default=MIN_TRAINING_GAMES,
        help=(
            f'Minimum completed games before predicting '
            f'(default: {MIN_TRAINING_GAMES})'
        )
    )
    parser.add_argument(
        '--sweep', action='store_true',
        help='Sweep training window sizes (0..max_prior) to find optimal window'
    )
    parser.add_argument(
        '--max_prior', type=int, default=6,
        help='Maximum number of prior seasons to sweep (default: 6)'
    )
    args = parser.parse_args()

    MIN_TRAINING_GAMES = args.min_games

    # ------------------------------------------------------------------
    # Pre-load ALL requested years (needed for both modes)
    # ------------------------------------------------------------------
    all_schedules = {}
    all_stats = {}

    EARLIEST_YEAR = 2010   # oldest year with odds + stats data available

    needed_years = set(args.years)
    if args.sweep:
        # For the sweep we also need prior years' data; clamp to earliest available
        for yr in list(needed_years):
            for offset in range(1, args.max_prior + 1):
                py = yr - offset
                if py >= EARLIEST_YEAR:
                    needed_years.add(py)

    for year in sorted(needed_years):
        team_stats = load_team_stats(year)
        schedule = load_schedule(year)
        if team_stats and schedule:
            all_schedules[year] = schedule
            all_stats[year] = team_stats
        else:
            print(f'  [INFO] {year}: missing data (stats={bool(team_stats)}, sched={bool(schedule)})')

    # ------------------------------------------------------------------
    # SWEEP MODE
    # ------------------------------------------------------------------
    if args.sweep:
        # Only test years that are in args.years AND have data loaded
        test_years = sorted(yr for yr in args.years if yr in all_schedules)
        if not test_years:
            print('No test years with data available. Exiting.')
            sys.exit(1)

        print(
            f'\nSweeping training windows 0–{args.max_prior} '
            f'for test years: {test_years}'
        )
        sweep_results = sweep_training_windows(
            test_years, all_schedules, all_stats,
            max_prior_years=args.max_prior,
            num_runs=args.runs,
        )
        print_window_sweep_summary(sweep_results)

        if args.save:
            out_path = 'C:/Users/Bobby/mlb_sweep_results.csv'
            rows = []
            for results in sweep_results.values():
                rows.extend(results)
            if rows:
                df = pd.DataFrame(rows)
                df.drop(columns=['_totals'], errors='ignore', inplace=True)
                df.to_csv(out_path, index=False)
                print(f'  Sweep results saved to: {out_path}')
        return

    # ------------------------------------------------------------------
    # STANDARD BACKTEST MODE
    # ------------------------------------------------------------------
    all_results = []

    for year in sorted(args.years):
        if year not in all_schedules:
            print(f'  [SKIP] {year}: data not loaded.')
            continue

        schedule = all_schedules[year]
        team_stats = all_stats[year]

        print(f'\n[{year}] Backtesting {len(schedule)} scheduled games...')
        year_results = backtest_year(
            year, schedule, team_stats, num_runs=args.runs
        )
        print(f'  Predicted: {len(year_results)} games.')
        all_results.extend(year_results)

    if not all_results:
        print('\nNo results generated.')
        sys.exit(1)

    print_summary(all_results)

    if args.save:
        out_path = 'C:/Users/Bobby/mlb_backtest_results.csv'
        df = pd.DataFrame(all_results)
        df.drop(columns=['_totals'], errors='ignore', inplace=True)
        df.to_csv(out_path, index=False)
        print(f'  Full results saved to: {out_path}')


if __name__ == '__main__':
    main()
