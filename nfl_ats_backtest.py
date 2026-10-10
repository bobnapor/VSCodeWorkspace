"""
NFL ATS Backtest Script

Backtests the ATS prediction model against historical spreads.
You provide a CSV of historical games WITH actual scores and spreads;
the script simulates predicting each game using only data available
at the time, then measures cover accuracy.

Historical spreads CSV format (e.g. spreads_historical.csv):
  week, year, home_team, away_team, spread, home_score, away_score

  spread      : Vegas line (negative = home favored)
  home_score  : actual final home score
  away_score  : actual final away score

The script calculates whether each team covered and compares
that to the model's prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import logging
import concurrent.futures
import os
from datetime import datetime
import time

# Import core functions from the main prediction script
from nflpicks_v2_0_multithreaded_web import (
    load_data_for_year,
    preprocess_data,
    predict_game,
    translate_week_number,
    get_last_regular_season_week,
    off_url,
    def_url,
    games_url,
)

# Path to your historical spreads file
HISTORICAL_SPREADS_FILE = 'spreads_historical.csv'


def load_historical_spreads(filepath):
    """
    Load historical spreads with actual scores from CSV.

    Required columns:
      week, year, home_team, away_team, spread, home_score, away_score
    """
    df = pd.read_csv(filepath)
    required = {
        'week', 'year', 'home_team', 'away_team',
        'spread', 'home_score', 'away_score'
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in historical file: {missing}")

    df['week'] = df['week'].astype(str)
    df['year'] = df['year'].astype(int)
    df['spread'] = df['spread'].astype(float)
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    # Compute actual margin (home - away) and whether home covered
    df['actual_margin'] = df['home_score'] - df['away_score']
    # Home covers if actual_margin > spread (e.g. won by more than the line)
    df['home_covered'] = df['actual_margin'] > df['spread']
    # Push: actual_margin == spread
    df['push'] = df['actual_margin'] == df['spread']

    return df


def run_ats_backtest_iteration(iteration, cached_data, games_df, test_year, min_test_week_num):
    """
    Run a single backtest iteration. Trains only on data available
    before the earliest test week, then predicts each game.
    """
    multi_year_combined_stats = cached_data['stats'].copy()
    all_games = cached_data['all_games'].copy()

    # Translate week numbers
    all_games['week_num_int'] = all_games.apply(
        lambda row: translate_week_number(row['week_number'], row['year']),
        axis=1
    )
    all_games['year_int'] = all_games['year'].astype(int)
    all_games = all_games[all_games['week_num_int'].notna()].copy()
    all_games['week_num_int'] = all_games['week_num_int'].astype(int)

    # Training data: completed games strictly before the test period
    def is_valid_training_game(row):
        last_reg_week = get_last_regular_season_week(row['year_int'])
        if row['score_difference'] == 0:
            return False
        if row['week_num_int'] == last_reg_week:
            return False
        if row['year_int'] < test_year:
            return True
        if row['year_int'] == test_year and row['week_num_int'] < min_test_week_num:
            return True
        return False

    training_games = all_games[all_games.apply(is_valid_training_game, axis=1)]

    merged_data, features = preprocess_data(
        multi_year_combined_stats, training_games
    )

    X = merged_data[features]
    y = merged_data['score_difference']

    X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X))
    X_train, _, y_train, _ = train_test_split(
        X_resampled, y_resampled, test_size=0.2
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    iteration_predictions = {}
    for idx, game in games_df.iterrows():
        try:
            base_pred = predict_game(
                game['home_team'], game['away_team'], str(game['year']),
                multi_year_combined_stats, model
            )
            noise = np.random.normal(0, 1.0)
            iteration_predictions[idx] = round(base_pred + noise, 1)
        except ValueError:
            pass

    return iteration_predictions


def calculate_ats_metrics(all_predictions, games_df, num_runs):
    """
    Aggregate predictions and compute ATS accuracy metrics.
    """
    results = []

    for idx, game in games_df.iterrows():
        preds = np.array(all_predictions.get(idx, []))
        if len(preds) == 0:
            continue

        spread = game['spread']
        median_margin = float(np.median(preds))
        mean_margin = float(np.mean(preds))
        home_cover_pct = float(np.mean(preds > spread)) * 100.0
        away_cover_pct = 100.0 - home_cover_pct
        edge = median_margin - spread

        # Model pick
        model_picks_home = home_cover_pct >= 50.0

        # Actual result
        actual_margin = game['actual_margin']
        home_covered = game['home_covered']
        is_push = game['push']

        # Did the model pick correctly?
        if is_push:
            pick_correct = None  # push - no result
        else:
            pick_correct = (model_picks_home == home_covered)

        # Spread label
        if spread <= 0:
            home_label = f"{game['home_team']} {spread:+.1f}"
            away_label = f"{game['away_team']} {-spread:+.1f}"
        else:
            home_label = f"{game['home_team']} +{spread:.1f}"
            away_label = f"{game['away_team']} -{spread:.1f}"

        model_pick_label = home_label if model_picks_home else away_label
        model_pick_pct = home_cover_pct if model_picks_home else away_cover_pct

        results.append({
            'week': game['week'],
            'year': game['year'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'spread': spread,
            'actual_margin': actual_margin,
            'home_score': game['home_score'],
            'away_score': game['away_score'],
            'median_margin': round(median_margin, 1),
            'mean_margin': round(mean_margin, 1),
            'edge': round(edge, 1),
            'home_cover_pct': round(home_cover_pct, 1),
            'away_cover_pct': round(away_cover_pct, 1),
            'model_pick': model_pick_label,
            'model_pick_pct': round(model_pick_pct, 1),
            'home_covered': home_covered,
            'is_push': is_push,
            'pick_correct': pick_correct,
        })

    return pd.DataFrame(results)


def main_ats_backtest(
    historical_file=HISTORICAL_SPREADS_FILE,
    num_runs=1000,
    use_web=True,
    start_year=2019,
):
    """
    Run ATS backtest against historical spreads + scores.

    Args:
        historical_file: CSV with week,year,home_team,away_team,
                         spread,home_score,away_score
        num_runs: Monte Carlo iterations
        use_web: True = fetch from web, False = local files
        start_year: Earliest year to load for training data
    """
    log_filename = (
        f'C:/Users/Bobby/nfl_ats_backtest_'
        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )

    logging.info(f"Logging to file: {log_filename}")
    logging.info(f"Loading historical spreads from: {historical_file}")

    games_df = load_historical_spreads(historical_file)
    logging.info(f"Loaded {len(games_df)} historical games")

    years_needed = sorted(games_df['year'].unique())
    test_year = int(games_df['year'].max())

    # Translate week numbers for test games to find the earliest test week
    games_df['week_num_int'] = games_df['week'].apply(
        lambda w: int(w) if str(w).isdigit() else
        translate_week_number(w, test_year)
    )
    min_test_week_num = int(games_df['week_num_int'].min())

    logging.info(
        f"Test year: {test_year}, weeks: {sorted(games_df['week'].unique())}"
    )

    # Pre-load all stats once
    logging.info("Loading team stats from source...")
    all_stats = []
    all_games_list = []

    for year in range(start_year, test_year + 1):
        logging.info(f"  Loading {year} data...")
        stats, games_past, games_future = load_data_for_year(
            year, off_url, def_url, games_url, use_web=use_web
        )
        all_stats.append(stats)
        all_games_list.append(games_past)
        if not games_future.empty:
            all_games_list.append(games_future)
        if use_web:
            time.sleep(1)

    cached_data = {
        'stats': pd.concat(all_stats, ignore_index=True),
        'all_games': pd.concat(all_games_list, ignore_index=True),
    }
    logging.info("Data loading complete!")

    # Run backtest iterations
    all_predictions = {}
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    logging.info(f"Using {max_workers} worker processes")
    logging.info(f"Running {num_runs} iterations...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_ats_backtest_iteration,
                i, cached_data, games_df, test_year, min_test_week_num
            ): i
            for i in range(num_runs)
        }

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            try:
                iter_preds = future.result()
                for idx, pred in iter_preds.items():
                    all_predictions.setdefault(idx, []).append(pred)
            except Exception as e:
                logging.error(f"Iteration error: {e}")
            if completed % 100 == 0:
                logging.info(
                    f"  Progress: {completed}/{num_runs} complete"
                )

    # Calculate metrics
    results_df = calculate_ats_metrics(all_predictions, games_df, num_runs)

    if results_df.empty:
        logging.warning("No results to report!")
        return results_df

    # Summary
    decidable = results_df[results_df['pick_correct'].notna()]
    total = len(decidable)
    correct = decidable['pick_correct'].sum()
    mae = results_df['edge'].abs().mean()

    logging.info("")
    logging.info("=" * 80)
    logging.info("ATS BACKTEST SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total games (excluding pushes): {total}")
    logging.info(
        f"ATS Record: {int(correct)}-{total - int(correct)} "
        f"({100.0 * correct / total:.1f}%)"
    )
    logging.info(f"Pushes: {results_df['is_push'].sum()}")
    logging.info(f"Mean |edge| vs spread: {mae:.2f} pts")

    # By week
    logging.info("")
    logging.info("ATS ACCURACY BY WEEK")
    logging.info("-" * 40)
    for week, grp in decidable.groupby('week'):
        w_correct = grp['pick_correct'].sum()
        w_total = len(grp)
        logging.info(
            f"  Week {week}: {int(w_correct)}/{w_total} "
            f"({100.0 * w_correct / w_total:.1f}%)"
        )

    # Individual game results
    logging.info("")
    logging.info("=" * 80)
    logging.info("INDIVIDUAL GAME RESULTS")
    logging.info("=" * 80)
    logging.info(
        f"{'Game':<40} {'Spread':>7} {'Model':>7} "
        f"{'Edge':>6} {'Score':>10} {'Pick':<28} {'Result'}"
    )
    logging.info("-" * 80)
    for _, row in results_df.iterrows():
        game_str = f"{row['home_team']} vs {row['away_team']}"
        score_str = f"{int(row['home_score'])}-{int(row['away_score'])}"
        if row['is_push']:
            result_str = "PUSH"
        elif row['pick_correct']:
            result_str = "WIN"
        else:
            result_str = "LOSS"
        logging.info(
            f"{game_str:<40} {row['spread']:>+7.1f} "
            f"{row['median_margin']:>+7.1f} {row['edge']:>+6.1f} "
            f"{score_str:>10} {row['model_pick']:<28} {result_str}"
        )

    # Save CSV
    csv_filename = (
        f'C:/Users/Bobby/nfl_ats_backtest_results_'
        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )
    results_df.to_csv(csv_filename, index=False)
    logging.info(f"\nResults saved to: {csv_filename}")
    logging.info("Backtest complete!")

    return results_df


if __name__ == "__main__":
    main_ats_backtest(
        historical_file=HISTORICAL_SPREADS_FILE,
        num_runs=1000,
        use_web=True,
        start_year=2019,
    )
