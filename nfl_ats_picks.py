"""
NFL Against-the-Spread (ATS) Prediction Platform

Loads Vegas spreads from spreads.csv, fetches team stats from the web,
and predicts cover probability for each game using Monte Carlo simulation.

Spread convention (in spreads.csv):
  spread < 0  -> home team is favored by that many points
                 e.g. spread = -3.0 means home team is -3 favorite
  spread > 0  -> away team is favored
                 e.g. spread = 3.0 means home team is +3 underdog

Cover logic:
  Home team covers if:  predicted_margin > spread  (margin = home - away)
  Away team covers if:  predicted_margin < spread
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

# Path to your spreads spreadsheet
SPREADS_FILE = 'spreads.csv'


def load_spreads(filepath=SPREADS_FILE):
    """
    Load spreads from CSV file.

    Expected columns: week, year, home_team, away_team, spread
    spread < 0 means home team is favored (e.g. -3.0 = home -3)
    spread > 0 means away team is favored (e.g. +3.0 = home +3)
    """
    df = pd.read_csv(filepath)
    required = {'week', 'year', 'home_team', 'away_team', 'spread'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"spreads.csv is missing columns: {missing}")
    df['week'] = df['week'].astype(str)
    df['year'] = df['year'].astype(int)
    df['spread'] = df['spread'].astype(float)
    return df


def run_ats_iteration(iteration, cached_data, games_with_spreads):
    """
    Run a single Monte Carlo iteration. Returns per-game predicted margin.

    Args:
        iteration: iteration number
        cached_data: dict with 'stats' and 'all_games' DataFrames
        games_with_spreads: DataFrame with week, year, home_team, away_team, spread
    """
    multi_year_combined_stats = cached_data['stats'].copy()
    all_games = cached_data['all_games'].copy()

    # Translate week numbers to integers
    all_games['week_num_int'] = all_games.apply(
        lambda row: translate_week_number(row['week_number'], row['year']),
        axis=1
    )
    all_games['year_int'] = all_games['year'].astype(int)
    all_games = all_games[all_games['week_num_int'].notna()].copy()
    all_games['week_num_int'] = all_games['week_num_int'].astype(int)

    # Use all completed games (score != 0) as training data,
    # excluding the last regular season week of each year
    def is_valid_training_game(row):
        last_reg_week = get_last_regular_season_week(row['year_int'])
        return (
            row['score_difference'] != 0
            and row['week_num_int'] != last_reg_week
        )

    training_games = all_games[all_games.apply(is_valid_training_game, axis=1)]

    merged_data, features = preprocess_data(
        multi_year_combined_stats, training_games
    )

    X = merged_data[features]
    y = merged_data['score_difference']

    X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X))
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    iteration_predictions = {}
    for idx, game in games_with_spreads.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        year_str = str(game['year'])
        try:
            base_pred = predict_game(
                home_team, away_team, year_str,
                multi_year_combined_stats, model
            )
            noise = np.random.normal(0, 1.0)
            iteration_predictions[idx] = round(base_pred + noise, 1)
        except ValueError:
            pass

    return iteration_predictions


def run_ats_predictions(
    games_with_spreads,
    cached_data,
    num_runs=1000
):
    """
    Run num_runs Monte Carlo iterations and aggregate cover probabilities.

    Returns a DataFrame with one row per game including:
      - mean/median predicted margin
      - cover probability for home and away team
      - recommended pick and confidence
    """
    all_predictions = {}  # idx -> list of predicted margins

    max_workers = min(32, (os.cpu_count() or 1) * 2)
    logging.info(f"Using {max_workers} worker processes")
    logging.info(f"Running {num_runs} iterations...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_ats_iteration, i, cached_data, games_with_spreads
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
                    f"  Progress: {completed}/{num_runs} iterations complete"
                )

    # Build results
    results = []
    for idx, game in games_with_spreads.iterrows():
        preds = np.array(all_predictions.get(idx, []))
        if len(preds) == 0:
            logging.warning(
                f"No predictions for {game['home_team']} vs {game['away_team']}"
            )
            continue

        spread = game['spread']
        mean_margin = round(float(np.mean(preds)), 1)
        median_margin = round(float(np.median(preds)), 1)
        std_dev = round(float(np.std(preds)), 1)

        # Cover probability:
        # Home covers if predicted margin > spread
        # Away covers if predicted margin < spread
        home_cover_pct = round(100.0 * np.mean(preds > spread), 1)
        away_cover_pct = round(100.0 - home_cover_pct, 1)

        # Edge = how many points the model thinks vs the spread
        edge = round(median_margin - spread, 1)

        # Pick the side with > 50% cover probability
        if home_cover_pct >= 50:
            pick = game['home_team']
            pick_cover_pct = home_cover_pct
            pick_label = f"{game['home_team']} ({spread:+.1f})"
        else:
            pick = game['away_team']
            pick_cover_pct = away_cover_pct
            away_spread = -spread
            pick_label = f"{game['away_team']} ({away_spread:+.1f})"

        # Confidence tier
        if pick_cover_pct >= 70:
            confidence = 'HIGH'
        elif pick_cover_pct >= 60:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        results.append({
            'week': game['week'],
            'year': game['year'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'spread': spread,
            'mean_margin': mean_margin,
            'median_margin': median_margin,
            'std_dev': std_dev,
            'edge': edge,
            'home_cover_pct': home_cover_pct,
            'away_cover_pct': away_cover_pct,
            'pick': pick,
            'pick_label': pick_label,
            'confidence': confidence,
        })

    return pd.DataFrame(results)


def print_picks(results_df):
    """Log a formatted ATS picks report."""
    logging.info("")
    logging.info("=" * 80)
    logging.info("NFL ATS PICKS")
    logging.info("=" * 80)
    logging.info(
        f"{'Game':<45} {'Spread':>7} {'Model':>7} {'Edge':>6} "
        f"{'Cover%':>7} {'Pick':<30} {'Conf'}"
    )
    logging.info("-" * 80)

    for _, row in results_df.iterrows():
        game_str = f"{row['home_team']} vs {row['away_team']}"
        spread_str = f"{row['spread']:+.1f}"
        model_str = f"{row['median_margin']:+.1f}"
        edge_str = f"{row['edge']:+.1f}"
        cover_str = f"{row['home_cover_pct']:.0f}%"
        logging.info(
            f"{game_str:<45} {spread_str:>7} {model_str:>7} {edge_str:>6} "
            f"{cover_str:>7} {row['pick_label']:<30} {row['confidence']}"
        )

    logging.info("")
    logging.info("HIGH confidence picks:")
    high = results_df[results_df['confidence'] == 'HIGH']
    if high.empty:
        logging.info("  None this week")
    for _, row in high.iterrows():
        logging.info(f"  {row['pick_label']}  ({row['home_cover_pct']:.0f}% cover)")

    logging.info("")
    logging.info("MEDIUM confidence picks:")
    med = results_df[results_df['confidence'] == 'MEDIUM']
    if med.empty:
        logging.info("  None this week")
    for _, row in med.iterrows():
        logging.info(f"  {row['pick_label']}  ({row['home_cover_pct']:.0f}% cover)")


def main(
    spreads_file=SPREADS_FILE,
    num_runs=1000,
    use_web=True,
    start_year=2019,
):
    """
    Main entry point for the ATS prediction platform.

    Args:
        spreads_file: Path to your spreads CSV
        num_runs: Number of Monte Carlo iterations
        use_web: True = fetch stats from web, False = use local files
        start_year: Earliest year to load for training data
    """
    log_filename = (
        f'C:/Users/Bobby/nfl_ats_picks_'
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
    logging.info(f"Loading spreads from: {spreads_file}")

    # Load spreads
    games_with_spreads = load_spreads(spreads_file)
    logging.info(f"Loaded {len(games_with_spreads)} games")

    # Determine what years of stats we need
    years_needed = sorted(games_with_spreads['year'].unique())
    end_year = max(years_needed)

    # Pre-load all stats data once
    logging.info("Loading team stats from source...")
    all_stats = []
    all_games = []

    for year in range(start_year, end_year + 1):
        logging.info(f"  Loading {year} data...")
        stats, games_past, games_future = load_data_for_year(
            year, off_url, def_url, games_url, use_web=use_web
        )
        all_stats.append(stats)
        all_games.append(games_past)
        if not games_future.empty:
            all_games.append(games_future)
        if use_web:
            time.sleep(1)

    cached_data = {
        'stats': pd.concat(all_stats, ignore_index=True),
        'all_games': pd.concat(all_games, ignore_index=True),
    }
    logging.info("Data loading complete!")

    # Run predictions
    results_df = run_ats_predictions(
        games_with_spreads, cached_data, num_runs=num_runs
    )

    # Print formatted picks
    print_picks(results_df)

    # Save to CSV
    csv_filename = (
        f'C:/Users/Bobby/nfl_ats_picks_'
        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )
    results_df.to_csv(csv_filename, index=False)
    logging.info(f"Results saved to: {csv_filename}")

    return results_df


if __name__ == "__main__":
    main(
        spreads_file=SPREADS_FILE,
        num_runs=1000,
        use_web=True,
        start_year=2019,
    )
