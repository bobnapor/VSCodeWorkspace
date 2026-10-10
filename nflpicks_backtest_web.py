"""
NFL Picks Backtest Script (Web Version)

This script imports core functions from nflpicks_v2_0_multithreaded_web.py
to ensure consistency between predictions and backtesting.
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
import nflpicks_v2_0_multithreaded_web as picks_web
from nflpicks_v2_0_multithreaded_web import (
    load_data_for_year,
    preprocess_data,
    predict_game,
    translate_week_number,
    get_last_regular_season_week,
    get_current_season_year,
    LOOKBACK_YEARS,
    off_url,
    def_url,
    games_url
)


def run_backtest_iteration(iteration, cached_data, test_year, test_weeks):
    """Run a single iteration of the backtest simulation
    
    Args:
        iteration: iteration number
        cached_data: Pre-loaded data dict with 'stats', 'all_games' DataFrames
        test_year: Year to test on
        test_weeks: List of numeric weeks to test (e.g., [1, 2, 3, ...])
    """
    
    # Use pre-loaded cached data
    multi_year_combined_stats = cached_data['stats'].copy()
    all_games = cached_data['all_games'].copy()
    
    # Translate week numbers (including playoff weeks) to integers
    all_games['week_num_int'] = all_games.apply(
        lambda row: translate_week_number(row['week_number'], row['year']), axis=1
    )
    all_games['year_int'] = all_games['year'].astype(int)
    
    # Filter out rows where week translation failed (unknown week names)
    all_games = all_games[all_games['week_num_int'].notna()].copy()
    all_games['week_num_int'] = all_games['week_num_int'].astype(int)
    
    # Only use games with scores (completed games)
    completed_games = all_games[all_games['score_difference'] != 0]
    
    # Get test games (games we want to predict)
    test_games = completed_games[
        (completed_games['year_int'] == test_year) & 
        (completed_games['week_num_int'].isin(test_weeks))
    ]
    
    # Get training games: all completed games BEFORE each test week
    # For backtesting, we need to be careful to only use data available at prediction time
    # We'll train on all games before the earliest test week
    min_test_week = min(test_weeks)
    
    def is_valid_training_game(row):
        game_year = row['year_int']
        game_week = row['week_num_int']
        last_reg_week = get_last_regular_season_week(game_year)
        
        # Exclude the last regular season week from training
        if game_week == last_reg_week:
            return False
        
        # Include if before test year, or same year but earlier than earliest test week
        if game_year < test_year:
            return True
        if game_year == test_year and game_week < min_test_week:
            return True
        return False
    
    training_games = completed_games[completed_games.apply(is_valid_training_game, axis=1)]

    # Prepare the training data
    merged_data, features = preprocess_data(multi_year_combined_stats, training_games)

    # Split the data into train and test sets
    X = merged_data[features]
    y = merged_data['score_difference']

    # Bootstrap resample the data for more variation
    X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X))
    
    # Random train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on test games
    iteration_predictions = {}
    game_info = {}
    
    for index, game in test_games.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        input_week = game['week_number']
        input_year = game['year']
        actual_score_difference = game['score_difference']
        
        try:
            # Get base prediction
            base_prediction = predict_game(home_team, away_team, input_year, multi_year_combined_stats, model)
            
            # Add random noise
            noise = np.random.normal(0, 1.0)
            projected_score_difference = round(base_prediction + noise, 1)
            
            iteration_predictions[index] = projected_score_difference
            game_info[index] = {
                'home_team': home_team,
                'away_team': away_team,
                'week': input_week,
                'week_num': game['week_num_int'],
                'year': input_year,
                'actual_score_difference': actual_score_difference
            }
        except ValueError as e:
            pass
    
    return iteration_predictions, game_info


def calculate_backtest_metrics(all_predictions, game_details, num_runs):
    """Calculate backtest accuracy metrics"""
    results = []
    
    for game_index in sorted(all_predictions.keys()):
        predictions_array = np.array(all_predictions[game_index])
        median_pred = np.median(predictions_array)
        mean_pred = np.mean(predictions_array)
        
        if game_index in game_details:
            info = game_details[game_index]
            actual = info['actual_score_difference']
            
            # Calculate metrics
            error = median_pred - actual
            abs_error = abs(error)
            
            # Did we predict the correct winner?
            predicted_winner_correct = (median_pred > 0 and actual > 0) or (median_pred < 0 and actual < 0) or (median_pred == 0 and actual == 0)
            
            # Was prediction within spread?
            within_3 = abs_error <= 3
            within_7 = abs_error <= 7
            within_10 = abs_error <= 10
            
            results.append({
                'game_index': game_index,
                'week': info['week'],
                'week_num': info['week_num'],
                'year': info['year'],
                'home_team': info['home_team'],
                'away_team': info['away_team'],
                'predicted': round(median_pred, 1),
                'actual': actual,
                'error': round(error, 1),
                'abs_error': round(abs_error, 1),
                'correct_winner': predicted_winner_correct,
                'within_3': within_3,
                'within_7': within_7,
                'within_10': within_10
            })
    
    return pd.DataFrame(results)


def shift_stats_to_following_season(stats):
    """Relabel each season's team stats as the following season's, following franchise renames."""
    renames = {
        'Oakland Raiders': 'Las Vegas Raiders',
        'Washington Redskins': 'Washington Football Team',
        'Washington Football Team': 'Washington Commanders',
    }
    shifted = []
    for year_str, prior in stats.groupby('year'):
        next_year = int(year_str) + 1
        next_teams = set(stats.loc[stats['year'] == str(next_year), 'team'])
        prior = prior.copy()
        if next_teams:
            prior['team'] = prior['team'].map(lambda t: t if t in next_teams else renames.get(t, t))
        prior['year'] = str(next_year)
        shifted.append(prior)
    return pd.concat(shifted, ignore_index=True)


def main_backtest(num_runs=10000, test_year=None, test_weeks=None, use_web=True, start_year=None,
                  use_prior_season_stats=True):
    """
    Backtest the model on past games using web data
    
    Args:
        num_runs: Number of simulation iterations
        test_year: Year to test on (default: the current season)
        test_weeks: List of weeks to test (e.g., [1, 2, 3, 4, 5]). Default: every regular season week
                    already played in test_year.
                    For playoffs use 19=WildCard, 20=Division, 21=ConfChamp, 22=SuperBowl (for 2021+)
        use_web: True to fetch from web, False to use local files
        start_year: First year to load for training data (default: LOOKBACK_YEARS before test_year)
        use_prior_season_stats: Features for a season's games come from the previous season's team stats.
                    The available stats are season totals, which include the game being predicted,
                    so using them directly leaks the result. Games from start_year have no prior
                    season and are skipped.
    """
    if test_year is None:
        test_year = get_current_season_year()
    if start_year is None:
        start_year = test_year - LOOKBACK_YEARS
    
    # Setup logging
    log_filename = f'C:/Users/Bobby/nfl_backtest_web_{test_year}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
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
    logging.info(f"Data source: {'Web URLs' if use_web else 'Local files'}")
    logging.info(f"Running {num_runs} iterations")
    
    # Pre-load all data once in main process
    logging.info("Loading data from source...")
    all_years_team_stats_arr = []
    all_games_arr = []
    test_year_past = pd.DataFrame()

    for year in range(start_year, test_year + 1):
        logging.info(f"  Loading {year} data...")
        single_year_combined_stats, single_year_games_past, single_year_games_future = load_data_for_year(
            year, off_url, def_url, games_url, use_web=use_web
        )
        all_years_team_stats_arr.append(single_year_combined_stats)
        all_games_arr.append(single_year_games_past)
        if not single_year_games_future.empty:
            all_games_arr.append(single_year_games_future)
        if year == test_year:
            test_year_past = single_year_games_past
        
        if use_web and not picks_web._web_blocked:
            time.sleep(1)  # Rate limiting for web requests

    cached_data = {
        'stats': pd.concat(all_years_team_stats_arr, ignore_index=True),
        'all_games': pd.concat(all_games_arr, ignore_index=True)
    }
    if use_prior_season_stats:
        cached_data['stats'] = shift_stats_to_following_season(cached_data['stats'])
        logging.info("Using prior-season team stats for every game so a game's own result is never in its features")
    logging.info("Data loading complete!")

    if test_weeks is None:
        last_regular_week = get_last_regular_season_week(test_year)
        played_weeks = {translate_week_number(w, test_year) for w in test_year_past['week_number']} if not test_year_past.empty else set()
        test_weeks = sorted(w for w in played_weeks if w is not None and w <= last_regular_week)
    if not test_weeks:
        logging.warning(f"No played games found for {test_year}; nothing to backtest")
        return pd.DataFrame()
    logging.info(f"Backtesting {test_year} season, weeks: {test_weeks}")
    
    averaged_predictions = {}
    all_predictions = {}
    game_details = {}
    
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    logging.info(f"Using {max_workers} worker processes")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_backtest_iteration, i, cached_data, test_year, test_weeks): i 
            for i in range(num_runs)
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            iteration = futures[future]
            completed += 1
            try:
                iteration_predictions, game_info = future.result()
                
                if not game_details:
                    game_details = game_info
                
                for game_index, pred in iteration_predictions.items():
                    if game_index in averaged_predictions:
                        averaged_predictions[game_index] += pred
                        all_predictions[game_index].append(pred)
                    else:
                        averaged_predictions[game_index] = pred
                        all_predictions[game_index] = [pred]

                if completed % 100 == 0:
                    logging.info(f"Progress: {completed}/{num_runs} iterations completed")

            except Exception as e:
                logging.error(f"Iteration {iteration} generated an exception: {e}")
    
    # Calculate backtest metrics
    results_df = calculate_backtest_metrics(all_predictions, game_details, num_runs)
    
    if results_df.empty:
        logging.warning("No games found to backtest!")
        return results_df
    
    # Log summary statistics
    logging.info("")
    logging.info("=" * 80)
    logging.info("BACKTEST RESULTS SUMMARY")
    logging.info("=" * 80)
    
    total_games = len(results_df)
    correct_winners = results_df['correct_winner'].sum()
    within_3 = results_df['within_3'].sum()
    within_7 = results_df['within_7'].sum()
    within_10 = results_df['within_10'].sum()
    
    mae = results_df['abs_error'].mean()
    mse = (results_df['error'] ** 2).mean()
    rmse = np.sqrt(mse)
    
    logging.info(f"")
    logging.info(f"Total games tested: {total_games}")
    logging.info(f"")
    logging.info(f"Winner Prediction Accuracy: {correct_winners}/{total_games} ({100*correct_winners/total_games:.1f}%)")
    logging.info(f"")
    logging.info(f"Spread Accuracy:")
    logging.info(f"  Within 3 points: {within_3}/{total_games} ({100*within_3/total_games:.1f}%)")
    logging.info(f"  Within 7 points: {within_7}/{total_games} ({100*within_7/total_games:.1f}%)")
    logging.info(f"  Within 10 points: {within_10}/{total_games} ({100*within_10/total_games:.1f}%)")
    logging.info(f"")
    logging.info(f"Error Metrics:")
    logging.info(f"  Mean Absolute Error (MAE): {mae:.2f} points")
    logging.info(f"  Root Mean Squared Error (RMSE): {rmse:.2f} points")
    
    # Log individual game results
    logging.info("")
    logging.info("=" * 80)
    logging.info("INDIVIDUAL GAME RESULTS")
    logging.info("=" * 80)
    logging.info("Week|Year|Home|Away|Predicted|Actual|Error|Correct")
    
    for _, row in results_df.sort_values(['week_num', 'game_index']).iterrows():
        winner_mark = "Y" if row['correct_winner'] else "N"
        logging.info(
            f"{row['week']}|{row['year']}|{row['home_team']}|{row['away_team']}|"
            f"{row['predicted']}|{row['actual']}|{row['error']}|{winner_mark}"
        )
    
    # Save results to CSV
    csv_filename = f'C:/Users/Bobby/nfl_backtest_web_results_{test_year}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results_df.to_csv(csv_filename, index=False)
    logging.info(f"")
    logging.info(f"Detailed results saved to: {csv_filename}")
    
    # Summary by week
    logging.info("")
    logging.info("=" * 80)
    logging.info("ACCURACY BY WEEK")
    logging.info("=" * 80)
    
    week_summary = results_df.groupby('week').agg({
        'correct_winner': ['sum', 'count'],
        'abs_error': 'mean'
    }).round(2)
    week_summary.columns = ['correct', 'total', 'avg_error']
    week_summary['accuracy'] = (100 * week_summary['correct'] / week_summary['total']).round(1)
    
    for week, row in week_summary.iterrows():
        logging.info(f"Week {week}: {int(row['correct'])}/{int(row['total'])} correct ({row['accuracy']}%), Avg Error: {row['avg_error']:.1f}")
    
    logging.info("")
    logging.info("Backtest complete!")
    
    return results_df


if __name__ == "__main__":
    # Configuration
    TEST_YEAR = None  # None = current season; set a year to override
    TEST_WEEKS = None  # None = every regular season week already played; set a list to override
    # For playoffs, use: [19, 20, 21, 22] for WildCard, Division, ConfChamp, SuperBowl (2021+ seasons)
    NUM_RUNS = 10000
    USE_WEB = True  # Set to False to use local HTML files
    START_YEAR = None  # None = LOOKBACK_YEARS before the test year; set a year to override
    
    results = main_backtest(
        num_runs=NUM_RUNS,
        test_year=TEST_YEAR,
        test_weeks=TEST_WEEKS,
        use_web=USE_WEB,
        start_year=START_YEAR
    )
