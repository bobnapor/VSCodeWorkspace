import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4 import Comment
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import logging
import concurrent.futures
import os
from datetime import datetime
import requests
import time

# URL templates for web scraping
games_url = 'https://www.pro-football-reference.com/years/yyyy/games.htm'
def_url = 'https://www.pro-football-reference.com/years/yyyy/opp.htm'
off_url = 'https://www.pro-football-reference.com/years/yyyy/'

# Optional: Keep file templates as fallback
file_dir = 'C:/Users/Bobby/Downloads/NFL_Stats/'
games_template = file_dir + 'yyyy NFL Regular Season Schedule _ Pro-Football-Reference.com.html'
def_template = file_dir + 'yyyy NFL Opposition & Defensive Statistics _ Pro-Football-Reference.com.html'
off_template = file_dir + 'yyyy NFL Standings & Team Stats _ Pro-Football-Reference.com.html'

# Columns to use for stats
stat_columns_to_use = {
    'team',
    'pass_yds',
    'rush_yds',
    'turnovers'
}


def is_numeric_value(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def fetch_url_with_retry(url, max_retries=3, delay=2):
    """Fetch URL content with retry logic and rate limiting"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                raise e
    return None


def get_offense_stats(full_offense_soup, year_str):
    single_year_offense = []
    for comment in full_offense_soup.find_all(string=lambda text: isinstance(text, Comment)):
        offense_soup = BeautifulSoup(comment, 'html.parser')
        offense_table = offense_soup.find('table', id='team_stats')
        if offense_table:
            offense_rows = offense_table.find('tbody').find_all('tr')
            for offense_row in offense_rows:
                num_games = int(next((col.text for col in offense_row.find_all('td') if col['data-stat'] == 'g'), '0'))
                single_team_offense = {col['data-stat']: '0' if col.text.strip() == '' else str(int(col.text)/num_games) if is_numeric_value(col.text) else col.text for col in offense_row.find_all('td') if col['data-stat'] in stat_columns_to_use}
                team = single_team_offense.get('team')
                if team:
                    single_year_offense.append(single_team_offense)
    return pd.DataFrame(single_year_offense)


def get_defense_stats(full_defense_soup, year_str):
    single_year_defense = []
    defense_table = full_defense_soup.find('table', id='team_stats')
    if not defense_table:
        return pd.DataFrame()
    defense_rows = defense_table.find('tbody').find_all('tr')
    for defense_row in defense_rows:
        num_games = int(next((col.text for col in defense_row.find_all('td') if col['data-stat'] == 'g'), '0'))
        single_team_defense = {f'def_{col["data-stat"]}': '0' if col.text.strip() == '' else str(int(col.text)/num_games) if is_numeric_value(col.text) else col.text for col in defense_row.find_all('td') if col['data-stat'] in stat_columns_to_use}
        team = single_team_defense.get('def_team')
        if team:
            single_year_defense.append(single_team_defense)
    return pd.DataFrame(single_year_defense)


def get_single_year_team_stats(single_year_offense_df, single_year_defense_df, year_str):
    single_year_defense_df.rename(columns={'def_team':'team'}, inplace=True)
    single_year_combined_stats = pd.merge(single_year_offense_df, single_year_defense_df, on='team')
    single_year_combined_stats['year'] = year_str
    return single_year_combined_stats


def get_games(full_games_soup, year_str):
    """Parse games from HTML and return past and future games DataFrames"""
    games_table = full_games_soup.find('table', id='games')
    games_list_past = []
    games_list_future = []

    for game_row in games_table.find('tbody').find_all('tr'):
        this_game_week_num = game_row.find_all('th')[0].text.strip()
        game_stats = {col['data-stat']: col.text for col in game_row.find_all('td')}
        if not game_stats or 'game_date' not in game_stats or 'winner' not in game_stats or 'loser' not in game_stats or this_game_week_num == '':
            continue

        game_date = game_stats['game_date']
        away_team = game_stats['winner'] if game_stats.get('game_location') == '@' else game_stats['loser']
        home_team = game_stats['loser'] if away_team == game_stats['winner'] else game_stats['winner']

        if game_stats['pts_win'] != '' and game_stats['pts_lose'] != '':
            winner_score = int(game_stats['pts_win'])
            loser_score = int(game_stats['pts_lose'])
            score_difference = winner_score - loser_score if home_team == game_stats['winner'] else loser_score - winner_score

            games_list_past.append({
                'week_number': this_game_week_num,
                'game_date': game_date,
                'away_team': away_team,
                'home_team': home_team,
                'score_difference': score_difference,
                'year': year_str
            })
        else:
            games_list_future.append({
                'week_number': this_game_week_num,
                'game_date': game_date,
                'away_team': away_team,
                'home_team': home_team,
                'score_difference': 0,
                'year': year_str
            })

    return pd.DataFrame(games_list_past), pd.DataFrame(games_list_future)


def preprocess_data(football_data, games):
    merged_data = games.merge(
        football_data, how='left', left_on=['year', 'home_team'], right_on=['year', 'team']
    ).merge(
        football_data, how='left', left_on=['year', 'away_team'], right_on=['year', 'team'], suffixes=('_home', '_away')
    )

    features = [
        'pass_yds_home', 'def_pass_yds_home', 'rush_yds_home', 'def_rush_yds_home', 'turnovers_home', 'def_turnovers_home',
        'pass_yds_away', 'def_pass_yds_away', 'rush_yds_away', 'def_rush_yds_away', 'turnovers_away', 'def_turnovers_away'
    ]
    merged_data = merged_data.dropna(subset=features + ['score_difference'])

    return merged_data, features


def predict_game(home_team, away_team, year, football_data, model):
    home_stats = football_data[(football_data['team'] == home_team) & (football_data['year'] == year)]
    away_stats = football_data[(football_data['team'] == away_team) & (football_data['year'] == year)]

    if home_stats.empty or away_stats.empty:
        raise ValueError("Team statistics for the specified year not found.")

    input_features = pd.DataFrame([{
        'pass_yds_home': home_stats['pass_yds'].values[0],
        'def_pass_yds_home': home_stats['def_pass_yds'].values[0],
        'rush_yds_home': home_stats['rush_yds'].values[0],
        'def_rush_yds_home': home_stats['def_rush_yds'].values[0],
        'turnovers_home': home_stats['turnovers'].values[0],
        'def_turnovers_home': home_stats['def_turnovers'].values[0],
        'pass_yds_away': away_stats['pass_yds'].values[0],
        'def_pass_yds_away': away_stats['def_pass_yds'].values[0],
        'rush_yds_away': away_stats['rush_yds'].values[0],
        'def_rush_yds_away': away_stats['def_rush_yds'].values[0],
        'turnovers_away': away_stats['turnovers'].values[0],
        'def_turnovers_away': away_stats['def_turnovers'].values[0]
    }])

    predicted_score_difference = model.predict(input_features)[0]
    return predicted_score_difference


def load_data_for_year(year, use_web=True):
    """Load data for a year from web URLs or local files"""
    year_str = str(year)

    if use_web:
        # Fetch from web
        off_html = fetch_url_with_retry(off_url.replace('yyyy', year_str))
        offense_soup = BeautifulSoup(off_html, 'html.parser')
        single_year_offense = get_offense_stats(offense_soup, year_str)

        def_html = fetch_url_with_retry(def_url.replace('yyyy', year_str))
        defense_soup = BeautifulSoup(def_html, 'html.parser')
        single_year_defense = get_defense_stats(defense_soup, year_str)

        single_year_combined_stats = get_single_year_team_stats(
            single_year_offense, single_year_defense, year_str
        )

        games_html = fetch_url_with_retry(games_url.replace('yyyy', year_str))
        games_soup = BeautifulSoup(games_html, 'html.parser')
        single_year_games_past, single_year_games_future = get_games(games_soup, year_str)
    else:
        # Fallback to local files
        with open(off_template.replace('yyyy', year_str)) as offense_file:
            offense_soup = BeautifulSoup(offense_file.read(), 'html.parser')
            single_year_offense = get_offense_stats(offense_soup, year_str)

        with open(def_template.replace('yyyy', year_str)) as defense_file:
            defense_soup = BeautifulSoup(defense_file.read(), 'html.parser')
            single_year_defense = get_defense_stats(defense_soup, year_str)

        single_year_combined_stats = get_single_year_team_stats(
            single_year_offense, single_year_defense, year_str
        )

        with open(games_template.replace('yyyy', year_str)) as games_file:
            games_soup = BeautifulSoup(games_file.read(), 'html.parser')
            single_year_games_past, single_year_games_future = get_games(games_soup, year_str)

    return single_year_combined_stats, single_year_games_past, single_year_games_future


def translate_week_number(week_str, year):
    """Translate playoff week names to numeric values based on season year.
    
    2021+ seasons (18-game regular season):
        WildCard -> 19, Division -> 20, ConfChamp -> 21, SuperBowl -> 22
    2020 and earlier (17-game regular season):
        WildCard -> 18, Division -> 19, ConfChamp -> 20, SuperBowl -> 21
    """
    year_int = int(year)
    
    # Check if already numeric
    if week_str.isdigit():
        return int(week_str)
    
    # Playoff week translations
    if year_int >= 2021:
        # 18-game season (2021+)
        translations = {
            'WildCard': 19,
            'Division': 20,
            'ConfChamp': 21,
            'SuperBowl': 22
        }
    else:
        # 17-game season (2020 and earlier)
        translations = {
            'WildCard': 18,
            'Division': 19,
            'ConfChamp': 20,
            'SuperBowl': 21
        }
    
    return translations.get(week_str, None)


def get_last_regular_season_week(year):
    """Get the last regular season week for a given year.
    
    2021+ seasons: 18 weeks
    2020 and earlier: 17 weeks
    """
    year_int = int(year)
    return 18 if year_int >= 2021 else 17


def run_backtest_iteration(iteration, cached_data, test_year, test_weeks, include_playoffs):
    """Run a single iteration of the backtest simulation
    
    Args:
        iteration: iteration number
        cached_data: Pre-loaded data dict with 'stats', 'all_games' DataFrames
        test_year: Year to test on
        test_weeks: List of numeric weeks to test (e.g., [1, 2, 3, ...])
        include_playoffs: Whether test_weeks includes playoff weeks
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


def main_backtest(num_runs=1000, test_year=2024, test_weeks=None, include_playoffs=False, use_web=True, start_year=2019):
    """
    Backtest the model on past games using web data
    
    Args:
        num_runs: Number of simulation iterations
        test_year: Year to test on
        test_weeks: List of weeks to test (e.g., [1, 2, 3, 4, 5]). 
                    For playoffs use 19=WildCard, 20=Division, 21=ConfChamp, 22=SuperBowl (for 2021+)
        include_playoffs: Set True if test_weeks includes playoff weeks (19-22)
        use_web: True to fetch from web, False to use local files
        start_year: First year to load for training data
    """
    if test_weeks is None:
        test_weeks = list(range(1, 19))  # All regular season weeks for 2021+ seasons
    
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
    logging.info(f"Backtesting {test_year} season, weeks: {test_weeks}")
    logging.info(f"Running {num_runs} iterations")
    
    # Pre-load all data once in main process
    logging.info("Loading data from source...")
    all_years_team_stats_arr = []
    all_games_arr = []

    for year in range(start_year, test_year + 1):
        logging.info(f"  Loading {year} data...")
        single_year_combined_stats, single_year_games_past, single_year_games_future = load_data_for_year(
            year, use_web=use_web
        )
        all_years_team_stats_arr.append(single_year_combined_stats)
        all_games_arr.append(single_year_games_past)
        if not single_year_games_future.empty:
            all_games_arr.append(single_year_games_future)
        
        if use_web:
            time.sleep(1)  # Rate limiting for web requests

    cached_data = {
        'stats': pd.concat(all_years_team_stats_arr, ignore_index=True),
        'all_games': pd.concat(all_games_arr, ignore_index=True)
    }
    logging.info("Data loading complete!")
    
    averaged_predictions = {}
    all_predictions = {}
    game_details = {}
    
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    logging.info(f"Using {max_workers} worker processes")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_backtest_iteration, i, cached_data, test_year, test_weeks, include_playoffs): i 
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
    TEST_YEAR = 2024
    TEST_WEEKS = list(range(1, 19))  # Regular season weeks 1-18
    # For playoffs, use: [19, 20, 21, 22] for WildCard, Division, ConfChamp, SuperBowl (2021+ seasons)
    NUM_RUNS = 1000
    USE_WEB = True  # Set to False to use local HTML files
    START_YEAR = 2019  # First year to load for training data
    
    results = main_backtest(
        num_runs=NUM_RUNS,
        test_year=TEST_YEAR,
        test_weeks=TEST_WEEKS,
        include_playoffs=False,
        use_web=USE_WEB,
        start_year=START_YEAR
    )
