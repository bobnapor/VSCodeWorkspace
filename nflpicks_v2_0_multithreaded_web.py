import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4 import Comment
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
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

# Global cache for web data (shared across iterations)
_web_data_cache = {}

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


def calculate_prediction_intervals(model, X, y, X_test, n_iterations=100, alpha=0.05):
    predictions = []
    for _ in range(n_iterations):
        X_resample, y_resample = resample(X, y)
        model.fit(X_resample, y_resample)
        predictions.append(model.predict(X_test))

    predictions = np.array(predictions)
    lower_bound = np.percentile(predictions, 100 * alpha / 2, axis=0)
    upper_bound = np.percentile(predictions, 100 * (1 - alpha / 2), axis=0)

    return lower_bound, upper_bound


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


def load_data_for_year(year, off_url, def_url, games_url, use_web=True):
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


def get_projection_measures(averaged_predictions, all_predictions, num_runs, game_details):
    for game_index in sorted(averaged_predictions.keys()):
        predictions_array = np.array(all_predictions[game_index])
        average_pred = round(averaged_predictions[game_index] / num_runs, 1)
        std_dev = round(np.std(predictions_array), 2)
        median_pred = round(np.median(predictions_array), 1)

        if game_index in game_details:
            info = game_details[game_index]
            logging.info(f"{game_index}|{info['week']}|{info['year']}|{info['home_team']}|{info['away_team']}|{average_pred}|{median_pred}|{std_dev}| Actual: {info['actual_score_difference']}"            )
        else:
            logging.info(f"Game {game_index}: Mean:{average_pred}|Median:{median_pred}|StdDev:{std_dev}")


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


def run_single_iteration(iteration, cached_data, predict_year, predict_week):
    """Run a single iteration of the simulation
    
    Args:
        iteration: iteration number
        cached_data: Pre-loaded data dict with 'stats', 'all_games' DataFrames
        predict_year: Year to predict (e.g., 2025)
        predict_week: Week to predict (e.g., 21 for ConfChamp)
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
    
    # Get games to predict: all games in the specified week/year
    games_to_predict = all_games[
        (all_games['year_int'] == predict_year) & 
        (all_games['week_num_int'] == predict_week)
    ]
    
    # Get training games: all games BEFORE the prediction week
    # Exclude the last regular season week of each year from training data
    def is_valid_training_game(row):
        game_year = row['year_int']
        game_week = row['week_num_int']
        last_reg_week = get_last_regular_season_week(game_year)
        
        # Exclude the last regular season week
        if game_week == last_reg_week:
            return False
        
        # Include if before prediction year, or same year but earlier week
        if game_year < predict_year:
            return True
        if game_year == predict_year and game_week < predict_week:
            return True
        return False
    
    training_games = all_games[all_games.apply(is_valid_training_game, axis=1)]
    
    # Only use games with scores for training
    training_games = training_games[training_games['score_difference'] != 0]

    # Prepare the training data
    merged_data, features = preprocess_data(multi_year_combined_stats, training_games)

    # Split the data into train and test sets
    X = merged_data[features]
    y = merged_data['score_difference']

    # Bootstrap resample the data for more variation
    X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X))
    
    # Random train/test split (no random_state for different splits each iteration)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

    # Define the models
    models = {
        'Linear Regression': LinearRegression(),
    }

    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Predict and evaluate the model
        y_pred = model.predict(X_test)
        score_rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        score_mae = mean_absolute_error(y_test, y_pred)
        score_r2 = r2_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'rmse': score_rmse,
            'mae': score_mae,
            'r2': score_r2
        }

    best_model_name = min(results, key=lambda k: results[k]['mae'])
    best_model = results[best_model_name]['model']

    iteration_predictions = {}
    game_info = {}  # Store game details for logging later
    
    for index, game in games_to_predict.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        input_week = game['week_number']
        input_year = game['year']
        actual_score_difference = game['score_difference']  # May be 0 if game hasn't been played
        actual_score_difference = game['score_difference']
        
        try:
            # Get base prediction
            base_prediction = predict_game(home_team, away_team, input_year, multi_year_combined_stats, best_model)
            
            # OPTION 2: Add random noise to predictions (normal distribution with std dev of 1.0)
            noise = np.random.normal(0, 1.0)
            projected_score_difference = round(base_prediction + noise, 1)
            
            iteration_predictions[index] = projected_score_difference
            # Store game info for final logging
            game_info[index] = {
                'home_team': home_team,
                'away_team': away_team,
                'week': input_week,
                'year': input_year,
                'actual_score_difference': actual_score_difference
            }
        except ValueError as e:
            logging.error(f"Could not predict score difference for {home_team} vs {away_team}: {e}")
    
    return iteration_predictions, game_info


def main(num_runs, predict_year=2025, predict_week=21, use_web=True):
    # Setup logging to file - only in main process
    log_filename = f'C:/Users/Bobby/nfl_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also print to console
        ],
        force=True  # Override any existing configuration
    )
    
    logging.info(f"Logging to file: {log_filename}")
    logging.info(f"Data source: {'Web URLs' if use_web else 'Local files'}")
    logging.info(f"Predicting: Year {predict_year}, Week {predict_week}")
    
    # Pre-load all data once in main process
    logging.info("Loading data from source...")
    all_years_team_stats_arr = []
    all_games_arr = []  # Combined past and future games

    for year in range(2019, 2026):
        logging.info(f"  Loading {year} data...")
        single_year_combined_stats, single_year_games_past, single_year_games_future = load_data_for_year(
            year, off_url, def_url, games_url, use_web=use_web
        )
        all_years_team_stats_arr.append(single_year_combined_stats)
        # Combine past and future games into one list
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
    all_predictions = {}  # Store all predictions for each game for std dev and median
    game_details = {}  # Store game details from first iteration
    
    # Determine number of worker threads (use CPU count)
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    logging.info(f"Using {max_workers} worker processes")
    logging.info(f"Running {num_runs} iterations with bootstrap resampling and random noise")
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all iterations - pass cached data and prediction parameters
        futures = {
            executor.submit(run_single_iteration, i, cached_data, predict_year, predict_week): i 
            for i in range(num_runs)
        }
        
        # Collect results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            iteration = futures[future]
            completed += 1
            try:
                iteration_predictions, game_info = future.result()
                
                # Store game details from first completed iteration
                if not game_details:
                    game_details = game_info
                
                # Accumulate predictions
                for game_index, pred in iteration_predictions.items():
                    if game_index in averaged_predictions:
                        averaged_predictions[game_index] += pred
                        all_predictions[game_index].append(pred)
                    else:
                        averaged_predictions[game_index] = pred
                        all_predictions[game_index] = [pred]

                if completed % 100 == 0:
                    logging.info(f"Progress: {completed}/{num_runs} iterations completed")
                    get_projection_measures(averaged_predictions, all_predictions, completed, game_details)

            except Exception as e:
                logging.error(f"Iteration {iteration} generated an exception: {e}")

    # Calculate and log final averages with team names, std dev, and median
    logging.info("\n=== FINAL AVERAGED PREDICTIONS ===")
    for game_index in sorted(averaged_predictions.keys()):
        predictions_array = np.array(all_predictions[game_index])
        average_pred = round(averaged_predictions[game_index] / num_runs, 1)
        std_dev = round(np.std(predictions_array), 2)
        median_pred = round(np.median(predictions_array), 1)
        
        if game_index in game_details:
            info = game_details[game_index]
            logging.info(
                f"{game_index}|{info['week']}|{info['year']}|{info['home_team']}|{info['away_team']}|{average_pred}|{median_pred}|{std_dev}| Actual: {info['actual_score_difference']}"
            )
        else:
            logging.info(f"Game {game_index}: Mean:{average_pred}|Median:{median_pred}|StdDev:{std_dev}")

    logging.info("Projections complete!")


if __name__ == "__main__":
    # Configuration
    PREDICT_YEAR = 2025
    PREDICT_WEEK = 16
    NUM_RUNS = 10000
    USE_WEB = True  # Set to False to use local HTML files
    
    main(
        num_runs=NUM_RUNS,
        predict_year=PREDICT_YEAR,
        predict_week=PREDICT_WEEK,
        use_web=USE_WEB
    )
