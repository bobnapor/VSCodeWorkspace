import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4 import Comment
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample
import logging
import concurrent.futures
import os
from datetime import datetime

# File directory setup
file_dir = 'C:/Users/Bobby/Downloads/CFB_Stats/'
games_template = file_dir + 'yyyy College Football Schedule and Results _ College Football at Sports-Reference.com.html'
def_template = file_dir + 'yyyy College Football Team Defense _ College Football at Sports-Reference.com.html'
off_template = file_dir + 'yyyy College Football Team Offense _ College Football at Sports-Reference.com.html'

# Team name mappings
variable_school_names = {
    'Alabama-Birmingham': 'UAB',
    'Southern Methodist': 'SMU',
    'Central Florida': 'UCF',
    'Southern California': 'USC',
    'Louisiana State': 'LSU',
    'Pittsburgh': 'Pitt',
    'Mississippi': 'Ole Miss',
    'Texas-El Paso': 'UTEP',
    'Texas-San Antonio': 'UTSA',
    'Nevada-Las Vegas': 'UNLV',
    'Texas Christian': 'TCU',
    'Bowling Green': 'Bowling Green State'
}

# Columns to use for stats
stat_columns_to_use = {
    'team',
    'pass_yds',
    'rush_yds',
    'turnovers'
}


def is_numeric_value(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def normalize_team_name(team_name):
    return variable_school_names.get(team_name, team_name)


def calculate_def_turnovers(stats_dict):
    fumbles = float(stats_dict.get('opp_fumbles_lost', '0') or '0')
    interceptions = float(stats_dict.get('opp_pass_int', '0') or '0')
    return fumbles + interceptions


def calculate_off_turnovers(stats_dict):
    fumbles = float(stats_dict.get('fumbles_lost', '0') or '0')
    interceptions = float(stats_dict.get('pass_int', '0') or '0')
    return fumbles + interceptions


def get_offense_stats(full_offense_soup, year_str):
    single_year_offense = []
    offense_table = full_offense_soup.find('table', id='offense')
    if not offense_table:
        return pd.DataFrame()
        
    for offense_row in offense_table.find('tbody').find_all('tr'):
        cols = {col['data-stat']: col.text.strip() for col in offense_row.find_all('td')}
        if not cols:
            continue
            
        team_name = cols.get('school_name')
        if not team_name:
            continue
            
        team_name = normalize_team_name(team_name)
        
        num_games = float(cols.get('g', '0') or '0')
        if num_games == 0:
            continue

        try:
            single_team_offense = {
                'team': team_name,
                'pass_yds': str(float(cols.get('pass_yds', '0') or '0')),
                'rush_yds': str(float(cols.get('rush_yds', '0') or '0')),
                'turnovers': str(calculate_off_turnovers(cols))
            }
            single_year_offense.append(single_team_offense)
        except Exception as e:
            logging.error(f"Error processing offensive stats for {team_name}: {str(e)}")
            continue
            
    return pd.DataFrame(single_year_offense)


def get_defense_stats(full_defense_soup, year_str):
    single_year_defense = []
    defense_table = full_defense_soup.find('table', id='defense')
    if not defense_table:
        return pd.DataFrame()
        
    for defense_row in defense_table.find('tbody').find_all('tr'):
        cols = {col['data-stat']: col.text.strip() for col in defense_row.find_all('td')}
        if not cols:
            continue
            
        team_name = cols.get('school_name')
        if not team_name:
            continue
            
        team_name = normalize_team_name(team_name)
        
        num_games = float(cols.get('g', '0') or '0')
        if num_games == 0:
            continue

        try:
            single_team_defense = {
                'def_team': team_name,
                'def_pass_yds': str(float(cols.get('opp_pass_yds', '0') or '0')),
                'def_rush_yds': str(float(cols.get('opp_rush_yds', '0') or '0')),
                'def_turnovers': str(calculate_def_turnovers(cols))
            }
            single_year_defense.append(single_team_defense)
        except Exception as e:
            logging.error(f"Error processing defensive stats for {team_name}: {str(e)}")
            continue
            
    return pd.DataFrame(single_year_defense)


def get_single_year_team_stats(single_year_offense_df, single_year_defense_df, year_str):
    single_year_defense_df.rename(columns={'def_team': 'team'}, inplace=True)
    single_year_combined_stats = pd.merge(single_year_offense_df, single_year_defense_df, on='team')
    single_year_combined_stats['year'] = year_str
    return single_year_combined_stats


def clean_team_name(team_name):
    """Remove ranking prefix (e.g., '(22)') and extra whitespace from team names"""
    if not team_name:
        return team_name
    
    # Remove ranking prefix like '(22)' or '(1)' and any non-breaking spaces
    import re
    # Pattern matches: optional opening paren, digits, closing paren, any whitespace (including \xa0)
    cleaned = re.sub(r'^\(\d+\)\s*', '', team_name)
    # Also remove any remaining non-breaking spaces
    cleaned = cleaned.replace('\xa0', ' ').strip()
    
    return cleaned


def get_games(full_games_soup, year_str):
    games_table = full_games_soup.find('table', id='schedule')
    games_list_past = []
    games_list_future = []

    if not games_table:
        logging.error("Could not find schedule table")
        return pd.DataFrame(), pd.DataFrame()
        
    tbody = games_table.find('tbody')
    if not tbody:
        logging.error("Could not find tbody in schedule table")
        return pd.DataFrame(), pd.DataFrame()
        
    for game_row in tbody.find_all('tr'):
        cols = {col['data-stat']: col.text.strip() for col in game_row.find_all('td')}
        
        if not cols or 'date_game' not in cols:
            continue
        
        # Get week number from th element
        week_number = cols.get('week_number')
        if not week_number:
            continue

        # Clean team names to remove rankings and normalize
        winner_raw = cols.get('winner_school_name', '')
        loser_raw = cols.get('loser_school_name', '')
        
        winner = normalize_team_name(clean_team_name(winner_raw))
        loser = normalize_team_name(clean_team_name(loser_raw))
        
        if not winner or not loser:
            continue
            
        game_location = cols.get('game_location', '')   #need to figure out how to handle neutral sites
        away_team = winner if game_location == '@' else loser
        home_team = loser if away_team == winner else winner
            
        if cols.get('winner_points') and cols.get('loser_points') and cols.get('winner_points') != '0' and cols.get('loser_points') != '0':
            winner_score = int(cols['winner_points'])
            loser_score = int(cols['loser_points'])
            score_difference = winner_score - loser_score if home_team == winner else loser_score - winner_score

            games_list_past.append({
                'week_number': week_number,
                'game_date': cols['date_game'],
                'away_team': away_team,
                'home_team': home_team,
                'score_difference': score_difference,
                'year': year_str
            })
        else:
            games_list_future.append({
                'week_number': week_number,
                'game_date': cols['date_game'],
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


def load_data_for_year(year, off_template, def_template, games_template):
    year_str = str(year)

    with open(off_template.replace('yyyy', year_str)) as offense_file:
        offense_soup = BeautifulSoup(offense_file.read(), 'html.parser')
        single_year_offense = get_offense_stats(offense_soup, year_str)

    with open(def_template.replace('yyyy', year_str)) as defense_file:
        defense_soup = BeautifulSoup(defense_file.read(), 'html.parser')
        single_year_defense = get_defense_stats(defense_soup, year_str)

    single_year_combined_stats = get_single_year_team_stats(single_year_offense, single_year_defense, year_str)

    with open(games_template.replace('yyyy', year_str)) as games_file:
        games_soup = BeautifulSoup(games_file.read(), 'html.parser')
        single_year_games_past, single_year_games_future = get_games(games_soup, year_str)

    return single_year_combined_stats, single_year_games_past, single_year_games_future


def run_single_iteration(iteration, off_template, def_template, games_template, predict_start_year_inc, predict_end_year_exc, predict_start_week_inc, predict_end_week_exc):
    """Run a single iteration of the simulation"""
    all_years_team_stats_arr = []
    all_games_history_arr = []
    all_games_future_arr = []

    for year in range(2018, 2026):
        single_year_combined_stats, single_year_games_past, single_year_games_future = load_data_for_year(
            year, off_template, def_template, games_template
        )
        all_years_team_stats_arr.append(single_year_combined_stats)
        all_games_history_arr.append(single_year_games_past)
        all_games_future_arr.append(single_year_games_future)

    multi_year_combined_stats = pd.concat(all_years_team_stats_arr, ignore_index=True)
    multi_year_games_history = pd.concat(all_games_history_arr, ignore_index=True)
    multi_year_games_future = pd.concat(all_games_future_arr, ignore_index=True)

    # Prepare the data
    merged_data, features = preprocess_data(multi_year_combined_stats, multi_year_games_history)

    # Split the data into train and test sets
    X = merged_data[features]
    y = merged_data['score_difference']

    # OPTION 1: Bootstrap resample the data for more variation
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

    input_games_by_year_and_week = []
    for input_year in range(predict_start_year_inc, predict_end_year_exc):
        for input_week in range(predict_start_week_inc, predict_end_week_exc):
            input_games_by_year_and_week.append(multi_year_games_future[
                (multi_year_games_future['week_number'] == str(input_week)) & 
                (multi_year_games_future['year'] == str(input_year))
            ])

    input_games = pd.concat(input_games_by_year_and_week, ignore_index=True)
    iteration_predictions = {}
    game_info = {}  # Store game details for logging later
    
    for index, game in input_games.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        input_week = game['week_number']
        input_year = game['year']
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
                'game_date': game['game_date'],
                'actual_score_difference': actual_score_difference
            }
        except ValueError as e:
            logging.error(f"Could not predict score difference for {home_team} vs {away_team}: {e}")
    
    return iteration_predictions, game_info


def main(num_runs, predict_start_year_inc=2024, predict_end_year_exc=2025, predict_start_week_inc=15, predict_end_week_exc=16):
    # Setup logging to file - only in main process
    log_filename = f'C:/Users/Bobby/cfb_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
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
    
    averaged_predictions = {}
    all_predictions = {}  # Store all predictions for each game for std dev and median
    game_details = {}  # Store game details from first iteration
    
    # Determine number of worker threads (use CPU count)
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    logging.info(f"Using {max_workers} worker processes")
    logging.info(f"Running {num_runs} iterations with bootstrap resampling and random noise")
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all iterations
        futures = {
            executor.submit(run_single_iteration, i, off_template, def_template, games_template, 
                          predict_start_year_inc, predict_end_year_exc, predict_start_week_inc, predict_end_week_exc): i 
            for i in range(num_runs)
        }
        
        # Collect results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            iteration = futures[future]
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
                
                completed += 1
                if completed % 100 == 0:
                    logging.info(f"Progress: {completed}/{num_runs} iterations completed")
                    
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
                f"{game_index}|{info['week']}|{info['game_date']}|{info['year']}|{info['away_team']}|{info['home_team']}|{average_pred}|{median_pred}|{std_dev}|Actual:{info['actual_score_difference']}"
            )
        else:
            logging.info(f"Game {game_index}: Mean:{average_pred}|Median:{median_pred}|StdDev:{std_dev}")

    logging.info("Projections complete!")


if __name__ == "__main__":
    main(num_runs=1, predict_start_year_inc=2025, predict_end_year_exc=2026, predict_start_week_inc=17, predict_end_week_exc=22)