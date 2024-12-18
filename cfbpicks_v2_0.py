import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4 import Comment
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample
import logging
from concurrent.futures import ProcessPoolExecutor
import functools
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File directory setup
file_dir = 'C:/Users/Bobby/Downloads/CFB_Stats/'
games_template = file_dir + 'yyyy College Football Schedule and Results _ College Football at Sports-Reference.com.html'
def_template = file_dir + 'yyyy College Football Team Defense _ College Football at Sports-Reference.com.html'
off_template = file_dir + 'yyyy College Football Team Offense _ College Football at Sports-Reference.com.html'

# Team name mappings
variable_school_names = {
    'Alabama-Birmingham':'UAB',
    'Southern Methodist':'SMU',
    'Central Florida':'UCF',
    'Southern California':'USC',
    'Louisiana State':'LSU',
    'Pittsburgh':'Pitt',
    'Mississippi':'Ole Miss',
    'Texas-El Paso':'UTEP',
    'Texas-San Antonio':'UTSA',
    'Nevada-Las Vegas':'UNLV',
    'Texas Christian':'TCU',
    'Bowling Green':'Bowling Green State'
}

# Months mapping for date processing
months = {
    'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
    'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12
}

# Consolidated stats to use
stat_columns_to_use = frozenset({
    'team',
    'pass_yds',    # Both offense and defense will track these
    'rush_yds',    # Both offense and defense will track these
    'turnovers',   # Will combine fumbles_lost and pass_int for offense, same for defense
    'g'            # Number of games played for per-game calculation
})

def is_numeric_value(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def normalize_team_name(team_name):
    return variable_school_names.get(team_name, team_name)

def calculate_turnovers(stats_dict):
    fumbles = float(stats_dict.get('fumbles_lost', '0') or '0')
    interceptions = float(stats_dict.get('pass_int', '0') or '0')
    return fumbles + interceptions

@functools.lru_cache(maxsize=128)
def get_offense_stats(html_content, year_str):
    single_year_offense = []
    soup = BeautifulSoup(html_content, 'html.parser')
    offense_table = soup.find('table', id='offense')
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

        # Calculate per-game stats
        try:
            single_team_offense = {
                'team': team_name,
                'pass_yds': float(cols.get('pass_yds', '0') or '0') / num_games,
                'rush_yds': float(cols.get('rush_yds', '0') or '0') / num_games,
                'turnovers': calculate_turnovers(cols) / num_games,
                'g': num_games
            }
            single_year_offense.append(single_team_offense)
        except Exception as e:
            logging.error(f"Error processing offensive stats for {team_name}: {str(e)}")
            continue
            
    return pd.DataFrame(single_year_offense)

@functools.lru_cache(maxsize=128)
def get_defense_stats(html_content, year_str):
    single_year_defense = []
    soup = BeautifulSoup(html_content, 'html.parser')
    defense_table = soup.find('table', id='defense')
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

        # Calculate per-game stats
        try:
            single_team_defense = {
                'team': team_name,
                'def_pass_yds': float(cols.get('pass_yds', '0') or '0') / num_games,
                'def_rush_yds': float(cols.get('rush_yds', '0') or '0') / num_games,
                'def_turnovers': calculate_turnovers(cols) / num_games,
                'g': num_games
            }
            single_year_defense.append(single_team_defense)
        except Exception as e:
            logging.error(f"Error processing defensive stats for {team_name}: {str(e)}")
            continue
            
    return pd.DataFrame(single_year_defense)


@functools.lru_cache(maxsize=128)
def get_games(html_content, year_str):
    soup = BeautifulSoup(html_content, 'html.parser')
    games_table = soup.find('table', id='schedule')
    if not games_table:
        logging.error("Could not find schedule table")
        return pd.DataFrame(), pd.DataFrame()
        
    logging.info(f"Found schedule table for {year_str}")
    
    games_list_past = []
    games_list_future = []
    
    tbody = games_table.find('tbody')
    if not tbody:
        logging.error("Could not find tbody in schedule table")
        return pd.DataFrame(), pd.DataFrame()
        
    rows = tbody.find_all('tr')
    logging.info(f"Found {len(rows)} rows in schedule table")
    
    for game_row in rows:
        cols = {col['data-stat']: col.text.strip() for col in game_row.find_all('td')}
        logging.debug(f"Processing row with columns: {cols.keys()}")
        
        if not cols or 'date_game' not in cols:
            logging.debug("Skipping row - no date_game found")
            continue
        
        # Get week number from th element
        week_num = game_row.find('th', {'data-stat': 'week_number'})
        if week_num:
            week_number = week_num.text.strip()
            logging.debug(f"Found week number: {week_number}")
        else:
            logging.debug("No week number found in row")
            continue
            
        winner = normalize_team_name(cols.get('winner_school_name', ''))
        loser = normalize_team_name(cols.get('loser_school_name', ''))
        
        if not winner or not loser:
            logging.debug(f"Skipping row - missing winner or loser: winner={winner}, loser={loser}")
            continue
            
        game_time = cols.get('time_game', '12:00 PM')
        if game_time == '':
            game_time = '12:00 PM'
            
        game_location = cols.get('game_location', '')
        away_team = winner if game_location == '@' else loser
        home_team = loser if away_team == winner else winner
            
        game_data = {
            'week_number': week_number,
            'game_date': cols['date_game'],
            'away_team': away_team,
            'home_team': home_team,
            'year': year_str
        }
        
        if not cols.get('winner_points') or not cols.get('loser_points'):
            game_data['score_difference'] = 0
            games_list_future.append(game_data)
            logging.debug("Added to future games list")
        else:
            winner_score = int(cols['winner_points'])
            loser_score = int(cols['loser_points'])
            score_difference = winner_score - loser_score if home_team == winner else loser_score - winner_score
            game_data['score_difference'] = score_difference
            games_list_past.append(game_data)
            logging.debug("Added to past games list")
    
    logging.info(f"Processed {len(games_list_past)} past games and {len(games_list_future)} future games")
    
    past_df = pd.DataFrame(games_list_past)
    future_df = pd.DataFrame(games_list_future)
    
    if not past_df.empty:
        past_df['year'] = year_str
    if not future_df.empty:
        future_df['year'] = year_str
        
    return past_df, future_df


def load_data_for_year(year):
    year_str = str(year)
    try:
        logging.info(f"Attempting to load files for year {year_str}")
        
        offense_path = off_template.replace('yyyy', year_str)
        defense_path = def_template.replace('yyyy', year_str)
        games_path = games_template.replace('yyyy', year_str)
        
        logging.info(f"Checking paths:\nOffense: {offense_path}\nDefense: {defense_path}\nGames: {games_path}")
        
        with open(off_template.replace('yyyy', year_str)) as f:
            offense_content = f.read()
            logging.info(f"Successfully loaded offense file for {year_str}")
            offense_df = get_offense_stats(offense_content, year_str)
            logging.info(f"Processed offense data for {year_str}. DataFrame empty? {offense_df.empty}")
            
        with open(def_template.replace('yyyy', year_str)) as f:
            defense_content = f.read()
            logging.info(f"Successfully loaded defense file for {year_str}")
            defense_df = get_defense_stats(defense_content, year_str)
            logging.info(f"Processed defense data for {year_str}. DataFrame empty? {defense_df.empty}")
            
        with open(games_template.replace('yyyy', year_str)) as f:
            games_content = f.read()
            logging.info(f"Successfully loaded games file for {year_str}")
            games_past, games_future = get_games(games_content, year_str)
            logging.info(f"Processed games data for {year_str}. Past games empty? {games_past.empty}, Future games empty? {games_future.empty}")
        
        if offense_df.empty or defense_df.empty:
            logging.error(f"Empty DataFrames for year {year_str}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Merge stats and ensure year column exists
        combined_stats = pd.merge(offense_df, defense_df, on='team', how='inner')
        combined_stats['year'] = year_str
        
        # Ensure all DataFrames have the year column
        if not games_past.empty:
            games_past['year'] = year_str
        if not games_future.empty:
            games_future['year'] = year_str
        
        logging.info(f"Successfully processed all data for year {year_str}")
        return combined_stats, games_past, games_future
        
    except Exception as e:
        logging.error(f"Error processing year {year}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def process_single_run(iteration, years_range, start_week, end_week):
    logging.info(f"Running simulation {iteration + 1}")
    
    all_data = []
    for year in years_range:
        data = load_data_for_year(year)
        if not data[0].empty and not data[1].empty:
            all_data.append(data)
    
    if not all_data:
        logging.error("No valid data loaded")
        return {}
        
    multi_year_combined_stats = pd.concat([data[0] for data in all_data], ignore_index=True)
    multi_year_games_history = pd.concat([data[1] for data in all_data], ignore_index=True)
    multi_year_games_future = pd.concat([data[2] for data in all_data], ignore_index=True)
    
    # Ensure all required columns exist
    required_columns = ['year', 'home_team', 'away_team', 'score_difference', 'week_number']
    for col in required_columns:
        if col not in multi_year_games_future.columns:
            logging.error(f"Missing required column: {col}")
            return {}
    
    # Prepare the data
    merged_data = multi_year_games_history.merge(
        multi_year_combined_stats, how='left', left_on=['year', 'home_team'], right_on=['year', 'team']
    ).merge(
        multi_year_combined_stats, how='left', left_on=['year', 'away_team'], right_on=['year', 'team'],
        suffixes=('_home', '_away')
    )
    
    features = [
        'pass_yds_home', 'def_pass_yds_home', 'rush_yds_home', 'def_rush_yds_home',
        'turnovers_home', 'def_turnovers_home', 'pass_yds_away', 'def_pass_yds_away',
        'rush_yds_away', 'def_rush_yds_away', 'turnovers_away', 'def_turnovers_away'
    ]
    
    # Drop rows with missing values
    merged_data = merged_data.dropna(subset=features + ['score_difference'])
    
    if merged_data.empty:
        logging.error("No valid training data after merging")
        return {}
        
    X = merged_data[features]
    y = merged_data['score_difference']
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Process future games
    predictions = {}
    try:
        future_games = multi_year_games_future[
            (multi_year_games_future['year'].astype(str) >= '2024') & 
            (multi_year_games_future['week_number'].astype(int) >= int(start_week)) &
            (multi_year_games_future['week_number'].astype(int) < int(end_week))
        ]
    except Exception as e:
        logging.error(f"Error filtering future games: {str(e)}")
        return {}

    for idx, game in future_games.iterrows():
        try:
            home_stats = multi_year_combined_stats[
                (multi_year_combined_stats['team'] == game['home_team']) & 
                (multi_year_combined_stats['year'] == game['year'])
            ]
            away_stats = multi_year_combined_stats[
                (multi_year_combined_stats['team'] == game['away_team']) & 
                (multi_year_combined_stats['year'] == game['year'])
            ]
            
            if home_stats.empty or away_stats.empty:
                continue
                
            input_features_dict = {}
            for stat in ['pass_yds', 'def_pass_yds', 'rush_yds', 'def_rush_yds', 'turnovers', 'def_turnovers']:
                input_features_dict[f"{stat}_home"] = home_stats[stat].iloc[0]
                input_features_dict[f"{stat}_away"] = away_stats[stat].iloc[0]
            
            input_features = pd.DataFrame([input_features_dict])
            pred = model.predict(input_features)[0]
            
            predictions[idx] = {
                'week': game['week_number'],
                'date': game['game_date'],
                'away_team': game['away_team'],
                'home_team': game['home_team'],
                'prediction': pred
            }
            
        except Exception as e:
            logging.error(f"Error processing game {idx}: {str(e)}")
            continue
    
    return predictions


def main(num_runs=1, start_week=15, end_week=16):
    years_range = range(2018, 2024)
    
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_run, i, years_range, start_week, end_week)
            for i in range(num_runs)
        ]
        
        all_predictions = {}
        for future in futures:
            try:
                predictions = future.result()
                for idx, pred_data in predictions.items():
                    if idx in all_predictions:
                        all_predictions[idx]['predictions'].append(pred_data['prediction'])
                    else:
                        all_predictions[idx] = {
                            'week': pred_data['week'],
                            'date': pred_data['date'],
                            'away_team': pred_data['away_team'],
                            'home_team': pred_data['home_team'],
                            'predictions': [pred_data['prediction']]
                        }
            except Exception as e:
                logging.error(f"Error processing simulation: {str(e)}")
                continue
    
    # Calculate and log final averages
    for idx, game_data in all_predictions.items():
        if game_data['predictions']:
            preds = game_data['predictions']
            avg_pred = round(np.mean(preds), 1)
            std_dev = round(np.std(preds), 2)
            logging.info(f"{game_data['week']}|{game_data['date']}|{game_data['away_team']}|{avg_pred}|{game_data['home_team']}|{std_dev}")

if __name__ == "__main__":
    main()