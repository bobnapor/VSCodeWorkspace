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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File directory setup
file_dir = 'C:/Users/Bobby/Downloads/NFL_Stats/'
games_template = file_dir + 'yyyy NFL Regular Season Schedule _ Pro-Football-Reference.com.html'
def_template = file_dir + 'yyyy NFL Opposition & Defensive Statistics _ Pro-Football-Reference.com.html'
off_template = file_dir + 'yyyy NFL Standings & Team Stats _ Pro-Football-Reference.com.html'

# Columns to use for stats
stat_columns_to_use = frozenset({
    'team',
    'pass_yds',
    'rush_yds',
    'turnovers'
})

def is_numeric_value(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

@functools.lru_cache(maxsize=128)
def get_offense_stats(html_content, year_str):
    single_year_offense = []
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        offense_soup = BeautifulSoup(comment, 'html.parser')
        offense_table = offense_soup.find('table', id='team_stats')
        if offense_table:
            offense_rows = offense_table.find('tbody').find_all('tr')
            for offense_row in offense_rows:
                cols = {col['data-stat']: col.text.strip() for col in offense_row.find_all('td') if col['data-stat'] in stat_columns_to_use or col['data-stat'] == 'g'}
                if 'team' not in cols:
                    continue
                    
                num_games = float(cols.get('g', '0') or '0')
                if num_games == 0:
                    continue
                    
                single_team_offense = {
                    stat: '0' if not cols.get(stat) else 
                    str(float(cols[stat])/num_games) if is_numeric_value(cols[stat]) else 
                    cols[stat] 
                    for stat in stat_columns_to_use if stat in cols
                }
                
                if single_team_offense.get('team'):  # Only add if team exists
                    single_year_offense.append(single_team_offense)
                
    return pd.DataFrame(single_year_offense)

@functools.lru_cache(maxsize=128)
def get_defense_stats(html_content, year_str):
    soup = BeautifulSoup(html_content, 'html.parser')
    single_year_defense = []
    defense_table = soup.find('table', id='team_stats')
    if not defense_table:
        return pd.DataFrame()
        
    defense_rows = defense_table.find('tbody').find_all('tr')
    for defense_row in defense_rows:
        cols = {col['data-stat']: col.text.strip() for col in defense_row.find_all('td') if col['data-stat'] in stat_columns_to_use or col['data-stat'] == 'g'}
        if 'team' not in cols:
            continue
            
        num_games = float(cols.get('g', '0') or '0')
        if num_games == 0:
            continue
            
        single_team_defense = {
            f'def_{stat}': '0' if not cols.get(stat) else 
            str(float(cols[stat])/num_games) if is_numeric_value(cols[stat]) else 
            cols[stat] 
            for stat in stat_columns_to_use if stat in cols
        }
        
        if single_team_defense.get('def_team'):  # Only add if team exists
            single_year_defense.append(single_team_defense)
        
    return pd.DataFrame(single_year_defense)

@functools.lru_cache(maxsize=128)
def get_games(html_content, year_str):
    soup = BeautifulSoup(html_content, 'html.parser')
    games_table = soup.find('table', id='games')
    if not games_table:
        return pd.DataFrame(), pd.DataFrame()
        
    games_list_past = []
    games_list_future = []
    
    tbody = games_table.find('tbody')
    if not tbody:
        return pd.DataFrame(), pd.DataFrame()
        
    for game_row in tbody.find_all('tr'):
        th = game_row.find('th')
        if not th:
            continue
        this_game_week_num = th.text.strip()
        if not this_game_week_num:
            continue
            
        cols = {col['data-stat']: col.text.strip() for col in game_row.find_all('td')}
        if not cols or 'game_date' not in cols or 'winner' not in cols or 'loser' not in cols:
            continue
            
        away_team = cols['winner'] if cols.get('game_location') == '@' else cols['loser']
        home_team = cols['loser'] if away_team == cols['winner'] else cols['winner']
        
        if cols.get('pts_win') and cols.get('pts_lose'):
            winner_score = int(cols['pts_win'])
            loser_score = int(cols['pts_lose'])
            score_difference = winner_score - loser_score if home_team == cols['winner'] else loser_score - winner_score
            
            games_list_past.append({
                'week_number': this_game_week_num,
                'game_date': cols['game_date'],
                'away_team': away_team,
                'home_team': home_team,
                'score_difference': score_difference,
                'year': year_str
            })
        else:
            games_list_future.append({
                'week_number': this_game_week_num,
                'game_date': cols['game_date'],
                'away_team': away_team,
                'home_team': home_team,
                'score_difference': 0,
                'year': year_str
            })
            
    return pd.DataFrame(games_list_past), pd.DataFrame(games_list_future)

def load_data_for_year(year):
    year_str = str(year)
    try:
        # Read files once and cache content
        with open(off_template.replace('yyyy', year_str)) as f:
            offense_content = f.read()
        with open(def_template.replace('yyyy', year_str)) as f:
            defense_content = f.read()
        with open(games_template.replace('yyyy', year_str)) as f:
            games_content = f.read()
            
        offense_df = get_offense_stats(offense_content, year_str)
        defense_df = get_defense_stats(defense_content, year_str)
        
        # Merge stats
        defense_df.rename(columns={'def_team':'team'}, inplace=True)
        combined_stats = pd.merge(offense_df, defense_df, on='team', how='inner')
        combined_stats['year'] = year_str
        
        games_past, games_future = get_games(games_content, year_str)
        
        return combined_stats, games_past, games_future
        
    except Exception as e:
        logging.error(f"Error processing year {year}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def process_single_run(iteration, years_range, features):
    logging.info(f"Running simulation {iteration + 1}")
    
    # Load and process data for all years
    all_data = [load_data_for_year(year) for year in years_range]
    
    # Filter out empty DataFrames
    all_data = [data for data in all_data if not data[0].empty and not data[1].empty]
    
    if not all_data:
        logging.error("No valid data loaded")
        return {}
    
    multi_year_combined_stats = pd.concat([data[0] for data in all_data], ignore_index=True)
    multi_year_games_history = pd.concat([data[1] for data in all_data], ignore_index=True)
    multi_year_games_future = pd.concat([data[2] for data in all_data], ignore_index=True)
    
    # Prepare the data
    merged_data = multi_year_games_history.merge(
        multi_year_combined_stats, how='left', left_on=['year', 'home_team'], right_on=['year', 'team']
    ).merge(
        multi_year_combined_stats, how='left', left_on=['year', 'away_team'], right_on=['year', 'team'], 
        suffixes=('_home', '_away')
    )
    
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
    future_games = multi_year_games_future[
        (multi_year_games_future['year'].astype(str) >= '2024') & 
        (multi_year_games_future['week_number'].astype(int) >= 15) &
        (multi_year_games_future['week_number'].astype(int) < 16)
    ]
    
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
            
            # Create input features dictionary - Python 3.8 compatible version
            input_features_dict = {}
            # Add home stats
            for stat in ['pass_yds', 'def_pass_yds', 'rush_yds', 'def_rush_yds', 'turnovers', 'def_turnovers']:
                input_features_dict[f"{stat}_home"] = home_stats[stat].iloc[0]
            # Add away stats
            for stat in ['pass_yds', 'def_pass_yds', 'rush_yds', 'def_rush_yds', 'turnovers', 'def_turnovers']:
                input_features_dict[f"{stat}_away"] = away_stats[stat].iloc[0]
            
            input_features = pd.DataFrame([input_features_dict])
            
            pred = model.predict(input_features)[0]
            predictions[idx] = pred
            
        except Exception as e:
            logging.error(f"Error processing game {idx}: {str(e)}")
            continue
    
    return predictions

def main(num_runs):
    years_range = range(2018, 2025)
    features = [
        'pass_yds_home', 'def_pass_yds_home', 'rush_yds_home', 'def_rush_yds_home', 
        'turnovers_home', 'def_turnovers_home', 'pass_yds_away', 'def_pass_yds_away', 
        'rush_yds_away', 'def_rush_yds_away', 'turnovers_away', 'def_turnovers_away'
    ]
    
    # Calculate chunk size for parallel processing
    chunk_size = max(1, num_runs // (ProcessPoolExecutor()._max_workers or 1))
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_run, i, years_range, features)
            for i in range(num_runs)
        ]
        
        # Collect results
        all_predictions = {}
        for future in futures:
            try:
                predictions = future.result()
                for idx, pred in predictions.items():
                    if idx in all_predictions:
                        all_predictions[idx].append(pred)
                    else:
                        all_predictions[idx] = [pred]
            except Exception as e:
                logging.error(f"Error processing simulation: {str(e)}")
                continue
    
    # Calculate and log final averages
    for idx, preds in all_predictions.items():
        if preds:  # Only process if we have predictions
            avg_pred = round(np.mean(preds), 1)
            std_dev = round(np.std(preds), 2)
            logging.info(f"{idx}|{avg_pred}|{std_dev}")

if __name__ == "__main__":
    main(10000)