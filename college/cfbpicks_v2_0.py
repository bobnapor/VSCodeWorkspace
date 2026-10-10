import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4 import Comment
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample
import logging
import concurrent.futures
import os
from datetime import datetime
from functools import lru_cache

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
    'turnovers',
    'points',
    'pass_cmp_pct',
    'pass_td',
    'rush_yds_per_att',
    'rush_td',
    'tot_plays',
    'tot_yds',
    'tot_yds_per_play',
    'first_down_pass',
    'first_down_rush',
    'first_down'
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

        try:
            single_team_offense = {
                'team': team_name,
                'pass_yds': str(float(cols.get('pass_yds', '0') or '0')),
                'rush_yds': str(float(cols.get('rush_yds', '0') or '0')),
                'turnovers': str(calculate_off_turnovers(cols)),
                'points': str(float(cols.get('points', '0') or '0')),
                'pass_cmp_pct': str(float(cols.get('pass_cmp_pct', '0') or '0')),
                'pass_td': str(float(cols.get('pass_td', '0') or '0')),
                'rush_yds_per_att': str(float(cols.get('rush_yds_per_att', '0') or '0')),
                'rush_td': str(float(cols.get('rush_td', '0') or '0')),
                'tot_plays': str(float(cols.get('plays_offense', '0') or '0')),
                'tot_yds': str(float(cols.get('tot_yds_offense', '0') or '0')),
                'tot_yds_per_play': str(float(cols.get('tot_yds_per_play', '0') or '0')),
                'first_down_pass': str(float(cols.get('first_down_pass', '0') or '0')),
                'first_down_rush': str(float(cols.get('first_down_rush', '0') or '0')),
                'first_down': str(float(cols.get('first_down', '0') or '0')),
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

        try:
            single_team_defense = {
                'def_team': team_name,
                'def_pass_yds': str(float(cols.get('opp_pass_yds', '0') or '0')),
                'def_rush_yds': str(float(cols.get('opp_rush_yds', '0') or '0')),
                'def_turnovers': str(calculate_def_turnovers(cols)),
                'def_points': str(float(cols.get('opp_points', '0') or '0')),
                'def_pass_cmp_pct': str(float(cols.get('opp_pass_cmp_pct', '0') or '0')),
                'def_pass_td': str(float(cols.get('opp_pass_td', '0') or '0')),
                'def_rush_yds_per_att': str(float(cols.get('opp_rush_yds_per_att', '0') or '0')),
                'def_rush_td': str(float(cols.get('opp_rush_td', '0') or '0')),
                'def_tot_plays': str(float(cols.get('plays_defense', '0') or '0')),
                'def_tot_yds': str(float(cols.get('tot_yds_defense', '0') or '0')),
                'def_tot_yds_per_play': str(float(cols.get('opp_tot_yds_per_play', '0') or '0')),
                'def_first_down_pass': str(float(cols.get('opp_first_down_pass', '0') or '0')),
                'def_first_down_rush': str(float(cols.get('opp_first_down_rush', '0') or '0')),
                'def_first_down': str(float(cols.get('opp_first_down', '0') or '0')),
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
            
        game_location = cols.get('game_location', '')
        
        # Handle neutral sites
        is_neutral = (game_location == 'N')
        
        if is_neutral:
            # For neutral sites, use the order from the data
            away_team = loser
            home_team = winner
        else:
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
                'is_neutral': 1 if is_neutral else 0,
                'year': year_str
            })
        else:
            games_list_future.append({
                'week_number': week_number,
                'game_date': cols['date_game'],
                'away_team': away_team,
                'home_team': home_team,
                'score_difference': 0,
                'is_neutral': 1 if is_neutral else 0,
                'year': year_str
            })

    return pd.DataFrame(games_list_past), pd.DataFrame(games_list_future)


# OPTIMIZATION: Cache win percentage calculations
@lru_cache(maxsize=2048)
def calculate_win_percentage_cached(team, year, games_tuple):
    """Calculate win percentage with caching"""
    wins = 0
    total = 0
    for game_data in games_tuple:
        home_team, away_team, score_diff, game_year = game_data
        if game_year != year:
            continue
        if home_team == team or away_team == team:
            total += 1
            if (home_team == team and score_diff > 0) or (away_team == team and score_diff < 0):
                wins += 1
    
    return wins / total if total > 0 else 0.5


def calculate_win_percentage(games_history, team, year):
    """Calculate win percentage for a team in a given year"""
    # Convert to hashable tuple for caching
    games_tuple = tuple(
        (row['home_team'], row['away_team'], row['score_difference'], row['year'])
        for _, row in games_history.iterrows()
    )
    return calculate_win_percentage_cached(team, year, games_tuple)


# OPTIMIZATION: Vectorized SOS calculation
def calculate_strength_of_schedule_fast(games_history, team_stats):
    """Calculate average opponent strength for each team - vectorized version"""
    sos_dict = {}
    
    # Create lookup dictionaries for fast access
    team_points = team_stats.set_index(['team', 'year'])['points'].astype(float).to_dict()
    team_def_points = team_stats.set_index(['team', 'year'])['def_points'].astype(float).to_dict()
    
    # Group by team and year
    for (team, year), group in team_stats.groupby(['team', 'year']):
        # Get all games this team played
        team_games = games_history[
            ((games_history['home_team'] == team) | (games_history['away_team'] == team)) &
            (games_history['year'] == year)
        ]
        
        if len(team_games) == 0:
            sos_dict[(team, year)] = {'sos_offense': 30.0, 'sos_defense': 30.0}
            continue
        
        # Get opponents
        opponents = []
        for _, game in team_games.iterrows():
            opponent = game['away_team'] if game['home_team'] == team else game['home_team']
            opponents.append(opponent)
        
        # Lookup opponent stats
        opp_points = [team_points.get((opp, year), 30.0) for opp in opponents]
        opp_def_points = [team_def_points.get((opp, year), 30.0) for opp in opponents]
        
        sos_dict[(team, year)] = {
            'sos_offense': np.mean(opp_points) if opp_points else 30.0,
            'sos_defense': np.mean(opp_def_points) if opp_def_points else 30.0
        }
    
    return sos_dict


def add_advanced_metrics(team_stats, games_history):
    """Add win percentage, SOS, and opponent-adjusted stats"""
    
    # OPTIMIZATION: Pre-calculate all win percentages
    win_pct_dict = {}
    for team in team_stats['team'].unique():
        for year in team_stats['year'].unique():
            win_pct_dict[(team, year)] = calculate_win_percentage(games_history, team, year)
    
    team_stats['win_pct'] = team_stats.apply(
        lambda row: win_pct_dict.get((row['team'], row['year']), 0.5),
        axis=1
    )
    
    # Use faster SOS calculation
    sos_dict = calculate_strength_of_schedule_fast(games_history, team_stats)
    
    team_stats['sos_offense'] = team_stats.apply(
        lambda row: sos_dict.get((row['team'], row['year']), {}).get('sos_offense', 30.0), axis=1
    )
    
    team_stats['sos_defense'] = team_stats.apply(
        lambda row: sos_dict.get((row['team'], row['year']), {}).get('sos_defense', 30.0), axis=1
    )
    
    # OPTIMIZATION: Vectorize adjusted stats calculations
    team_stats['adj_points'] = np.where(
        team_stats['sos_defense'] > 0,
        team_stats['points'].astype(float) * (30.0 / team_stats['sos_defense']),
        team_stats['points'].astype(float)
    )
    
    team_stats['adj_tot_yds'] = np.where(
        team_stats['sos_defense'] > 0,
        team_stats['tot_yds'].astype(float) * (30.0 / team_stats['sos_defense']),
        team_stats['tot_yds'].astype(float)
    )
    
    team_stats['adj_tot_yds_per_play'] = np.where(
        team_stats['sos_defense'] > 0,
        team_stats['tot_yds_per_play'].astype(float) * (30.0 / team_stats['sos_defense']),
        team_stats['tot_yds_per_play'].astype(float)
    )
    
    team_stats['adj_def_points'] = np.where(
        team_stats['sos_offense'] > 0,
        team_stats['def_points'].astype(float) * (30.0 / team_stats['sos_offense']),
        team_stats['def_points'].astype(float)
    )
    
    team_stats['adj_def_tot_yds'] = np.where(
        team_stats['sos_offense'] > 0,
        team_stats['def_tot_yds'].astype(float) * (30.0 / team_stats['sos_offense']),
        team_stats['def_tot_yds'].astype(float)
    )
    
    team_stats['adj_def_tot_yds_per_play'] = np.where(
        team_stats['sos_offense'] > 0,
        team_stats['def_tot_yds_per_play'].astype(float) * (30.0 / team_stats['sos_offense']),
        team_stats['def_tot_yds_per_play'].astype(float)
    )
    
    return team_stats


def preprocess_data(football_data, games):
    # Add advanced metrics
    football_data = add_advanced_metrics(football_data, games)
    
    merged_data = games.merge(
        football_data, how='left', left_on=['year', 'home_team'], right_on=['year', 'team']
    ).merge(
        football_data, how='left', left_on=['year', 'away_team'], right_on=['year', 'team'], suffixes=('_home', '_away')
    )

    # All features from stat_columns_to_use plus new advanced metrics
    features = [
        # Home team offense
        'pass_yds_home', 'rush_yds_home', 'turnovers_home', 'points_home',
        'pass_cmp_pct_home', 'pass_td_home', 'rush_yds_per_att_home', 'rush_td_home',
        'tot_plays_home', 'tot_yds_home', 'tot_yds_per_play_home',
        'first_down_pass_home', 'first_down_rush_home', 'first_down_home',
        # Home team defense
        'def_pass_yds_home', 'def_rush_yds_home', 'def_turnovers_home', 'def_points_home',
        'def_pass_cmp_pct_home', 'def_pass_td_home', 'def_rush_yds_per_att_home', 'def_rush_td_home',
        'def_tot_plays_home', 'def_tot_yds_home', 'def_tot_yds_per_play_home',
        'def_first_down_pass_home', 'def_first_down_rush_home', 'def_first_down_home',
        # Away team offense
        'pass_yds_away', 'rush_yds_away', 'turnovers_away', 'points_away',
        'pass_cmp_pct_away', 'pass_td_away', 'rush_yds_per_att_away', 'rush_td_away',
        'tot_plays_away', 'tot_yds_away', 'tot_yds_per_play_away',
        'first_down_pass_away', 'first_down_rush_away', 'first_down_away',
        # Away team defense
        'def_pass_yds_away', 'def_rush_yds_away', 'def_turnovers_away', 'def_points_away',
        'def_pass_cmp_pct_away', 'def_pass_td_away', 'def_rush_yds_per_att_away', 'def_rush_td_away',
        'def_tot_plays_away', 'def_tot_yds_away', 'def_tot_yds_per_play_away',
        'def_first_down_pass_away', 'def_first_down_rush_away', 'def_first_down_away',
        # Advanced metrics
        'win_pct_home', 'win_pct_away',
        'sos_offense_home', 'sos_defense_home',
        'sos_offense_away', 'sos_defense_away',
        'adj_points_home', 'adj_tot_yds_home', 'adj_tot_yds_per_play_home',
        'adj_points_away', 'adj_tot_yds_away', 'adj_tot_yds_per_play_away',
        'adj_def_points_home', 'adj_def_tot_yds_home', 'adj_def_tot_yds_per_play_home',
        'adj_def_points_away', 'adj_def_tot_yds_away', 'adj_def_tot_yds_per_play_away',
        # Neutral site indicator
        'is_neutral',
    ]
    
    merged_data = merged_data.dropna(subset=features + ['score_difference'])

    return merged_data, features


def predict_game(home_team, away_team, year, football_data, model, is_neutral=0):
    home_stats = football_data[(football_data['team'] == home_team) & (football_data['year'] == year)]
    away_stats = football_data[(football_data['team'] == away_team) & (football_data['year'] == year)]

    if home_stats.empty or away_stats.empty:
        raise ValueError("Team statistics for the specified year not found.")

    input_features = pd.DataFrame([{
        # Home team offense
        'pass_yds_home': home_stats['pass_yds'].values[0],
        'rush_yds_home': home_stats['rush_yds'].values[0],
        'turnovers_home': home_stats['turnovers'].values[0],
        'points_home': home_stats['points'].values[0],
        'pass_cmp_pct_home': home_stats['pass_cmp_pct'].values[0],
        'pass_td_home': home_stats['pass_td'].values[0],
        'rush_yds_per_att_home': home_stats['rush_yds_per_att'].values[0],
        'rush_td_home': home_stats['rush_td'].values[0],
        'tot_plays_home': home_stats['tot_plays'].values[0],
        'tot_yds_home': home_stats['tot_yds'].values[0],
        'tot_yds_per_play_home': home_stats['tot_yds_per_play'].values[0],
        'first_down_pass_home': home_stats['first_down_pass'].values[0],
        'first_down_rush_home': home_stats['first_down_rush'].values[0],
        'first_down_home': home_stats['first_down'].values[0],
        # Home team defense
        'def_pass_yds_home': home_stats['def_pass_yds'].values[0],
        'def_rush_yds_home': home_stats['def_rush_yds'].values[0],
        'def_turnovers_home': home_stats['def_turnovers'].values[0],
        'def_points_home': home_stats['def_points'].values[0],
        'def_pass_cmp_pct_home': home_stats['def_pass_cmp_pct'].values[0],
        'def_pass_td_home': home_stats['def_pass_td'].values[0],
        'def_rush_yds_per_att_home': home_stats['def_rush_yds_per_att'].values[0],
        'def_rush_td_home': home_stats['def_rush_td'].values[0],
        'def_tot_plays_home': home_stats['def_tot_plays'].values[0],
        'def_tot_yds_home': home_stats['def_tot_yds'].values[0],
        'def_tot_yds_per_play_home': home_stats['def_tot_yds_per_play'].values[0],
        'def_first_down_pass_home': home_stats['def_first_down_pass'].values[0],
        'def_first_down_rush_home': home_stats['def_first_down_rush'].values[0],
        'def_first_down_home': home_stats['def_first_down'].values[0],
        # Away team offense
        'pass_yds_away': away_stats['pass_yds'].values[0],
        'rush_yds_away': away_stats['rush_yds'].values[0],
        'turnovers_away': away_stats['turnovers'].values[0],
        'points_away': away_stats['points'].values[0],
        'pass_cmp_pct_away': away_stats['pass_cmp_pct'].values[0],
        'pass_td_away': away_stats['pass_td'].values[0],
        'rush_yds_per_att_away': away_stats['rush_yds_per_att'].values[0],
        'rush_td_away': away_stats['rush_td'].values[0],
        'tot_plays_away': away_stats['tot_plays'].values[0],
        'tot_yds_away': away_stats['tot_yds'].values[0],
        'tot_yds_per_play_away': away_stats['tot_yds_per_play'].values[0],
        'first_down_pass_away': away_stats['first_down_pass'].values[0],
        'first_down_rush_away': away_stats['first_down_rush'].values[0],
        'first_down_away': away_stats['first_down'].values[0],
        # Away team defense
        'def_pass_yds_away': away_stats['def_pass_yds'].values[0],
        'def_rush_yds_away': away_stats['def_rush_yds'].values[0],
        'def_turnovers_away': away_stats['def_turnovers'].values[0],
        'def_points_away': away_stats['def_points'].values[0],
        'def_pass_cmp_pct_away': away_stats['def_pass_cmp_pct'].values[0],
        'def_pass_td_away': away_stats['def_pass_td'].values[0],
        'def_rush_yds_per_att_away': away_stats['def_rush_yds_per_att'].values[0],
        'def_rush_td_away': away_stats['def_rush_td'].values[0],
        'def_tot_plays_away': away_stats['def_tot_plays'].values[0],
        'def_tot_yds_away': away_stats['def_tot_yds'].values[0],
        'def_tot_yds_per_play_away': away_stats['def_tot_yds_per_play'].values[0],
        'def_first_down_pass_away': away_stats['def_first_down_pass'].values[0],
        'def_first_down_rush_away': away_stats['def_first_down_rush'].values[0],
        'def_first_down_away': away_stats['def_first_down'].values[0],
        # Advanced metrics
        'win_pct_home': home_stats['win_pct'].values[0],
        'win_pct_away': away_stats['win_pct'].values[0],
        'sos_offense_home': home_stats['sos_offense'].values[0],
        'sos_defense_home': home_stats['sos_defense'].values[0],
        'sos_offense_away': away_stats['sos_offense'].values[0],
        'sos_defense_away': away_stats['sos_defense'].values[0],
        'adj_points_home': home_stats['adj_points'].values[0],
        'adj_tot_yds_home': home_stats['adj_tot_yds'].values[0],
        'adj_tot_yds_per_play_home': home_stats['adj_tot_yds_per_play'].values[0],
        'adj_points_away': away_stats['adj_points'].values[0],
        'adj_tot_yds_away': away_stats['adj_tot_yds'].values[0],
        'adj_tot_yds_per_play_away': away_stats['adj_tot_yds_per_play'].values[0],
        'adj_def_points_home': home_stats['adj_def_points'].values[0],
        'adj_def_tot_yds_home': home_stats['adj_def_tot_yds'].values[0],
        'adj_def_tot_yds_per_play_home': home_stats['adj_def_tot_yds_per_play'].values[0],
        'adj_def_points_away': away_stats['adj_def_points'].values[0],
        'adj_def_tot_yds_away': away_stats['adj_def_tot_yds'].values[0],
        'adj_def_tot_yds_per_play_away': away_stats['adj_def_tot_yds_per_play'].values[0],
        # Neutral site
        'is_neutral': is_neutral,
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


# OPTIMIZATION: Pre-compute advanced metrics once, outside the iteration loop
_cached_advanced_metrics = {}

def load_all_data_once(off_template, def_template, games_template):
    """Load and preprocess all data once, to be reused across iterations"""
    global _cached_advanced_metrics
    
    cache_key = 'all_data'
    if cache_key in _cached_advanced_metrics:
        return _cached_advanced_metrics[cache_key]
    
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
    
    # Pre-compute advanced metrics once
    multi_year_combined_stats_with_metrics = add_advanced_metrics(
        multi_year_combined_stats.copy(), 
        multi_year_games_history
    )
    
    _cached_advanced_metrics[cache_key] = (
        multi_year_combined_stats_with_metrics,
        multi_year_games_history,
        multi_year_games_future
    )
    
    return _cached_advanced_metrics[cache_key]


def run_single_iteration(iteration, off_template, def_template, games_template, predict_start_year_inc, predict_end_year_exc, predict_start_week_inc, predict_end_week_exc):
    """Run a single iteration of the simulation"""
    
    # OPTIMIZATION: Use pre-loaded data
    multi_year_combined_stats, multi_year_games_history, multi_year_games_future = load_all_data_once(
        off_template, def_template, games_template
    )

    # Prepare the data
    merged_data, features = preprocess_data(multi_year_combined_stats, multi_year_games_history)

    # Calculate and subtract average home field advantage from training data
    home_advantage = merged_data['score_difference'].mean()
    
    # Split the data into train and test sets
    X = merged_data[features]
    y = merged_data['score_difference'] - home_advantage
    
    # Bootstrap resample the data for more variation
    X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X))
    
    # Random train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

    # OPTIMIZATION: Reduced Random Forest complexity and skip cross-validation
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=50,      # Reduced from 100
            max_depth=8,          # Reduced from 10
            min_samples_split=10, # Increased to reduce complexity
            n_jobs=1,             # Each worker handles one model
            random_state=None
        )
    }

    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # Predict and evaluate the model (skip cross-validation for speed)
        y_pred = model.predict(X_test)
        score_mae = mean_absolute_error(y_test, y_pred)

        results[name] = {
            'model': model,
            'mae': score_mae,
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
    game_info = {}
    
    for index, game in input_games.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        input_week = game['week_number']
        input_year = game['year']
        actual_score_difference = game['score_difference']
        is_neutral = game.get('is_neutral', 0)
        
        try:
            base_prediction = predict_game(home_team, away_team, input_year, multi_year_combined_stats, best_model, is_neutral)
            
            if is_neutral == 1:
                adjusted_prediction = base_prediction
            else:
                adjusted_prediction = base_prediction + 2
            
            noise = np.random.normal(0, 1.0)
            projected_score_difference = round(adjusted_prediction + noise, 1)
            
            iteration_predictions[index] = projected_score_difference
            game_info[index] = {
                'home_team': home_team,
                'away_team': away_team,
                'week': input_week,
                'year': input_year,
                'game_date': game['game_date'],
                'actual_score_difference': actual_score_difference,
                'best_model': best_model_name,
                'is_neutral': is_neutral
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
    model_usage = {}  # Track which models are being selected
    
    # Determine number of worker threads (use CPU count)
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    logging.info(f"Using {max_workers} worker processes")
    logging.info(f"Running {num_runs} iterations with bootstrap resampling and random noise")
    logging.info(f"Using advanced metrics: Win%, SOS, Opponent-Adjusted Stats, and Neutral Site Indicator")
    logging.info(f"Optimizations: Data pre-loading, vectorized calculations, reduced RF complexity")
    
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
                
                # Track model usage
                for idx, info in game_info.items():
                    model_name = info.get('best_model', 'Unknown')
                    model_usage[model_name] = model_usage.get(model_name, 0) + 1
                
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
    
    # Log model selection statistics
    logging.info("\n=== MODEL SELECTION STATISTICS ===")
    total_selections = sum(model_usage.values())
    for model_name, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_selections * 100) if total_selections > 0 else 0
        logging.info(f"{model_name}: {count} times ({percentage:.1f}%)")
    
    # Calculate and log final averages with team names, std dev, and median
    logging.info("\n=== FINAL AVERAGED PREDICTIONS ===")
    for game_index in sorted(averaged_predictions.keys()):
        predictions_array = np.array(all_predictions[game_index])
        average_pred = round(averaged_predictions[game_index] / num_runs, 1)
        std_dev = round(np.std(predictions_array), 2)
        median_pred = round(np.median(predictions_array), 1)
        
        if game_index in game_details:
            info = game_details[game_index]
            neutral_flag = 'N' if info.get('is_neutral', 0) == 1 else ''
            logging.info(
                f"{game_index}|{info['week']}|{info['game_date']}|{info['year']}|{info['away_team']}|{info['home_team']}|{average_pred}|{median_pred}|{std_dev}|{neutral_flag}|Actual:{info['actual_score_difference']}"
            )
        else:
            logging.info(f"Game {game_index}: Mean:{average_pred}|Median:{median_pred}|StdDev:{std_dev}")

    logging.info("Projections complete!")


if __name__ == "__main__":
    main(num_runs=10, predict_start_year_inc=2025, predict_end_year_exc=2026, predict_start_week_inc=17, predict_end_week_exc=22)