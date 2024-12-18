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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File directory setup
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
                    logging.info(f'Extracted offensive data for the {year_str} {team}')
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
            logging.info(f'Extracted defense data for the {year_str} {team}')
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


def main(num_runs):
    averaged_predictions = {}

    for iteration in range(num_runs):
        logging.info(f"Running simulation {iteration + 1}")
        all_years_team_stats_arr = []
        all_games_history_arr = []
        all_games_future_arr = []

        for year in range(2018, 2025):
            #try:
                single_year_combined_stats, single_year_games_past, single_year_games_future = load_data_for_year(year, off_template, def_template, games_template)
                all_years_team_stats_arr.append(single_year_combined_stats)
                all_games_history_arr.append(single_year_games_past)
                all_games_future_arr.append(single_year_games_future)
            #except Exception as e:
            #    logging.error(f"Failed to process data for year {year}: {e}")
            #    continue

        multi_year_combined_stats = pd.concat(all_years_team_stats_arr, ignore_index=True)
        multi_year_games_history = pd.concat(all_games_history_arr, ignore_index=True)
        multi_year_games_future = pd.concat(all_games_future_arr, ignore_index=True)

        # Prepare the data
        merged_data, features = preprocess_data(multi_year_combined_stats, multi_year_games_history)

        # Split the data into train and test sets
        X = merged_data[features]
        y = merged_data['score_difference']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the models
        models = {
            'Linear Regression': LinearRegression(),
            #'Decision Tree': DecisionTreeRegressor(random_state=42),
            #'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),  # Reduced number of trees for faster fitting
            #'SVR': SVR()
        }

        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            logging.info(f"{name} Cross-validation scores: {cv_scores}")
            logging.info(f"{name} Mean cross-validation score: {np.mean(cv_scores)}")

            # Predict and evaluate the model
            y_pred = model.predict(X_test)
            score_rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
            score_mae = mean_absolute_error(y_test, y_pred)
            score_r2 = r2_score(y_test, y_pred)
            logging.info(f"{name} Score Difference RMSE: {score_rmse}")
            logging.info(f"{name} Score Difference MAE: {score_mae}")
            logging.info(f"{name} Score Difference R^2: {score_r2}")

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
        logging.info(f"Best model based on MAE: {best_model_name}")

        # Calculate prediction intervals for the best model
        lower_bound, upper_bound = calculate_prediction_intervals(best_model, X_train, y_train, X_test, n_iterations=100)
        logging.info(f"{best_model_name} Prediction intervals (lower bound): {lower_bound[:5]}")
        logging.info(f"{best_model_name} Prediction intervals (upper bound): {upper_bound[:5]}")

        predict_start_year_inc = 2024
        predict_end_year_exc = 2025
        predict_start_week_inc = 15
        predict_end_week_exc = 16#predict_start_week_inc + 1   #change if want to do more than 1 week

        input_games_by_year_and_week = []
        for input_year in range(predict_start_year_inc, predict_end_year_exc):
            for input_week in range(predict_start_week_inc, predict_end_week_exc):
                input_games_by_year_and_week.append(multi_year_games_future[
                    (multi_year_games_future['week_number'] == str(input_week)) & 
                    (multi_year_games_future['year'] == str(input_year))
                ])
                #comment out above and uncomment below to gather actual score differences after the games
                #input_games_by_year_and_week.append(multi_year_games_history[
                #    (multi_year_games_history['week_number'] == str(input_week)) & 
                #    (multi_year_games_history['year'] == str(input_year))
                #])

        input_games = pd.concat(input_games_by_year_and_week, ignore_index=True)
        for index, game in input_games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            input_week = game['week_number']
            input_year = game['year']
            actual_score_difference = game['score_difference']
            projected_score_difference = 0
            try:
                projected_score_difference = round(predict_game(home_team, away_team, input_year, multi_year_combined_stats, best_model), 1)
                #TODO: also compute standard deviation and median to assert range and confidence in projection
                if index in averaged_predictions:
                    averaged_predictions[index] += projected_score_difference
                else:
                    averaged_predictions[index] = projected_score_difference
                logging.info(f"{index}|{home_team}|{away_team}|{input_week}|{input_year}|{projected_score_difference}|{actual_score_difference}")
            except ValueError as e:
                logging.error(f"Could not predict score difference for {home_team} vs {away_team} for week {input_week} of {input_year}: {e}")

        for game_index, total_pred in averaged_predictions.items():
            average_pred = round(total_pred / (iteration+1), 1)
            logging.info(f"{game_index}|{average_pred}")

        logging.info("Projections complete!")

    for game_index, total_pred in averaged_predictions.items():
        average_pred = round(total_pred / num_runs, 1)
        logging.info(f"{game_index}|{average_pred}")

if __name__ == "__main__":
    main(10000)  #5k took ~10 hours, 10k took ~19 hours
    