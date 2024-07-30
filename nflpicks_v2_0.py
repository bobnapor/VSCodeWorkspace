import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4 import Comment
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

file_dir = 'C:/Users/Bobby/Downloads/NFL_Stats/'
games_template = file_dir + 'yyyy NFL Weekly League Schedule _ Pro-Football-Reference.com.html'
def_template = file_dir + 'yyyy NFL Opposition & Defensive Statistics _ Pro-Football-Reference.com.html'
off_template = file_dir + 'yyyy NFL Standings & Team Stats _ Pro-Football-Reference.com.html'

stat_columns_to_use = {
    'team',
    'pass_yds',
    'rush_yds',
    'turnovers'
}

def get_offense_stats(full_offense_soup, year_str):
    single_year_offense = []
    for comment in full_offense_soup.find_all(string=lambda text: isinstance(text, Comment)):
        offense_soup = BeautifulSoup(comment, 'html.parser')
        offense_table = offense_soup.find('table', id='team_stats')
        if offense_table:
            offense_rows = offense_table.find('tbody').find_all('tr')
            for offense_row in offense_rows:
                single_team_offense = {col['data-stat']: col.text for col in offense_row.find_all('td') if col['data-stat'] in stat_columns_to_use}
                team = single_team_offense.get('team')
                if team:
                    single_year_offense.append(single_team_offense)
                    print(f'Extracted offensive data for the {year_str} {team}')
    df = pd.DataFrame(single_year_offense)
    return df

def get_defense_stats(full_defense_soup, year_str):
    single_year_defense = []
    defense_table = full_defense_soup.find('table', id='team_stats')
    if not defense_table:
        return pd.DataFrame()
    defense_rows = defense_table.find('tbody').find_all('tr')
    for defense_row in defense_rows:
        single_team_defense = {f'def_{col["data-stat"]}': col.text for col in defense_row.find_all('td') if col['data-stat'] in stat_columns_to_use}
        team = single_team_defense.get('def_team')
        if team:
            single_year_defense.append(single_team_defense)
            print(f'Extracted defense data for the {year_str} {team}')
    df = pd.DataFrame(single_year_defense)
    return df

def get_single_year_team_stats(single_year_offense_df, single_year_defense_df, year_str):
    single_year_defense_df.rename(columns={'def_team':'team'}, inplace=True)
    single_year_combined_stats = pd.merge(single_year_offense_df, single_year_defense_df, on='team')
    single_year_combined_stats['year'] = year_str
    return single_year_combined_stats

def get_games(full_games_soup, year_str):
    games_table = full_games_soup.find('table', id='games')
    games_list = []

    for game_row in games_table.find('tbody').find_all('tr'):
        this_game_week_num = game_row.find_all('th')[0].text.strip()
        game_stats = {col['data-stat']: col.text for col in game_row.find_all('td')}
        if not game_stats or 'game_date' not in game_stats or 'winner' not in game_stats or 'loser' not in game_stats or this_game_week_num == '':
            continue

        game_date = game_stats['game_date']
        away_team = game_stats['winner'] if game_stats.get('game_location') == '@' else game_stats['loser']
        home_team = game_stats['loser'] if away_team == game_stats['winner'] else game_stats['winner']

        winner_score = int(game_stats['pts_win'])
        loser_score = int(game_stats['pts_lose'])
        score_difference = winner_score - loser_score if home_team == game_stats['winner'] else loser_score - winner_score

        games_list.append({
            'week_number': this_game_week_num,
            'game_date': game_date,
            'away_team': away_team,
            'home_team': home_team,
            'score_difference': score_difference,
            'year': year_str
        })

    return pd.DataFrame(games_list)

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

def calculate_prediction_intervals(model, X, y, X_test, n_iterations=1000, alpha=0.05):
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

def main():
    all_years_team_stats_arr = []
    all_games_history_arr = []

    for year in range(2018, 2024):
        year_str = str(year)

        with open(off_template.replace('yyyy', year_str)) as offense_file:
            offense_soup = BeautifulSoup(offense_file.read(), 'html.parser')
            single_year_offense = get_offense_stats(offense_soup, year_str)

        with open(def_template.replace('yyyy', year_str)) as defense_file:
            defense_soup = BeautifulSoup(defense_file.read(), 'html.parser')
            single_year_defense = get_defense_stats(defense_soup, year_str)

        single_year_combined_stats = get_single_year_team_stats(single_year_offense, single_year_defense, year_str)
        all_years_team_stats_arr.append(single_year_combined_stats)

        with open(games_template.replace('yyyy', year_str)) as games_file:
            games_soup = BeautifulSoup(games_file.read(), 'html.parser')
            single_year_games = get_games(games_soup, year_str)
            all_games_history_arr.append(single_year_games)

    multi_year_combined_stats = pd.concat(all_years_team_stats_arr, ignore_index=True)
    multi_year_games_history = pd.concat(all_games_history_arr, ignore_index=True)

    # Prepare the data
    merged_data, features = preprocess_data(multi_year_combined_stats, multi_year_games_history)

    # Split the data into train and test sets
    X = merged_data[features]
    y = merged_data['score_difference']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores)}")

    # Prediction intervals
    lower_bound, upper_bound = calculate_prediction_intervals(model, X_train, y_train, X_test)
    print(f"Prediction intervals (lower bound): {lower_bound[:5]}")
    print(f"Prediction intervals (upper bound): {upper_bound[:5]}")

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    score_rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    score_mae = mean_absolute_error(y_test, y_pred)
    score_r2 = r2_score(y_test, y_pred)
    print(f"Score Difference RMSE: {score_rmse}")
    print(f"Score Difference MAE: {score_mae}")
    print(f"Score Difference R^2: {score_r2}")

    input_games_by_year_and_week = []

    for input_year in range(2018, 2024):
        for input_week in range(1, 19):
            input_games_by_year_and_week.append(multi_year_games_history[
                (multi_year_games_history['week_number'] == str(input_week)) & 
                (multi_year_games_history['year'] == str(input_year))
            ])

    input_games = pd.concat(input_games_by_year_and_week, ignore_index=True)
    for index, game in input_games.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        input_week = game['week_number']
        input_year = game['year']
        actual_score_difference = game['score_difference']
        try:
            predicted_score_difference = round(predict_game(home_team, away_team, input_year, multi_year_combined_stats, model), 1)
            print(f"{home_team}|{away_team}|{input_week}|{input_year}|{predicted_score_difference}|{actual_score_difference}")
        except ValueError as e:
            print(f"Could not predict score difference for {home_team} vs {away_team} for week {input_week} of {input_year}: {e}")

if __name__ == "__main__":
    main()
