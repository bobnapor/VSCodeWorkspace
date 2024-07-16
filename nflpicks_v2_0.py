import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4 import Comment
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(single_year_offense)
    return df


def get_defense_stats(full_defense_soup, year_str):
    single_year_defense = []
    defense_table = full_defense_soup.find('table', id='team_stats')
    if not defense_table:
        return pd.DataFrame()  # Return an empty DataFrame if the table is not found
    defense_rows = defense_table.find('tbody').find_all('tr')
    for defense_row in defense_rows:
        single_team_defense = {f'def_{col["data-stat"]}': col.text for col in defense_row.find_all('td') if col['data-stat'] in stat_columns_to_use}
        team = single_team_defense.get('def_team')
        if team:
            single_year_defense.append(single_team_defense)
            print(f'Extracted defense data for the {year_str} {team}')
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(single_year_defense)
    return df


def get_single_year_team_stats(single_year_offense_df, single_year_defense_df, year_str):
    single_year_defense_df.rename(columns={'def_team':'team'}, inplace=True)
    single_year_combined_stats = pd.merge(single_year_offense_df, single_year_defense_df, on='team')
    single_year_combined_stats.loc[:, 'year'] = year_str
    return single_year_combined_stats


def get_games(full_games_soup, year_str):
    games_table = full_games_soup.find('table', id='games')
    games_list = []

    for game_row in games_table.find('tbody').find_all('tr'):
        this_game_week_num = game_row.find_all('th')[0].text
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
            'game_date': game_date,
            'away_team': away_team,
            'home_team': home_team,
            'score_difference': score_difference,
            'year': year_str
        })

    return pd.DataFrame(games_list)


# Preprocessing function
def preprocess_data(football_data, games):
    # Merge the football_data with games to get the features for each game
    merged_data = games.merge(
        football_data, how='left', left_on=['date', 'home_team'], right_on=['year', 'team']
    ).merge(
        football_data, how='left', left_on=['date', 'away_team'], right_on=['year', 'team'], suffixes=('_home', '_away')
    )

    # Select relevant columns and drop rows with missing values
    features = [
        'pass_yds_home', 'def_pass_yds_home', 'rush_yds_home', 'def_rush_yds_home', 'turnovers_home', 'def_turnovers_home',
        'pass_yds_away', 'def_pass_yds_away', 'rush_yds_away', 'def_rush_yds_away', 'turnovers_away', 'def_turnovers_away'
    ]
    merged_data = merged_data.dropna(subset=features + ['home_score', 'away_score'])

    return merged_data, features


# Predict game function
def predict_game(home_team, away_team, year, football_data, games, model_home, model_away):
    game_data = games[(games['home_team'] == home_team) & (games['away_team'] == away_team) & (games['date'] == year)]
    if game_data.empty:
        return None

    game_features = football_data[(football_data['team'].isin([home_team, away_team])) & (football_data['year'] == year)]
    if game_features.shape[0] < 2:
        return None

    home_stats = game_features[game_features['team'] == home_team].iloc[0]
    away_stats = game_features[game_features['team'] == away_team].iloc[0]

    input_features = np.array([
        home_stats['off_passing_yds'], home_stats['def_passing_yds'], home_stats['off_rushing_yds'], home_stats['def_rushing_yds'], home_stats['off_turnovers'], home_stats['def_turnovers'],
        away_stats['off_passing_yds'], away_stats['def_passing_yds'], away_stats['off_rushing_yds'], away_stats['def_rushing_yds'], away_stats['off_turnovers'], away_stats['def_turnovers']
    ]).reshape(1, -1)

    pred_home_score = model_home.predict(input_features)[0]
    pred_away_score = model_away.predict(input_features)[0]

    return pred_home_score, pred_away_score


def main():
############################################
############################################

    #work in progress - integration

############################################
############################################
    all_years_team_stats_arr = []
    all_games_history_arr = []

    for year in range(2018, 2023):
        year_str = str(year)

        offense_file = open(off_template.replace('yyyy', year_str))
        offense_soup = BeautifulSoup(offense_file.read(), 'html.parser')
        single_year_offense = get_offense_stats(offense_soup, year_str)

        defense_file = open(def_template.replace('yyyy', year_str))
        defense_soup = BeautifulSoup(defense_file.read(), 'html.parser')
        single_year_defense = get_defense_stats(defense_soup, year_str)

        single_year_combined_stats = get_single_year_team_stats(single_year_offense, single_year_defense, year_str)
        print(single_year_combined_stats)
        all_years_team_stats_arr.append(single_year_combined_stats)

        games_file = open(games_template.replace('yyyy', year_str))
        games_soup = BeautifulSoup(games_file.read(), 'html.parser')
        single_year_games = get_games(games_soup, year_str)
        print(single_year_games)
        all_games_history_arr.append(single_year_games)

    multi_year_combined_stats = pd.concat(all_years_team_stats_arr, ignore_index=True)
    multi_year_games_history = pd.concat(all_games_history_arr, ignore_index=True)
    print(multi_year_combined_stats)
    print(multi_year_games_history)

############################################
############################################

    # Assuming football_data and games are already loaded into pandas DataFrames
    # Sample data from the previous completion
    # You might need to adjust the paths if reading from CSVs or other sources

    # Load or define your data here
    # For this example, let's assume football_data and games are loaded as pandas DataFrames

    # Prepare the data
    merged_data, features = preprocess_data(multi_year_combined_stats, multi_year_games_history)

    # Split the data into train and test sets
    X = merged_data[features]
    y_home = merged_data['home_score']
    y_away = merged_data['away_score']

    X_train, X_test, y_train_home, y_test_home, y_train_away, y_test_away = train_test_split(X, y_home, y_away, test_size=0.2, random_state=42)

    # Train the linear regression models
    model_home = LinearRegression()
    model_away = LinearRegression()

    model_home.fit(X_train, y_train_home)
    model_away.fit(X_train, y_train_away)

    # Predict the scores
    y_pred_home = model_home.predict(X_test)
    y_pred_away = model_away.predict(X_test)

    # Evaluate the models
    home_score_rmse = np.sqrt(np.mean((y_pred_home - y_test_home) ** 2))
    away_score_rmse = np.sqrt(np.mean((y_pred_away - y_test_away) ** 2))

    print(f"Home Score RMSE: {home_score_rmse}")
    print(f"Away Score RMSE: {away_score_rmse}")

    # Example prediction
    predicted_scores = predict_game('Team A', 'Team B', 2023, football_data, games, model_home, model_away)
    print(f"Predicted Scores for Team A vs Team B in 2023: {predicted_scores}")


if __name__ == "__main__":
    main()
