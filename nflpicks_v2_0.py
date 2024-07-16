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
        football_data, how='left', left_on=['year', 'home_team'], right_on=['year', 'team']
    ).merge(
        football_data, how='left', left_on=['year', 'away_team'], right_on=['year', 'team'], suffixes=('_home', '_away')
    )

    # Select relevant columns and drop rows with missing values
    features = [
        'pass_yds_home', 'def_pass_yds_home', 'rush_yds_home', 'def_rush_yds_home', 'turnovers_home', 'def_turnovers_home',
        'pass_yds_away', 'def_pass_yds_away', 'rush_yds_away', 'def_rush_yds_away', 'turnovers_away', 'def_turnovers_away'
    ]
    merged_data = merged_data.dropna(subset=features + ['score_difference'])

    return merged_data, features


# Predict game function
def predict_game(home_team, away_team, year, football_data, model):
    # Get the team statistics
    home_stats = football_data[(football_data['team'] == home_team) & (football_data['year'] == year)]
    away_stats = football_data[(football_data['team'] == away_team) & (football_data['year'] == year)]

    if home_stats.empty or away_stats.empty:
        raise ValueError("Team statistics for the specified year not found.")

    # Create the input feature vector
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

    # Predict the score difference
    predicted_score_difference = model.predict(input_features)[0]
    return predicted_score_difference


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
    y = merged_data['score_difference']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the score differences
    y_pred = model.predict(X_test)

    # Evaluate the model
    score_rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

    print(f"Score Difference RMSE: {score_rmse}")

    # Example prediction
    predicted_score_difference = predict_game('Team A', 'Team B', 2023, multi_year_combined_stats, model)
    print(f"Predicted Score Difference for Team A vs Team B in 2023: {predicted_score_difference}")


if __name__ == "__main__":
    main()
