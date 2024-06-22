import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
        'off_passing_yds_home', 'def_passing_yds_home', 'off_rushing_yds_home', 'def_rushing_yds_home', 'off_turnovers_home', 'def_turnovers_home',
        'off_passing_yds_away', 'def_passing_yds_away', 'off_rushing_yds_away', 'def_rushing_yds_away', 'off_turnovers_away', 'def_turnovers_away'
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
    # Assuming football_data and games are already loaded into pandas DataFrames
    # Sample data from the previous completion
    # You might need to adjust the paths if reading from CSVs or other sources

    # Load or define your data here
    # For this example, let's assume football_data and games are loaded as pandas DataFrames

    # Prepare the data
    merged_data, features = preprocess_data(football_data, games)

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
