import numpy as np
#import requests
from bs4 import BeautifulSoup
from bs4 import Comment
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
#import nfl_ats_utils as nfl_utils --- get all functions in this file

#2019-20 NBA Season Summary _ Basketball-Reference.com
#December 2019-20 NBA Schedule and Results _ Basketball-Reference.com

file_dir = 'C:/Users/bobna/Downloads/NBA_Stats/'
games_template = file_dir + 'month yyyy-yy NBA Schedule and Results _ Basketball-Reference.com.html'
stats_template = file_dir + 'yyyy-yy NBA Season Summary _ Basketball-Reference.com.html'

months = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
months_abbr = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
season_months = {'October':10, 'November':11, 'December':12, 'January':1, 'February':2, 'March':3, 'April':4}

start_time = datetime.now()

off_stat_names_to_use = {
}

def_stat_names_to_use = {
}


def strip_chars_from_stat(stat_to_alter):
    return stat_to_alter.replace('-','').replace(' ', '').replace(',','').replace('.','')


def populate_inputs(year_stats, off_inputs, def_inputs):
    num_games_played = float(year_stats['g'])
    stats_used = dict()
    stat_num = 0
    for (stat_name, stat_value) in year_stats.items():
        formatted_stat = strip_chars_from_stat(stat_value)
        if formatted_stat.isdigit():
            per_game_stat = float(stat_value)/num_games_played
            if stat_name in off_stat_names_to_use:
                off_inputs.append(per_game_stat)
                stats_used[stat_num] = stat_name
                stat_num += 1
            elif stat_name in def_stat_names_to_use:
                def_inputs.append(per_game_stat)
                stats_used[stat_num] = stat_name
                stat_num += 1
    return stats_used


def get_off_stats(full_offense_soup):
    single_year_offense = dict()
    for comment in full_offense_soup.find_all(string=lambda text: isinstance(text, Comment)):
        offense_soup = BeautifulSoup(comment.string, 'html.parser')
        offense_tables = offense_soup.find_all('table', id='team-stats-per_game')
        if len(offense_tables) < 1:
            continue
        else:
            offense_table = offense_tables[0]
            offense_rows = offense_table.find_all('tbody')[0].find_all('tr')
            for offense_row in offense_rows:
                single_team_offense = dict()
                for offense_column in offense_row.find_all('td'):
                    column_name = offense_column['data-stat']
                    single_team_offense[column_name] = offense_column.text
                team = single_team_offense['team_name']
                single_year_offense[team] = single_team_offense
                print('Extracted offensive data for the ' + str(year) + ' ' + team)
    return single_year_offense


def get_def_stats(full_defense_soup):
    single_year_defense = dict()
    for comment in full_defense_soup.find_all(string=lambda text: isinstance(text, Comment)):
        defense_soup = BeautifulSoup(comment.string, 'html.parser')
        defense_tables = defense_soup.find_all('table', id='opponent-stats-per_game')
        if len(defense_tables) < 1:
            continue
        else:
            defense_table = defense_tables[0]
            defense_rows = defense_table.find_all('tbody')[0].find_all('tr')
            for defense_row in defense_rows:
                single_team_defense = dict()
                for defense_column in defense_row.find_all('td'):
                    column_name = defense_column['data-stat']
                    single_team_defense[column_name] = defense_column.text
                team = single_team_defense['team_name']
                single_year_defense[team] = single_team_defense
                print('Extracted defensive data for the ' + str(year) + ' ' + team)
    return single_year_defense


def is_game_in_future(program_start_time, game_date, game_time):
    game_time_split = game_time.split(':')
    game_hours = int(game_time_split[0])
    game_minutes = int(game_time_split[1][:2])
    game_am_pm = game_time_split[1][2:]

    if game_am_pm == 'p' and game_hours != 12:
        game_hours += 12
    elif game_am_pm == 'a' and game_hours == 12:
        game_hours = 0

    game_date_split = game_date.replace(',', '').split()
    game_year = int(game_date_split[3])
    game_month = game_date_split[1]
    game_month_number = months_abbr[game_month]
    game_day_number = int(game_date_split[2])

    game_datetime = datetime(year=game_year, month=game_month_number, day=game_day_number, hour=game_hours, minute=game_minutes)

    return game_datetime > program_start_time


def get_model_inputs(full_games_soup, single_year_stats, year):
    game_counter = 0
    games_table = full_games_soup.find_all('table', id='schedule')[0]
    inputs = []
    outputs = []
    stat_names_used = dict()
    for game_row in games_table.find_all('tbody')[0].find_all('tr'):
        if game_counter == 256:
            return inputs, outputs, stat_names_used

        winner_off_inputs = []
        winner_def_inputs = []
        loser_off_inputs = []
        loser_def_inputs = []
        single_game = dict()

        game_date = game_row.find('th').a.text
        single_game['game_date'] = game_date
        game_stat_columns = game_row.find_all('td')

        if len(game_stat_columns) < 1:
            continue

        game_counter += 1

        for game_stat_column in game_stat_columns:
            game_column_name = game_stat_column['data-stat']
            single_game[game_column_name] = game_stat_column.text

        if is_game_in_future(start_time, game_date, single_game['game_start_time']):
            return inputs, outputs, stat_names_used

        #not done by winner and loser, honestly dont think i care -> just predicting team1 score vs team2 score
        winner = single_game['visitor_team_name']
        winner_stats = single_year_stats[winner]
        stat_names_used_tmp = populate_inputs(winner_stats, winner_off_inputs, winner_def_inputs)
        if len(stat_names_used_tmp) == len(off_stat_names_to_use) + len(def_stat_names_to_use):
            stat_names_used = stat_names_used_tmp

        loser = single_game['home_team_name']
        loser_stats = single_year_stats[loser]
        populate_inputs(loser_stats, loser_off_inputs, loser_def_inputs)

        winner_inputs = []
        loser_inputs = []
        winner_inputs.extend(winner_off_inputs)
        winner_inputs.extend(loser_def_inputs)
        loser_inputs.extend(loser_off_inputs)
        loser_inputs.extend(winner_def_inputs)

        winner_score = int(single_game['visitor_pts'])  #should rename all these vars as im not doing winner and loser
        loser_score = int(single_game['home_pts'])
        inputs.append(winner_inputs)
        outputs.append(winner_score)
        inputs.append(loser_inputs)
        outputs.append(loser_score)
    return inputs, outputs, stat_names_used


def predict_weekly_scores(linear_regression_model, week_num_target):
    future_games_file = open(games_template.replace('yyyy', '2019'))
    future_games_soup = BeautifulSoup(future_games_file.read(), 'html.parser')
    games_table = future_games_soup.find_all('table', id='games')[0]
    for game_row in games_table.find_all('tbody')[0].find_all('tr'):

        week_num = game_row.find_all('th')[0].text

        if week_num != week_num_target:
            continue
        else:
            team1_off_inputs = []
            team1_def_inputs = []
            team2_off_inputs = []
            team2_def_inputs = []
            game_to_predict = dict()

            game_stat_columns = game_row.find_all('td')

            if len(game_stat_columns) < 1:
                continue

            for game_stat_column in game_stat_columns:
                game_column_name = game_stat_column['data-stat']
                game_to_predict[game_column_name] = game_stat_column.text

            team1 = game_to_predict['winner']
            team2 = game_to_predict['loser']
            team1_year_stats = year_stats[2019][team1]
            team2_year_stats = year_stats[2019][team2]

            populate_inputs(team1_year_stats, team1_off_inputs, team1_def_inputs)
            populate_inputs(team2_year_stats, team2_off_inputs, team2_def_inputs)

            team1_off_inputs.extend(team2_def_inputs)
            team2_off_inputs.extend(team1_def_inputs)
            team1_inputs = []
            team2_inputs = []
            team1_inputs.append(team1_off_inputs)
            team2_inputs.append(team2_off_inputs)

            team1_pred = linear_regression_model.predict(team1_inputs)
            team2_pred = linear_regression_model.predict(team2_inputs)
            print(team1 + ':' + str(team1_pred[0]) + ':' + team2 + ':' + str(team2_pred[0]))


year_stats = dict()
x_input = []
y_input = []
stat_names_used = dict()

for year in range(2019, 2020):
    single_year_stats = dict()

    next_year = str(year + 1)[2:]

    stats_file = open(stats_template.replace('yyyy', str(year)).replace('yy', next_year), 'rb')
    stats_soup = BeautifulSoup(stats_file.read(), 'html.parser')
    single_year_offense = get_off_stats(stats_soup)

    single_year_defense = get_def_stats(stats_soup)

    for team in single_year_offense:
        single_team_stats = dict()
        team_offense = single_year_offense[team]
        team_defense = single_year_defense[team]
        for stat_name in team_offense:
            single_team_stats[stat_name] = team_offense[stat_name]
        for stat_name in team_defense:
            single_team_stats[stat_name] = team_defense[stat_name]
        single_year_stats[team] = single_team_stats

    year_stats[year] = single_year_stats

    for month in season_months:
        games_file = open(games_template.replace('yyyy', str(year)).replace('yy', next_year).replace('month', month), 'rb')
        games_soup = BeautifulSoup(games_file.read(), 'html.parser')
        inputs, outputs, stat_names_used = get_model_inputs(games_soup, single_year_stats, year)

    for input_stat in inputs:
        x_input.append(input_stat)
    for output_score in outputs:
        y_input.append(output_score)


x, y = np.array(x_input), np.array(y_input)
model = LinearRegression().fit(x, y)

#dataset = pd.read_csv('C:\\Users\\bobna\\OneDrive\\Documents') #missing a parser file?

#coeff_df = pd.DataFrame(model.coef_, stat_names_used, columns=['Coefficient'])
#print(coeff_df)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

predict_weekly_scores(model, '16')

for num_stat in range(0, len(x_input[0])):
    x_plot = []
    x_label = stat_names_used[num_stat]
    for outcome in range(0, len(x_input)):
        x_plot.append(x_input[outcome][num_stat])

    plt.scatter(x_plot, y_input)
    plt.xlabel(x_label)
    plt.ylabel('single game points')
    #plt.show()

print('Completed!')
