import numpy as np
#import requests
from bs4 import BeautifulSoup
from bs4 import Comment
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime

file_dir = 'C:/Users/bobna/Downloads/CFB_Stats/'
games_template = file_dir + 'yyyy Schedule and Results _ College Football at Sports-Reference.com.html'
def_template = file_dir + 'yyyy Team Defense _ College Football at Sports-Reference.com.html'
off_template = file_dir + 'yyyy Team Offense _ College Football at Sports-Reference.com.html'

months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}

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
    'Nevada-Las Vegas':'UNLV'
}

start_time = datetime.now()

off_stat_names_to_use = {
    'points',
    'pass_cmp_pct',
    'pass_yds',
    'pass_td',
    'rush_yds',
    'rush_yds_per_att',
    'rush_td',
    'tot_plays',
    'tot_yds',
    'tot_yds_per_play',
    'first_down_pass',
    'first_down_rush',
    'first_down',
    'fumbles_lost',
    'pass_int',
}


def_stat_names_to_use = {
    'def_opp_points',
    'def_opp_pass_cmp',
    'def_opp_pass_cmp_pct',
    'def_opp_pass_yds',
    'def_opp_pass_td',
    'def_opp_rush_att',
    'def_opp_rush_yds',
    'def_opp_rush_yds_per_att',
    'def_opp_rush_td',
    'def_opp_tot_plays',
    'def_opp_tot_yds',
    'def_opp_tot_yds_per_play',
    'def_opp_first_down_pass',
    'def_opp_first_down_rush',
    'def_opp_first_down_penalty',
    'def_opp_first_down',
    'def_opp_penalty_yds',
    'def_opp_fumbles_lost',
    'def_opp_pass_int',
}


def strip_chars_from_stat(stat_to_alter):
    return stat_to_alter.replace('-','').replace(' ', '').replace(',','').replace('.','')


def populate_inputs(year_stats, off_inputs, def_inputs):
    stats_used = dict()
    stat_num = 0

    for (stat_name, stat_value) in year_stats.items():
        formatted_stat = strip_chars_from_stat(stat_value)
        if formatted_stat.isdigit():
            per_game_stat = float(stat_value)
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
    offense_table = offense_soup.find_all('table', id='offense')[0]
    for offense_row in offense_table.find_all('tbody')[0].find_all('tr'):
        if len(offense_row.find_all('td')) < 1:
            continue
        single_team_offense = dict()
        for offense_column in offense_row.find_all('td'):
            column_name = offense_column['data-stat']
            single_team_offense[column_name] = offense_column.text
        team = single_team_offense['school_name']
        single_year_offense[team] = single_team_offense
        print('Extracted offensive data for the ' + str(year) + ' ' + team)
    return single_year_offense


def get_def_stats(full_defense_soup):   #combine with above - parameterize offense vs defense specifics
    single_year_defense = dict()
    defense_table = full_defense_soup.find_all('table', id='defense')[0]
    for defense_row in defense_table.find_all('tbody')[0].find_all('tr'):
        if len(defense_row.find_all('td')) < 1:
            continue
        single_team_defense = dict()
        for defense_column in defense_row.find_all('td'):
            column_name = 'def_' + defense_column['data-stat']
            single_team_defense[column_name] = defense_column.text
        team = single_team_defense['def_school_name']
        single_year_defense[team] = single_team_defense
        print('Extracted defense data for the ' + str(year) + ' ' + team)
    return single_year_defense


def is_game_in_future(program_start_time, game_date, game_time):
    game_time_split = game_time.split(':')
    game_hours = int(game_time_split[0])
    game_minutes = int(game_time_split[1][:2])
    game_am_pm = str.strip(game_time_split[1][2:])

    if game_am_pm == 'PM' and game_hours != 12:
        game_hours += 12
    elif game_am_pm == 'AM' and game_hours == 12:
        game_hours = 0

    game_date_split = game_date.split()
    game_year = int(game_date_split[2])
    game_month = game_date_split[0]
    game_month_number = months[game_month]
    game_day_number = int(game_date_split[1].replace(',', '')) #strip comma

    game_datetime = datetime(year=game_year, month=game_month_number, day=game_day_number)#, hour=game_hours, minute=game_minutes)

    return game_datetime.date() >= datetime.today().date()


def get_model_inputs(full_games_soup, single_year_stats, year):
    inputs = []
    outputs = []
    stat_names_used = dict()

    games_table = full_games_soup.find_all('table', id='schedule')[0]
    for game_row in games_table.find_all('tbody')[0].find_all('tr'):
        winner_off_inputs = []
        winner_def_inputs = []
        loser_off_inputs = []
        loser_def_inputs = []
        single_game = dict()

        game_stat_columns = game_row.find_all('td')

        if len(game_stat_columns) < 1:
            continue

        for game_stat_column in game_stat_columns:
            game_column_name = game_stat_column['data-stat']
            if game_stat_column.a is not None and (game_column_name == 'winner_school_name' or game_column_name == 'loser_school_name'):
                single_game[game_column_name] = game_stat_column.a.text
            else:
                single_game[game_column_name] = game_stat_column.text

        game_time = single_game['time_game']
        if game_time == '':
            game_time = '12:00 PM'

        if is_game_in_future(start_time, single_game['date_game'], game_time):
            return inputs, outputs, stat_names_used

        winner = single_game['winner_school_name']
        loser = single_game['loser_school_name']

        if winner in variable_school_names:
            winner = variable_school_names[winner]
        if loser in variable_school_names:
            loser = variable_school_names[loser]

        if winner not in single_year_stats or loser not in single_year_stats:
            continue

        winner_stats = single_year_stats[winner]
        stat_names_used_tmp = populate_inputs(winner_stats, winner_off_inputs, winner_def_inputs)
        if len(stat_names_used_tmp) == len(off_stat_names_to_use) + len(def_stat_names_to_use):
            stat_names_used = stat_names_used_tmp

        loser_stats = single_year_stats[loser]
        populate_inputs(loser_stats, loser_off_inputs, loser_def_inputs)

        winner_inputs = []
        loser_inputs = []
        winner_inputs.extend(winner_off_inputs)
        winner_inputs.extend(loser_def_inputs)
        loser_inputs.extend(loser_off_inputs)
        loser_inputs.extend(winner_def_inputs)

        #print(single_game)

        if single_game['winner_points'] == '' or single_game['loser_points'] == '':
            continue

        winner_score = int(single_game['winner_points'])
        loser_score = int(single_game['loser_points'])
        inputs.append(winner_inputs)
        outputs.append(winner_score)
        inputs.append(loser_inputs)
        outputs.append(loser_score)
    return inputs, outputs, stat_names_used


def predict_weekly_scores(linear_regression_model, week_num_target):
    future_games_file = open(games_template.replace('yyyy', '2022'))
    future_games_soup = BeautifulSoup(future_games_file.read(), 'html.parser')

    games_table = future_games_soup.find_all('table', id='schedule')[0]
    for game_row in games_table.find_all('tbody')[0].find_all('tr'):

        game_stat_columns = game_row.find_all('td')

        if len(game_stat_columns) < 1:
            continue

        game_to_predict = dict()
        for game_stat_column in game_stat_columns:
            game_column_name = game_stat_column['data-stat']
            if game_stat_column.a is not None and (game_column_name == 'winner_school_name' or game_column_name == 'loser_school_name'):
                game_to_predict[game_column_name] = game_stat_column.a.text
            else:
                game_to_predict[game_column_name] = game_stat_column.text

        #if not is_game_in_future(start_time, game_to_predict['date_game'], game_to_predict['time_game']):
        #    continue
        if game_to_predict['week_number'] != week_num_target:
            continue
        else:
            team1_off_inputs = []
            team1_def_inputs = []
            team2_off_inputs = []
            team2_def_inputs = []

            team1 = game_to_predict['winner_school_name']
            team2 = game_to_predict['loser_school_name']
            if team1 in variable_school_names:
                team1 = variable_school_names[team1]
            if team2 in variable_school_names:
                team2 = variable_school_names[team2]

            team1StatsPresent = team1 in year_stats[2022]
            team2StatsPresent = team2 in year_stats[2022]

            game_time = game_to_predict['time_game']
            if game_time == '':
                game_time = '12:00 PM'

            if not team1StatsPresent or not team2StatsPresent:
                print(week_num_target + '|' + str(game_to_predict['date_game']) + ' ' + game_time + '|' + team1 + '|' + str(team1StatsPresent) + '|' + team2 + '|' + str(team2StatsPresent))
            else:
                team1_year_stats = year_stats[2022][team1]
                team2_year_stats = year_stats[2022][team2]

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
                print(week_num_target + '|' + str(game_to_predict['date_game']) + ' ' + game_time + '|' + team1 + '|' + str(round(team1_pred[0])) + '|' + team2 + '|' + str(round(team2_pred[0])))


year_stats = dict()
x_input = []
y_input = []
stat_names_used = dict()

for year in range(2016, 2023):
    single_year_stats = dict()

    offense_file = open(off_template.replace('yyyy', str(year)))
    offense_soup = BeautifulSoup(offense_file.read(), 'html.parser')
    single_year_offense = get_off_stats(offense_soup)

    defense_file = open(def_template.replace('yyyy', str(year)))
    defense_soup = BeautifulSoup(defense_file.read(), 'html.parser')
    single_year_defense = get_def_stats(defense_soup)

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

    games_file = open(games_template.replace('yyyy', str(year)))
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

predict_weekly_scores(model, '7')

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
