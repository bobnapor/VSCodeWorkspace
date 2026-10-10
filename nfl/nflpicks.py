import numpy as np
#import requests
from bs4 import BeautifulSoup
from bs4 import Comment
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
#import nfl_ats_utils as nfl_utils --- get all functions in this file


url_template = 'https://www.pro-football-reference.com/years/yyyy/'
#def_url = 'https://www.pro-football-reference.com/years/2009/opp.htm'
#games_url = 'https://www.pro-football-reference.com/years/2019/games.htm'


#TODO: for use when online - need to expand on this for the other pages
#for year in range(2009,2020):
#    year_url = url_template.replace('yyyy', str(year))
#    year_req = requests.get(year_url)
#    print(year_req.status_code)
#    year_soup = BeautifulSoup(year_req.content, 'html.parser')
#    print(year_soup.prettify())

file_dir = 'C:/Users/Bobby/Downloads/NFL_Stats/'
games_template = file_dir + 'yyyy NFL Weekly League Schedule _ Pro-Football-Reference.com.html'
def_template = file_dir + 'yyyy NFL Opposition & Defensive Statistics _ Pro-Football-Reference.com.html'
off_template = file_dir + 'yyyy NFL Standings & Team Stats _ Pro-Football-Reference.com.html'

months = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}

start_time = datetime.now()

off_stat_names_to_use = {
    'points',
    'plays_offense',
    'pass_yds',
    'pass_td',
    'pass_int',
    'pass_fd',
    'rush_fd',
    'score_pct',
    'turnover_pct'
}

#figure out way to also only use stats common across all years (def_exp_pts_def_tot and exp_pts_tot for example blank in 2019)
def_stat_names_to_use = {
    'def_points',
    'def_fumbles_lost',
    'def_pass_yds',
    'def_pass_td',
    'def_pass_int',
    'def_pass_fd',
    'def_rush_td',
    'def_rush_yds_per_att',
    'def_rush_fd',
    'def_turnover_pct'
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
        offense_tables = offense_soup.find_all('table', id='team_stats')
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
                team = single_team_offense['team']
                single_year_offense[team] = single_team_offense
                print('Extracted offensive data for the ' + str(year) + ' ' + team)
    return single_year_offense


def get_off_stats_new(full_offense_soup):
    single_year_offense = []
    for comment in full_offense_soup.find_all(string=lambda text: isinstance(text, Comment)):
        offense_soup = BeautifulSoup(comment, 'html.parser')
        offense_table = offense_soup.find('table', id='team_stats')
        if offense_table:
            offense_rows = offense_table.find('tbody').find_all('tr')
            for offense_row in offense_rows:
                single_team_offense = {col['data-stat']: col.text for col in offense_row.find_all('td')}
                team = single_team_offense.get('team')
                if team:
                    single_year_offense.append(single_team_offense)
                    print(f'Extracted offensive data for the {team}')

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(single_year_offense)
    return df


def get_def_stats_new(full_defense_soup):
    single_year_defense = []
    defense_table = full_defense_soup.find('table', id='team_stats')
    if not defense_table:
        return pd.DataFrame()  # Return an empty DataFrame if the table is not found
    defense_rows = defense_table.find('tbody').find_all('tr')
    for defense_row in defense_rows:
        single_team_defense = {f'def_{col["data-stat"]}': col.text for col in defense_row.find_all('td')}
        team = single_team_defense.get('def_team')
        if team:
            single_year_defense.append(single_team_defense)
            print(f'Extracted defense data for the {team}')
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(single_year_defense)
    return df


def get_def_stats(full_defense_soup):
    single_year_defense = dict()
    defense_table = full_defense_soup.find_all('table', id='team_stats')[0]
    for defense_row in defense_table.find_all('tbody')[0].find_all('tr'):
        single_team_defense = dict()
        for defense_column in defense_row.find_all('td'):
            column_name = 'def_' + defense_column['data-stat']
            single_team_defense[column_name] = defense_column.text
        team = single_team_defense['def_team']
        single_year_defense[team] = single_team_defense
        print('Extracted defense data for the ' + str(year) + ' ' + team)
    return single_year_defense


def is_game_in_future(game_date):
    game_date_split = game_date.split('-')
    game_year = int(game_date_split[0])
    game_month = int(game_date_split[1])
    game_day = int(game_date_split[2])

    game_datetime = datetime(year=game_year, month=game_month, day=game_day)

    return game_datetime.date() >= datetime.today().date()


#ideas - get rid of week 17...and include playoffs
#TODO: change from using the season averages to compiling weighted averages from each game stats
def get_model_inputs(full_games_soup, single_year_stats, year):
    game_counter = 0
    games_table = full_games_soup.find_all('table', id='games')[0]
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
        
        this_game_week_num = game_row.find_all('th')[0].text
        game_stat_columns = game_row.find_all('td')

        if len(game_stat_columns) < 1:
            continue

        game_counter += 1

        for game_stat_column in game_stat_columns:
            game_column_name = game_stat_column['data-stat']
            single_game[game_column_name] = game_stat_column.text

        if is_game_in_future(single_game['game_date']):
            return inputs, outputs, stat_names_used
        
        winner = single_game['winner']
        winner_stats = single_year_stats[winner]
        stat_names_used_tmp = populate_inputs(winner_stats, winner_off_inputs, winner_def_inputs)
        if len(stat_names_used_tmp) == len(off_stat_names_to_use) + len(def_stat_names_to_use):
            stat_names_used = stat_names_used_tmp

        loser = single_game['loser']
        loser_stats = single_year_stats[loser]
        populate_inputs(loser_stats, loser_off_inputs, loser_def_inputs)

        winner_inputs = []
        loser_inputs = []
        winner_inputs.extend(winner_off_inputs)
        winner_inputs.extend(loser_def_inputs)
        loser_inputs.extend(loser_off_inputs)
        loser_inputs.extend(winner_def_inputs)

        #print(single_game)
        winner_score = int(single_game['pts_win'])
        loser_score = int(single_game['pts_lose'])
        inputs.append(winner_inputs)
        outputs.append(winner_score)
        inputs.append(loser_inputs)
        outputs.append(loser_score)

        if year == 2023:
            gamedatetime = single_game['game_date'] + ' ' + single_game['gametime']
            if single_game['game_location'] == '@':
                print(gamedatetime + winner + loser + '|' + single_game['game_date'] + '|' + winner + '|' + str(winner_score) + '|' + loser + '|' + str(loser_score))
            else:
                print(gamedatetime + loser + winner + '|' + single_game['game_date'] + '|' + loser + '|' + str(loser_score) + '|' + winner + '|' + str(winner_score))

    return inputs, outputs, stat_names_used


def predict_weekly_scores(linear_regression_model, week_num_target):
    future_games_file = open(games_template.replace('yyyy', '2023'))
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

            team1_year_stats = year_stats[2023][team1]
            team2_year_stats = year_stats[2023][team2]

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
            print(week_num + '|' + str(game_to_predict['game_date']) + ' ' + game_to_predict['gametime'] + '|' + team1 + '|' + str(round(team1_pred[0])) + '|' + team2 + '|' + str(round(team2_pred[0])))


year_stats = dict()
x_input = []
y_input = []
stat_names_used = dict()
curr_run_week_num = '4'

for year in range(2018, 2024):
    single_year_stats = dict()

    offense_file = open(off_template.replace('yyyy', str(year)))
    offense_soup = BeautifulSoup(offense_file.read(), 'html.parser')
    single_year_offense = get_off_stats_new(offense_soup)

    defense_file = open(def_template.replace('yyyy', str(year)))
    defense_soup = BeautifulSoup(defense_file.read(), 'html.parser')
    single_year_defense = get_def_stats_new(defense_soup)

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

#for week in range(1, 18):
    #print('Week: ' + str(week))
    #predict_weekly_scores(model, str(week))

predict_weekly_scores(model, curr_run_week_num)

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
