#import pandas as pd
import numpy as np
#import requests
from bs4 import BeautifulSoup
from bs4 import Comment
from sklearn.linear_model import LinearRegression


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

#for testing
#local_url = 'C:/Users/bobna/Downloads/2019_nfl_stats.html'  #for testing
#page = open(local_url)                                      #for testing
#soup = BeautifulSoup(page.read(), 'html.parser')            #for testing


file_dir = 'C:/Users/bobna/Downloads/NFL_Stats/'
local_games_template = 'yyyy_weekly_schedule.html'
local_def_template = 'yyyy NFL Opposition & Defensive Statistics _ Pro-Football-Reference.com.html'
local_off_template = 'yyyy NFL Standings & Team Stats _ Pro-Football-Reference.com.html'

year_stats = dict()
#column_names = []

x_input = []
y_input = []

off_stat_names_to_use = {
    'points',
    'yds_per_play_offense',
    'fumbles_lost',
    'first_down',
    'pass_yds',
    'pass_int',
    'pass_net_yds_per_att',
    'rush_yds',
    'rush_yds_per_att',
    'penalties',
    'score_pct',
    'turnover_pct'
}

def_stat_names_to_use = {
    'def_points',
    'def_yds_per_play_offense',
    'def_fumbles_lost',
    'def_first_down',
    'def_pass_yds',
    'def_pass_int',
    'def_pass_net_yds_per_att',
    'def_rush_yds',
    'def_rush_yds_per_att',
    'def_penalties',
    'def_score_pct',
    'def_turnover_pct'
}

for year in range(2009, 2020):
    local_off_url = file_dir + local_off_template.replace('yyyy', str(year))
    local_def_url = file_dir + local_def_template.replace('yyyy', str(year))
    local_games_url = file_dir + local_games_template.replace('yyyy', str(year))

    single_year_stats = dict()
    local_off_file = open(local_off_url)
    local_off_soup = BeautifulSoup(local_off_file.read(), 'html.parser')
    for comment in local_off_soup.find_all(string=lambda text: isinstance(text, Comment)):
        off_stats_soup = BeautifulSoup(comment.string, 'html.parser')
        off_stats_tables = off_stats_soup.find_all('table', id='team_stats')
        if len(off_stats_tables) < 1:
            continue
        else:
            off_stats_table = off_stats_tables[0]
            off_stats_rows = off_stats_table.find_all('tbody')[0].find_all('tr')
            for off_stats_row in off_stats_rows:
                single_team_stats = dict()
                for off_stat_column in off_stats_row.find_all('td'):
                    #TODO: if column name not present in dictionary yet, place it in column_names
                    column_name = off_stat_column['data-stat']
                    single_team_stats[column_name] = off_stat_column.text
                team = single_team_stats['team']
                single_year_stats[team] = single_team_stats
                print('Extracted data for the ' + str(year) + ' ' + team)

    local_def_file = open(local_def_url)
    local_def_soup = BeautifulSoup(local_def_file.read(), 'html.parser')
    def_stats_table = local_def_soup.find_all('table', id='team_stats')[0]
    for def_stats_row in def_stats_table.find_all('tbody')[0].find_all('tr'):
        single_team_stats_def = dict()
        for def_stat_column in def_stats_row.find_all('td'):
            column_name = 'def_' + def_stat_column['data-stat']
            single_team_stats_def[column_name] = def_stat_column.text
        team = single_team_stats_def['def_team']
        for def_stat_key in single_team_stats_def:
            single_year_stats[team][def_stat_key] = single_team_stats_def[def_stat_key]

    local_games_file = open(local_games_url)
    local_games_soup = BeautifulSoup(local_games_file.read(), 'html.parser')
    games_table = local_games_soup.find_all('table', id='games')[0]
    game_counter = 0
    for game_row in games_table.find_all('tbody')[0].find_all('tr'):
        if game_counter == 256:
            continue

        winner_off_inputs = []
        winner_def_inputs = []
        loser_off_inputs = []
        loser_def_inputs = []
        single_game = dict()

        game_stat_columns = game_row.find_all('td')

        if len(game_stat_columns) < 1:
            continue

        game_counter += 1

        for game_stat_column in game_stat_columns:
            game_column_name = game_stat_column['data-stat']
            single_game[game_column_name] = game_stat_column.text

        if single_game['boxscore_word'] == 'preview':
            continue

        winner = single_game['winner']
        winner_stats = single_year_stats[winner]
        for (stat_name, stat_value) in winner_stats.items():
            checkable_stat = stat_value.replace('-','').replace(' ', '').replace(',','').replace('.','')
            if checkable_stat.isdigit() and stat_name in off_stat_names_to_use:
                winner_off_inputs.append(float(stat_value)/float(winner_stats['g']))
            if checkable_stat.isdigit() and stat_name in def_stat_names_to_use:
                winner_def_inputs.append(float(stat_value)/float(winner_stats['g']))

        loser = single_game['loser']
        loser_stats = single_year_stats[loser]
        for (stat_name, stat_value) in loser_stats.items():
            checkable_stat = stat_value.replace('-','').replace(' ', '').replace(',','').replace('.','')
            if checkable_stat.isdigit() and stat_name in off_stat_names_to_use:
                loser_off_inputs.append(float(stat_value)/float(loser_stats['g']))
            if checkable_stat.isdigit() and stat_name in def_stat_names_to_use:
                loser_def_inputs.append(float(stat_value)/float(loser_stats['g']))

        winner_inputs = []
        loser_inputs = []
        winner_inputs.extend(winner_off_inputs)
        winner_inputs.extend(loser_def_inputs)
        loser_inputs.extend(loser_off_inputs)
        loser_inputs.extend(winner_def_inputs)

        winner_score = int(single_game['pts_win'])
        loser_score = int(single_game['pts_lose'])
        x_input.append(winner_inputs)
        y_input.append(winner_score)
        #y_input.append(winner_score-loser_score)
        x_input.append(loser_inputs)
        y_input.append(loser_score)

    year_stats[year] = single_year_stats

x, y = np.array(x_input), np.array(y_input)
model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

future_games_url = file_dir + local_games_template.replace('yyyy', '2019')
future_games_file = open(future_games_url)
future_games_soup = BeautifulSoup(future_games_file.read(), 'html.parser')
games_table = local_games_soup.find_all('table', id='games')[0]
game_counter = 0
for game_row in games_table.find_all('tbody')[0].find_all('tr'):

    week_num = game_row.find_all('th')[0].text

    if week_num != '12':
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
            single_game[game_column_name] = game_stat_column.text

        team1 = single_game['winner']
        team2 = single_game['loser']
        team1_year_stats = year_stats[2019][team1]
        team2_year_stats = year_stats[2019][team2]

        for (stat_name, stat_value) in team1_year_stats.items():
            checkable_stat = stat_value.replace('-','').replace(' ', '').replace(',','').replace('.','')
            if checkable_stat.isdigit() and stat_name in off_stat_names_to_use:
                team1_off_inputs.append(float(stat_value)/float(team1_year_stats['g']))
            if checkable_stat.isdigit() and stat_name in def_stat_names_to_use:
                team1_def_inputs.append(float(stat_value)/float(team1_year_stats['g']))

        for (stat_name, stat_value) in team2_year_stats.items():
            checkable_stat = stat_value.replace('-','').replace(' ', '').replace(',','').replace('.','')
            if checkable_stat.isdigit() and stat_name in off_stat_names_to_use:
                team2_off_inputs.append(float(stat_value)/float(team2_year_stats['g']))
            if checkable_stat.isdigit() and stat_name in def_stat_names_to_use:
                team2_def_inputs.append(float(stat_value)/float(team2_year_stats['g']))

        team1_off_inputs.extend(team2_def_inputs)
        team2_off_inputs.extend(team1_def_inputs)
        team1_inputs = []
        team2_inputs = []
        team1_inputs.append(team1_off_inputs)
        team2_inputs.append(team2_off_inputs)

        team1_pred = model.predict(team1_inputs)
        team2_pred = model.predict(team2_inputs)
        print('team:', team1, '; predicted score:', team1_pred)
        print('team:', team2, '; predicted score:', team2_pred)
        print('\n')


print('Completed!')
