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

[k,v]:
off and def = [year, [team, [stat1, stat2, stat3,...]]]

year_stats = dict(dict(dict()))
column_names = []

for year in range(2009, 2020):
    local_off_url = file_dir + local_off_template.replace('yyyy', str(year))
    #local_def_url = file_dir + local_def_template.replace('yyyy', str(year))
    #local_games_url = file_dir + local_games_template.replace('yyyy', str(year))

    local_off_file = open(local_off_url)
    local_off_soup = BeautifulSoup(local_off_file.read(), 'html.parser')
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        off_stats_soup = BeautifulSoup(comment.string, 'html.parser')
        off_stats_table = off_stats_soup.find_all('table', id='team_stats')[0]
        off_stats_rows = off_stats_table.find_all('tbody')[0].find_all('tr')
        num_teams = 0
        for off_stats_row in off_stats_rows:
            for off_stat_column in off_stats_row.find_all('td'):
                year_stats[year][team][column_name] = off_stat_column.text #TODO: must do something like the below i think
                #column_name = value_column['data-stat']
                #    if column_name not in year_stats[]:
                #        year_stats[column_name] = [value_column.text]
                #        column_names.append(column_name)
                #    else:
                #        year_stats[column_name].append(value_column.text)
                print('Extracted data for the ' + year + ' ' + one_year_stats['team'][num_teams])
                num_teams += 1


    #local_def_file = open(local_def_url)
    #local_def_soup = BeautifulSoup(local_def_file.read(), 'html.parser')

    #local_games_file = open(local_games_url)
    #local_games_soup = BeautifulSoup(local_games_file.read(), 'html.parser')
    #games_table = local_games_soup.find_all('table', id='games')

    print(one_year_stats)



#i need two arrs for games 1-n
    #x_input = [input_stats_1, input_stats_2, ... , input_stats_n]
    #y_input = [points_1, points_2, ... , points_n]

#TODO: indent section below to run once per year
#TODO: need to collect aggregated stats per year per team, as well as values for defenses they played each game and their score for that game
#x inputs will be year stats of team + def stats of opp team per game
#y inputs will be points scored by that team in that game
#comments = soup.find_all(string=lambda text: isinstance(text, Comment))
#for comment in comments:
#    table_soup = BeautifulSoup(comment.string, 'html.parser')
#    tables = table_soup.find_all('table', id='team_stats')
#    if len(tables) > 0:
#        team_stats_table = tables[0]
#
#        one_year_stats = dict()
#        column_names = []
#
#        tbodies = team_stats_table.find_all('tbody')
#        if len(tbodies) > 0:
#            tbody = tbodies[0]
#            body_rows = tbody.find_all('tr')
#            if len(body_rows) > 0:
#                num_teams = 0
#                for body_row in body_rows:
#                    #print(body_row.prettify())
#                    value_columns = body_row.find_all('td')
#                    for value_column in value_columns:
#                        column_name = value_column['data-stat']
#                        if column_name not in one_year_stats:
#                            one_year_stats[column_name] = [value_column.text]
#                            column_names.append(column_name)
#                        else:
#                            one_year_stats[column_name].append(value_column.text)
#                    print('Extracted data for ' + one_year_stats['team'][num_teams])
#                    num_teams += 1


print('Completed!')
