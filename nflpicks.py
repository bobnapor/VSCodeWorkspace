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

year_stats = dict(dict(dict()))
column_names = []

for year in range(2009, 2020):
    local_off_url = file_dir + local_off_template.replace('yyyy', str(year))
    #local_def_url = file_dir + local_def_template.replace('yyyy', str(year))
    #local_games_url = file_dir + local_games_template.replace('yyyy', str(year))

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
                    column_name = off_stat_column['data-stat']
                    single_team_stats[column_name] = off_stat_column.text
                team = single_team_stats['team']
                single_year_stats[team] = single_team_stats
                print('Extracted data for the ' + str(year) + ' ' + team)

    year_stats[year] = single_year_stats

print(year_stats)

print('Completed!')
