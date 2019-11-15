import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from bs4 import Comment
from sklearn.linear_model import LinearRegression


#TODO:also need defense - not included in the below page
url_template = 'https://www.pro-football-reference.com/years/yyyy/'
#def_url = 'https://www.pro-football-reference.com/years/2009/opp.htm'
#games_url = 'https://www.pro-football-reference.com/years/2019/games.htm'

local_url = 'C:/Users/bobna/Downloads/2019_nfl_stats.html'  #for testing
page = open(local_url)                                      #for testing
soup = BeautifulSoup(page.read(), 'html.parser')            #for testing

for year in range(2009,2020):
    year_url = url_template.replace('yyyy', str(year))
    year_req = requests.get(year_url)
    print(year_req.status_code)
    year_soup = BeautifulSoup(year_req.content, 'html.parser')
    print(year_soup.prettify())

#TODO: indent section below to run once per year
#TODO: need to collect aggregated stats per year per team, as well as values for defenses they played each game and their score for that game
#x inputs will be year stats of team + def stats of opp team per game
#y inputs will be points scored by that team in that game
comments = soup.find_all(string=lambda text: isinstance(text, Comment))
for comment in comments:
    table_soup = BeautifulSoup(comment.string, 'html.parser')
    tables = table_soup.find_all('table', id='team_stats')
    if len(tables) > 0:
        team_stats_table = tables[0]

        one_year_stats = dict()
        column_names = []

        tbodies = team_stats_table.find_all('tbody')
        if len(tbodies) > 0:
            tbody = tbodies[0]
            body_rows = tbody.find_all('tr')
            if len(body_rows) > 0:
                num_teams = 0
                for body_row in body_rows:
                    #print(body_row.prettify())
                    value_columns = body_row.find_all('td')
                    for value_column in value_columns:
                        column_name = value_column['data-stat']
                        if column_name not in one_year_stats:
                            one_year_stats[column_name] = [value_column.text]
                            column_names.append(column_name)
                        else:
                            one_year_stats[column_name].append(value_column.text)
                    print('Extracted data for ' + one_year_stats['team'][num_teams])
                    num_teams+=1


print('Completed!')