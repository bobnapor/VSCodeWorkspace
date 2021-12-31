import numpy as np
import requests
from bs4 import BeautifulSoup
from bs4 import Comment
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta

schedule_template = 'https://fantasy.espn.com/football/league/schedule?leagueId=175990&seasonId=yyyy'
draft_template = 'https://fantasy.espn.com/football/league/draftrecap?leagueId=175990&seasonId=yyyy'
standings_template = 'https://fantasy.espn.com/football/league/standings?leagueId=175990&seasonId=yyyy'

#TODO: for use when online - need to expand on this for the other pages
#for year in range(2009,2020):
#    year_url = url_template.replace('yyyy', str(year))
#    year_req = requests.get(year_url)
#    print(year_req.status_code)
#    year_soup = BeautifulSoup(year_req.content, 'html.parser')
#    print(year_soup.prettify())


dict_example = dict()
arr_example = []

for year in range(2009, 2020):
    schedule_url = schedule_template.replace('yyyy', str(year))
    schedule_request = requests.get(schedule_url)
    schedule_soup = BeautifulSoup(schedule_request.content, 'html.parser')
    print(schedule_soup.prettify())


print('Completed!')