"""
TODO: Put in later for basic UI
import tkinter as tk
from tkinter import filedialog
def getExcel():
    global df
    import_file_path = filedialog.askopenfilename()
    df = pd.read_excel(import_file_path)
    print(df)
    print(df.describe())

root = tk.Tk()
canvas1 = tk.Canvas(root, width=300, height=300, bg='lightsteelblue')
canvas1.pack()

browseButton_Excel = tk.Button(text='Import Excel File', command=getExcel, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 150, window=browseButton_Excel)
root.mainloop()
"""

import pandas as pd
import csv
import sys
import os.path
from datetime import datetime


def write_draft_pick(file_name, player_name, drafted_price):
    with open(file_name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        draft_pick_iter = [player_name, drafted_price]
        csvwriter.writerow(draft_pick_iter)


def compute_player_value(player, extra_point_value, extra_ppg_by_pos):
    extra_ppg = player['EXTRA_PPG']
    player_value = (extra_point_value * extra_ppg) + 1
    player['VALUE'] = round(player_value)
    if extra_ppg_by_pos[player['POS']] <= 0:
        pos_rel_value = 0
    else:
        pos_rel_value = extra_ppg / extra_ppg_by_pos[player['POS']]
    player['POS_REL_VALUE'] = '{0:.0f}%'.format(pos_rel_value*100)
    return player


def write_player_vals(players_idx):
    timestr = datetime.now().strftime('%Y%m%d-%H_%M_%S_%f')[:-3]
    filename = 'C:/Users/bobna/OneDrive/Documents/fantasyfootball/output/draft_values_{}.csv'.format(timestr)
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['NAME', 'TEAM', 'POS', 'PPG', 'EXTRA_PPG', 'VALUE', 'POS_REL_VALUE']
        csvwriter.writerow(header)
        for player in players_idx.values():
            name = player['NAME']
            team = player['TEAM']
            pos = player['POS']
            ppg = player['PPG']
            extra_ppg = player['EXTRA_PPG']
            value = player['VALUE']
            pos_rel_value = player['POS_REL_VALUE']
            player_iter = [name, team, pos, ppg, extra_ppg, value, pos_rel_value]
            csvwriter.writerow(player_iter)


def update_all_player_vals(players_idx, extra_point_value, extra_ppg_by_pos):
    for idx in players_idx:
        compute_player_value(players_idx[idx], extra_point_value, extra_ppg_by_pos)
    write_player_vals(players_idx)


def get_extra_point_val(total_extra_dollars, total_extra_ppg):
    extra_point_value = total_extra_dollars / total_extra_ppg
    return extra_point_value


def gather_extra_points(players_idx, baselines):
    total_extra_ppg = 0
    extra_ppg_by_pos = dict()

    for idx in players_idx:
        player = players_idx[idx]
        extra_ppg_col = 'EXTRA_PPG'
        position = player['POS']
        ppg = player['PPG']
        extra_ppg = ppg - baselines[position]
        if extra_ppg > 0:
            player[extra_ppg_col] = extra_ppg
        else:
            player[extra_ppg_col] = 0

        if position in extra_ppg_by_pos:
            extra_ppg_by_pos[position] += player[extra_ppg_col]
        else:
            extra_ppg_by_pos[position] = player[extra_ppg_col]

        total_extra_ppg += player[extra_ppg_col]

    return (total_extra_ppg, extra_ppg_by_pos)


def gather_baselines(players_idx):
    baseline_ppg_by_pos = dict()
    pos_player_counts = dict()
    for idx in players_idx:
        player = players_idx[idx]
        position = player['POS']
        points = player['PPG']
        if position in pos_player_counts:
            pos_player_counts[position] += 1
        else:
            pos_player_counts[position] = 1

        count = pos_player_counts[position]
        is_baseline = False

        if position == 'QB' and count == 20:
            is_baseline = True
        elif position == 'RB' and count == 30:
            is_baseline = True
        elif position == 'WR' and count == 30:
            is_baseline = True
        elif position == 'TE' and count == 10:
            is_baseline = True
        elif position == 'K' and count == 10:
            is_baseline = True
        elif position == 'D/ST' and count == 10:
            is_baseline = True

        if is_baseline:
            baseline_ppg_by_pos[position] = points
    return baseline_ppg_by_pos


def read_players(file_path):
    players_df = pd.read_excel(file_path)
    players_idx = players_df.to_dict('index')
    return players_idx


def replay_picks(file_name, players_idx, baselines, total_extra_dollars, total_extra_ppg):
    print('Replaying any previously occured picks from ' + file_name)
    extra_point_val = get_extra_point_val(total_extra_dollars, total_extra_ppg)
    if os.path.exists(file_name):
        with open(file_name, 'r', newline='') as csvfile:
            draft_pick_reader = csv.reader(csvfile)
            for row in draft_pick_reader:
                pick_name = row[0]
                pick_price = row[1]
                print('Attempting to remove {} for {}'.format(pick_name, pick_price))
                player_removed = remove_drafted_player(players_idx, pick_name)

                if player_removed:
                    print('Successfully removed {} for {}'.format(pick_name, pick_price))
                    (total_extra_ppg, extra_ppg_by_pos) = gather_extra_points(players_idx, baselines)
                    total_extra_dollars -= int(pick_price)
                    extra_point_val = get_extra_point_val(total_extra_dollars, total_extra_ppg)
                    update_all_player_vals(players_idx, extra_point_val, extra_ppg_by_pos)
    return (players_idx, total_extra_dollars, total_extra_ppg, extra_point_val)


def remove_drafted_player(players_idx, player_drafted):
    player_found = False
    for idx in players_idx:
        player = players_idx[idx]
        if player['NAME'] == player_drafted:
            del players_idx[idx]
            player_found = True
            break
    return player_found


def main(file_path):
    print('Initializing data...')
    total_dollars = (10 * 200)
    players_idx = read_players(file_path)
    baselines = gather_baselines(players_idx)
    (total_extra_ppg, extra_ppg_by_pos) = gather_extra_points(players_idx, baselines)
    extra_point_val = get_extra_point_val(total_dollars, total_extra_ppg)
    update_all_player_vals(players_idx, extra_point_val, extra_ppg_by_pos)

    keepers_file = 'C:/Users/bobna/OneDrive/Documents/fantasyfootball/input/ff_2020_keepers.csv'
    draft_picks_file = 'C:/Users/bobna/OneDrive/Documents/fantasyfootball/output/draft_picks.csv'

    (players_idx, total_dollars, total_extra_ppg, extra_point_val) = replay_picks(keepers_file, players_idx, baselines, total_dollars, total_extra_ppg)
    (players_idx, total_dollars, total_extra_ppg, extra_point_val) = replay_picks(draft_picks_file, players_idx, baselines, total_dollars, total_extra_ppg)

    while True:
        if total_dollars <= 0:
            break
        player_drafted = input('Player Drafted: ')
        if player_drafted.lower() == 'quit':
            break
        drafted_price = input('Drafted Price: ')
        if drafted_price.lower() == 'quit':
            break

        player_removed = remove_drafted_player(players_idx, player_drafted)

        if player_removed:
            print('Successfully removed {} for {}'.format(player_drafted, drafted_price))
            (total_extra_ppg, extra_ppg_by_pos) = gather_extra_points(players_idx, baselines)
            total_dollars -= int(drafted_price)
            extra_point_val = get_extra_point_val(total_dollars, total_extra_ppg)
            update_all_player_vals(players_idx, extra_point_val, extra_ppg_by_pos)
            write_draft_pick(draft_picks_file, player_drafted, drafted_price)
        else:
            print('Player not found, try again')

    print('Completed!')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main('C:/Users/bobna/OneDrive/Documents/fantasyfootball/input/ff_2020_players.xlsx')
