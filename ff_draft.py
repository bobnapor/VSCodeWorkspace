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


def write_draft_pick(player_name, drafted_price):
    filename = 'C:/Users/Bobby/Documents/draft_picks.csv'
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        draft_pick_iter = [player_name, drafted_price]
        csvwriter.writerow(draft_pick_iter)


def write_player_values(players_idx, extra_point_value):
    filename = 'C:/Users/Bobby/Documents/player_values.csv'
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for idx in players_idx:
            player = players_idx[idx]
            extra_ppg = player['EXTRA_PPG']
            player_value = (extra_point_value * extra_ppg) + 1
            player['VALUE'] = player_value
            player_value_iter = [player['NAME'], player_value]
            csvwriter.writerow(player_value_iter)


def get_extra_point_val(total_extra_dollars, total_extra_ppg):
    extra_point_value = total_extra_dollars / total_extra_ppg
    return extra_point_value


def gather_extra_points(players_idx, baselines):
    total_extra_ppg = 0

    for idx in players_idx:
        player = players_idx[idx]
        position = player['POS']
        ppg = player['PPG']
        extra_ppg = ppg - baselines[position]
        if extra_ppg > 0:
            player['EXTRA_PPG'] = extra_ppg
        else:
            player['EXTRA_PPG'] = 0
        total_extra_ppg += player['EXTRA_PPG']

    return total_extra_ppg


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


def replay_picks(players_idx, baselines, total_extra_dollars, total_extra_ppg):
    print('Replaying any previously occured picks...')
    file_path = 'C:/Users/Bobby/Documents/draft_picks.csv'
    extra_point_val = get_extra_point_val(total_extra_dollars, total_extra_ppg)
    if os.path.exists(file_path):
        with open(file_path, 'r', newline='') as csvfile:
            draft_pick_reader = csv.reader(csvfile)
            for row in draft_pick_reader:
                pick_name = row[0]
                pick_price = row[1]

                player_removed = remove_drafted_player(players_idx, pick_name)

                if player_removed:
                    total_extra_ppg = gather_extra_points(players_idx, baselines)
                    total_extra_dollars -= int(pick_price)
                    extra_point_val = get_extra_point_val(total_extra_dollars, total_extra_ppg)
                    write_player_values(players_idx, extra_point_val)
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
    total_extra_dollars = (10 * 200) - (10 * 16)
    players_idx = read_players(file_path)
    baselines = gather_baselines(players_idx)
    total_extra_ppg = gather_extra_points(players_idx, baselines)
    extra_point_val = get_extra_point_val(total_extra_dollars, total_extra_ppg)
    write_player_values(players_idx, extra_point_val)

    (players_idx, total_extra_dollars, total_extra_ppg, extra_point_val) = replay_picks(players_idx, baselines, total_extra_dollars, total_extra_ppg)

    while True:
        player_drafted = input('Player Drafted: ')
        if player_drafted.lower() == 'quit':
            break
        drafted_price = input('Drafted Price: ')
        if drafted_price.lower() == 'quit':
            break

        player_removed = remove_drafted_player(players_idx, player_drafted)

        if player_removed:
            total_extra_ppg = gather_extra_points(players_idx, baselines)
            total_extra_dollars -= int(drafted_price)
            extra_point_val = get_extra_point_val(total_extra_dollars, total_extra_ppg)
            write_player_values(players_idx, extra_point_val)
            write_draft_pick(player_drafted, drafted_price)
        else:
            print('Player not found, try again')

    print('Completed!')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main('C:/Users/Bobby/Documents/ff_2020_players.xlsx')
