#import tkinter as tk
#from tkinter import filedialog

"""
To put in later for UI work
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

#read excel using hardcoded path
import pandas as pd
import csv

players_file = 'C:/Users/Bobby/Documents/ff_2020_players.xlsx'
players_df = pd.read_excel(players_file)
players_idx = players_df.to_dict('index')

total_ppg_by_pos = dict()
baseline_ppg_by_pos = dict()
pos_player_counts = dict()

for idx in players_idx:
    player = players_idx[idx]
    position = player['POS']
    points = player['PPG']
    if position in total_ppg_by_pos:
        total_ppg_by_pos[position] += points
    else:
        total_ppg_by_pos[position] = points

    if position in pos_player_counts:
        pos_player_counts[position] += 1
    else:
        pos_player_counts[position] = 1

    count = pos_player_counts[position]
    is_baseline = False

    if position == 'QB' and count == 30:
        is_baseline = True
    elif position == 'RB' and count == 50:
        is_baseline = True
    elif position == 'WR' and count == 50:
        is_baseline = True
    elif position == 'TE' and count == 10:
        is_baseline = True
    elif position == 'K' and count == 10:
        is_baseline = True
    elif position == 'D/ST' and count == 10:
        is_baseline = True

    if is_baseline:
        baseline_ppg_by_pos[position] = points

total_extra_ppg_by_pos = dict()
total_extra_ppg = 0
pos_player_counts.clear()

for idx in players_idx:
    player = players_idx[idx]
    position = player['POS']
    ppg = player['PPG']
    extra_ppg = ppg - baseline_ppg_by_pos[position]
    if extra_ppg > 0:
        player['EXTRA_PPG'] = extra_ppg
    else:
        player['EXTRA_PPG'] = 0
    total_extra_ppg += player['EXTRA_PPG']
    if position in total_extra_ppg_by_pos:
        total_extra_ppg_by_pos[position] += player['EXTRA_PPG']
    else:
        total_extra_ppg_by_pos[position] = player['EXTRA_PPG']


total_extra_dollars = (10 * 200) - (10 * 16)
extra_point_value = total_extra_dollars / total_extra_ppg

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


print(players_idx)
print('Completed!')
