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
players_file = 'C:/Users/Bobby/Documents/ff_2020_players.xlsx'
players_df = pd.read_excel(players_file)
players_idx = players_df.to_dict('index')

total_points_by_pos = dict()

for idx in players_idx:
    player = players_idx[idx]
    position = player['POS']
    points = player['POINTS']
    if position in total_points_by_pos:
        total_points_by_pos[position] += points
    else:
        total_points_by_pos[position] = points

print(total_points_by_pos)
print('Completed!')
