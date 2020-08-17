import tkinter as tk
from tkinter import filedialog
import pandas as pd


def getExcel():
    global df
    import_file_path = filedialog.askopenfilename()
    df = pd.read_excel(import_file_path)
    print(df)
    print(df.describe())


#example pandas DataFrame methods:
#sorted_by_gross = movies.sort_values(['Gross Earnings'], ascending=False)
#movies.describe()

root = tk.Tk()
canvas1 = tk.Canvas(root, width=300, height=300, bg='lightsteelblue')
canvas1.pack()

browseButton_Excel = tk.Button(text='Import Excel File', command=getExcel, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 150, window=browseButton_Excel)
root.mainloop()

#read excel using hardcoded path
#players_file = 'C:/Users/Bobby/Documents/ff_2020_players.xlsx'
#df = pd.read_excel(players_file)    #pip install xlrd
#print(df)
