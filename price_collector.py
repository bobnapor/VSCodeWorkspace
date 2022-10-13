import csv

filename = 'C:/Users/Bobby/OneDrive/Desktop/Price Upload Project/Spreadsheet 1- Website Items.csv'

#TODO: 1. loop through spreadsheet 1 and collect sku's
#TODO: 2. if sku from sheet 1 exists in sheet 2, collect montgomery price from sheet 2 and mark up 20% (make variable later) -> take min of marked up and column AS
#TODO: 3. if sku from sheet 1 exists in sheet 3, mark column O up 20% and take that
#TODO: see notes doc

fields = []
rows = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)
    
    print("Total no. of rows: %d"%(csvreader.line_num))
 
# printing the field names
print('Field names are:' + ', '.join(field for field in fields))
 
# printing first 5 rows
print('\nFirst 5 rows are:\n')
for row in rows[:5]:
    # parsing each column of a row
    for col in row:
        print("%10s"%col,end=" "),
    print('\n')