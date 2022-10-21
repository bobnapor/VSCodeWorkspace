import csv

filenames = []
filenames.append('C:/Users/Bobby/OneDrive/Desktop/Price Upload Project/Spreadsheet 1- Website Items.csv')
filenames.append('C:/Users/Bobby/OneDrive/Desktop/Price Upload Project/Spreadsheet 2 - New Pricing.csv')
filenames.append('C:/Users/Bobby/OneDrive/Desktop/Price Upload Project/Spreadsheet 3- ORS Pricing.csv')

#TODO: 1. loop through spreadsheet 1 and collect sku's
#TODO: 2. if sku from sheet 1 exists in sheet 2, collect montgomery price from sheet 2 and mark up 20% (make variable later) -> take min of marked up and column AS
#TODO: 3. if sku from sheet 1 exists in sheet 3, mark column O up 20% and take that
#TODO: see notes doc

sheet3_sku_col_name = 'Part Number'

fileCount = 0
for filename in filenames:
    fields = dict()
    rows = []
    encoding = "utf8"

    if(fileCount == 2):
        encoding = "latin1"

    with open(filename, 'r', encoding=encoding) as csvfile:
        csvreader = csv.reader(csvfile)
        field_num = 0

        for field in next(csvreader):
            fields[field] = field_num
            field_num += 1

        for row in csvreader:
            #print('SKU = ' + str(row[fields.get('SKU')]))
            rows.append(row)
            #print(row)

        print("Total no. of rows: %d"%(csvreader.line_num))
    
    # printing the field names
    print('Field names are:' + ', '.join(field for field in fields))

    if(fileCount == 2):
        print(sheet3_sku_col_name + ' field number is = ' + str(fields.get(sheet3_sku_col_name)))
    else:
        print('SKU field number is = ' + str(fields.get('SKU')))
    
    # printing first 5 rows
    print('\nFirst 5 rows are:\n')
    #for row in rows[:5]:
        # parsing each column of a row
    #    for col in row:
            #print("%10s"%col,end=" "),
        #print('\n')
    fileCount += 1