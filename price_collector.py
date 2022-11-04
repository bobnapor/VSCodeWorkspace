from contextlib import AbstractAsyncContextManager
import csv
from re import A

filenames = []
filenames.append('C:/Users/bobna/OneDrive/Desktop/Price Upload Project/Spreadsheet 1 - Website Items.csv')
filenames.append('C:/Users/bobna/OneDrive/Desktop/Price Upload Project/Spreadsheet 2 - New Pricing.csv')
filenames.append('C:/Users/bobna/OneDrive/Desktop/Price Upload Project/Spreadsheet 3 - ORS Pricing.csv')

updates_output_file = 'C:/Users/bobna/OneDrive/Desktop/Price Upload Project/updates.csv'
no_updates_output_file = 'C:/Users/bobna/OneDrive/Desktop/Price Upload Project/no_updates.csv'

#TODO: 1. loop through spreadsheet 1 and collect sku's
#TODO: 2. if sku from sheet 1 exists in sheet 2, collect 'MONTGOMERY' price from sheet 2 and mark up 20% (make variable later) -> take min of marked up and column 'MAPP PRICE'
#TODO: 3. if sku from sheet 1 exists in sheet 3, mark column 'Std Pkg Cust Cost' up 20% and take that
#TODO: see notes doc

file_2_sku_price_map = dict()
file_2_fields = dict()
with open(filenames[1], 'r', encoding="utf-8-sig") as csvfile:
    csvreader = csv.reader(csvfile)
    field_num = 0

    for field in next(csvreader):
        file_2_fields[field] = field_num
        field_num += 1

    file_2_sku_price_map = dict()
    for row in csvreader:
        marked_up_montgomery_price = float(row[file_2_fields.get('MONTGOMERY')]) * 1.2
        mapp_price_str = row[file_2_fields.get('MAPP PRICE')]
        mapp_price = 0.0
        if(mapp_price_str != ''):
            mapp_price = float(mapp_price_str)
        sheet2_price = min(marked_up_montgomery_price, mapp_price)
        file_2_sku_price_map[str(row[file_2_fields.get('SKU')])] = sheet2_price

file_3_sku_price_map = dict()
file_3_fields = dict()
with open(filenames[2], 'r', encoding="utf-8-sig") as csvfile:
    csvreader = csv.reader(csvfile)
    field_num = 0

    for field in next(csvreader):
        file_3_fields[field] = field_num
        field_num += 1

    file_3_sku_price_map = dict()
    for row in csvreader:
        marked_up_std_pkg_cust_cost = float(row[file_3_fields.get('Std Pkg Cust Cost')]) * 1.2
        file_3_sku_price_map[str(row[file_3_fields.get('Part Number')])] = marked_up_std_pkg_cust_cost

file_1_fields = dict()
with open(filenames[0], 'r', encoding="utf-8-sig") as csvfile:
    csvreader = csv.reader(csvfile)
    field_num = 0

    for field in next(csvreader):
        file_1_fields[field] = field_num
        field_num += 1

    for row in csvreader:
        sku = str(row[file_1_fields.get('SKU')])
        price = float(row[file_1_fields.get('Price')])
        sku_sheet2_price_str = file_2_sku_price_map.get(sku)
        sku_sheet3_price_str = file_3_sku_price_map.get(sku)

        sku_sheet2_price = price
        sku_sheet3_price = price

        #TODO need to handle when sku not present in sheet 2 or sheet 3, setting above to price might be hiding some records

        if sku_sheet2_price_str is not None:
            sku_sheet2_price = float(sku_sheet2_price_str)

        if sku_sheet3_price_str is not None:
            sku_sheet3_price = float(sku_sheet3_price_str)

        if sku_sheet2_price == sku_sheet3_price:
            if sku_sheet3_price == price:
                with open(no_updates_output_file, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    no_update_line_to_write = [row[0], row[1], row[2], row[3], row[4], row[5], 'new price equals original']
                    csvwriter.writerow(no_update_line_to_write)
            else:
                with open(updates_output_file, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    update_line_to_write = [row[0], row[1], row[2], row[3], row[4], sku_sheet3_price]
                    csvwriter.writerow(update_line_to_write)
        else:
            if sku_sheet2_price != price and sku_sheet3_price != price:
                print('do something')
            elif sku_sheet2_price == price and sku_sheet3_price != price:
                with open(updates_output_file, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    update_line_to_write = [row[0], row[1], row[2], row[3], row[4], sku_sheet3_price]
                    csvwriter.writerow(update_line_to_write)
            elif sku_sheet2_price != price and sku_sheet3_price == price:
                with open(updates_output_file, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    update_line_to_write = [row[0], row[1], row[2], row[3], row[4], sku_sheet2_price]
                    csvwriter.writerow(update_line_to_write)