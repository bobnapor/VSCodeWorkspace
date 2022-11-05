#!/usr/bin/env python

import csv
import os.path
import sys
from datetime import datetime


def delete_file_if_exists(file_to_delete):
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)


def write_to_file(output_file, sku, title, brand, part_number1, part_number2, price_to_write, note):
    with open(output_file, 'a', newline='', encoding="utf-8") as output_csvfile:
        output_csvwriter = csv.writer(output_csvfile)
        if note is None:
            line_to_write = [sku, title, brand, part_number1, part_number2, price_to_write]
        else:
            line_to_write = [sku, title, brand, part_number1, part_number2, price_to_write, note]
        output_csvwriter.writerow(line_to_write)
    output_csvfile.close()


def collect_sheet2_prices(price_file2):
    sku_price_map = dict()
    fields = dict()
    with open(price_file2, 'r', encoding="utf-8-sig") as price_file2_csv:
        csvreader_price_file2 = csv.reader(price_file2_csv)
        field_num = 0

        for field in next(csvreader_price_file2):
            fields[field] = field_num
            field_num += 1

        for row in csvreader_price_file2:
            marked_up_montgomery_price = float(row[fields.get('MONTGOMERY')]) * 1.2
            mapp_price_str = row[fields.get('MAPP PRICE')]
            mapp_price = 0.0
            if(mapp_price_str != ''):
                mapp_price = float(mapp_price_str)
            sheet2_price = min(marked_up_montgomery_price, mapp_price)
            sku_price_map[str(row[fields.get('SKU')])] = sheet2_price
    return sku_price_map


def collect_sheet3_prices(price_file3):
    sku_price_map = dict()
    fields = dict()
    with open(price_file3, 'r', encoding="unicode_escape") as price_file3_csv:
        csvreader_price_file3 = csv.reader(price_file3_csv)
        field_num = 0

        for field in next(csvreader_price_file3):
            fields[field] = field_num
            field_num += 1

        for row in csvreader_price_file3:
            marked_up_std_pkg_cust_cost = float(row[fields.get('Std Pkg Cust Cost')]) * 1.2
            sku_price_map[str(row[fields.get('Part Number')])] = marked_up_std_pkg_cust_cost
    return sku_price_map


def main(website_items_sheet, sheet2, sheet3, output_dir=os.path.expanduser("~/OneDrive/Desktop/")):
    timestr = datetime.now().strftime('%Y%m%d-%H_%M_%S')
    updates_output_file = "{}updates_{}.csv".format(output_dir, timestr)
    no_updates_output_file = "{}no_updates_{}.csv".format(output_dir, timestr)
    conflicting_updates_file = "{}conflicting_updates_{}.csv".format(output_dir, timestr)

    delete_file_if_exists(updates_output_file)
    delete_file_if_exists(no_updates_output_file)
    delete_file_if_exists(conflicting_updates_file)

    file_2_sku_price_map = collect_sheet2_prices(sheet2)
    file_3_sku_price_map = collect_sheet3_prices(sheet3)

    with open(website_items_sheet, 'r', encoding="utf-8-sig") as csvfile:
        file_1_fields = dict()
        csvreader = csv.reader(csvfile)
        field_num = 0

        for field in next(csvreader):
            file_1_fields[field] = field_num
            field_num += 1

        for row in csvreader:
            sku = str(row[file_1_fields.get('SKU')])
            original_price = float(row[file_1_fields.get('Price')])
            title = str(row[file_1_fields.get('Title')])
            brand = str(row[file_1_fields.get('Product Brand')])
            part_number1 = str(row[3])
            part_number2 = str(row[4])
            sku_sheet2_price_str = file_2_sku_price_map.get(sku)
            sku_sheet3_price_str = file_3_sku_price_map.get(sku)

            sku_sheet2_price = original_price
            sku_sheet3_price = original_price

            if sku_sheet3_price_str is None and sku_sheet2_price_str is None:
                write_to_file(no_updates_output_file, sku, title, brand, part_number1, part_number2, original_price, 'sku not found in sheet2 or sheet3')
            else:
                if sku_sheet2_price_str is not None:
                    sku_sheet2_price = float(sku_sheet2_price_str)

                if sku_sheet3_price_str is not None:
                    sku_sheet3_price = float(sku_sheet3_price_str)

                if sku_sheet2_price == sku_sheet3_price:
                    if sku_sheet3_price == original_price:
                        write_to_file(no_updates_output_file, sku, title, brand, part_number1, part_number2, original_price, 'new price equals original')
                    elif sku_sheet3_price == 0:
                        write_to_file(no_updates_output_file, sku, title, brand, part_number1, part_number2, original_price, 'new price is 0')
                    else:
                        write_to_file(updates_output_file, sku, title, brand, part_number1, part_number2, sku_sheet3_price, '')
                else:
                    if sku_sheet2_price != original_price and sku_sheet3_price != original_price:
                        write_to_file(conflicting_updates_file, sku, title, brand, part_number1, part_number2, sku_sheet2_price, sku_sheet3_price)
                    elif sku_sheet2_price == original_price and sku_sheet3_price != original_price:
                        if sku_sheet3_price == 0:
                            write_to_file(no_updates_output_file, sku, title, brand, part_number1, part_number2, original_price, 'new price is 0')
                        else:
                            write_to_file(updates_output_file, sku, title, brand, part_number1, part_number2, sku_sheet3_price, '')
                    elif sku_sheet2_price != original_price and sku_sheet3_price == original_price:
                        if sku_sheet2_price == 0:
                            write_to_file(no_updates_output_file, sku, title, brand, part_number1, part_number2, original_price, 'new price is 0')
                        else:
                            write_to_file(updates_output_file, sku, title, brand, part_number1, part_number2, sku_sheet2_price, '')


if __name__ == '__main__':
    if(sys.argv[1] == "test"):
        main('C:/Users/Bobby/OneDrive/Desktop/Price Upload Project/Spreadsheet 1 - Website Items.csv', 'C:/Users/Bobby/OneDrive/Desktop/Price Upload Project/Spreadsheet 2 - New Pricing.csv', 'C:/Users/Bobby/OneDrive/Desktop/Price Upload Project/Spreadsheet 3 - ORS Pricing.csv')
    else:
        if len(sys.argv) == 4:
            main(sys.argv[1], sys.argv[2], sys.argv[3])
        elif len(sys.argv) == 5:
            main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            print("Invalid number of input arguments.  Please supply at least the full paths of the 3 product files: Website Items, New Pricing, ORS Pricing")
