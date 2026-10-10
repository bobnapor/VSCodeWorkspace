import calendar

#for idk in calendar.month_name:
    #print(idk)
months = {'January':1, 'February':2}
for (k, v) in months.items():
    print('month=', k, 'month_num=', v)
print(months)
print(calendar.month_name[1])
print(calendar.February)