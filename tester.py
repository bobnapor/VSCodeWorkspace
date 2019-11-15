import random


def populate_file(file_name):
    output = open(file_name, "w+")
    for team_num in range(1, 33, 1):
        qbr = random.randint(0, 100)
        text_team_num = str(team_num)
        output.write('team' + text_team_num + ',qb' + text_team_num + ',' + str(qbr) + '\n')


def read_file(file_name):
    file_to_read = open(file_name, "r")
    line = file_to_read.read() ##read just reads in whole file, readline does one line at a time, what does readlines do?
    print(line)
    #split_line = l.split(',')


def fib(n):
    if(n < 2):
        return n
    return fib(n-1) + fib(n-2)


print(fib(1))

#file_name = "stats.txt"
#populate_file(file_name)
#read_file(file_name)
