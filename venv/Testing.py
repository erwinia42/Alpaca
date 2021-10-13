import csv, random, math, time
from Config import *

def get_test_cases(n):
    cases = []
    rootn = math.ceil(math.sqrt(n))

    for _ in range(rootn):
        with open(DATA_PATH, "r") as csvFile:
            reader = csv.reader(csvFile)
            noOfStocks = 501

            randomStock = random.randint(0, noOfStocks-1)
            for _ in range(randomStock):
                next(reader)

            for row in reader:
                for i in range(rootn):
                    if len(row) > 366:
                        randomDay = random.randint(1, len(row) - 366)
                    else:
                        print("FAILED")
                        break
                    cases.append([row[randomDay], row[randomDay + 1], row[randomDay + 7], row[randomDay+30], row[randomDay+90], row[randomDay+365]])
                break

    return cases[:n]

x = get_test_cases(10)
ys = map (lambda l:l[0], x)
xs = map (lambda l:l[1:], x)
print(list(ys), list(xs))