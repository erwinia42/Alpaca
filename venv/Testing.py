import csv, random, math, time
from Config import *

def get_test_cases(n):
    cases = []
    rootn = math.ceil(math.sqrt(n))

    for _ in range(rootn):
        with open(DATA_PATH, "r") as csvFile:
            reader = csv.reader(csvFile)
            noOfStocks = 497

            randomStock = random.randint(0, noOfStocks-1)
            for _ in range(randomStock):
                next(reader)

            for row in reader:
                for i in range(rootn):
                    if len(row) > 396:
                        randomDay = random.randint(1, len(row) - 396)
                    else:
                        print("FAILED")
                        break
                    currPrice = float(row[randomDay])
                    values = list (map (lambda d:float(d)/currPrice, row[randomDay+30:randomDay+394:7]))
                    cases.append(values)
                break

    return cases[:n]

def get_training_data(n):
    x = get_test_cases(n)
    ys = list(map(growth, x))
    xs = x
    return ys, xs

def growth(l):
    if 1 > l[0]:
        return 1
    else:
        return 0
