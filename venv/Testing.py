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
                    testPrice = float(row[randomDay + 30])
                    dataPoints = [31, 37, 60, 90, 180, 360]
                    values = [percentage_change(currPrice, testPrice)]
                    values.extend(list (map (lambda n:percentage_change(testPrice, float(row[randomDay + n])), dataPoints)))
                    cases.append(values)
                break

    return cases[:n]

def percentage_change(currentVal, prevVal):
    return ((currentVal / prevVal) - 1) * 100

def get_training_data(n):
    x = get_test_cases(n)
    ys = list(map(growth, x))
    xs = list(map(lambda x:x[1:], x))
    return ys, xs

def growth(l):
    if l[0] > 5:
        return 1
    elif l[0] < -5:
        return -1
    else:
        return 0
