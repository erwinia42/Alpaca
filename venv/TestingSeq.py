import csv, random, math, time
from Config import *

def get_test_cases():
    cases = []

    with open(DATA_PATH, "r") as csvFile:
        reader = csv.reader(csvFile)

        for row in reader:
            if len(row) < 362:
                continue
                print("ERROR, DATA TOO SHORT FOR SYMBOL", row[0])
            else:
                startDay = 1
                while(startDay + 22 + DATA_POINTS[-1] < len(row)):

                    targetPrice = float(row[startDay])
                    testPrice = float(row[startDay + 22])

                    dataPoints = [point + 22 for point in DATA_POINTS]

                    values = [percentage_change(targetPrice, testPrice)]
                    values.extend(list(map(lambda n: percentage_change(testPrice, float(row[startDay + n])), dataPoints)))

                    cases.append(values)
                    startDay += 1


    return cases

def percentage_change(currentVal, prevVal):
    return ((currentVal / prevVal) - 1) * 100

def get_training_data():
    x = get_test_cases()
    ys = list(map(growth, x))
    xs = list(map(lambda x:x[1:], x))
    ys, xs = shuffle(ys, xs)
    return ys, xs

def growth(l):
    if l[0] > 3:
        return 1
    elif l[0] < -3:
        return -1
    else:
        return 0

def get_test_train_data(p):
    ys, xs = get_training_data()
    testys, testxs = [], []

    for i in range(len(xs) // p):
        randomIndex = random.randint(0, len(xs) - 1)
        testys.append(ys.pop(randomIndex))
        testxs.append(xs.pop(randomIndex))

    return ys, xs, testys, testxs

def shuffle(l1, l2):

    randomList = [random.random() for i in range(len(l1))]

    l1, l2 = list(zip(randomList, l1)), list(zip(randomList, l2))
    l1.sort()
    l2.sort()

    return list(map(snd, l1)), list(map(snd, l2))

def snd(a):
    _, a = a
    return a
