from NN import train_network, nn_structure, ask_network, get_params
from TestingSeq import get_training_data, percentage_change
from Data import updateData
from Config import *

import csv, requests, json, numpy



def get_todays_data():
    dataPoints = [1,2,5,22,44,66,130,260]
    companies = {}
    with open(DATA_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 331:
                for i in range(1):
                    todaysPrice = float(row[i+1])
                    data = list(map(lambda n: percentage_change(todaysPrice, float(row[i+n+1])), dataPoints))
                    symbol = row[0]
                    companies[symbol + str(i)] = data
            else:
                continue

    return companies

try:
    params = get_params()
except:
    print("couldnt find network params, training new network")
    ys, xs = get_training_data()
    params, _ = train_network(ys, xs, nn_structure, 0.01, 10)

companies = get_todays_data()

suggestedBuy = []
for symbol, data in companies.items():
    expected = ask_network(data, params, nn_structure)

    print(expected, symbol)
    if expected > 0.40:
        suggestedBuy.append(symbol)

print(suggestedBuy)

paramJSON = {}
for key, value in params.items():
    paramJSON[key] = value.tolist()

with open("params.json", "w") as f:
    json.dump(paramJSON, f, sort_keys=False, indent=4)

