from NN import train_network, nn_structure, ask_network, get_params, write_params
from TestingSeq import get_training_data, percentage_change
from Data import updateData, get_todays_data
from Config import *
from Position import Position, get_positions
import Alpaca

import csv, requests, json, numpy, datetime

def get_suggestions():
    try:
        params = get_params()
    except:
        print("couldnt find network params, training new network")
        ys, xs = get_training_data()
        params, _ = train_network(ys, xs, nn_structure, 0.001, 10)
        write_params(params)

    companies = get_todays_data()

    suggestedBuy = []
    for symbol, data in companies.items():
        expected = ask_network(data["Data"], params, nn_structure)

        if expected > 0.70:
            suggestedBuy.append({"Symbol": symbol, "Price": data["Price"]})

    return suggestedBuy

def open_position(symbol, amount, limitPrice):
    today = datetime.date.today()
    Alpaca.open_position(symbol, amount, limitPrice)
    position = Position(symbol, amount, str(today), str(today + datetime.timedelta(days=30)))
    position.write()

def close_position(position):
    Alpaca.close_position(position.symbol, position.amount)
    position.close()

def update_positions():
    currPositions = get_positions()
    for pos in currPositions:
        if datetime.datetime.strptime(pos.closeDate, "%Y-%m-%d") < datetime.datetime.today():
            print(f"Selling {pos.amount} stocks of {pos.symbol}")
            close_position(pos)
    newStocks = get_suggestions()

    for suggestion in newStocks:
        qty = max(1, 1000 // suggestion["Price"])
        print(f'Buying {qty} stocks of {suggestion["Symbol"]}')
        open_position(suggestion["Symbol"], qty, suggestion["Price"]*1.01)

if __name__ == '__main__':
    update_positions()
