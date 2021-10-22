import json, datetime
from Config import *


def get_positions():
    positionJSON = None

    with open(POSITION_PATH) as f:
        positionJSON = json.load(f)

    positions = []
    for position in positionJSON["Positions"]:
        positions.append(Position(position["Symbol"], position["Amount"], position["Open"], position["Close"]))

    return positions

class Position():
    def write(self):
        positions = None

        with open(POSITION_PATH) as f:
            positions = json.load(f)

        position = {
            "Symbol": self.symbol,
            "Amount": self.amount,
            "Open": self.openDate,
            "Close": self.closeDate
        }

        positions["Positions"].append(position)
        positions["Updated"] = str(datetime.date.today())

        with open(POSITION_PATH, "w") as f:
            json.dump(positions, f, indent=4)

    def __init__(self, symbol, amount, openDate, closeDate):
        self.symbol = symbol
        self.amount = amount
        self.openDate = openDate
        self.closeDate = closeDate

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        else:
            return (self.symbol == other.symbol
                    and self.amount == other.amount
                    and self.openDate == other.openDate
                    and self.closeDate == other.closeDate)

    def close(position):
        positions = get_positions()
        clearJSON()
        for pos in positions:
            if position == pos:
                positions.remove(position)
                break
        for pos in positions:
            pos.write()

def clearJSON():
    with open(POSITION_PATH, "r") as f:
        old = json.load(f)

    old["Positions"] = []

    with open(POSITION_PATH, "w") as f:
        json.dump(old, f)


