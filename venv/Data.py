from Config import *
from TestingSeq import percentage_change
import json, csv, datetime, requests

def get_close(symbol, years):
    BARS_URL = DATA_URL + "/v2/stocks/"+symbol+"/bars"

    today = datetime.date.today()

    d = []

    for i in range(years // 3):
        yesterday = today - datetime.timedelta(days = 1)
        endOfData = today - datetime.timedelta(days = 261 * 3)

        data = {
            "start" : str(endOfData),
            "end" : str(yesterday),
            "timeframe" : "1Day"
        }
        
        r = requests.get(BARS_URL, params = data, headers = Headers)
        rd = json.loads(r.content)["bars"]

        if rd:
            rd.reverse()
            d.extend(rd)
        else:
            break

        today = today - datetime.timedelta(days = (261 * 3))
    if not d:
        return []

    returnData = list(map(lambda d:d["c"], d))
    return returnData


def nameSort(x):
    return x["Name"]


def combine(*indexes):
    allStocks = []
    for index in indexes:
        for stock in index:
            alreadyExists = False
            for currStock in allStocks:
                if stock["Symbol"] == currStock["Symbol"]:
                    alreadyExists = True
                    break
            if not alreadyExists:
                allStocks.append({"Symbol": stock["Symbol"], "Name": stock["Name"]})
    allStocks.sort(key=nameSort)
    return allStocks


def updateData():
    allCompanies = combine(SP500json, Nasdaq100json)

    with open(DATA_PATH, "w", newline="") as f:
        writer = csv.writer(f)

        for company in allCompanies:
            print(company)
            symbol = company["Symbol"]

            row = [symbol] + list(map(lambda s:float(s), get_close(symbol, 20)))

            if len(row) > 22 + DATA_POINTS[-1]:
                writer.writerow(row)

def get_todays_data(shift = 0):
    companies = {}
    with open(DATA_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > DATA_POINTS[-1] + 1 + shift:
                todaysPrice = float(row[1 + shift])
                data = list(map(lambda n: percentage_change(todaysPrice, float(row[1+shift+n])), DATA_POINTS))
                symbol = row[0]
                companies[symbol] = {"Data": data, "Price":todaysPrice}
            else:
                continue

    return companies


if __name__ == '__main__':
    updateData()
