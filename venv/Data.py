from Config import *
import requests, json, csv, datetime

Headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}

ACCOUNT_URL = BASE_URL + "/v2/account"
ORDERS_URL = BASE_URL + "/v2/orders"

def get_account():
    r = requests.get(ACCOUNT_URL, headers=Headers)
    return json.loads(r.content)

def create_order(symbol, qty, side="buy", type="market", time_in_force="day"):
    data = {
        "symbol" : symbol,
        "qty": qty,
        "side": side,
        "type": type,
        "time_in_force": time_in_force
    }
    o = requests.post(ORDERS_URL, json=data, headers=Headers)
    return json.loads(o.content)

def get_open(symbol, years):
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
            d.extend(rd)
        else:
            break

        today = today - datetime.timedelta(days = (261 * 3) + 1)
           
    return map(lambda d:d["o"], d)

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

            row = [symbol] + list(map(lambda s:float(s), list(get_open(symbol, 20))[::-1]))

            if len(row) > 362:
                writer.writerow(row)

if __name__ == '__main__':
    updateData()
