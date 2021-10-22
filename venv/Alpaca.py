import requests, json
from Config import *

Headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}

ACCOUNT_URL = BASE_URL + "/v2/account"
ORDERS_URL = BASE_URL + "/v2/orders"
POSITION_URL = BASE_URL + "/v2/positions"

def get_account():
    r = requests.get(ACCOUNT_URL, headers=Headers)
    return json.loads(r.content)

def get_positions():
    r = requests.get(POSITION_URL, headers=Headers)
    return json.loads(r.content)

def open_position(symbol, stocks, limitPrice):
    bodyParams = {
        "symbol": symbol,
        "qty": str(stocks),
        "side": "buy",
        "type": "limit",
        "time_in_force": "day",
        "limit_price": str(limitPrice)
    }
    r = requests.post(ORDERS_URL, json=bodyParams, headers=Headers)
    return json.loads(r.content)

def close_position(symbol, stocks):
    bodyParams = {
        "symbol": symbol,
        "qty": str(stocks),
        "side": "sell",
        "type": "market",
        "time_in_force": "gtc"
    }
    r = requests.post(ORDERS_URL, json=bodyParams, headers=Headers)
    return json.loads(r.content)

