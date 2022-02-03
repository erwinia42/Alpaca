from Config import *
from Data import get_todays_data
from Trading import get_suggestions
import csv, random

def getSimSuggestions(shift, data):
    companies = get_todays_data(shift=shift)
    suggestions = get_suggestions(companies)
        
    return suggestions
    
    
def simulateDay(shift, data):
    suggestions = getSimSuggestions(shift + 22, data)

    expectedReturn = 0
    invested = 0

    for company in suggestions:
        symbol = company["Symbol"]

        if symbol in data:
            sellPrice = data[symbol][shift]
            buyPrice = company["Price"] * 1.01
            conf = company["Confidence"]
            noOfStocksBought = max(0, (3 ** ((conf - 0.6) * 10)) * (conf > 0.6) / buyPrice)

            expectedReturn += (sellPrice - buyPrice) * noOfStocksBought
            invested += buyPrice * noOfStocksBought

    return expectedReturn, invested
    

def get_testData(n):
    data = {}
    with open(DATA_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > n:
                data[row[0]] = list(map(float, row[1:]))
    return data

def sumdict(dict):
    sum = 0
    for _, value in dict.items():
        sum += value
    return sum

if __name__ == '__main__':
    years = 5
    n = 261
    #Start + n maximum value of
    d = get_testData(n)

    for year in range(years - 1, -1, -1):
        avgInvests = []
        expectedReturn = 0
        investTracker = {str(i + 1): 0 for i in range(30)}
        for i in range(year * n, (year+1) * n):
            dayReturn, invested = simulateDay(i, d)
            investTracker[str((i % 30) + 1)] = invested
            if dayReturn:
                expectedReturn += dayReturn
            if i > 30:
                avgInvests.append(sumdict(investTracker))

        avgInvest = sum(avgInvests) / len(avgInvests)
        maxInvest = max(avgInvests)

        print("Year:", year + 1, avgInvest, maxInvest, f"{round(100 * (expectedReturn / avgInvest), 1)}%")
