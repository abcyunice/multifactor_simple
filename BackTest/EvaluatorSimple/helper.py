'''
change the dict to df
change the args_list to df with isInList

'''

import pandas as pd


def dictToDf(d):
    return pd.DataFrame(d)


def getDf(time_, stock, isInList, start_date, return_rate, factor):
    '''
    get the df for next evaluate
    :param time_: date_list
    :param stock: stock_list
    :param isInList: if stock in HS300, isInList=1 else 0
    :param start_date: the date start to evaluate
    :param y1: return_rate
    :param y2: factor
    :return: df  columns :[date, return_rate, factor,stock,isIn]
    '''
    df = pd.DataFrame({"date": time_, "factor": factor, "return_rate": return_rate, "stock": stock, "isIn": isInList})
    df = df[(df["date"] >= start_date) & (df["isIn"] == 1)]
    df.sort_values(by="date", inplace=True)
    return df
