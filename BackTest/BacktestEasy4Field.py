import pandas as pd
import numpy as np


def helper(df):
    df["field_mean"] = df.groupby(["field", "date"]).mean()
    return df


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


def meanIc(df, half_life=252):
    '''

    :param df: dataframe [date, return_rate, factor]
    :param half_life:
    :return: meanic
    '''
    ic = df.groupby("date").apply(lambda X: np.corrcoef(X["return_rate"].values, X["factor"].values)[0, 1])
    ic.sort_index(inplace=True)
    x = (1 / 2) ** (1 / half_life)
    df_index = sorted(df.index.unique())
    weight_list = [x ** (i - 1) for i in range(len(df_index))]
    return abs(np.sum(ic * weight_list) / np.sum(weight_list))


def getLongShortEasy(df, r=0.1):
    '''

    :param df: dataframe [date, return_rate, factor]
    :param r: ratio of  long / short
    :return: df dataframe [date, rlong, rshort]
            rshort: mean return rate of the short
    '''
    df["factor_rank"] = df.groupby("date")["factor"].apply(lambda x: x.rank() / len(x))

    df.loc[df["factor_rank"] > 1 - r, "rlong"] = df.loc[df["factor_rank"] > 1 - r, "return_rate"]
    df.loc[df["factor_rank"] < r, "rshort"] = df.loc[df["factor_rank"] < r, "return_rate"]

    df_r = df.groupby("date")[["rlong", "rshort"]].mean()
    return df_r


def getIcLag(df, ic_times=10):
    '''
    get the corr of lag ic and return_rate
    :param df: dataframe , columns [date, stock, factor, return_rate]
    :param ic_times: range of nums of lagged
    :return: df , columns [lag, ic_lag]
    '''
    ic_lag_list = []
    for i in range(ic_times):
        df["factor_"] = df.groupby("stock")["factor"].shift(i)
        df_tmp = df.dropna(inplace=False)
        ic_lag = df_tmp.groupby("date").apply(lambda x: np.corrcoef(x["factor_"], x["return_rate"])[0, 1])
        ic_lag_list.append(ic_lag.mean())

    df = pd.DataFrame({"lag": range(ic_times), "ic_lag": ic_lag_list})
    return df


# Turnover rate
def getLongOnlyEasy(df, cost_long=0.003, cost_short=0.003, r=0.1):
    '''
    get the Long only return_rate and turnover rate of the factor model
    this function should be depreciated !!!
    :param df: df columns [date, return_rate, factor]
    :param cost_long: cost of long ,ratio
    :param cost_short: cost of short , ratio
    :param r:ratio of long and short
    :return:df_r, df_turnover
            df_r :index is date, columns is return_rate
            df_turnover: index is date, columns is turnover_rate

    '''
    df["factor_rank"] = df.groupby("date")["factor"].apply(lambda x: x.rank() / len(x))

    # Long weight
    df["weight"] = 0

    def adjust_weight(df):
        df.loc[df["factor_rank"] > 1 - r, "weight"] = 1 / len(df[df["factor_rank"] > 1 - r])
        return df[["weight"]]

    df["weight"] = df.groupby("date")[["factor_rank", "weight"]].apply(adjust_weight)

    '''
    calculate the turnover rate
    The problem is: if we use the shift, the date next is not the next trade day because
    the stocks in HS300 may change. So we get the columns last_trade_day to calculate the
    last weight.
    '''
    df["weight_last_day"] = 0
    df["weight_last_day"] = df.groupby("stock")["weight"].shift(1)
    df["last_day"] = df.groupby("stock")["date"].shift(1)

    trade_dates = sorted(list(set(df["date"])))
    trade_date_dict = {day2: day1 for day1, day2 in zip(trade_dates[:-1], trade_dates[1:])}
    df["last_trade_day"] = df["date"].apply(lambda x: trade_date_dict.get(x, np.NaN))
    df.loc[df["last_trade_day"] != df["last_day"], "weight_last_day"] = 0
    df["delta_weight"] = df["weight"] - df["weight_last_day"]

    '''
    The second problem is if the stock is not in the next HS300, we should get the fee of sold it

    '''
    trade_date_dict = {day1: day2 for day1, day2 in zip(trade_dates[:-1], trade_dates[1:])}
    df["next_trade_day"] = df["date"].apply(lambda x: trade_date_dict.get(x, np.NaN))
    df["next_day"] = df.groupby("stock")["date"].shift(-1)

    # next day sold. If the day is last date, we cannot sold.
    is_sold = (df["next_day"] != df["next_trade_day"])
    df["turnover_loss"] = 0
    df.loc[is_sold, "turnover_loss"] = df.loc[is_sold, "weight"] * cost_short
    df_turnover_loss = df.groupby("next_trade_day")["turnover_loss"].sum()

    # calculate the turnover rate, the turnover cost is in this month.
    df_r = df.groupby("date")[["return_rate", "weight", "delta_weight"]]. \
        apply(lambda x: (x["weight"] * x["return_rate"]).sum() -
                        (x.loc[x["delta_weight"] > 0, "delta_weight"] * cost_long).sum() +
                        (x.loc[x["delta_weight"] < 0, "delta_weight"] * cost_short).sum()
              ).to_frame("return_rate_gross")

    df_r = pd.merge(df_r, df_turnover_loss, how="left", left_index=True, right_index=True)
    df_r.fillna(value=0, inplace=True)
    df_r["return_rate"] = df_r["return_rate_gross"] - df_r["turnover_loss"]
    del df_r["turnover_loss"]
    del df_r["return_rate_gross"]

    # np.abs(x["weight_last_day"]).sum() != 1,means the stock may be sold yesterday and buy None today
    # the first day we need to adjust the turnover ratio
    df_turnover = df.groupby("date")[["delta_weight", "weight_last_day"]].apply(
        lambda x: (np.abs(x["delta_weight"]).sum() +
                   1 - np.abs(x["weight_last_day"]).sum()) / 2)

    df_turnover[0] = 0.5

    return df_r, df_turnover


def getLongWeight(df, r=0.1):
    '''
    get the long weight of the factor model
    :param df : columns:[date, stock, return_rate ,factor]
    :param r : ratio of long

    :return:df: columns:[date, stock, y1, y2, weight]
    '''
    df.sort_values(by="date", inplace=True)
    df["factor_rank"] = df.groupby("date")["factor"].apply(lambda x: x.rank() / len(x))

    # Long Short weight
    df["weight"] = 0

    def adjust_weight(df):
        df.loc[df["factor_rank"] > 1 - r, "weight"] = 1 / len(df[df["factor_rank"] > 1 - r])
        return df[["weight"]]

    df["weight"] = df.groupby("date")[["factor_rank", "weight"]].apply(adjust_weight)

    del df["factor_rank"]
    del df["factor"]

    return df


def getLongReturn(df, long_cost=0.003, short_cost=0.003):
    '''
    get the long return of the model
    :param df: dataframe, columns[date, stock, return_rate ,weight]
           long_cost : long cost
    :return: df_new : dataframe columns [date, return]
    '''
    trade_dates = sorted(list(set(df["date"])))
    trade_date_dict = {day1: day2 for day1, day2 in zip(trade_dates[:-1], trade_dates[1:])}
    df["next_trade_day"] = df["date"].apply(lambda x: trade_date_dict.get(x, pd.NA))

    df_ = pd.merge(df, df.dropna(inplace=False), left_on=["date", "stock"], right_on=["next_trade_day", "stock"],
                   how="outer")
    df_.fillna({"weight_x": 0, "weight_y": 0}, inplace=True)
    df_["delta_weight"] = df_["weight_x"] - df_["weight_y"]

    # if date_x is nan, then the next_trade_day_y cant be found in date_x,
    # means the the date before next_trade_day_y we want to buy the stock
    # and we sold it next day of next_trade_day_y.
    df_.loc[pd.isna(df_["date_x"]), "date_x"] = df_.loc[pd.isna(df_["date_x"]), "next_trade_day_y"]
    df_.dropna(subset=["date_x"], inplace=True)

    df_return = df_.groupby("date_x").apply(
        lambda df: (df["weight_x"] * df["return_rate_x"]).sum() -
                   df.loc[df["delta_weight"] > 0, "delta_weight"].sum() * long_cost +
                   df.loc[df["delta_weight"] < 0, "delta_weight"].sum() * short_cost)

    return df_return


def getTurnOver(df):
    '''
    get the turnover rate of the model
    :param df: dataframe [date ,stock, weight]
    :return: df_turnover : dataframe [date, turnover_rate]
    '''
    trade_dates = sorted(list(set(df["date"])))
    trade_date_dict = {day1: day2 for day1, day2 in zip(trade_dates[:-1], trade_dates[1:])}
    df["next_trade_day"] = df["date"].apply(lambda x: trade_date_dict.get(x, pd.NA))

    df_ = pd.merge(df, df.dropna(inplace=False), left_on=["date", "stock"], right_on=["next_trade_day", "stock"],
                   how="outer")
    df_.fillna({"weight_x": 0, "weight_y": 0}, inplace=True)
    df_["delta_weight"] = df_["weight_x"] - df_["weight_y"]

    # if date_x is nan, then the next_trade_day_y cant be found in date_x,
    # means the the date before next_trade_day_y we want to buy the stock
    # and we sold it next day of next_trade_day_y.
    df_.loc[pd.isna(df_["date_x"]), "date_x"] = df_.loc[pd.isna(df_["date_x"]), "next_trade_day_y"]
    df_.dropna(subset=["date_x"], inplace=True)

    df_turnover = df_.groupby("date_x").apply(
        lambda x: (np.abs(x["delta_weight"]).sum() / 2))

    return df_turnover


if __name__ == "__main__":
    df = pd.read_hdf("../datasets/data_factor_monthly.h5")

    df["code"] = np.random.randint(0, 6, [len(df), 1])
    df_test = pd.DataFrame(
        {"date": df.index, "stock": df["code"], "return_rate": df["pct_chg"],
         "factor": df["macd"], "field": df["field_"]})

    df_test["rank_"] = df_test.groupby(["field", "date"])["return_rate"].rank()
    print(df_test)

    def helper(df):
        s = pd.DataFrame(df.groupby(["field", "date"])["return_rate"].mean())
        s.columns = ["return_adj"]
        s.reset_index(drop=False)

        df_ = pd.merge(df, s, how="inner", on=["field", "date"])

        s = pd.DataFrame(df.groupby(["field", "date"])["factor"].mean())
        s.columns = ["factor_adj"]
        s.reset_index(drop=False)
        df_ = pd.merge(df_, s, how="inner", on=["field", "date"])

        df_ = df_[["date", "stock", "return_adj", "factor_adj"]]
        df_.columns = ["date", "stock", "return_rate", "factor"]

        return df_


    helper(df_test)
    # method 1
    # df_weight = getLongWeight(df_test, r=0.1)
    # df_tmp = getLongReturn(df_weight)
    # df_tmp_turnover = getTurnOver(df_weight)
    #
    # method 2
    # df_tmp2, df_turnover = getLongOnlyEasy(df["TradingDay"], df["InnerCode"])(df["return_rate"], df["OpenPrice"])

    # # test return rate
    # print(df_tmp)
    # print(df_tmp2)
    # print(np.abs(df_tmp - df_tmp2["return_rate"]).sum())

    # # test turnover
    # print(df_turnover)
    # print(df_tmp_turnover)
    #
    # print(abs(df_tmp_turnover - df_turnover).sum())
