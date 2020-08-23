import numpy as np
import pandas as pd

from collections.abc import Mapping

from .base import ReturnEvaluator
from .helper import dictToDf


class AbsoluteReturnEvaluator(ReturnEvaluator):
    def __init__(self, long_cost=0.0003, short_cost=0.0003):
        super().__init__()
        self.long_cost = long_cost
        self.short_cost = short_cost

    def _check_data(self, X):
        super()._check_data(X)

    def evaluate(self, df):
        '''
        get the long return of the model
        :param df: dataframe, columns[date, stock, return_rate ,weight]
               long_cost : long cost
        :return: df_new : dataframe columns [date, return]
        '''
        df = df.copy()
        self._check_data(df)
        if isinstance(df, Mapping):
            df = dictToDf(df)

        trade_dates = sorted(list(set(df["date"])))
        trade_date_dict = {day1: day2 for day1, day2 in zip(trade_dates[:-1], trade_dates[1:])}
        df["next_trade_day"] = df["date"].apply(lambda x: trade_date_dict.get(x, np.nan))
        df2 = df.dropna(inplace=False).copy()
        df2["next_trade_day"] = pd.to_datetime(df2["next_trade_day"])

        df_ = pd.merge(df, df2, left_on=["date", "stock"], right_on=["next_trade_day", "stock"],
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
                       df.loc[df["delta_weight"] > 0, "delta_weight"].sum() * self.long_cost +
                       df.loc[df["delta_weight"] < 0, "delta_weight"].sum() * self.short_cost)
        df_return = pd.DataFrame(df_return)
        df_return.reset_index(inplace=True)

        df_return.columns = ["date", "return"]
        return df_return

    @property
    def params(self):
        return self._params


class ExcessReturnEvaluator(ReturnEvaluator):
    def __init__(self, long_cost=0.003, short_cost=0.003):
        super().__init__()
        self.long_cost = long_cost
        self.short_cost = short_cost

    def _check_data(self, X):
        super()._check_data(X)

    def evaluate(self, df):
        '''
        get the long return of the model
        :param df: dataframe, columns[date, stock, return_rate ,weight]
               long_cost : long cost
        :return: df_new : dataframe columns [date, return]
        '''
        df = df.copy()
        self._check_data(df)
        if isinstance(df, Mapping):
            df = dictToDf(df)

        myabsolute_return = AbsoluteReturnEvaluator(self.long_cost, self.short_cost)
        df_return = myabsolute_return.evaluate(df)

        # 计算平均收益
        df_market = pd.DataFrame(df.groupby("date")["return_rate"].mean())
        df_market.reset_index(inplace=True)
        df_market.columns = ["date", "return_market"]

        df_all = pd.merge(df_return, df_market, on="date")
        df_all["excess_return"] = df_all["return"] - df_all["return_market"]

        return df_all

    @property
    def params(self):
        return self._params
