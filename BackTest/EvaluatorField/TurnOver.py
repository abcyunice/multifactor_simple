import numpy as np
import pandas as pd

from collections.abc import Mapping

from .base import TurnoverEvaluator
from .helper import dictToDf


class TurnoverRateEvaluator(TurnoverEvaluator):
    def __init__(self):
        super().__init__()

    def _check_data(self, X):
        super()._check_data(X)

    def evaluate(self, df):
        '''
        get the turnover rate of the model
        :param df: dataframe [date ,stock, weight]
        :return: df_turnover : dataframe [date, turnover_rate]
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

        df_turnover = df_.groupby("date_x").apply(
            lambda x: (np.abs(x["delta_weight"]).sum() / 2))

        df_turnover = pd.DataFrame(df_turnover)
        df_turnover.reset_index(inplace=True)

        df_turnover.columns = ["date", "turnover_rate"]

        return df_turnover

    @property
    def params(self):
        return self._params
