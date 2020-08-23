'''The func for factor model
include: mean_ic, ic_lag, weight calculator for long longshort group
'''

import numpy as np
import pandas as pd

from collections.abc import Mapping

from .base import FactorEvaluator
from .helper import dictToDf, IndustryNeu


class MeanIc(FactorEvaluator):
    def __init__(self, half_life=252):
        super().__init__()
        self._params = {"date", "return_rate", "factor", "field"}
        self.half_life = half_life

    def _check_data(self, X):
        super()._check_data(X)

    def evaluate(self, df):
        '''
        :param df: dataframe [date, return_rate, factor, field]
        :param half_life: half life of the weight
        :return: meanic
        '''

        self._check_data(df)
        if isinstance(df, Mapping):
            df = dictToDf(df)

        df = IndustryNeu(df)

        ic = df.groupby("date").apply(lambda X: np.corrcoef(X["return_rate"].values, X["factor"].values)[0, 1])
        ic.sort_index(inplace=True)
        weight = (1 / 2) ** (1 / self.half_life)
        weight_list = np.array([weight ** (i - 1) for i in range(len(ic))])
        return np.sum(ic * weight_list) / np.sum(weight_list)

    @property
    def params(self):
        return self._params


class IcLag(FactorEvaluator):
    def __init__(self, ic_times=10):
        super().__init__()
        self._params = {"date", "return_rate", "factor", "field"}
        self.ic_times = ic_times

    def _check_data(self, X):
        super()._check_data(X)

    def evaluate(self, df):
        '''
        get the corr of lag ic and return_rate
        :param df: dataframe , columns [date, stock, factor, return_rate]
        :param ic_times: range of nums of lagged
        :return: df , columns [lag, ic_lag]
        '''
        df = df.copy()
        self._check_data(df)
        if isinstance(df, Mapping):
            df = dictToDf(df)

        ic_lag_list = []
        for i in range(self.ic_times):
            df["factor_"] = df.groupby("stock")["factor"].shift(i)
            df_tmp = df.dropna(inplace=False)
            df_tmp = IndustryNeu(df_tmp)
            ic_lag = df_tmp.groupby("date").apply(lambda x: np.corrcoef(x["factor_"], x["return_rate"])[0, 1])
            ic_lag_list.append(ic_lag.mean())

        df = pd.DataFrame({"lag": range(self.ic_times), "ic_lag": ic_lag_list})
        return df

    @property
    def params(self):
        return self._params


class LongWeight(FactorEvaluator):
    def __init__(self, r=0.1):
        super().__init__()
        self.r = r

    def _check_data(self, X):
        super()._check_data(X)

    def evaluate(self, df):
        '''
        get the long weight of the factor model
        :param df : columns:[date, stock, return_rate ,factor,field]
        :param r : ratio of long

        :return:df: columns:[date, stock, y1, y2, weight]
        '''
        df = df.copy()
        self._check_data(df)
        if isinstance(df, Mapping):
            df = dictToDf(df)

        df.sort_values(by="date", inplace=True)
        df["factor_rank"] = df.groupby("date", "field")["factor"].apply(lambda x: x.rank() / len(x))

        # Long Short weight
        df["weight"] = 0

        def adjust_weight(df):
            df.loc[df["factor_rank"] > 1 - self.r, "weight"] = 1 / len(df[df["factor_rank"] > 1 - self.r])
            return df[["weight"]]

        df["weight"] = df.groupby("date")[["factor_rank", "weight"]].apply(adjust_weight)

        del df["factor_rank"]
        del df["factor"]

        df.reset_index(inplace=True)

        return df

    @property
    def params(self):
        return self._params


class LongShortWeight(FactorEvaluator):
    def __init__(self, r=0.1):
        super().__init__()
        self.r = r

    def _check_data(self, X):
        super()._check_data(X)

    def evaluate(self, df):
        '''
        get the long weight of the factor model
        :param df : columns:[date, stock, return_rate, factor, field]
        :param r : ratio of long

        :return:df: columns:[date, stock, y1, y2, weight]
        '''
        df = df.copy()
        self._check_data(df)
        if isinstance(df, Mapping):
            df = dictToDf(df)

        df.sort_values(by="date", inplace=True)
        df["factor_rank"] = df.groupby("date")["factor"].apply(lambda x: x.rank() / len(x))

        # Long Short weight
        df["weight"] = 0

        def adjust_weight(df):
            df.loc[df["factor_rank"] > 1 - self.r, "weight"] = 1 / len(df[df["factor_rank"] > 1 - self.r])
            df.loc[df["factor_rank"] <= self.r, "weight"] = -1 / len(df[df["factor_rank"] <= self.r])
            return df[["weight"]]

        df["weight"] = df.groupby("date", "field")[["factor_rank", "weight"]].apply(adjust_weight)

        del df["factor_rank"]
        del df["factor"]

        df.reset_index(inplace=True)

        return df

    @property
    def params(self):
        return self._params


class GroupWeight(FactorEvaluator):
    def __init__(self, r_start, r_end):
        super().__init__()
        self.r_start = r_start
        self.r_end = r_end

    def _check_data(self, X):
        super()._check_data(X)

    def evaluate(self, df):
        '''
        get the long weight of the factor model
        :param df : columns:[date, stock, return_rate , factor, field]
        :param r : ratio of long

        :return:df: columns:[date, stock, y1, y2, weight]
        '''
        df = df.copy()
        self._check_data(df)
        if isinstance(df, Mapping):
            df = dictToDf(df)

        df.sort_values(by="date", inplace=True)
        df["factor_rank"] = df.groupby("date", "field")["factor"].apply(lambda x: x.rank() / len(x))

        # Long Short weight
        df["weight"] = 0

        def adjust_weight(df):
            df.loc[(df["factor_rank"] > self.r_start) & (df["factor_rank"] <= self.r_end), "weight"] = \
                1 / len(df[(df["factor_rank"] > self.r_start) & (df["factor_rank"] <= self.r_end)])
            return df[["weight"]]

        df["weight"] = df.groupby("date")[["factor_rank", "weight"]].apply(adjust_weight)

        del df["factor_rank"]
        del df["factor"]

        df.reset_index(inplace=True)

        return df

    @property
    def params(self):
        return self._params
