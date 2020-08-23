# encoding=utf8

import pandas as pd
import numpy as np

from BackTest.EvaluatorSimple.ReturnRate import ExcessReturnEvaluator

from dateutil.relativedelta import relativedelta
import datetime


class TrainModel(object):
    def __init__(self, pipeline, param_grid, cv=5):
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.cv = cv

    def getResample(self, df, sd_cols):
        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        df.set_index("trade_date", inplace=True, drop=True)
        df_ = df.groupby("code").resample("M").last()

        # 标准化
        df_[sd_cols] = df_[sd_cols].groupby("trade_date").apply(
            lambda x: (x - x.mean(axis=0)) / x.std(axis=0))

        df_["sh_chg"] = df[["sh_chg", "code"]].groupby("code").resample("M").apply(
            lambda x: (x + 1).prod() - 1 if len(x) > 0 else 0)
        df_["pct_chg"] = df[["pct_chg", "code"]].groupby("code").resample("M").apply(
            lambda x: (x + 1).prod() - 1 if len(x) > 0 else 0)

        # 注意 ,resample 会排序
        df_["pct_chg"] = df_["pct_chg"].groupby("code").shift(-1)

        del df_["code"]

        df_.reset_index(inplace=True)
        df_.dropna(inplace=True)

        return df_

    def getXy(self, df, start_date, end_date, X_columns, y_columns):
        df = df.copy()
        df = df[(df["trade_date"] < end_date) & (df["trade_date"] >= start_date)]
        # add time delta
        delta_days = (end_date - start_date).days
        df["delta_time"] = (df["trade_date"] - start_date).apply(lambda x: x.days / delta_days)
        return np.hstack([df[X_columns].values.astype(np.float32), df[["delta_time"]].values.astype(np.float32)]), \
               df[y_columns].values.astype(np.float32).reshape(-1)

    def trainModel(self, train_X, train_y):
        grid_model = GridSearchCV(self.pipeline, param_grid=self.param_grid, cv=self.cv)
        grid_model.fit(train_X, train_y)
        return grid_model

    def getWeightDF(self, df, train_start_date, train_end_date, test_start_date, test_end_date, X_columns, y_columns):
        df = df.copy()

        train_X, train_y = self.getXy(df, train_start_date, train_end_date, X_columns, y_columns)
        model = self.trainModel(train_X, train_y)

        test_X, test_y = self.getXy(df, test_start_date, test_end_date, X_columns, y_columns)

        if len(test_X) == 0:
            return df

        weight_test = model.best_estimator_.predict(test_X)
        weight_big = np.quantile(weight_test, 0.9)

        if len(weight_test[weight_test > weight_big]) == 0:
            weight_test[:] = 1 / len(weight_test)
        else:
            weight_test[weight_test <= weight_big] = 0
            weight_test[weight_test > weight_big] = 1 / len(weight_test[weight_test > weight_big])
        df.loc[(df["trade_date"] < test_end_date) & (df["trade_date"] >= test_start_date), "weight"] = weight_test

        return df


class RollingTrain(object):
    def __init__(self, df, pipeline, param_grid, X_cols, y_cols, sd_cols, train_start_date, \
                 train_end_date, test_start_date, test_end_date):
        self.df = df.copy()
        self.pipeline = pipeline
        self.param_grid = param_grid

        self.X_cols = X_cols
        self.y_cols = y_cols
        self.sd_cols = sd_cols

        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date

    def train(self, delta_months=8 * 12):
        model = TrainModel(self.pipeline, self.param_grid)

        start_date = self.train_start_date
        end_date = self.test_end_date + relativedelta(months=delta_months)
        df = self.df[(self.df["trade_date"] < end_date) & (self.df["trade_date"] >= start_date)].copy()
        df = model.getResample(df, self.sd_cols)

        for i in range(delta_months + 1):
            model = TrainModel(self.pipeline, self.param_grid)
            train_start_date = self.train_start_date + relativedelta(months=i)
            train_end_date = self.train_end_date + relativedelta(months=i)
            test_start_date = self.test_start_date + relativedelta(months=i)
            test_end_date = self.test_end_date + relativedelta(months=i)
            df = model.getWeightDF(df, train_start_date, train_end_date, test_start_date, test_end_date, X_cols, y_cols)

        df.dropna(inplace=True)
        df_ = df[["trade_date", "code", "pct_chg", "weight"]]
        df_.columns = ["date", "stock", "return_rate", "weight"]
        df_.reset_index(drop=True, inplace=True)

        myevaluator = ExcessReturnEvaluator(long_cost=0.003, short_cost=0.003)
        df_all = myevaluator.evaluate(df_)

        df_all["nav_market"] = (df_all["return_market"] + 1).cumprod()
        df_all["nav_my"] = (df_all["return"] + 1).cumprod()

        return df_all

    def plotNav(self, df_all):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(df_all["date"], df_all["nav_market"], c="r", label="market")
        plt.plot(df_all["date"], df_all["nav_my"], c="b", label="my")

        plt.legend()
        plt.savefig("../result/modelfirst/XGB.jpg", dpi=100)


if __name__ == "__main__":
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_hdf("../../datasets/data_factor_monthly.h5")
    df.reset_index(drop=False, inplace=True)
    fields = list(set(df["field_"]))

    # one - hot
    for field in fields[:-1]:
        df[field] = (df["field_"] == field).astype(np.int)
    del df["field_"]

    X_cols = [c for c in df.columns if c not in ["trade_date", "code", "pct_chg"]]
    y_cols = ["pct_chg"]
    sd_cols = [c for c in df.columns if c not in fields + ["sh_chg", "trade_date", "code", "pct_chg"]]

    train_start_date = datetime.datetime(2010, 1, 1)
    train_end_date = datetime.datetime(2011, 1, 1)
    test_start_date = datetime.datetime(2011, 1, 1)
    test_end_date = datetime.datetime(2011, 2, 1)

    # estimator = Pipeline([("Lasso", Lasso(max_iter=10000))])
    # param_grid = {'Lasso__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
    # estimator = Pipeline([("Ridge", Ridge(max_iter=10000))])
    # param_grid = {'Ridge__alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000]}
    # estimator = Pipeline([("RF", RandomForestRegressor(n_estimators=1000, max_features="log2"))])
    # param_grid = {'RF__max_leaf_nodes': [2, 4, 6]}

    estimator = Pipeline([("xgb", xgb.XGBRegressor())])
    param_grid = {"xgb__gamma": [0.1, 0.3, 0.5], "xgb__n_estimators": [100, 1000]}

    rt = RollingTrain(df, estimator, param_grid, X_cols, y_cols, sd_cols, train_start_date,
                      train_end_date, test_start_date, test_end_date)
    df_all = rt.train()
    rt.plotNav(df_all)

    print(df_all)
    df_all.to_csv("./df_all_monthly.csv", index=False)

