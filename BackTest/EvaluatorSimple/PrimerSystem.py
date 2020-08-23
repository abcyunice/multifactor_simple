from BackTest.EvaluatorSimple.ReturnRate import AbsoluteReturnEvaluator, ExcessReturnEvaluator
from BackTest.EvaluatorSimple.TurnOver import TurnoverRateEvaluator
from BackTest.EvaluatorSimple.helper import getDf
from BackTest.EvaluatorSimple.FactorModel import LongWeight, LongShortWeight, IcLag, GroupWeight, MeanIc
from BackTest.EvaluatorSimple.base import EvaluatorSystem

import pandas as pd
from math import sqrt


class PrimerFactorValueEvaluators(EvaluatorSystem):
    def __init__(self, long_cost=0.0003, short_cost=0.0003):
        super().__init__()
        self.long_cost = long_cost
        self.short_cost = short_cost

    def _check_data(self, X):
        super()._check_data(X)

    def evaluate(self, df):
        '''

        :param df: dataframe columns ["date", "stock", "return_rate", "factor"]
        :return:
        '''
        df = df.copy()
        self._check_data(df)

        # mean_ic
        my_ic = MeanIc()
        mean_ic = my_ic.evaluate(df)

        # 计算平均收益
        df_market = pd.DataFrame(df.groupby("date")["return_rate"].mean())
        df_market.reset_index(inplace=True)
        df_market.columns = ["date", "return_market"]

        # 添加weight, 多头和空头
        longweight = LongWeight()
        df_long = longweight.evaluate(df)
        longshortweight = LongShortWeight()
        df_longshort = longshortweight.evaluate(df)

        # 换手率
        turnover = TurnoverRateEvaluator()
        df_turnover = turnover.evaluate(df_long)

        # 平均换手率
        turnover_rate_annual = df_turnover["turnover_rate"].mean() * 252

        # 多头收益率,多空收益率
        return_evaluator = AbsoluteReturnEvaluator(long_cost=self.long_cost, short_cost=self.short_cost)
        df_long = return_evaluator.evaluate(df_long)
        df_longshort = return_evaluator.evaluate(df_longshort)

        # 平均超额收益率,平均多空收益率
        longshort_mean = df_longshort["return"].mean() * 252
        df_all = pd.merge(df_long, df_market, on="date")
        df_all["excess_rate"] = df_all["return"] - df_all["return_market"]
        excess_mean = df_all["excess_rate"].mean() * 252
        # information ratio
        info_ratio = excess_mean / (df_all["excess_rate"].std() * sqrt(252))

        # market_return_mean
        market_return_mean = df_all["return_market"].mean() * 252

        # result
        res = {"年化多空收益": longshort_mean, "年化超额收益": excess_mean, "信息比率": info_ratio,
               "年化市场收益": market_return_mean, "年均换手率": turnover_rate_annual, "ic均值": mean_ic}

        return res


class PrimerFactorDFEvaluators(EvaluatorSystem):
    def __init__(self, long_cost=0.0003, short_cost=0.0003):
        super().__init__()
        self.long_cost = long_cost
        self.short_cost = short_cost

    def _check_data(self, X):
        super()._check_data(X)

    def evaluate(self, df):
        '''
        :param df: dataframe columns ["date", "stock", "return_rate", "factor"]
        :return: df lag_ic, [ic, ic_lag]
                return_monthly,  [date:index, return, return_market, excess_return]
                group return [group, return annual]
        '''
        df = df.copy()
        self._check_data(df)

        # ic_lag
        ic_lag = IcLag()
        df_ic = ic_lag.evaluate(df)

        # return long
        longweight = LongWeight()
        df_long = longweight.evaluate(df)

        # 多头月均收益率
        return_evaluator = ExcessReturnEvaluator(long_cost=self.long_cost, short_cost=self.short_cost)
        df_long = return_evaluator.evaluate(df_long)
        df_long["date"] = pd.to_datetime(df_long["date"])
        df_long.set_index("date", inplace=True, drop=True)
        df_long = df_long.resample("M").apply(lambda x: (x + 1).cumprod().tolist()[-1] - 1)

        # 组间单调性
        return_list = []
        for r in range(9, -1, -1):
            r_start, r_end = r / 10, r / 10 + 0.1
            mygroupweight = GroupWeight(r_start, r_end)
            df_tmp = mygroupweight.evaluate(df)
            df_return = AbsoluteReturnEvaluator().evaluate(df_tmp)
            return_list.append(df_return["return"].mean() * 252)

        return_group = pd.DataFrame({"group": range(1, 11), "return_annual": return_list})

        return df_ic, df_long, return_group


class PrimerFactorPlotEvaluators(EvaluatorSystem):
    def __init__(self, long_cost=0.0003, short_cost=0.0003, factor_name="MyFactor"):
        super().__init__()
        self.long_cost = long_cost
        self.short_cost = short_cost
        self.factor_name = factor_name

    def _check_data(self, X):
        super()._check_data(X)

    def evaluate(self, df):
        import matplotlib.pyplot as plt
        import seaborn as sns

        value_evaluators = PrimerFactorValueEvaluators()
        res_value = value_evaluators.evaluate(df)

        df_evaluators = PrimerFactorDFEvaluators()
        df_ic, df_long, return_group = df_evaluators.evaluate(df)

        fig = plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")

        # lag ic
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(df_ic["lag"], df_ic["ic_lag"])
        ax1.set_xlabel("lag")
        ax1.set_ylabel("ic_lag")
        ax1.set_title("IC_LAG")
        plt.tight_layout()

        # df_long
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(df_long.index, df_long["excess_return"])
        ax2.set_xlabel("date")
        ax2.set_ylabel("excess_return_monthly")
        ax2.set_title("Long Excess Return Monthly")
        plt.tight_layout()

        # return_group
        ax3 = fig.add_subplot(2, 2, 3)
        sns.barplot(return_group["group"], return_group["return_annual"], alpha=0.8, color='red')
        ax3.set_xlabel("group")
        ax3.set_ylabel("absolute_return_annual")
        ax3.set_title("Group Return")
        plt.tight_layout()

        # table
        ax4 = fig.add_subplot(2, 2, 4)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        str_ = self.factor_name.replace("\t", "  ")
        plt.text(x=0.3, y=0.7, s=str_, fontsize=10)
        for i, key_value in enumerate(res_value.items()):
            plt.text(x=0.4, y=0.6 - 0.1 * i, s=key_value[0] + " : " + str(round(key_value[1], 2)), fontsize=10)
        plt.axis("off")

        plt.tight_layout()

        return fig
