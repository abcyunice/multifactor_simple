'''
重构了代码，pct_chg并不考虑信息泄露的问题，目的是为了接下来的计算

'''

import tushare as ts
import pandas as pd
import numpy as np
import datetime
import pickle

from Helper.Index import common


class GetStockMonthly(object):
    def __init__(self, mytoken):
        """
        储存最终结果

        """
        ts.set_token(mytoken)
        self.last_df = pd.DataFrame()
        self.except_code_list = []

    def get_data_helper(self, pro, code, regress_month):
        df_sh = self.getSH()
        df_daily = self.getDaily(code)
        df_daily_basic = self.getDailyBasic(pro, code)
        df123 = pd.concat([df_sh, df_daily, df_daily_basic], axis=1, join="inner")

        df_f = self.getFina(pro, code)
        df_f = df_f.resample("M").last()
        df_f.fillna(method="ffill", inplace=True)
        df_rollingreg = self.getRollingReg(df_daily[["pct_chg"]], df_sh, regress_month)
        df12345 = pd.concat([df123, df_f, df_rollingreg], axis=1, join="inner")
        df12345.dropna(inplace=True)

        return df12345

    def getSH(self):
        """
        上证指数序列
        return today

        """
        sh_res = ts.pro_bar(ts_code='000001.SH', adj='qfq', asset="I", freq="M")
        sh_array = sh_res["pct_chg"] / 100
        sh_df = pd.DataFrame({"trade_date": sh_res["trade_date"], "sh_chg": sh_array})
        sh_df["trade_date"] = pd.to_datetime(sh_df["trade_date"])
        sh_df.set_index("trade_date", inplace=True, drop=True)
        sh_df = sh_df.resample("M").last()
        return sh_df

    def getDaily(self, code):
        """
        交易数据
        trade_date:交易日
        amount:成交量
        pct_chg:涨跌幅百分比（用百分比表示，之后会除以100）

        """
        df_daily = ts.pro_bar(ts_code=code, adj='qfq', ma=[5, 10, 20], freq="M")
        df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"])
        df_daily["pct_chg"] = df_daily["pct_chg"] / 100
        df_daily.sort_values(by="trade_date", inplace=True)

        myindex = common.TechnicalIndicator()
        df_daily = myindex.getIndex(df_daily)
        df_daily = df_daily[["trade_date", "amount", "macd", "k", "d", "j", "pct_chg",
                             "ma5", "ma10", "ma20"]]
        df_daily.set_index("trade_date", inplace=True, drop=True)
        df_daily = df_daily.resample("M").last()

        return df_daily

    def getDailyBasic(self, pro, code):
        """
        每日指标
        pb:pb
        circ_mv:流通市值
        pe_ttm:pe_ttm

        """
        df_daily_basic = pro.daily_basic(ts_code=code, fields="trade_date,pb,ps_ttm,pe_ttm,circ_mv,turnover_rate_f")
        df_daily_basic["trade_date"] = pd.to_datetime(df_daily_basic["trade_date"])
        df_daily_basic.set_index("trade_date", inplace=True)
        df_daily_basic1 = df_daily_basic.resample("M").last()
        df_daily_basic1.columns = [col + "_last" for col in df_daily_basic1.columns]

        df_daily_basic2 = df_daily_basic.resample("M").mean()
        df_daily_basic2.columns = [col + "_mean" for col in df_daily_basic2.columns]

        df_daily_basic = pd.merge(df_daily_basic1, df_daily_basic2, left_index=True, right_index=True)
        return df_daily_basic

    def getFina(self, pro, code):
        """
        财务数据
        end_date:报告期(注意区分公告日期)
        roe:roe
        q_gr_yoy:营业总收入同比增长率(%)(单季度)，
        注意这里没有除以100，因为之后会标准化
        debt_to_assets:资产负债率

        """
        df_fina = pro.fina_indicator(ts_code=code, \
                                     fields='end_date,dt_eps,current_ratio,quick_ratio,fcfe_ps,\
                                            grossprofit_margin,op_of_gr,roe,roa,debt_to_assets,ocf_to_debt,\
                                            dt_eps_yoy,q_gr_yoy,q_profit_yoy,equity_yoy')
        df_fina["trade_date"] = pd.to_datetime(df_fina["end_date"])
        del df_fina["end_date"]
        df_fina.set_index("trade_date", inplace=True, drop=True)

        return df_fina

    def getRollingReg(self, df, sh_df, regress_month):
        """
          计算beta,残差平方和
          过去20天滚动回归
          同样注意防止信息泄露

          """
        df_all = pd.merge(df, sh_df, left_index=True, right_index=True, how="inner")

        df_all.sort_values(by="trade_date", inplace=True)
        df_all.dropna(inplace=True)
        df_all.reset_index(drop=False, inplace=True)

        def get_beta_e(s1, s2):
            chg_my = s1
            sh_chg = s2
            model = np.polyfit(x=chg_my, y=sh_chg, deg=1)
            beta = model[0]  # beta
            get_array = np.array([model[0] * my_ + model[1] for my_ in sh_chg])
            e_array = chg_my - get_array
            var_e = np.sum(e_array ** 2) / (len(e_array) - 2)  # e2的无偏估计
            return beta, var_e

        for j in range(len(df_all) - regress_month):
            s1 = np.array(df_all.loc[j:j + regress_month - 1, "pct_chg"])
            s2 = np.array(df_all.loc[j:j + regress_month - 1, "sh_chg"])
            beta, var_e = get_beta_e(s1, s2)
            df_all.loc[j + regress_month - 1, "beta"] = beta
            df_all.loc[j + regress_month - 1, "var_e"] = var_e

        """
        计算动量,过去20月累计涨跌幅，包括今天

        """

        for k in range(len(df_all) - regress_month):
            df_all.loc[k + regress_month - 1, "momentum"] = (df_all.loc[k:k + regress_month - 1,
                                                             "pct_chg"] + 1).prod() - 1
        """
        计算滚动二十月的平均，合并

        """
        df_all.set_index(["trade_date"], drop=True, inplace=True)

        df_all.dropna(inplace=True)
        del df_all["pct_chg"]
        del df_all["sh_chg"]
        return df_all

    def get_data(self, mydata, regress_month):
        """
        获取所有股票面板数据，结果储存在一个大df中
        mydata格式:dataframe,columns:["code","field","field_"]
        regress_day:计算beta回归的时间窗口长度
        cum_day:自变量取值的时间，barra模型默认是1，理论上最好和调仓周期一致
        return: dataframe 注意pct_chg是当天的pct_chg !!!

        """
        start_time = datetime.datetime.now()
        pro = ts.pro_api()

        """
        获取数据
        注意pct_chg

        """
        df_result = pd.DataFrame()
        for i, code in enumerate(mydata["code"]):
            try:
                field = mydata.loc[i, "field_"]
                df = self.get_data_helper(pro, code, regress_month)
                df["field_"] = field
                df["code"] = code
                df_result = df_result.append(df)

            except Exception as ex:
                print(ex, code)
                self.except_code_list.append(code)

            """输出，记录时间"""
            end_time = datetime.datetime.now()
            delta_time = end_time - start_time

            if i % 50 == 0:
                print("读取了%d只股票,过去了%s,时间是%s" % (i + 1, delta_time, end_time.strftime("%Y-%m-%d %H:%M:%S")))

        return df_result, self.except_code_list

if __name__ == "__main__":
    mydata = pd.read_csv("../datasets/field.csv", encoding="gbk")
    mytoken = "82f17f80a6f62681bcbf105689d892a9559a4e65af4045adb472eaf0"
    pro = ts.pro_api()

    mygetstock = GetStockMonthly(mytoken)
    last_df, except_code_list = mygetstock.get_data(mydata, regress_month=20)

    print(last_df, except_code_list)

    last_df.to_hdf("../datasets/data_factor_monthly.h5", key="data_factor_monthly")
    with open("../datasets/except_code_list.pkl", "wb+") as f:
        pickle.dump(except_code_list, f)
