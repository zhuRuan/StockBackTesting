# 加载需要的库
import datetime
import multiprocessing
import os
import sys
import time
from multiprocessing.managers import BaseManager
from tqdm import tqdm
import backtrader as bt
import numpy as np
import pandas as pd
import tensorflow as tf
from backtrader.feeds import PandasData
from keras import backend as K
from keras.models import load_model
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler

from stock_backtesting_data.handle_df import mad

'''
获得某一股票的全部数据
输入：code--该股票对应的ts_code
输出：df_stock--该股票的全部数据，存为df
'''


def treat_df(df3):
    # 命名
    columns = df3.columns
    # 缺失值处理
    df = df3[~df3['pre_close'].isin([0, np.NAN])]
    df = df[~df['close'].isin([0, np.NAN])]
    df = df[~df['datetime'].isin([np.NAN])]
    df = df[df['pre_close'] > 0]
    df['is_st']
    # # 计算收益率
    # df['earn_rate'] = df.apply(lambda x: x['close'] / x['pre_close'] - 1, axis=1)
    # df = df[df['earn_rate'] < 0.1]
    # df = df[df['earn_rate'] > -0.1]
    df = df.drop(df.columns[[0, 1]], axis=1)
    print(df)
    print(df.columns.to_list())
    df.columns = ['code'] + df.columns.to_list()[1:]
    df_list = []
    # 多线程加速
    pool = multiprocessing.Pool()
    for i in tqdm(range(1,9)):
        index_1 = (i-1)*df.shape[0]//9
        index_2 = i*df.shape[0]//9
        df_list.append(pool.apply_async(func=fill_mean_into_nan, args=(df.iloc[index_1:index_2,:],)))
    df_list.append(pool.apply_async(func=fill_mean_into_nan, args=(df.iloc[8*df.shape[0]//9:,:],)))
    pool.close()
    pool.join()
    res = pd.DataFrame()
    for result in tqdm(df_list):
        part_res = result.get()
        print(part_res)
        res = pd.concat([res,part_res],axis=0)
    print(res)
    print('数据预处理完毕')
    # for column in df.columns.to_list()[12:]:
    #     print(df[df[column].isin([np.nan])][column])
    #     df[df[column].isin([np.nan])][column] = df[column].mean()
    #     df[column] = mad(df[column])
        # print(df[df[column].isin([np.nan])][column])
    # print(columns_list)
    return res


def fill_mean_into_nan(df):
    for column in df.columns.to_list()[12:]:
        # print(df[df[column].isin([np.nan])][column])
        df[df[column].isin([np.nan])][column] = df[column].mean()
        df[column] = mad(df[column])
    return df


def get_stock_data(code, factor_list, df_all):
    df_stock = df_all[df_all['code'] == code]
    df_stock = df_stock[['datetime', 'open', 'close', 'high', 'low', 'volume', 'money', 'high_limit',
                         'low_limit', 'avg', 'pre_close'] + factor_list]
    df_stock.index = df_stock.datetime
    df_stock = df_stock.sort_index()
    return df_stock


@tf.autograph.experimental.do_not_convert
def r_square_oos(y_true, y_pred):
    SSR = K.sum(K.square(y_true - y_pred))
    SST = K.sum(K.square(y_true - K.mean(y_true)))
    return SSR / SST


def add_parameters(list_factor):
    lines = 'lines = ('
    params = 'params = ('
    i = 6
    for factor in list_factor:
        lines = lines + '\'' + str(factor) + '\','
        params = params + '(\'' + factor + '\',' + str(i) + '),'
        i += 1
    lines = lines + ')'
    params = params + ')'
    return lines, params


# 修改原数据加载模块，以便能够加载更多自定义的因子数据
class Addmoredata(PandasData):
    lines = ('money', 'high_limit', 'low_limit', 'avg', 'pre_close', 'net_operate_cash_flow_to_total_liability',
             'fifty_two_week_close_rank', 'BIAS20', 'eps_ttm', 'net_profit_to_total_operate_revenue_ttm',
             'margin_stability', 'VOL5', 'super_quick_ratio', 'leverage', 'total_asset_growth_rate',
             'cash_to_current_liability', 'cash_flow_to_price_ratio', 'BIAS60', 'net_operate_cash_flow_to_net_debt',
             'DEGM_8y', 'np_parent_company_owners_growth_rate', 'momentum', 'BIAS5', 'BBIC', 'CCI88', 'CCI20',
             'net_profit_growth_rate', 'roa_ttm', 'VOL10', 'VOL60', 'operating_revenue_growth_rate', 'BIAS10',
             'total_asset_turnover_rate', 'VOL20', 'roic_ttm', 'quick_ratio', 'ROAEBITTTM', 'gross_income_ratio',
             'roe_ttm', 'DAVOL5', 'HY008', 'HY009', 'HY006', 'HY007', 'HY004', 'HY005', 'HY002', 'HY003',
             'equity_to_fixed_asset_ratio', 'HY001', 'VSTD10', 'BR', 'PEG', 'size', 'VOSC', 'cfo_to_ev',
             'net_operate_cash_flow_to_asset', 'Price1Y', 'ROC12', 'Price3M', 'VOL240', 'equity_to_asset_ratio', 'MASS',
             'residual_volatility', 'debt_to_equity_ratio', 'Price1M', 'book_to_price_ratio', 'ROC6',
             'debt_to_tangible_equity_ratio', 'HY011', 'HY010', 'ROC120', 'VOL120', 'beta', 'ROC60', 'WVAD', 'ROC20',
             'sales_to_price_ratio', 'is_st', 'turnover_ratio801010', 'pe801010', 'pb801010', 'average_price801010',
             'money_ratio801010', 'circulating_market_cap801010', 'average_circulating_market_cap801010',
             'dividend_ratio801010', 'turnover_ratio801030', 'pe801030', 'pb801030', 'average_price801030',
             'money_ratio801030', 'circulating_market_cap801030', 'average_circulating_market_cap801030',
             'dividend_ratio801030', 'turnover_ratio801040', 'pe801040', 'pb801040', 'average_price801040',
             'money_ratio801040', 'circulating_market_cap801040', 'average_circulating_market_cap801040',
             'dividend_ratio801040', 'turnover_ratio801050', 'pe801050', 'pb801050', 'average_price801050',
             'money_ratio801050', 'circulating_market_cap801050', 'average_circulating_market_cap801050',
             'dividend_ratio801050', 'turnover_ratio801080', 'pe801080', 'pb801080', 'average_price801080',
             'money_ratio801080', 'circulating_market_cap801080', 'average_circulating_market_cap801080',
             'dividend_ratio801080', 'turnover_ratio801110', 'pe801110', 'pb801110', 'average_price801110',
             'money_ratio801110', 'circulating_market_cap801110', 'average_circulating_market_cap801110',
             'dividend_ratio801110', 'turnover_ratio801120', 'pe801120', 'pb801120', 'average_price801120',
             'money_ratio801120', 'circulating_market_cap801120', 'average_circulating_market_cap801120',
             'dividend_ratio801120', 'turnover_ratio801130', 'pe801130', 'pb801130', 'average_price801130',
             'money_ratio801130', 'circulating_market_cap801130', 'average_circulating_market_cap801130',
             'dividend_ratio801130', 'turnover_ratio801140', 'pe801140', 'pb801140', 'average_price801140',
             'money_ratio801140', 'circulating_market_cap801140', 'average_circulating_market_cap801140',
             'dividend_ratio801140', 'turnover_ratio801150', 'pe801150', 'pb801150', 'average_price801150',
             'money_ratio801150', 'circulating_market_cap801150', 'average_circulating_market_cap801150',
             'dividend_ratio801150', 'turnover_ratio801160', 'pe801160', 'pb801160', 'average_price801160',
             'money_ratio801160', 'circulating_market_cap801160', 'average_circulating_market_cap801160',
             'dividend_ratio801160', 'turnover_ratio801170', 'pe801170', 'pb801170', 'average_price801170',
             'money_ratio801170', 'circulating_market_cap801170', 'average_circulating_market_cap801170',
             'dividend_ratio801170', 'turnover_ratio801180', 'pe801180', 'pb801180', 'average_price801180',
             'money_ratio801180', 'circulating_market_cap801180', 'average_circulating_market_cap801180',
             'dividend_ratio801180', 'turnover_ratio801200', 'pe801200', 'pb801200', 'average_price801200',
             'money_ratio801200', 'circulating_market_cap801200', 'average_circulating_market_cap801200',
             'dividend_ratio801200', 'turnover_ratio801210', 'pe801210', 'pb801210', 'average_price801210',
             'money_ratio801210', 'circulating_market_cap801210', 'average_circulating_market_cap801210',
             'dividend_ratio801210', 'turnover_ratio801230', 'pe801230', 'pb801230', 'average_price801230',
             'money_ratio801230', 'circulating_market_cap801230', 'average_circulating_market_cap801230',
             'dividend_ratio801230', 'turnover_ratio801710', 'pe801710', 'pb801710', 'average_price801710',
             'money_ratio801710', 'circulating_market_cap801710', 'average_circulating_market_cap801710',
             'dividend_ratio801710', 'turnover_ratio801720', 'pe801720', 'pb801720', 'average_price801720',
             'money_ratio801720', 'circulating_market_cap801720', 'average_circulating_market_cap801720',
             'dividend_ratio801720', 'turnover_ratio801730', 'pe801730', 'pb801730', 'average_price801730',
             'money_ratio801730', 'circulating_market_cap801730', 'average_circulating_market_cap801730',
             'dividend_ratio801730', 'turnover_ratio801740', 'pe801740', 'pb801740', 'average_price801740',
             'money_ratio801740', 'circulating_market_cap801740', 'average_circulating_market_cap801740',
             'dividend_ratio801740', 'turnover_ratio801750', 'pe801750', 'pb801750', 'average_price801750',
             'money_ratio801750', 'circulating_market_cap801750', 'average_circulating_market_cap801750',
             'dividend_ratio801750', 'turnover_ratio801760', 'pe801760', 'pb801760', 'average_price801760',
             'money_ratio801760', 'circulating_market_cap801760', 'average_circulating_market_cap801760',
             'dividend_ratio801760', 'turnover_ratio801770', 'pe801770', 'pb801770', 'average_price801770',
             'money_ratio801770', 'circulating_market_cap801770', 'average_circulating_market_cap801770',
             'dividend_ratio801770', 'turnover_ratio801780', 'pe801780', 'pb801780', 'average_price801780',
             'money_ratio801780', 'circulating_market_cap801780', 'average_circulating_market_cap801780',
             'dividend_ratio801780', 'turnover_ratio801790', 'pe801790', 'pb801790', 'average_price801790',
             'money_ratio801790', 'circulating_market_cap801790', 'average_circulating_market_cap801790',
             'dividend_ratio801790', 'turnover_ratio801880', 'pe801880', 'pb801880', 'average_price801880',
             'money_ratio801880', 'circulating_market_cap801880', 'average_circulating_market_cap801880',
             'dividend_ratio801880', 'turnover_ratio801890', 'pe801890', 'pb801890', 'average_price801890',
             'money_ratio801890', 'circulating_market_cap801890', 'average_circulating_market_cap801890',
             'dividend_ratio801890', 'index000001', 'index000300', 'index399001', 'index399006', 'index399101',
             'index399102', 'index399106',)
    params = (('money', 6), ('high_limit', 7), ('low_limit', 8), ('avg', 9), ('pre_close', 10),
              ('net_operate_cash_flow_to_total_liability', 11), ('fifty_two_week_close_rank', 12), ('BIAS20', 13),
              ('eps_ttm', 14), ('net_profit_to_total_operate_revenue_ttm', 15), ('margin_stability', 16), ('VOL5', 17),
              ('super_quick_ratio', 18), ('leverage', 19), ('total_asset_growth_rate', 20),
              ('cash_to_current_liability', 21), ('cash_flow_to_price_ratio', 22), ('BIAS60', 23),
              ('net_operate_cash_flow_to_net_debt', 24), ('DEGM_8y', 25), ('np_parent_company_owners_growth_rate', 26),
              ('momentum', 27), ('BIAS5', 28), ('BBIC', 29), ('CCI88', 30), ('CCI20', 31),
              ('net_profit_growth_rate', 32), ('roa_ttm', 33), ('VOL10', 34), ('VOL60', 35),
              ('operating_revenue_growth_rate', 36), ('BIAS10', 37), ('total_asset_turnover_rate', 38), ('VOL20', 39),
              ('roic_ttm', 40), ('quick_ratio', 41), ('ROAEBITTTM', 42), ('gross_income_ratio', 43), ('roe_ttm', 44),
              ('DAVOL5', 45), ('HY008', 46), ('HY009', 47), ('HY006', 48), ('HY007', 49), ('HY004', 50), ('HY005', 51),
              ('HY002', 52), ('HY003', 53), ('equity_to_fixed_asset_ratio', 54), ('HY001', 55), ('VSTD10', 56),
              ('BR', 57), ('PEG', 58), ('size', 59), ('VOSC', 60), ('cfo_to_ev', 61),
              ('net_operate_cash_flow_to_asset', 62), ('Price1Y', 63), ('ROC12', 64), ('Price3M', 65), ('VOL240', 66),
              ('equity_to_asset_ratio', 67), ('MASS', 68), ('residual_volatility', 69), ('debt_to_equity_ratio', 70),
              ('Price1M', 71), ('book_to_price_ratio', 72), ('ROC6', 73), ('debt_to_tangible_equity_ratio', 74),
              ('HY011', 75), ('HY010', 76), ('ROC120', 77), ('VOL120', 78), ('beta', 79), ('ROC60', 80), ('WVAD', 81),
              ('ROC20', 82), ('sales_to_price_ratio', 83), ('is_st', 84), ('turnover_ratio801010', 85),
              ('pe801010', 86), ('pb801010', 87), ('average_price801010', 88), ('money_ratio801010', 89),
              ('circulating_market_cap801010', 90), ('average_circulating_market_cap801010', 91),
              ('dividend_ratio801010', 92), ('turnover_ratio801030', 93), ('pe801030', 94), ('pb801030', 95),
              ('average_price801030', 96), ('money_ratio801030', 97), ('circulating_market_cap801030', 98),
              ('average_circulating_market_cap801030', 99), ('dividend_ratio801030', 100),
              ('turnover_ratio801040', 101), ('pe801040', 102), ('pb801040', 103), ('average_price801040', 104),
              ('money_ratio801040', 105), ('circulating_market_cap801040', 106),
              ('average_circulating_market_cap801040', 107), ('dividend_ratio801040', 108),
              ('turnover_ratio801050', 109), ('pe801050', 110), ('pb801050', 111), ('average_price801050', 112),
              ('money_ratio801050', 113), ('circulating_market_cap801050', 114),
              ('average_circulating_market_cap801050', 115), ('dividend_ratio801050', 116),
              ('turnover_ratio801080', 117), ('pe801080', 118), ('pb801080', 119), ('average_price801080', 120),
              ('money_ratio801080', 121), ('circulating_market_cap801080', 122),
              ('average_circulating_market_cap801080', 123), ('dividend_ratio801080', 124),
              ('turnover_ratio801110', 125), ('pe801110', 126), ('pb801110', 127), ('average_price801110', 128),
              ('money_ratio801110', 129), ('circulating_market_cap801110', 130),
              ('average_circulating_market_cap801110', 131), ('dividend_ratio801110', 132),
              ('turnover_ratio801120', 133), ('pe801120', 134), ('pb801120', 135), ('average_price801120', 136),
              ('money_ratio801120', 137), ('circulating_market_cap801120', 138),
              ('average_circulating_market_cap801120', 139), ('dividend_ratio801120', 140),
              ('turnover_ratio801130', 141), ('pe801130', 142), ('pb801130', 143), ('average_price801130', 144),
              ('money_ratio801130', 145), ('circulating_market_cap801130', 146),
              ('average_circulating_market_cap801130', 147), ('dividend_ratio801130', 148),
              ('turnover_ratio801140', 149), ('pe801140', 150), ('pb801140', 151), ('average_price801140', 152),
              ('money_ratio801140', 153), ('circulating_market_cap801140', 154),
              ('average_circulating_market_cap801140', 155), ('dividend_ratio801140', 156),
              ('turnover_ratio801150', 157), ('pe801150', 158), ('pb801150', 159), ('average_price801150', 160),
              ('money_ratio801150', 161), ('circulating_market_cap801150', 162),
              ('average_circulating_market_cap801150', 163), ('dividend_ratio801150', 164),
              ('turnover_ratio801160', 165), ('pe801160', 166), ('pb801160', 167), ('average_price801160', 168),
              ('money_ratio801160', 169), ('circulating_market_cap801160', 170),
              ('average_circulating_market_cap801160', 171), ('dividend_ratio801160', 172),
              ('turnover_ratio801170', 173), ('pe801170', 174), ('pb801170', 175), ('average_price801170', 176),
              ('money_ratio801170', 177), ('circulating_market_cap801170', 178),
              ('average_circulating_market_cap801170', 179), ('dividend_ratio801170', 180),
              ('turnover_ratio801180', 181), ('pe801180', 182), ('pb801180', 183), ('average_price801180', 184),
              ('money_ratio801180', 185), ('circulating_market_cap801180', 186),
              ('average_circulating_market_cap801180', 187), ('dividend_ratio801180', 188),
              ('turnover_ratio801200', 189), ('pe801200', 190), ('pb801200', 191), ('average_price801200', 192),
              ('money_ratio801200', 193), ('circulating_market_cap801200', 194),
              ('average_circulating_market_cap801200', 195), ('dividend_ratio801200', 196),
              ('turnover_ratio801210', 197), ('pe801210', 198), ('pb801210', 199), ('average_price801210', 200),
              ('money_ratio801210', 201), ('circulating_market_cap801210', 202),
              ('average_circulating_market_cap801210', 203), ('dividend_ratio801210', 204),
              ('turnover_ratio801230', 205), ('pe801230', 206), ('pb801230', 207), ('average_price801230', 208),
              ('money_ratio801230', 209), ('circulating_market_cap801230', 210),
              ('average_circulating_market_cap801230', 211), ('dividend_ratio801230', 212),
              ('turnover_ratio801710', 213), ('pe801710', 214), ('pb801710', 215), ('average_price801710', 216),
              ('money_ratio801710', 217), ('circulating_market_cap801710', 218),
              ('average_circulating_market_cap801710', 219), ('dividend_ratio801710', 220),
              ('turnover_ratio801720', 221), ('pe801720', 222), ('pb801720', 223), ('average_price801720', 224),
              ('money_ratio801720', 225), ('circulating_market_cap801720', 226),
              ('average_circulating_market_cap801720', 227), ('dividend_ratio801720', 228),
              ('turnover_ratio801730', 229), ('pe801730', 230), ('pb801730', 231), ('average_price801730', 232),
              ('money_ratio801730', 233), ('circulating_market_cap801730', 234),
              ('average_circulating_market_cap801730', 235), ('dividend_ratio801730', 236),
              ('turnover_ratio801740', 237), ('pe801740', 238), ('pb801740', 239), ('average_price801740', 240),
              ('money_ratio801740', 241), ('circulating_market_cap801740', 242),
              ('average_circulating_market_cap801740', 243), ('dividend_ratio801740', 244),
              ('turnover_ratio801750', 245), ('pe801750', 246), ('pb801750', 247), ('average_price801750', 248),
              ('money_ratio801750', 249), ('circulating_market_cap801750', 250),
              ('average_circulating_market_cap801750', 251), ('dividend_ratio801750', 252),
              ('turnover_ratio801760', 253), ('pe801760', 254), ('pb801760', 255), ('average_price801760', 256),
              ('money_ratio801760', 257), ('circulating_market_cap801760', 258),
              ('average_circulating_market_cap801760', 259), ('dividend_ratio801760', 260),
              ('turnover_ratio801770', 261), ('pe801770', 262), ('pb801770', 263), ('average_price801770', 264),
              ('money_ratio801770', 265), ('circulating_market_cap801770', 266),
              ('average_circulating_market_cap801770', 267), ('dividend_ratio801770', 268),
              ('turnover_ratio801780', 269), ('pe801780', 270), ('pb801780', 271), ('average_price801780', 272),
              ('money_ratio801780', 273), ('circulating_market_cap801780', 274),
              ('average_circulating_market_cap801780', 275), ('dividend_ratio801780', 276),
              ('turnover_ratio801790', 277), ('pe801790', 278), ('pb801790', 279), ('average_price801790', 280),
              ('money_ratio801790', 281), ('circulating_market_cap801790', 282),
              ('average_circulating_market_cap801790', 283), ('dividend_ratio801790', 284),
              ('turnover_ratio801880', 285), ('pe801880', 286), ('pb801880', 287), ('average_price801880', 288),
              ('money_ratio801880', 289), ('circulating_market_cap801880', 290),
              ('average_circulating_market_cap801880', 291), ('dividend_ratio801880', 292),
              ('turnover_ratio801890', 293), ('pe801890', 294), ('pb801890', 295), ('average_price801890', 296),
              ('money_ratio801890', 297), ('circulating_market_cap801890', 298),
              ('average_circulating_market_cap801890', 299), ('dividend_ratio801890', 300), ('index000001', 301),
              ('index000300', 302), ('index399001', 303), ('index399006', 304), ('index399101', 305),
              ('index399102', 306), ('index399106', 307),)

    # 设置佣金和印花税率


class stampDutyCommissionScheme(bt.CommInfoBase):
    '''
    本佣金模式下，买入股票仅支付佣金，卖出股票支付佣金和印花税.    
    '''
    params = (
        ('stamp_duty', 0.001),  # 印花税率
        ('commission', 0.0005),  # 佣金率
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        '''
        If size is greater than 0, this indicates a long / buying of shares.
        If size is less than 0, it idicates a short / selling of shares.
        '''
        # print('self.p.commission', str(self.p.commission))
        if size > 0:  # 买入，不考虑印花税
            return size * price * self.p.commission * 100
        elif size < 0:  # 卖出，考虑印花税
            return - size * price * (self.p.stamp_duty + self.p.commission * 100)
        else:
            return 0

        # 编写策略


class momentum_factor_strategy(bt.Strategy):
    # interval-换仓间隔，stocknum-持仓股票数, 均值时间长度长度
    params = (("interval", 1), ("stocknum", 10),)

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('{}, {}'.format(dt.isoformat(), txt))

    def __init__(self):

        # 已清洗过的每日可用股票列表
        self.df_byday = pd.read_csv(csv_name3)
        # 交易天数，用于判断是否交易
        self.bar_num = 0
        # 上次交易股票的列表
        self.last = []
        # 记录以往订单，在调仓日要全部取消未成交的订单
        self.order_list = []

        # 记录现有的持仓
        self.stock_position_list = []

    def prenext(self):

        self.next()

    def next(self):
        # 记录交易日期
        self.bar_num += 1
        # 获取当前账户价值
        total_value = self.broker.getvalue()
        print("当天日期:{}".format(str(self.datas[0].datetime.date(0))))
        self.log('当前总市值 %.2f。' % (total_value))
        self.log('当前总收益率 {:.2%}'.format((self.broker.getvalue() / startcash) - 1))
        # 计算当日是否调仓
        if self.bar_num % self.p.interval == 0 and self.bar_num > 1 * self.p.interval and self.datas[0].datetime.date(
                0) < datetime.date(2022, 9, 12):
            # 得到当天的时间
            current_date = self.datas[0].datetime.date(0)
            print("交易日日期:{}".format(str(self.datas[0].datetime.date(0))))
            # 获得上一调仓日和下一调仓日时间
            prev_date = self.datas[0].datetime.date(-self.p.interval)
            # next_date = self.datas[0].datetime.date(self.p.interval)
            # next_next_date = self.datas[0].datetime.date(self.p.interval + 1)
            # 获取当日可行股票池
            stocklist = self.get_valid_list_day(current_date)
            # 获取上一调仓日可行股票池
            stocklist_p = self.get_valid_list_day(prev_date)
            # #获取下个调仓日可行股票池
            # stocklist_next = self.get_valid_list_day(next_date)
            # print(stocklist_next)
            # # 获取下下个调仓日可行股票池
            # stocklist_next_next = self.get_valid_list_day(next_next_date)

            # 计算本期因子数据df_fac，并清洗
            df_fac = self.get_df_fac(stocklist=stocklist, factor_list=list_factor, prev=0)
            print(df_fac)
            df_fac = df_fac.dropna(axis=0, how='any')

            # 计算上期因子数据df_faxc_p，并清洗
            df_fac_p = self.get_df_fac(stocklist=stocklist_p, factor_list=list_factor, prev=1)
            df_fac_p = df_fac_p.dropna(axis=0, how='any')

            # 本期因子排列命名
            df_fac.columns = ['code', 'momentum_value', 'pre_close'] + list_factor
            df_fac.index = df_fac.code.values

            # 上期因子排列命名
            df_fac_p.columns = ['code', 'momentum_value', 'pre_close'] + list_factor

            df_fac_p.index = df_fac_p.code.values

            # 舍弃X_p和Y中不同的index（股票代码）
            # 先去除X_p比Y多的index
            diffIndex = df_fac_p.index.difference(df_fac.index)
            print(stocklist)
            print(df_fac_p)
            print(df_fac)
            # 删除整行
            df_fac_p = df_fac_p.drop(diffIndex, errors='ignore')
            df_fac = df_fac.drop(diffIndex, errors='ignore')

            # 然后去除Y比X_p多的index
            np.set_printoptions(suppress=True)
            diffIndex = df_fac.index.difference(df_fac_p.index)
            df_fac_p = df_fac_p.drop(diffIndex, errors='ignore')
            df_fac = df_fac.drop(diffIndex, errors='ignore')
            # X_p是上一期的因子值，X是本期因子值，Y是回归目标
            X_p = df_fac_p[['momentum_value','pre_close'] + list_factor[:len(list_factor) - 1]]  # 过去因子数据
            X = df_fac[['momentum_value','pre_close'] + list_factor[:len(list_factor) - 1]]  # 当前因子数据
            Y = df_fac[['momentum_value']]
            Y_long = df_fac[['Price1M']]

            # 将因子值与Y值均进行标准化
            rbX = MinMaxScaler(feature_range=(0, 1))
            robust_x_train = rbX.fit_transform(X_p)
            robust_x_predict = rbX.transform(X)
            rbY = MinMaxScaler(feature_range=(0, 1))
            rbY2 = MinMaxScaler(feature_range=(0, 1))
            Y_after_transform = rbY.fit_transform(Y)
            Y_long_after_transform = rbY2.fit_transform(Y_long)
            np.set_printoptions(suppress=True)

            # 转化为三维数据
            # reshape input to be 3D [samples, timesteps, features]
            train_X = robust_x_train.reshape((robust_x_train.shape[0], 1, robust_x_train.shape[1]))
            test_X = robust_x_predict.reshape((robust_x_predict.shape[0], 1, robust_x_predict.shape[1]))

            train_y = Y_after_transform.flatten()

            # # 用上期因子值与本期回报率进行训练、
            LSTM.fit(x=train_X, y=train_y, batch_size=5000, epochs=50,verbose=10000)
            # sgdr_long_term_revenue.partial_fit(X=robust_x_train, y=(Y_long_after_transform))

            LSTM_pred = LSTM.predict(test_X)
            # sgdr_long_pred = sgdr_long_term_revenue.predict(robust_x_predict)

            a = rbY.inverse_transform(LSTM_pred.reshape(-1, 1))
            # a2 = rbY2.inverse_transform(sgdr_long_pred.reshape(-1,1))
            df_fac['pred'] = a
            # df_fac['pred_long'] = a2

            # 找出所有st股
            st_list = df_fac[df_fac['is_st'] == True]['code'].to_list()

            # 找出所有涨停股票
            trading_list = df_fac[df_fac['momentum_value'] >= 9.95]['code'].to_list()

            # 按照预测得到的下期收益进行排序
            df_fac.sort_values(by="pred", inplace=True, ascending=False)
            # 取预测收益>0且排序靠前的stocknum只股票做多
            df_fac_pos = df_fac[df_fac['pred'] > 0]
            # df_fac_long_pos_pred = df_fac[df_fac['pred_long'] > 0]
            # df_fac_long_pos = df_fac[df_fac['Price1M'] > 0]
            sort_list_pos = df_fac_pos['code'].tolist()
            # positive_long_term_list = df_fac_long_pos['code'].to_list()
            # positive_long_term_list_next = df_fac_long_pos_pred['code'].to_list()
            # 根据预测收益率线性分配仓位
            long_list = sort_list_pos[:self.p.stocknum]  # 收益大于0的股票列表的排名在交易数量之内的股票列表
            long_list_yield_total = df_fac_pos['pred'].tolist()  # 收益大于0的所有股票的预测值
            long_list_yield = df_fac_pos['pred'].tolist()[:self.p.stocknum]  # 收益率大于0的所有股票的预测值中排名靠前的
            position_list = []
            sum_yield_total = np.sum(long_list_yield)
            for stock_yield, stock in zip(long_list_yield, long_list):
                percent = (0.95 * 0.03)  # 想要设定的仓位
                stock_position = self.getposition(data=stock).size
                portfolio_value = self.broker.getvalue()
                percent_now = stock_position / portfolio_value  # 获取当前的仓位
                if percent_now + percent >= 0.2:
                    percent = 0
                if stock in st_list or stock in trading_list:
                    percent = 0
                position_list.append(percent)

            # 取消以往所下订单（已成交的不会起作用）
            for o in self.order_list:
                self.cancel(o)
            # 重置订单列表
            self.order_list = []

            # 若上期交易股票未出现在本期交易列表中，则平仓
            # 即将退市的股票也清仓
            for i in self.stock_position_list:
                if i not in sort_list_pos:  # i not in positive_long_term_list and i not in positive_long_term_list_next):
                    self.stock_position_list.remove(i)
                    d = self.getdatabyname(i)
                    if self.getposition(d).size > 0:
                        print('sell 平仓', d._name, self.getposition(d).size)
                        o = self.close(data=d)
                        self.order_list.append(o)  # 记录订单
                # elif i not in stocklist_next or i not in stocklist_next_next:
                #     d = self.getdatabyname(i)
                #     if self.getposition(d).size > 0 :
                #         print('sell 强制平仓', d._name, self.getposition(d).size)
                #         o = self.close(data=d)
                #         self.order_list.append(o)  # 记录订单

            # 对long_list中股票做多
            if len(long_list):

                # 依次买入
                for d, position_of_stock in zip(long_list, position_list):
                    data = self.getdatabyname(d)
                    # 若不在上期持仓列表中，则加入列表
                    if d not in self.stock_position_list:
                        self.stock_position_list.append(d)
                    # 按次日开盘价计算下单量，下单量是100的整数倍
                    target_value = position_of_stock * total_value
                    size1 = int(abs(target_value / data.open[0] // 100 * 100))
                    o = self.order_target_size(data=d, target=size1)  # 留百分之五用于支付费用
                    # 记录订单
                    self.order_list.append(o)

    # 交易日志
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Value: %.2f, Comm %.2f, Size: %.2f, Stock: %s' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm,
                     order.executed.size,
                     order.data._name))
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Value: %.2f, Comm %.2f, Size: %.2f, Stock: %s' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          order.executed.size,
                          order.data._name))

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log('TRADE PROFIT, GROSS %.2f, NET %.2f' %
                     (trade.pnl, trade.pnlcomm))

    # 求个股某周期因子平均值，prev为是否为前一周期（0：否，1：是）
    def get_df_fac(self, stocklist, factor_list, prev=0):

        # 新建df_fac用于存储计算得到的因子值
        factor_list = ['pre_close'] + factor_list
        column_list = ['code', 'momentum_value'] + factor_list
        df_fac = pd.DataFrame(columns=column_list)

        for stock in stocklist:
            data = self.getdatabyname(stock)
            # 声明了一个用于存储 一定区间内所有因子的 list 具体结构为 [[],[],...,[]]
            factor_MA_list = []
            factor_list_copy = factor_list.copy()
            for factor in factor_list_copy:
                factor_list2 = []
                factor_MA_list.append(factor_list2)
            # 获取当期因子值得平均数
            factor_list_copy.remove('is_st')
            column_list = ['code', 'momentum_value'] + factor_list_copy
            for i in range(0, self.p.interval):
                for factor_list1, factor in zip(factor_MA_list[:len(factor_MA_list)-1], factor_list_copy):
                    if factor != 'is_st':
                        str = 'factor_list1.append(data.' + factor + '[-i - prev * self.p.interval])'
                        exec(str)
            # st无法取均值，单独获取
            factor_MA_list[len(factor_MA_list) - 1] = []
            factor_MA_list[len(factor_MA_list) - 1].append(data.is_st[0])
            # 计算当期动量
            try:
                sell_ = data.close[0 - prev * self.p.interval]  # 若prev = 0，则返回当前时点的开盘价
                buy_ = data.close[-1 - prev * self.p.interval]
                stock_momentum = sell_ / buy_ - 1
            except IndexError:
                stock_momentum = np.nan
            except ValueError:
                stock_momentum = np.nan
            except ZeroDivisionError:
                stock_momentum = np.nan
            # 最后返回的具体数值列表
            factor_MA_number_list = []
            factor_MA_number_list.append(stock)
            factor_MA_number_list.append(stock_momentum)
            for factor_list1 in factor_MA_list:
                factor_MA_number_list.append(np.mean(factor_list1))
            new = pd.DataFrame([factor_MA_number_list], index=[1], columns=column_list+['is_st'])
            df_fac = pd.concat([df_fac, new], ignore_index=True)
        # # print(df_fac["is_st"])
        # for column in df_fac.columns.to_list():
        #     print(df_fac[df_fac[column].isin([np.nan])][column])
        return df_fac

    # 获取当日可行股票池
    def get_valid_list_day(self, current_date):
        self.df_byday['Date'] = pd.to_datetime(self.df_byday['Date'], errors='coerce', format='%Y-%m-%d')
        current_date = datetime.datetime.strptime(str(current_date), '%Y-%m-%d')
        df_day = self.df_byday[self.df_byday['Date'] == current_date]
        stocklist = list(df_day["code"])
        # stocklist = literal_eval(df_day['stocklist'].tolist()[0])
        return stocklist


def get_new_pkl(file_dir, model_for_what):
    file_lists = os.listdir(file_dir)
    file_dict = {}
    for i in file_lists:  # 遍历所有文件
        if model_for_what in i:
            ctime = os.stat(os.path.join(file_dir, i)).st_ctime
            file_dict[ctime] = i  # 添加创建时间和文件名到字典
    max_ctime = max(file_dict.keys())  # 取值最大的时间
    print("已读取最新模型参数文件： ", file_dict[max_ctime])  # 打印出最新文件名
    return os.path.join(file_dir, file_dict[max_ctime])


def add_data(stock_list, list_factor, start_date, end_date, dataframe):
    feed = []
    for s in tqdm(stock_list):
        feed.append(Addmoredata(dataname=get_stock_data(s, list_factor, dataframe), plot=False, fromdate=start_date,
                       todate=end_date))
    # cerebro.adddata(feed, s)
    # print('success')
    return feed, stock_list

def split_list(lst, n):
    return (lst[i::n] for i in range(n))


if __name__ == '__main__':
    ##########################
    # 主程序开始
    ##########################
    begin_time = time.time()
    # 实例化模型'
    loss_model = 'squared_error'

    # sgdr = SGDRegressor(loss=loss_model, penalty='l2', learning_rate='adaptive')
    sgdr_long_term_revenue = SGDRegressor(loss=loss_model, penalty='l2', learning_rate='adaptive')
    LSTM = load_model(get_new_pkl(file_dir='../workplace/', model_for_what='LSTM'),
                      custom_objects={'r_square_oos': r_square_oos})
    # sgdr_long_term_revenue = joblib.load(filename=get_new_pkl(file_dir='../workplace/', model_for_what='1days_sgdr_long'))

    # csv文件的版本号
    index = 'V1_202206-0913year'
    # 起始日期
    start_date = datetime.datetime(2022, 6, 1)
    end_date = datetime.datetime(2022, 9, 13)
    # benchmark股票
    # bench_stock = '000001.XSHG'

    csv_name1 = '../data_stocks/' + index + '_stock_list.csv'  # 股票列表
    csv_name2 = '../data_stocks/' + index + '_final_data.csv'  # 股票数据
    csv_name3 = '../data_stocks/' + index + '_stock_valid.csv'  # 股票有效列表

    # 获取已清洗好的全A股列表
    stocklist_allA = pd.read_csv(csv_name1)
    stocklist_allA.fillna(0)
    stocklist_allA = stocklist_allA['0'].tolist()

    # 获取已清洗好的全A股所有数据
    df_all_0 = pd.read_csv(csv_name2)
    print(df_all_0)
    begin_time1 = time.time()
    df_all = treat_df(df_all_0)
    end_time1 = time.time()
    print('数据预处理共用时:',end_time1-begin_time1)
    print(df_all)
    df_all['datetime'] = pd.to_datetime(df_all['datetime'], format='%Y-%m-%d', errors='coerce')
    basic_information = ['code', 'datetime', 'open', 'close', 'high', 'low', 'volume', 'money', 'high_limit',
                         'low_limit', 'avg', 'pre_close']
    list_factor = df_all.columns.to_list()[12:]
    columns = basic_information + list_factor
    df_all.columns = columns



    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker = bt.brokers.BackBroker(shortcash=True)  # 0.5%的滑点
    # 考虑印花税和佣金印花税为0
    # comminfo = stampDutyCommissionScheme(stamp_duty=0.003, commission=0.0001)
    comminfo = stampDutyCommissionScheme(stamp_duty=0, commission=0)
    cerebro.broker.addcommissioninfo(comminfo)
    lines, params = add_parameters(columns[7:])
    print((lines), (params))


    begin_time2 = time.time()
    for s in stocklist_allA:
        feed = Addmoredata(dataname=get_stock_data(s, list_factor, df_all), plot=False,
                           fromdate=start_date, todate=end_date)
        # #
        cerebro.adddata(feed, s)
    # 多进程加速

    # feed_list = []
    # pool = multiprocessing.Pool(7)  # 创建一个7个进程的进程池
    # for s in split_list(stocklist_allA,7):
    #     feed_list.append(pool.apply_async(func=add_data, args=(s, list_factor, start_date, end_date, df_all,)))
    #
    # pool.close()
    # pool.join()
    #
    # for result in tqdm(feed_list):
    #     part_res = result.get()
    #     print(1)
    #     print(len(part_res[0]))
    #     for i in range(len(part_res[0])):
    #         cerebro.adddata(part_res[0][i],part_res[1][i])



    end_time2 = time.time()
    print('读取数据用时：', end_time2 - begin_time2)

    cerebro.broker.setcash(2000000.0)
    # 防止下单时现金不够被拒绝。只在执行时检查现金够不够。
    cerebro.broker.set_checksubmit(True)
    # 添加相应的费用，杠杆率
    # 获取策略运行的指标
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    startcash = cerebro.broker.getvalue()
    cerebro.addstrategy(momentum_factor_strategy)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.Trades)
    cerebro.addobserver(bt.observers.TimeReturn)

    # 添加Analyzer
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        timeframe=bt.TimeFrame.Days,
        riskfreerate=0.02,
        annualize=True,
        _name='sharp_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    thestrats = cerebro.run()
    thestrat = thestrats[0]
    # 输出分析器结果字典
    # 保存模型
    save_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    filename = "../workplace/1day_LSTM_" + loss_model + '_' + save_time + '.h5'  # 未注明具体调仓周期者，调仓周期为2天
    # filename2 = "../workplace/3days_sgdr_long_" + loss_model + '_' + save_time + '.pkl'
    LSTM.save(filename)
    # joblib.dump(sgdr_long_term_revenue, filename2)

    print('Sharpe Ratio:', thestrat.analyzers.sharp_ratio.get_analysis())
    print('DrawDown:', thestrat.analyzers.drawdown.get_analysis())

    # 进一步从字典中取出需要的值
    print('Sharpe Ratio:',
          thestrat.analyzers.sharp_ratio.get_analysis()['sharperatio'])
    print('Max DrawDown:',
          thestrat.analyzers.drawdown.get_analysis()['max']['drawdown'])

    # 打印各个分析器内容
    for a in thestrat.analyzers:
        a.print()
    cerebro.plot()
    # 获取回测结束后的总资金
    portvalue = cerebro.broker.getvalue()
    pnl = portvalue - startcash
    # 打印结果
    print(f'总资金: {round(portvalue, 2)}')
    print(f'净收益: {round(pnl, 2)}')
    print('收益率：{:.2%}'.format(portvalue / startcash - 1))
    end_time = time.time()
    print("一共使用时间为:{}".format(end_time - begin_time))
