import numpy as np
import pandas as pd


def mad(factor):
    me = np.median(factor)
    mad = np.median(abs(factor - me))
    # 求出3倍中位数的上下限制
    up = me + (3 * 1.4826 * mad)
    down = me - (3 * 1.4826 * mad)
    # 利用3倍中位数的值去极值
    factor = np.where(factor > up, up, factor)
    factor = np.where(factor < down, down, factor)
    return factor


def treat_df(df3):
    # 命名
    columns = df3.columns
    # 缺失值处理
    print(df3.shape)
    df = df3[~df3['pre_close'].isin([0, np.NAN])]
    df = df[~df['close'].isin([0, np.NAN])]
    df = df[df['pre_close']>0]
    del df['is_st']
    print(df.shape)
    # 计算收益率
    df['earn_rate'] = df.apply(lambda x: x['close'] / x['pre_close'] - 1, axis=1)
    # df = df[df['earn_rate'] < 0.1]
    # df = df[df['earn_rate'] > -0.1]
    df = df[df.columns[14:]]
    print(df.columns.to_list())
    for column in df.columns.to_list()[:len(df.columns.to_list()) - 1]:
        print(df[df[column].isin([np.nan])][column])
        df[df[column].isin([np.nan])][column] = df[column].mean()
        df[column] = mad(df[column])
        print(df[df[column].isin([np.nan])][column])


    return df


def df_merge(df_list):
    new_df = pd.DataFrame()
    for df in df_list:
        new_df = pd.concat([new_df, pd.read_csv(df)])
    print(new_df)
    new_df.to_csv('../data_stocks/V1_2021_20220804_stock_data.csv')


# df_list = ['D:\Ruiwen\PythonProject\StockBackTesting\data_stocks\V1_202001-03year_stock_data.csv',
#            'D:\Ruiwen\PythonProject\StockBackTesting\data_stocks\V1_202004-09year_stock_data.csv',
#            'D:\Ruiwen\PythonProject\StockBackTesting\data_stocks\V1_202010-12year_stock_data.csv']
#
# df_merge(df_list)
# df = pd.read_csv(r'D:\Ruiwen\PythonProject\StockBackTesting\data_stocks\V1_2021-20220805year_final_data.csv')
# df['date_parsed'] = pd.to_datetime(df['datetime'],format='%Y-%m-%d')
# df_all1 = df[df['date_parsed']>=datetime.datetime.strptime('2022-04-01','%Y-%m-%d')]
# del df_all1['date_parsed']
# df_all1.to_csv('../data_stocks/V1_202204-20220804year_final_data.csv')