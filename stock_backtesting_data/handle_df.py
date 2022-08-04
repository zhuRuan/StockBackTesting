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
    df = df3[~df3['pre_close'].isin([0,np.NAN])]
    df = df[~df['close'].isin([0,np.NAN])]
    del df['is_st']
    print(df.shape)
    # 计算收益率
    df['earn_rate'] = df.apply(lambda x: x['close'] / x['pre_close'] - 1, axis=1)
    # df = df[df['earn_rate'] < 0.1]
    # df = df[df['earn_rate'] > -0.1]
    df = df[df.columns[14:]]
    print(df.columns.to_list())
    for column in df.columns.to_list()[:len(df.columns.to_list())-1]:
        df.loc[df[column].isin([np.nan]).index.to_list(),column] = df[column].mean()
        df[column] = mad(df[column])

    return df
