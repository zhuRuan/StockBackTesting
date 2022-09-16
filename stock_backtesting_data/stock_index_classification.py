from jqdatasdk import *
import pandas as pd
import numpy as np
from data.data_size_handle import reduce_mem_usage

def stock_classify (csv):
    auth('18620290503', 'gxqh2019')
    # df=pd.read_csv('../data_stocks/V1_202206-0913year_stocks_and_industry_data.csv')
    df = pd.read_csv(csv)
    df, nan_list = reduce_mem_usage(df)
    print(nan_list)
    valid_stocks = df['Unnamed: 0'].to_list() #将股票转为list形式储存
    index_pool_list = ['000001.XSHG','000300.XSHG','399001.XSHE','399006.XSHE','399101.XSHE','399102.XSHE','399106.XSHE'] #想要获取的指数池
    #上证指数,沪深300,深证成指,创业板指,中小综指,创业板综合指数,深证综指


    dict = {}
    for index in index_pool_list:
        TrueorFalse = []# 新建储存股票所属股票池列表
        stocks_in_index = get_index_stocks(index) #获取特定指数股票池中所含股票信息
        for stock in valid_stocks:
            if stock in stocks_in_index:
                TrueorFalse.append(1)
            else:
                TrueorFalse.append(0)

        dict['index'+str(index.split('.')[0])]= TrueorFalse
        df1= pd.DataFrame(dict)
        #df1 = df.loc[:, ~df1.columns.str.contains('^Unnamed')]
        df = pd.concat([df,df1],axis=1)
        dict={}

    index = 'V1_202206-0913year'
    final = '../data_stocks/' +index+ '_final_data.csv'
    df.to_csv(final)
    return final