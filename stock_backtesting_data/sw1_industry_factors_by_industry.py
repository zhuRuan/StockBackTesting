# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 16:44:14 2022

@author: CHEN JINGHUI
"""
import numpy as np
from jqdatasdk import *
from jqdatasdk import finance
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

def get_industry_data (start, end):
    auth('18620290503', 'gxqh2019')
    #申万一级行业因子
    SW1_industry = ['801010','801030','801040','801050','801080','801110',
    '801120','801130','801140','801150','801160','801170','801180','801200','801210',
    '801230','801710','801720','801730','801740','801750','801760','801770','801780',
    '801790','801880','801890']


    new_df = pd.DataFrame()
    code = ''
    all_industry_valuation = pd.DataFrame()
    all_industry_valuation['date'] = np.NaN

    # start = '2022-6-1'
    # end = '2022-9-13'
    # 版本号
    index = 'V1_' + start + '-' + end
    date_start = datetime.strptime(start, '%Y-%m-%d')
    date_end = datetime.strptime(end, '%Y-%m-%d')

    #将所有申万行业因子存入数据库
    for i in tqdm(SW1_industry):
        df=finance.run_query(query(finance.SW1_DAILY_VALUATION).filter(finance.SW1_DAILY_VALUATION.code==i,
        finance.SW1_DAILY_VALUATION.date >= start,finance.SW1_DAILY_VALUATION.date < end, ))
        code = i
        #new_df = pd.concat([new_df,df['date']],axis=1)
        new_df = pd.concat([new_df,df['date'],df.iloc[:,4:12]],axis=1)
        new_df.columns = ['date','turnover_ratio'+i,'pe'+i,'pb'+i,'average_price'+i,'money_ratio'+i,'circulating_market_cap'+i,
                            'average_circulating_market_cap'+i,'dividend_ratio'+i]
        #new_df.columns = new_col
        new_df['date'] = pd.to_datetime(new_df['date'])
        all_industry_valuation['date'] = pd.to_datetime(all_industry_valuation['date'])
        all_industry_valuation = pd.merge(all_industry_valuation,new_df,on='date',how='outer')
        new_df = pd.DataFrame()


    sw_macro_data = '../data_stocks/' +index+ '_sw1_macro_industry_data.csv'
    all_industry_valuation.to_csv(sw_macro_data)
    return sw_macro_data

