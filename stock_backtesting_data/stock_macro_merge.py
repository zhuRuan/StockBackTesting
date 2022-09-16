import pandas as pd
from datetime import datetime

#读取dataframe
from tqdm import tqdm

def merge_data(macro, stock_data):
    # df = pd.read_csv('../data_stocks/V1_202206-0913year_sw1_macro_industry_data.csv')
    # df2= pd.read_csv('../data_stocks/V1_202206-0913year_stock_data.csv')
    df = pd.read_csv(macro)
    df2 = pd.read_csv(stock_data)

    #删去无用列
    df = df.loc[:,~df.columns.str.contains("^Unnamed")]
    columns = df.columns.to_list()
    new_columns = []

    #修改列名
    for column in tqdm(columns):
        if column == 'date':
            new_columns.append('datetime')
        else:
            new_columns.append(column)

    df.columns = new_columns

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'].apply(lambda x : x.strftime('%Y-%m-%d'))

    #合并并存入数据
    df3= pd.merge(df2,df,on='datetime',how='outer')
    final_data = '../data_stocks/V1_202206-0913year_stocks_and_industry_data.csv'
    df3.to_csv(final_data)
    return final_data
