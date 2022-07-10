from jqdatasdk import *
import pandas as pd
from datetime import datetime, timedelta
csv1 = '../data_stocks/2_stock_data.csv' #所有行情
csv2 = '../data_stocks/2_stock_valid.csv'
csv3 = '../data_stocks/2_stock_list.csv'

def load_price(sec, start, end):


    df = pd.DataFrame(get_price(all_sec, start_date=start, end_date=end,
                                fields=['open', 'close', 'high', 'low',  'volume', 'money',
                                        'high_limit', 'low_limit', 'avg', 'pre_close'], fq='pre', skip_paused=False, frequency='daily', panel=False).values)
    df.columns = [ 'datetime','code','open', 'close', 'high', 'low', 'volume', 'money', 'high_limit', 'low_limit',
                  'avg', 'pre_close']

    df.fillna(0, inplace=True)
    df.index = df.iloc[:, 1].values
    del df['code']
    csv_name = csv1
    df.to_csv(csv_name, mode='a+')


auth('18620290503', 'gxqh2019')

#打印当前剩余多少条请求
print(get_query_count())
#获取股票列表
all_sec = list(get_all_securities(types=['stock']).index)
all_sef_pd = pd.DataFrame(all_sec)
all_sef_pd.to_csv(csv3)

all_date = []

start = '2022-03-01'
end = '2022-07-04'

datestart = datetime.strptime(start, '%Y-%m-%d')
dateend = datetime.strptime(end, '%Y-%m-%d')

load_price(all_sec, start=datestart, end=dateend)

trade_days = get_trade_days(start_date=start, end_date=end)

csv_name2 = csv2

#遍历每个日期，并生成每日可交易标的列表
for day in trade_days:
    all_sec_valid = list(get_all_securities(types=['stock'], date=day).index)
    all_sec_valid_pd = pd.DataFrame(all_sec_valid)
    all_sec_valid_pd.columns = [ 'code']
    all_sec_valid_pd['Date'] = day
    all_sec_valid_pd.to_csv(csv_name2, mode='a+')