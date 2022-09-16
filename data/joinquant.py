from jqdatasdk import *
import pandas as pd
from datetime import datetime, timedelta

auth('18620290503', 'gxqh2019')

print(get_query_count())

# all_sec = list(get_all_securities(types=['stock']).index)
# print(all_sec)
#
# def load_price(all_sec, start, end):
#     print(start)
#     df = pd.DataFrame(get_price(all_sec, start_date=start, end_date=end, fields=['open', 'close', 'high', 'low', 'factor', 'volume', 'money']).values)
#     df.columns = ['date', 'code', 'open', 'close', 'high', 'low', 'factor', 'volume', 'money']
#     df.index = df.iloc[:, 1].values
#     del df['code']
#     csv_name = '../data_futures/' + str(start)
#     df.to_csv()
#
# all_date = []
#
# start = '2022-01-01'
# end = '2022-08-21'
#
# datestart = datetime.strptime(start, '%Y-%m-%d')
# dateend = datetime.strptime(end, '%Y-%m-%d')
#
# while datestart < dateend:
#     datestart += timedelta(days=1)
#     all_date.append(datestart.strftime('%Y-%m-%d'))
# print(all_date)
#
# for day in all_date:
#     load_price(all_sec, start=day, end=day)
