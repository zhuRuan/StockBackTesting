from jqdatasdk import *
from jqdatasdk import finance
import pandas as pd
from datetime import datetime, timedelta



#申万一级行业因子
'''SW1_industry = ['801010','801020','801030','801040','801050','801060','801070','801080'
'801090','801100','801110','801120','801130','801140','801150','801160','801170','801180',
'801190','801200','801210','801220','801230','801710','801720','801730','801740','801750',
'801760','801770','801780','801790','801880','801890','801950','801960','801970','801980']'''

#获取每日申万行业估值因子
start = '2015-1-1'
end = '2022-7-29'
date_start = datetime.strptime(start, '%Y-%m-%d')
date_end = datetime.strptime(end, '%Y-%m-%d')
new_df = pd.DataFrame() 

for i in range(100000):
    df=finance.run_query(query(finance.SW1_DAILY_VALUATION).filter(finance.SW1_DAILY_VALUATION.date==date_start))
    new_df = pd.concat([new_df, df])
    date_start = date_start+timedelta(days=1)
    if date_start == date_end:
        break

#写入csv    
index = 'V1_2020year01-12'
csv4 = '../data_stocks/' +index+ 'sw_valuation.csv'    
new_df.to_csv(csv4)

