import pandas as pd
pd.set_option('display.notebook_repr_html', False)

#读取xlsx（相对路径）
df = pd.read_excel(io='../stock_list/0710 股票推薦.xlsx', sheet_name=None)

df['股票代碼']

