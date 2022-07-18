import pandas as pd

df = pd.read_csv('3002_2006_3577_2006.csv')

# total_product_list = df['upc'].to_list()  #总产品列表
# company_list = df['manufacturer']
# company_list = company_list.drop_duplicates() #去重，进而获得公司列表
# dict = {}
# for company in company_list.to_list():
#     new_df = df[df['manufacturer']==company] #获取dataframe中 'manufacturer'==company 的行
#     trueOrFalseList = []
#     product_list = new_df['upc'].to_list() #获取该公司的所有产品
#     for product in total_product_list:
#         if product in product_list:
#             trueOrFalseList.append(1)
#         else:
#             trueOrFalseList.append(0)
#     dict[company] = trueOrFalseList
# print(dict)