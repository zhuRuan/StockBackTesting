from joinquant_securities import get_base_data
from stock_macro_merge import merge_data
from sw1_industry_factors_by_industry import get_industry_data
from stock_index_classification import stock_classify

if __name__ == '__main__':
    start = '2022-08-06'
    end = '2022-09-16'
    index = 'V1_' + start + '-' + end
    stock_list = 'all_sec'

    #获得带基础因子的数据
    base_csv = get_base_data(start, end, stock_list)
    #获取申万行业因子数据
    sw_csv = get_industry_data(start, end)
    #申万数据与基础因子合并
    after_merge_csv = merge_data(sw_csv,base_csv, index)
    #股票指数分类
    stock_classify(after_merge_csv, index)