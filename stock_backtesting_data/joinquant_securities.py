import time
import traceback
import datetime

import numpy as np
from jqdatasdk import *
from tqdm import tqdm
import pandas as pd
'''
暂时未加入的因子有：
2022-07-16
动量因子V2：'MAC5','MAC10','MAC10','MAC20','MAC60','MAC120'
'''


def get_price_liwidis(start, end, all_sec):
    start_date0 = start
    df = pd.DataFrame()
    for i in range(0, 10):  # 目前没有需求读取十年以上的数据，故此设定

        end_date0 = start_date0 + datetime.timedelta(days=364)
        if end_date0 > end:
            end_date0 = end
        print('当前访问的数据截止日期为： ', end_date0)
        df = pd.concat(
            [df,
             get_price(all_sec, start_date=start_date0.strftime('%Y-%m-%d'), end_date=end_date0.strftime('%Y-%m-%d'),
                       fields=['open', 'close', 'high', 'low', 'volume', 'money',
                               'high_limit', 'low_limit', 'avg', 'pre_close'],
                       fq='pre', skip_paused=False, frequency='daily', panel=False)])
        if end_date0 == end:  # 日期遍历结束，循坏结束
            print('行情数据遍历结束，即将开始遍历因子数据')
            break
        start_date0 = end_date0 + datetime.timedelta(days=1)  # 年份跨越到下一年
    return df


# 获取所有股票每日因子
def load_factor_to_every_sec(all_sec, sec_info_list, trade_day_list, factors_list, csv3):
    start_time3 = time.time()
    df2 = sec_info_list
    df3 = load_stock_factors(all_sec, factors_list, trade_day_list)  # 读取因子列表
    df2 = pd.merge(df2, df3, on=['datetime', 'code'], how='outer')
    df2.index = df2.iloc[:, 1].values
    del df2['code']
    df2.to_csv(csv3)
    end_time3 = time.time()
    print("获取factor，并将factor与行情数据合并，使用时间为:{}".format(end_time3 - start_time3))
    return df2


# 获取指定日期区间的一支股票的因子值(可以是列表)：
def load_stock_factors(all_sec, factors, trade_day_list):
    start_time4 = time.time()
    df_return = pd.DataFrame()
    time_cost = 40
    try:
        for day in tqdm(trade_day_list):
            start_time5 = datetime.datetime.now()

            print('正在查询的因子日期:', day,
                  '   |||   完成进度：{:.2%}'.format((np.where(trade_day_list == day)[0][0] + 1) / len(trade_day_list)),
                  '   |||   预计剩余时间：', (1 - (np.where(trade_day_list == day)[0][0]) / len(trade_day_list)) * len(
                    trade_day_list) * time_cost)

            dict_factors = get_factor_values(securities=all_sec, factors=factors[:35], start_date=day, end_date=day)
            dict_factors2 = get_factor_values(securities=all_sec, factors=factors[35:], start_date=day, end_date=day)
            dict_factors.update(dict_factors2)
            df_st = get_extras('is_st', all_sec, day, day, df=True)
            new_df = pd.DataFrame()
            for key, value in dict_factors.items():
                df2 = pd.DataFrame(value.values.T, index=value.columns, columns=[key])
                new_df = pd.concat([new_df, df2], axis=1)
            new_df = pd.concat([new_df, pd.DataFrame(df_st.values.T, index=df_st.columns, columns=['is_st'])], axis=1)
            new_df['code'] = new_df.index
            new_df['datetime'] = day
            df_return = pd.concat([df_return, new_df])
            end_time5 = datetime.datetime.now()
            time_cost = end_time5 - start_time5
    except Exception as ex:
        print("获取过程中出现错误:%s" % ex)
        traceback.print_exc()

    print(df_return)
    df_return['datetime'] = pd.to_datetime(df_return['datetime'])
    end_time4 = time.time()
    print("获取factor，使用时间为：:{}".format(end_time4 - start_time4))
    return df_return


# 遍历每个日期，并生成每日可交易标的列表，并写入csv
def load_stock_list_valid(csv_name, trade_days, all_sec):
    valid_sum = pd.DataFrame(columns=['code', 'Date'])
    for day in trade_days:
        all_sec_valid = list(get_all_securities(types=['stock'], date=day).index)
        all_sec_valid_pd = pd.DataFrame(all_sec_valid)
        all_sec_valid_pd.columns = ['code']
        all_sec_valid_pd['Date'] = day
        valid_sum = pd.concat([valid_sum, all_sec_valid_pd], ignore_index=True)
    valid_sum = valid_sum[valid_sum['code'].isin(all_sec)]
    valid_sum.to_csv(csv_name)

'''
start:
开始日期，格式为'YYYY-M-D'
end:
结束日期，格式为'YYY-M-D'
stock_list: 
'all_sec'代表所有股票；
其他情况需传入完整股票名称list
'''
def get_base_data(start, end, stock_list):
    # 主程序开始
    begin_time = time.time()

    auth('18620290503', 'gxqh2019')
    # 起止日期
    start = '2022-6-1'
    end = '2022-9-13'
    date_start = datetime.datetime.strptime(start, '%Y-%m-%d')
    date_end = datetime.datetime.strptime(end, '%Y-%m-%d')
    # 版本号
    index = 'V1_' + start + '-' + end
    # 写入地址
    csv1 = '../data_stocks/' + index + '_stock_list.csv'
    csv2 = '../data_stocks/' + index + '_stock_valid.csv'
    csv3 = '../data_stocks/' + index + '_stock_data.csv'  # 所有行情
    # 因子列表
    factors_list = [
        # 基本面因子——V1
        # 盈利能力
        # 毛利率，净利润与营业总收入之比，总资产税前报酬率，权益回报率TTM,资产回报率TTM
        'gross_income_ratio', 'net_profit_to_total_operate_revenue_ttm', 'ROAEBITTTM', 'roe_ttm', 'roa_ttm',
        # 股东获利能力
        # 每股收益TTM
        'eps_ttm',
        # 运营能力
        # 总资产周转率
        'total_asset_turnover_rate',
        # 偿付能力
        # 速动比率,杠杆因子,超速动比率,经营活动产生现金流量净额/净债务,经营活动产生的现金流量净额/负债合计
        'quick_ratio', 'leverage', 'super_quick_ratio', 'net_operate_cash_flow_to_net_debt',
        'net_operate_cash_flow_to_total_liability',
        # 成长能力（同比成长）
        # 总资产增长率,营业收入增长率,净利润增长率,归属母公司股东的净利润增长率
        'total_asset_growth_rate', 'operating_revenue_growth_rate', 'net_profit_growth_rate',
        'np_parent_company_owners_growth_rate',
        # 现金流量能力
        # 现金比率,现金流市值比
        'cash_to_current_liability', 'cash_flow_to_price_ratio',
        # 风险因子
        # 盈利能力稳定性,长期毛利率增长,投资资本回报率TTM
        'margin_stability', 'DEGM_8y', 'roic_ttm',

        # 技术指标
        # 动量因子
        # BBI 动量,5日乖离率,10日乖离率,20日乖离率,60日乖离率,88日顺势指标,20日顺势指标,当前价格处于过去1年股价的位置,
        # 动量:描述了过去两年里相对强势的股票与弱势股票之间的差异
        'BBIC', 'BIAS5', 'BIAS10', 'BIAS20', 'BIAS60', 'CCI88', 'CCI20', 'fifty_two_week_close_rank', 'momentum',
        # # 动量因子V_2 20220716更新
        # #5日移动均线,	10日移动均线,	20日移动均线,60日移动均线,120日移动均线
        # 'MAC5','MAC10','MAC10','MAC20','MAC60','MAC120',
        # 换手率因子
        # 5日平均换手率,5日平均换手率与120日平均换手率:5日平均换手率 / 120日平均换手率,	10日平均换手率,20日平均换手率,60日平均换手率,120日平均换手率,240日平均换手率,意愿指标,威廉变异离散量,10日成交量标准差
        'VOL5', 'DAVOL5', 'VOL10', 'VOL20', 'VOL60', 'VOL120', 'VOL240', 'BR', 'WVAD', 'VSTD10',
        # 市值因子
        # 对数市值
        'size',
        # Beta因子
        # beta
        'beta',
        # 反趋向因子
        # 6日变动速率（Price Rate of Change）,12日变动速率,20日变动速率,60日变动速率,120日变动速率（Price Rate of Change）,成交量震荡
        'ROC6', 'ROC12', 'ROC20', 'ROC60', 'ROC120', 'VOSC',
        # 收益率因子
        # 当前股价除以过去一个月股价均值再减1,当前股价除以过去三个月股价均值再减1,...
        'Price1M', 'Price3M', 'Price1Y',
        # 波动率因子
        # 残差波动率因子,梅斯线
        'residual_volatility', 'MASS',

        # 估值因子
        # 有形净值债务率,产权比率,股东权益与固定资产比率,股东权益比率,营收市值比,账面市值比,市盈率相对盈利增长比率,经营活动产生的现金流量净额与企业价值之比TTM,总资产报酬率,总资产现金回收率
        'debt_to_tangible_equity_ratio', 'debt_to_equity_ratio', 'equity_to_fixed_asset_ratio', 'equity_to_asset_ratio',
        'sales_to_price_ratio', 'book_to_price_ratio', 'PEG', 'cfo_to_ev',
        'ROAEBITTTM', 'net_operate_cash_flow_to_asset',

        # 行业因子-V1
        # 能源指数,材料指数,工业指数,可选消费指数,日常消费指数,医疗保健指数,金融指数,信息技术指数,电信服务指数,公用事业指数,房地产指数,石油天然气设备与服务指数,综合性石油天然气指数,石油天然气勘探与生产指数
        # 石油与天然气的炼制和销售指数,基础化工指数,多元化工指数...
        'HY001', 'HY002', 'HY003', 'HY004', 'HY005', 'HY006', 'HY007', 'HY008', 'HY009', 'HY010', 'HY011',
    ]

    # 打印当前剩余多少条请求
    feedback = str(get_query_count())
    print('当前账户余额：' + feedback)

    # 获取并写入csv股票列表
    all_sec = stock_list
    if stock_list == 'all_sec':
        # 若传参为高级参数
        all_sec = list(get_all_securities(types=['stock']).index)

    # all_sec = list(get_index_stocks('000300.XSHG')) #+ list(get_index_stocks('399011.XSHE'))
    # all_sec = ['000568.XSHE', '000858.XSHE', '600519.XSHG', '600809.XSHG', '600702.XSHG', '000799.XSHE', '603919.XSHG', '002304.XSHE', '603589.XSHG', '000596.XSHE', '002077.XSHE', '688234.XSHG', '605111.XSHG', '688262.XSHG', '002079.XSHE', '688689.XSHG', '688037.XSHG', '603290.XSHG', '603893.XSHG', '300458.XSHE', '688711.XSHG', '688521.XSHG', '688595.XSHG', '300831.XSHE', '002371.XSHE', '300623.XSHE', '300046.XSHE', '688110.XSHG', '600460.XSHG', '688270.XSHG', '600360.XSHG', '600584.XSHG', '300604.XSHE', '688385.XSHG', '002049.XSHE', '300672.XSHE', '003026.XSHE', '600171.XSHG', '688167.XSHG', '600483.XSHG', '003816.XSHE', '600023.XSHG', '600167.XSHG', '600905.XSHG', '601016.XSHG', '601619.XSHG', '000875.XSHE', '000883.XSHE', '600157.XSHG', '600795.XSHG', '000027.XSHE', '002015.XSHE', '002608.XSHE', '000155.XSHE', '600032.XSHG', '000591.XSHE', '600098.XSHG', '603693.XSHG', '600780.XSHG']
    all_sef_pd = pd.DataFrame(all_sec)
    all_sef_pd.to_csv(csv1)
    print('股票列表写入完毕')

    # 获得交易日期
    trade_days = get_trade_days(start_date=start, end_date=end)

    # 获得每日可用股票
    load_stock_list_valid(csv2, trade_days, all_sec)
    print('每日可用股票列表写入完毕')

    # 获得每日行情及当日的因子
    # 行情
    df = get_price_liwidis(date_start, date_end, all_sec)
    df.columns = ['datetime', 'code', 'open', 'close', 'high', 'low', 'volume', 'money', 'high_limit', 'low_limit',
                  'avg', 'pre_close']
    # 因子
    df_new = load_factor_to_every_sec(all_sec, df, trade_days, factors_list, csv3)
    end_time = time.time()
    print("一共使用时间为:{}".format(end_time - begin_time))

    print('测试用数据：')
    pd.set_option('display.max_columns', None)  # 行

    test_list = ['000300.XSHG']
    df = get_price(test_list, start_date=start, end_date=end,
                   fields=['open', 'close', 'high', 'low', 'volume', 'money',
                           'high_limit', 'low_limit', 'avg', 'pre_close'],
                   fq='pre', skip_paused=False, frequency='daily', panel=False)

    print(df)
    df = get_factor_values(securities=test_list, factors=factors_list, start_date=start, end_date=start)
    print(df)

    return csv3
