import time

from jqdatasdk import *
import pandas as pd
from datetime import datetime, timedelta

'''
暂时未加入的因子有：
2022-07-16
动量因子V2：'MAC5','MAC10','MAC10','MAC20','MAC60','MAC120'

'''


# 获取所有股票每日因子
def load_factor_to_every_sec(all_sec, sec_info_list, start, end):
    start_time3 = time.time()
    df2 = sec_info_list
    df3 = load_stock_factors(all_sec, start, end, factors_list) #读取因子列表
    df2 = pd.merge(df2, df3, on=['datetime', 'code'], how='outer')
    df2.index = df.iloc[:, 1].values
    del df2['code']
    df2.to_csv(csv3, mode='a+')
    end_time3 = time.time()
    print("获取factor，并将factor与行情数据合并，使用时间为:{}".format(end_time3 - start_time3))
    return df2


# 获取指定日期区间的一支股票的因子值(可以是列表)：
def load_stock_factors(all_sec, start, end, factors):
    start_time4 = time.time()
    df_return = pd.DataFrame()
    for sec in all_sec:
        df = get_factor_values(securities=sec, factors=factors, start_date=start, end_date=end)
        factor_list = []
        code = sec
        new_df = pd.DataFrame()
        ST_or_not = get_extras('is_st', sec, start, end)
        ST_or_not.columns = ['is_st']
        for key, value in df.items():
            factor_list.append(key)
            value.columns = [key]
            new_df = pd.concat([new_df, value], axis=1)
        new_df = pd.concat([new_df, ST_or_not], axis=1)
        new_df['code'] = code
        new_df['datetime'] = new_df.index

        df_return = pd.concat([df_return, new_df])
    end_time4 = time.time()
    print("获取factor，使用时间为：:{}".format(end_time4 - start_time4))
    return df_return


# 遍历每个日期，并生成每日可交易标的列表
def load_stock_list_valid(csv_name):
    for day in trade_days:
        all_sec_valid = list(get_all_securities(types=['stock'], date=day).index)
        all_sec_valid_pd = pd.DataFrame(all_sec_valid)
        all_sec_valid_pd.columns = ['code']
        all_sec_valid_pd['Date'] = day
        all_sec_valid_pd.to_csv(csv_name, mode='a+')


if __name__ == '__main__':
    # 主程序开始
    begin_time = time.time()

    auth('18620290503', 'gxqh2019')
    # 起止日期
    start = '2020-01-01'
    end = '2020-12-31'
    date_start = datetime.strptime(start, '%Y-%m-%d')
    date_end = datetime.strptime(end, '%Y-%m-%d')
    # 版本号
    index = 'V1_2020year01-12'
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
    all_sec = list(get_all_securities(types=['stock']).index)
    all_sef_pd = pd.DataFrame(all_sec)
    all_sef_pd.to_csv(csv1)
    print('股票列表写入完毕')

    # 获得交易日期
    trade_days = get_trade_days(start_date=start, end_date=end)

    # 获得每日可用股票
    load_stock_list_valid(csv2)
    print('每日可用股票列表写入完毕')

    # 获得每日行情及当日的因子
    # 行情
    df = pd.DataFrame(get_price(all_sec, start_date=start, end_date=end,
                                fields=['open', 'close', 'high', 'low', 'volume', 'money',
                                        'high_limit', 'low_limit', 'avg', 'pre_close'],
                                fq='pre', skip_paused=False, frequency='daily', panel=False).values)
    df.columns = ['datetime', 'code', 'open', 'close', 'high', 'low', 'volume', 'money', 'high_limit', 'low_limit',
                  'avg', 'pre_close']
    df.fillna(0, inplace=True)
    # 因子
    df_new = load_factor_to_every_sec(all_sec, df, start, end)
    end_time = time.time()
    print("一共使用时间为:{}".format(end_time - begin_time))
