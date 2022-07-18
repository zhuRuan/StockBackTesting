# 加载需要的库
import datetime
import os
import time

import backtrader as bt
import joblib
import numpy as np
import pandas as pd
from backtrader.feeds import PandasData
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import SGDRegressor

'''
获得某一股票的全部数据
输入：code--该股票对应的ts_code
输出：df_stock--该股票的全部数据，存为df
'''


def get_stock_data(code, factor_list):
    df_stock = df_all[df_all['code'] == code]
    df_stock = df_stock[['datetime', 'open', 'close', 'high', 'low', 'volume', 'money', 'high_limit',
                         'low_limit', 'avg', 'pre_close'] + factor_list]
    df_stock.index = df_stock.datetime
    df_stock = df_stock.sort_index()
    return df_stock


# 修改原数据加载模块，以便能够加载更多自定义的因子数据
class Addmoredata(PandasData):
    lines = ('money', 'high_limit', 'low_limit', 'avg', 'pre_close',
             'gross_income_ratio', 'net_profit_to_total_operate_revenue_ttm', 'ROAEBITTTM', 'roe_ttm', 'roa_ttm',
             'eps_ttm', 'total_asset_turnover_rate', 'quick_ratio', 'leverage', 'super_quick_ratio',
             'net_operate_cash_flow_to_net_debt', 'net_operate_cash_flow_to_total_liability', 'total_asset_growth_rate',
             'operating_revenue_growth_rate', 'net_profit_growth_rate', 'np_parent_company_owners_growth_rate',
             'cash_to_current_liability', 'cash_flow_to_price_ratio', 'margin_stability', 'DEGM_8y',
             'roic_ttm', 'BBIC', 'BIAS5', 'BIAS10', 'BIAS20', 'BIAS60', 'CCI88', 'CCI20',
             'fifty_two_week_close_rank', 'momentum', 'VOL5', 'DAVOL5', 'VOL10', 'VOL20', 'VOL60', 'VOL120',
             'VOL240', 'BR', 'WVAD', 'VSTD10', 'size', 'beta', 'ROC6', 'ROC12', 'ROC20', 'ROC60', 'ROC120',
             'VOSC', 'Price1M', 'Price3M', 'Price1Y', 'residual_volatility', 'MASS', 'debt_to_tangible_equity_ratio',
             'debt_to_equity_ratio', 'equity_to_fixed_asset_ratio', 'equity_to_asset_ratio', 'sales_to_price_ratio',
             'book_to_price_ratio', 'PEG', 'cfo_to_ev', 'ROAEBITTTM', 'net_operate_cash_flow_to_asset', 'HY001',
             'HY002', 'HY003', 'HY004', 'HY005', 'HY006', 'HY007', 'HY008', 'HY009', 'HY010', 'HY011', 'is_st')
    params = (('money', 6), ('high_limit', 7), ('low_limit', 8), ('avg', 9), ('pre_close', 10),
              ('gross_income_ratio', -1), ('net_profit_to_total_operate_revenue_ttm', -1), ('ROAEBITTTM', -1),
              ('roe_ttm', -1), ('roa_ttm', -1), ('eps_ttm', -1), ('total_asset_turnover_rate', -1),
              ('quick_ratio', -1), ('leverage', -1), ('super_quick_ratio', -1),
              ('net_operate_cash_flow_to_net_debt', -1), ('net_operate_cash_flow_to_total_liability', -1),
              ('total_asset_growth_rate', -1), ('operating_revenue_growth_rate', -1), ('net_profit_growth_rate', -1),
              ('np_parent_company_owners_growth_rate', -1), ('cash_to_current_liability', -1),
              ('cash_flow_to_price_ratio', -1), ('margin_stability', -1), ('DEGM_8y', -1),
              ('roic_ttm', -1), ('BBIC', -1), ('BIAS5', -1), ('BIAS10', -1), ('BIAS20', -1),
              ('BIAS60', -1), ('CCI88', -1), ('CCI20', -1), ('fifty_two_week_close_rank', -1),
              ('momentum', -1), ('VOL5', -1), ('DAVOL5', -1), ('VOL10', -1), ('VOL20', -1), ('VOL60', -1),
              ('VOL120', -1), ('VOL240', -1), ('BR', -1), ('WVAD', -1), ('VSTD10', -1), ('size', -1),
              ('beta', -1), ('ROC6', -1), ('ROC12', -1), ('ROC20', -1), ('ROC60', -1), ('ROC120', -1),
              ('VOSC', -1), ('Price1M', -1), ('Price3M', -1), ('Price1Y', -1), ('residual_volatility', -1),
              ('MASS', -1), ('debt_to_tangible_equity_ratio', -1), ('debt_to_equity_ratio', -1),
              ('equity_to_fixed_asset_ratio', -1), ('equity_to_asset_ratio', -1),
              ('sales_to_price_ratio', -1), ('book_to_price_ratio', -1),
              ('PEG', -1), ('cfo_to_ev', -1), ('ROAEBITTTM', -1), ('net_operate_cash_flow_to_asset', -1),
              ('HY001', -1), ('HY002', -1), ('HY003', -1), ('HY004', -1), ('HY005', -1), ('HY006', -1),
              ('HY007', -1), ('HY008', -1), ('HY009', -1), ('HY010', -1), ('HY011', -1), ('is_st', -1)
              )

    # 设置佣金和印花税率


class stampDutyCommissionScheme(bt.CommInfoBase):
    '''
    本佣金模式下，买入股票仅支付佣金，卖出股票支付佣金和印花税.    
    '''
    params = (
        ('stamp_duty', 0.001),  # 印花税率
        ('commission', 0.0005),  # 佣金率
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        '''
        If size is greater than 0, this indicates a long / buying of shares.
        If size is less than 0, it idicates a short / selling of shares.
        '''
        # print('self.p.commission', str(self.p.commission))
        if size > 0:  # 买入，不考虑印花税
            return size * price * self.p.commission * 100
        elif size < 0:  # 卖出，考虑印花税
            return - size * price * (self.p.stamp_duty + self.p.commission * 100)
        else:
            return 0

        # 编写策略


class momentum_factor_strategy(bt.Strategy):
    # interval-换仓间隔，stocknum-持仓股票数
    params = (("interval", 1), ("stocknum", 30))

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('{}, {}'.format(dt.isoformat(), txt))

    def __init__(self):

        # 已清洗过的每日可用股票列表
        self.df_byday = pd.read_csv(csv_name3)
        # 交易天数，用于判断是否交易
        self.bar_num = 0
        # 上次交易股票的列表
        self.last = []
        # 记录以往订单，在调仓日要全部取消未成交的订单
        self.order_list = []

        # 记录现有的持仓
        self.stock_position_list = []

    def prenext(self):

        self.next()

    def next(self):
        # 记录交易日期
        self.bar_num += 1
        # 获取当前账户价值
        total_value = self.broker.getvalue()
        print("当天日期:{}".format(str(self.datas[0].datetime.date(0))))
        self.log('当前总市值 %.2f。' % (total_value))
        self.log('当前总收益率 {:.2%}'.format((self.broker.getvalue() / startcash) - 1))
        # 计算当日是否调仓
        if self.bar_num % self.p.interval == 0 and self.bar_num > 1 * self.p.interval and self.datas[0].datetime.date(
                0) < datetime.date(2022, 7, 8):
            # 得到当天的时间
            current_date = self.datas[0].datetime.date(0)
            print("交易日日期:{}".format(str(self.datas[0].datetime.date(0))))
            # 获得上一调仓日和下一调仓日时间
            prev_date = self.datas[0].datetime.date(-self.p.interval)
            # next_date = self.datas[0].datetime.date(self.p.interval)
            # next_next_date = self.datas[0].datetime.date(self.p.interval + 1)
            # 获取当日可行股票池
            stocklist = self.get_valid_list_day(current_date)
            # 获取上一调仓日可行股票池
            stocklist_p = self.get_valid_list_day(prev_date)
            # #获取下个调仓日可行股票池
            # stocklist_next = self.get_valid_list_day(next_date)
            # print(stocklist_next)
            # # 获取下下个调仓日可行股票池
            # stocklist_next_next = self.get_valid_list_day(next_next_date)

            # 计算本期因子数据df_fac，并清洗
            df_fac = self.get_df_fac(stocklist=stocklist, factor_list=list_factor, prev=0)
            df_fac = df_fac.dropna(axis=0, how='any')

            # 计算上期因子数据df_faxc_p，并清洗
            df_fac_p = self.get_df_fac(stocklist=stocklist_p, factor_list=list_factor, prev=1)
            df_fac_p = df_fac_p.dropna(axis=0, how='any')

            # 本期因子排列命名
            df_fac.columns = ['code', 'momentum_value'] + list_factor
            df_fac.index = df_fac.code.values

            # 上期因子排列命名
            df_fac_p.columns = ['code', 'momentum_value'] + list_factor

            df_fac_p.index = df_fac_p.code.values

            # 舍弃X_p和Y中不同的index（股票代码）
            # 先去除X_p比Y多的index
            diffIndex = df_fac_p.index.difference(df_fac.index)
            # 删除整行
            df_fac_p = df_fac_p.drop(diffIndex, errors='ignore')
            df_fac = df_fac.drop(diffIndex, errors='ignore')

            # 然后去除Y比X_p多的index
            np.set_printoptions(suppress=True)
            diffIndex = df_fac.index.difference(df_fac_p.index)
            df_fac_p = df_fac_p.drop(diffIndex, errors='ignore')
            df_fac = df_fac.drop(diffIndex, errors='ignore')
            # X_p是上一期的因子值，X是本期因子值，Y是回归目标
            X_p = df_fac_p[['momentum_value'] + list_factor[:len(list_factor)-1]]  # 过去因子数据
            X = df_fac[['momentum_value'] + list_factor[:len(list_factor)-1]]  # 当前因子数据
            Y = df_fac[['momentum_value']]
            Y_long = df_fac[['Price1M']]

            # 将因子值与Y值均进行标准化
            rbX = StandardScaler()
            robust_x_train = rbX.fit_transform(X_p)
            robust_x_predict = rbX.fit_transform(X)
            rbY = StandardScaler()
            rbY2 = StandardScaler()
            Y_after_transform = rbY.fit_transform(Y)
            Y_long_after_transform = rbY2.fit_transform(Y_long)
            np.set_printoptions(suppress=True)
            print('robust转换后的数据Y')
            print((Y))
            print((Y_long))
            # 用上期因子值与本期回报率进行训练、
            sgdr.partial_fit(X=robust_x_train, y=Y_after_transform)
            sgdr_long_term_revenue.partial_fit(X=robust_x_train, y=Y_long_after_transform)

            # 用本期因子值预测下期回报率
            sgdr_pred = sgdr.predict(robust_x_train)
            sgdr_long_pred = sgdr_long_term_revenue.predict(robust_x_train)
            a4 = rbY.inverse_transform(sgdr_pred.reshape(-1, 1))
            a5 = rbY2.inverse_transform(sgdr_long_pred.reshape(-1, 1))
            print("输入训练的数据直接进行预测，预测：")
            print(sgdr_pred)
            print(sgdr_long_pred)
            print('数据预测归一化还原后:')
            print(a4)
            print(a5)

            sgdr_pred = sgdr.predict(robust_x_predict)
            sgdr_long_pred = sgdr_long_term_revenue.predict(robust_x_predict)
            print('预测未来的数据逆归一化后:')
            print(sgdr_pred)
            print(sgdr_long_pred)

            a = rbY.inverse_transform(sgdr_pred.reshape(-1, 1))
            a2 = rbY2.inverse_transform(sgdr_long_pred.reshape(-1, 1))
            df_fac['pred'] = a
            df_fac['pred_long'] = a2
            print(a)
            print(a2)

            # 按照预测得到的下期收益进行排序
            df_fac.sort_values(by="pred", inplace=True, ascending=False)
            # 取预测收益>0且排序靠前的stocknum只股票做多
            df_fac_pos = df_fac[df_fac['pred'] > 0]
            df_fac_long_pos_pred = df_fac[df_fac['pred_long'] > 0]
            df_fac_long_pos = df_fac[df_fac['Price1M'] > 0]
            sort_list_pos = df_fac_pos['code'].tolist()
            positive_long_term_list = df_fac_long_pos['code'].to_list()
            positive_long_term_list_next = df_fac_long_pos_pred['code'].to_list()
            # 根据预测收益率线性分配仓位
            long_list = sort_list_pos[:self.p.stocknum]  # 收益大于0的股票列表的排名在交易数量之内的股票列表
            long_list_yield_total = df_fac_pos['pred'].tolist()  # 收益大于0的所有股票的预测值
            long_list_yield = df_fac_pos['pred'].tolist()[:self.p.stocknum]  # 收益率大于0的所有股票的预测值中排名靠前的
            position_list = []
            sum_yield = np.sum(long_list_yield)
            sum_yield_total = np.sum(long_list_yield_total)
            for stock_yield, stock in zip(long_list_yield, long_list):

                percent = 0.1 * (0.3 * stock_yield / sum_yield_total + 0.7 * stock_yield / 0.1)  # 想要设定的仓位
                stock_position = self.getposition(data=stock).size
                portfolio_value = self.broker.getvalue()
                percent_now = stock_position / portfolio_value  # 获取当前的仓位
                if percent_now + percent >= 0.2:
                    percent = 0
                position_list.append(percent)
            # print("long_list:")
            # print(long_list)

            # #取预测收益<0且排序靠后的stocknum只股票做空
            # df_fac_neg = df_fac[df_fac['pred']<0]
            # sort_list_neg = df_fac_neg['code'].tolist()
            # short_list=sort_list_neg[-self.p.stocknum:]
            # # print("short_list:")
            # # # print(short_list)

            # 取消以往所下订单（已成交的不会起作用）
            for o in self.order_list:
                self.cancel(o)
            # 重置订单列表
            self.order_list = []

            # 若上期交易股票未出现在本期交易列表中，则平仓
            # 即将退市的股票也清仓
            for i in self.stock_position_list:
                if (
                        i not in sort_list_pos and i not in positive_long_term_list and i not in positive_long_term_list_next):
                    self.stock_position_list.remove(i)
                    d = self.getdatabyname(i)
                    if self.getposition(d).size > 0:
                        print('sell 平仓', d._name, self.getposition(d).size)
                        o = self.close(data=d)
                        self.order_list.append(o)  # 记录订单
                # elif i not in stocklist_next or i not in stocklist_next_next:
                #     d = self.getdatabyname(i)
                #     if self.getposition(d).size > 0 :
                #         print('sell 强制平仓', d._name, self.getposition(d).size)
                #         o = self.close(data=d)
                #         self.order_list.append(o)  # 记录订单

            # 对long_list中股票做多
            if len(long_list):

                # 依次买入
                for d, position_of_stock in zip(long_list, position_list):
                    data = self.getdatabyname(d)
                    # 若不在上期持仓列表中，则加入列表
                    if d not in self.stock_position_list:
                        self.stock_position_list.append(d)
                    # 按次日开盘价计算下单量，下单量是100的整数倍
                    target_value = position_of_stock * total_value
                    size1 = int(abs(target_value / data.open[0] // 100 * 100))
                    o = self.order_target_size(data=d, target=size1)  # 留百分之五用于支付费用
                    # 记录订单
                    self.order_list.append(o)

    # 交易日志
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Value: %.2f, Comm %.2f, Size: %.2f, Stock: %s' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm,
                     order.executed.size,
                     order.data._name))
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Value: %.2f, Comm %.2f, Size: %.2f, Stock: %s' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          order.executed.size,
                          order.data._name))

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log('TRADE PROFIT, GROSS %.2f, NET %.2f' %
                     (trade.pnl, trade.pnlcomm))

    # 求个股某周期因子平均值，prev为是否为前一周期（0：否，1：是）
    def get_df_fac(self, stocklist, factor_list, prev=0):

        # 新建df_fac用于存储计算得到的因子值
        column_list = ['code', 'momentum_value'] + factor_list
        df_fac = pd.DataFrame(columns= column_list)
        for stock in stocklist:
            data = self.getdatabyname(stock)
            #声明了一个用于存储 一定区间内所有因子的 list 具体结构为 [[],[],...,[]]
            factor_MA_list = []
            for factor in factor_list:
                factor_list2 = []
                factor_MA_list.append(factor_list2)
            # 获取当期因子值得平均数
            for i in range(0, self.p.interval):
                for factor_list1, factor in zip(factor_MA_list, factor_list):
                    str = 'factor_list1.append(data.' + factor + '[-i - prev * self.p.interval])'
                    exec(str)
                #st无法取均值，单独获取
            factor_MA_list[len(factor_MA_list) - 1] = []
            factor_MA_list[len(factor_MA_list) - 1].append(data.is_st[0])
            # 计算当期动量
            try:
                sell_ = data.open[0 - prev * self.p.interval]  # 若prev = 0，则返回当前时点的开盘价
                buy_ = data.open[-1 - prev * self.p.interval]
                stock_momentum = sell_ / buy_ - 1
            except IndexError:
                stock_momentum = np.nan
            except ValueError:
                stock_momentum = np.nan
            except ZeroDivisionError:
                stock_momentum = np.nan
            #最后返回的具体数值列表
            factor_MA_number_list = []
            factor_MA_number_list.append(stock)
            factor_MA_number_list.append(stock_momentum)
            for factor_list1 in factor_MA_list:
                factor_MA_number_list.append(np.mean(factor_list1))
            new = pd.DataFrame([factor_MA_number_list], index=[1], columns= column_list)
            df_fac = pd.concat([df_fac, new], ignore_index=True)

        return df_fac

    # 获取当日可行股票池
    def get_valid_list_day(self, current_date):
        self.df_byday['Date'] = pd.to_datetime(self.df_byday['Date'], errors='coerce', format='%Y-%m-%d')
        current_date = datetime.datetime.strptime(str(current_date), '%Y-%m-%d')
        df_day = self.df_byday[self.df_byday['Date'] == current_date]
        stocklist = list(df_day["code"])
        # stocklist = literal_eval(df_day['stocklist'].tolist()[0])
        return stocklist


def get_new_pkl(file_dir, model_for_what):
    file_lists = os.listdir(file_dir)
    file_dict = {}
    for i in file_lists:  # 遍历所有文件
        if model_for_what in i:
            ctime = os.stat(os.path.join(file_dir, i)).st_ctime
            file_dict[ctime] = i  # 添加创建时间和文件名到字典
    max_ctime = max(file_dict.keys())  # 取值最大的时间
    print("已读取最新模型参数文件： ", file_dict[max_ctime])
    return os.path.join(file_dir, file_dict[max_ctime])  # 打印出最新文件名


if __name__ == '__main__':
    ##########################
    # 主程序开始
    ##########################
    begin_time = time.time()
    # 实例化模型'
    loss_model = 'squared_error'

    sgdr = SGDRegressor(loss=loss_model, penalty='l2', learning_rate='adaptive')
    sgdr_long_term_revenue = SGDRegressor(loss=loss_model, penalty='l2', learning_rate='adaptive')
    # sgdr = joblib.load(filename=get_new_pkl(file_dir='../workplace/', model_for_what='1days_sgdr_squared_error'))
    # sgdr_long_term_revenue = joblib.load(filename=get_new_pkl(file_dir='../workplace/', model_for_what='1days_sgdr_long'))

    # csv文件的版本号
    index = 'V1_2022year07-07'
    # 起始日期
    start_date = datetime.datetime(2022, 7, 1)
    end_date = datetime.datetime(2022, 7, 8)
    # benchmark股票
    # bench_stock = '000001.XSHG'

    csv_name1 = '../data_stocks/' + index + '_stock_list.csv'  # 股票列表
    csv_name2 = '../data_stocks/' + index + '_stock_data.csv'  # 股票数据
    csv_name3 = '../data_stocks/' + index + '_stock_valid.csv'  # 股票有效列表

    # 获取已清洗好的全A股列表
    stocklist_allA = pd.read_csv(csv_name1)
    stocklist_allA.fillna(0)
    stocklist_allA = stocklist_allA['0'].tolist()

    # 获取已清洗好的全A股所有数据
    df_all = pd.read_csv(csv_name2)
    df_all['datetime'] = pd.to_datetime(df_all['datetime'], format='%Y-%m-%d', errors='coerce')
    basic_information = ['code', 'datetime', 'open', 'close', 'high', 'low', 'volume', 'money', 'high_limit',
                         'low_limit', 'avg', 'pre_close']
    list_factor = df_all.columns.to_list()[12:]
    columns = basic_information + list_factor
    df_all.columns = columns

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker = bt.brokers.BackBroker(shortcash=True)  # 0.5%的滑点
    # 考虑印花税和佣金印花税为0
    comminfo = stampDutyCommissionScheme(stamp_duty=0.003, commission=0.0001)
    cerebro.broker.addcommissioninfo(comminfo)

    # # 多线程加速
    # pool = multiprocessing.Pool(10)
    for s in stocklist_allA:
        feed = Addmoredata(dataname=get_stock_data(s, factor_list=list_factor), plot=False,
                           fromdate=start_date, todate=end_date)
        # pool.apply_async(func=cerebro.adddata,args=[feed,s])
        cerebro.adddata(feed, name=s)
    # pool.close()
    # pool.join()

    cerebro.broker.setcash(2000000.0)
    # 防止下单时现金不够被拒绝。只在执行时检查现金够不够。
    cerebro.broker.set_checksubmit(False)
    # 添加相应的费用，杠杆率
    # 获取策略运行的指标
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    startcash = cerebro.broker.getvalue()
    cerebro.addstrategy(momentum_factor_strategy)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.Trades)
    cerebro.addobserver(bt.observers.TimeReturn)

    # 添加Analyzer
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        timeframe=bt.TimeFrame.Days,
        riskfreerate=0.02,
        annualize=True,
        _name='sharp_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    thestrats = cerebro.run()
    thestrat = thestrats[0]
    # 输出分析器结果字典
    # 保存模型
    save_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    filename = "../workplace/1days_sgdr_" + loss_model + '_' + save_time + '.pkl'  # 未注明具体调仓周期者，调仓周期为2天
    filename2 = "../workplace/1days_sgdr_long_" + loss_model + '_' + save_time + '.pkl'
    joblib.dump(sgdr, filename)
    joblib.dump(sgdr_long_term_revenue, filename2)

    print('Sharpe Ratio:', thestrat.analyzers.sharp_ratio.get_analysis())
    print('DrawDown:', thestrat.analyzers.drawdown.get_analysis())

    # 进一步从字典中取出需要的值
    print('Sharpe Ratio:',
          thestrat.analyzers.sharp_ratio.get_analysis()['sharperatio'])
    print('Max DrawDown:',
          thestrat.analyzers.drawdown.get_analysis()['max']['drawdown'])

    # 打印各个分析器内容
    for a in thestrat.analyzers:
        a.print()
    cerebro.plot()
    # 获取回测结束后的总资金
    portvalue = cerebro.broker.getvalue()
    pnl = portvalue - startcash
    # 打印结果
    print(f'总资金: {round(portvalue, 2)}')
    print(f'净收益: {round(pnl, 2)}')
    print('收益率：{:.2%}'.format(portvalue / startcash - 1))
    end_time = time.time()
    print("一共使用时间为:{}".format(end_time - begin_time))
