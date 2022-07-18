# %%
from stock_backtesting_data.handle_df import treat_df

import datetime

import joblib
import pandas as pd
# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, \
    explained_variance_score


def r_square_oos(y_true, y_pred):
    SSR = K.sum(K.square(y_true - y_pred))
    SST = K.sum(K.square(y_true - K.mean(y_true)))
    return SSR / SST


# 定义series_to_supervised()函数
# 将时间序列转换为监督学习问题
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    print(data.shape)
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def mad(factor):
    me = np.median(factor)
    mad = np.median(abs(factor - me))
    # 求出3倍中位数的上下限制
    up = me + (3 * 1.4826 * mad)
    down = me - (3 * 1.4826 * mad)
    # 利用3倍中位数的值去极值
    factor = np.where(factor > up, up, factor)
    factor = np.where(factor < down, down, factor)
    return factor





# %%

csv_name2 = r'D:\Ruiwen\PythonProject\StockBackTesting\data_stocks\V1_2015-202207year_final_data.csv'  # 股票数据
csv_name3 = r'D:\Ruiwen\PythonProject\StockBackTesting\data_stocks\V1_2020year01-12_stock_data.csv'
df_all1 = (pd.read_csv(csv_name2))
df_all1['date_parsed'] = pd.to_datetime(df_all1['datetime'],format='%Y-%m-%d')
df_all1 = df_all1[df_all1['date_parsed']<datetime.datetime.strptime('2022-01-01','%Y-%m-%d')]
del df_all1['date_parsed']

# %%
df_all = treat_df(df_all1)
# df_all2 = treat_df(pd.read_csv(csv_name3))
# df_all = pd.concat([df_all, df_all2])

# %%
plt.hist(x=df_all['earn_rate'],  # 指定绘图数据
         bins=100,  # 指定直方图中条块的个数
         color='steelblue',  # 指定直方图的填充色
         edgecolor='black'  # 指定直方图的边框色
         )
plt.xlabel('earn_rate')
plt.title('收益率分布')
plt.show()
# %%


# 获取DataFrame中的数据，形式为数组array形式
values = df_all.values

# 确保所有数据为float类型
values = values.astype('float32')
print(values.shape)

# 将时间序列转换为监督学习问题
reframed = series_to_supervised(values, 1, 1)
print(reframed.shape)
# 删除不想预测的特征列，这里只预测收益率
drop_list = []
for i in range(len(df_all.columns.to_list()), len(df_all.columns.to_list()) * 2 - 1):
    drop_list.append(i)
reframed.drop(reframed.columns[drop_list], axis=1, inplace=True)
# 打印数据的前5行
# print(reframed)
print(reframed.shape)
# %%
# 划分训练集和测试集
values = reframed.values
X = values[:, :-1]
y = values[:, -1]

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)

# = test[:, :-1], test[:, -1]
test_y_real = test_y

# 特征的归一化处理
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
# scaler_x = StandardScaler()
# scaler_y = StandardScaler()
train_X = scaler_x.fit_transform(train_X)
train_y = scaler_y.fit_transform(train_y.reshape(-1, 1))
test_X = scaler_x.transform(test_X)
test_y = scaler_y.transform(test_y.reshape(-1, 1))

# %%
plt.hist(x=test_y,  # 指定绘图数据
         bins=100,  # 指定直方图中条块的个数
         color='steelblue',  # 指定直方图的填充色
         edgecolor='black'  # 指定直方图的边框色
         )
plt.xlabel('earn_rate')
plt.title('收益率分布')
plt.show()

# %%
# 实例化模型'
print(train_X.shape)
print(train_y.shape)
loss_model = 'huber'
sgdr = SGDRegressor(loss=loss_model, penalty='l2', learning_rate='adaptive')
# sgdr = SVR(kernel="rbf", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
# sgdr = SVR()
# %%

sgdr.fit(train_X, train_y)
# %%
y_pred = sgdr.predict(test_X)
# print(y_test)
# print(y_pred)
# mean_squared_error
print('MSE为：', mean_squared_error(test_y, y_pred))
print('RMSE为：', np.sqrt(mean_squared_error(test_y, y_pred)))

# median_absolute_error
print('median_absolute_error为： ', median_absolute_error(test_y, y_pred))

# mean_absolute_error
print('mean_absolute_error为： ', mean_absolute_error(test_y, y_pred))

# explained_variance_score
print('explained_variance_score为： ', explained_variance_score(test_y, y_pred))
# print(1 - np.var(y_test - y_pred) / np.var(y_test))

# r2_score
print('r2为： ', r2_score(test_y, y_pred))
# print(1 - (np.sum((y_test - y_pred) ** 2)) / np.sum((y_test - np.mean(y_test)) ** 2))


# %%
y_test = scaler_y.inverse_transform(test_y)
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
print(y_test)
print(y_pred)
print('r2为： ', r2_score(y_test, y_pred))
save_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
filename = "../workplace/1days_sgdr_" + loss_model + '_' + save_time + '.pkl'
joblib.dump(sgdr, filename)
print('done')
