# %%
import pandas as pd
from stock_backtesting_data.handle_df import treat_df
from data.data_size_handle import reduce_mem_usage
# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
import datetime
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras import backend as K
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from FactorTesting.BackTrader_Multifactors_stock import get_new_pkl

@tf.autograph.experimental.do_not_convert
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


# csv_name3 = r'D:\Ruiwen\PythonProject\StockBackTesting\data_stocks\V1_2020year01-12_stock_data.csv'
df_all0,nan_list = reduce_mem_usage(pd.read_csv('data_stocks/V1_pro_final/2022_all_stocks_04-08/V1_2021-20220805year_final_data.csv'))
# %%
df_all = treat_df(df_all0)
# df_all2 = treat_df(pd.read_csv(csv_name3))
# df_all = pd.concat([df_all, df_all2])
# %%
# print(df_all0)
print(len(df_all.columns.to_list()))
LSTM = load_model(get_new_pkl(file_dir='workplace/', model_for_what='LSTM'),
                      custom_objects={'r_square_oos': r_square_oos})

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
print(df_all[ 'earn_rate'])

# 获取DataFrame中的数据，形式为数组array形式
values = df_all.values

# 确保所有数据为float类型
# values = values.astype('float32')
print(values.shape)

# 将时间序列转换为监督学习问题
reframed = series_to_supervised(values, 1, 1)
# 删除不想预测的特征列，这里只预测收益率
drop_list = []
for i in range(len(df_all.columns.to_list()), len(df_all.columns.to_list()) * 2 - 1):
    drop_list.append(i)
reframed.drop(reframed.columns[drop_list], axis=1, inplace=True)
# 打印数据的前5行
print(reframed)
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
         bins=15,  # 指定直方图中条块的个数
         color='steelblue',  # 指定直方图的填充色
         edgecolor='black'  # 指定直方图的边框色
         )
plt.xlabel('earn_rate')
plt.title('收益率分布')
plt.show()
# %%
# 转化为三维数据
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

train_y = train_y.flatten()

test_y = test_y.flatten()

# %%
print(train_X.shape, train_y.shape)
print(test_X.shape, test_y.shape)
print(train_y)

# %%
# 搭建LSTM模型
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)  # 使用loss作为监测数据，轮数设置为10
model = Sequential()
model.add(LSTM(units=200, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True,
               kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'))
model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'))
model.add(LSTM(units=32, return_sequences=False, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'))
model.add(Dense(1, activation='relu'))
keras.optimizers.Adam(lr=0.005)
model.compile(loss=r_square_oos, optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=300, batch_size=10000, validation_data=(test_X, test_y),
                    verbose=1, shuffle=True, use_multiprocessing=True, workers=10,
                    callbacks=callback)

# %%
# 保存模型
save_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
filename = "workplace/1day_LSTM_" + 'r_square_oos' + '_' + save_time + '.h5'  # 未注明具体调仓周期者，调仓周期为2天
# filename2 = "../workplace/3days_sgdr_long_" + loss_model + '_' + save_time + '.pkl'
LSTM.save(filename)
# 绘制损失图
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('LSTM_600000.SH', fontsize='12')
plt.ylabel('loss', fontsize='10')
plt.xlabel('epoch', fontsize='10')
plt.legend()
plt.show()
# %%
# 模型预测收益率
# test_y = scaler_y.fit_transform(test_y.reshape(-1,1))
print(test_X.shape)
y_predict = model.predict(test_X)
# %%
print(y_predict)
print(y_predict.shape)
print(test_y)
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# %%
# invert scaling for forecast
inv_y_predict = scaler_y.inverse_transform(y_predict)

# invert scaling for actual
# 将真实结果按比例反归一化
test_y = test_y.reshape((len(test_y), 1))
# inv_y_train = concatenate((test_X[:, :72], test_y), axis=1)
# inv_y_train = scaler_y.inverse_transform(test_y)
inv_y = scaler_y.inverse_transform(test_y)
print('反归一化后的预测结果：', inv_y_predict)
print('反归一化的真实结果:', inv_y)
print('没有反归一化的真实结果：', test_y_real)

# %%
line1 = plt.plot(inv_y, color='red', label='Original' )
line2 = plt.plot(inv_y_predict, color='green', label='Predict')
plt.setp(line1, linewidth=0.2)
plt.setp(line2, linewidth=0.2)
plt.xlabel('the number of test data')
plt.ylabel('earn_rate')
plt.title('2016.3—2017.12')
plt.legend()
plt.show()
# %%
# 回归评价指标
# calculate MSE 均方误差
mse = mean_squared_error(inv_y, inv_y_predict)
# calculate RMSE 均方根误差
rmse = sqrt(mean_squared_error(inv_y, inv_y_predict))
# calculate MAE 平均绝对误差
mae = mean_absolute_error(inv_y, inv_y_predict)
# calculate R square
r_square = r2_score(inv_y, inv_y_predict)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)
print('R_square: %.6f' % r_square)
