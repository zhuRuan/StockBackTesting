from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn import datasets
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

import pandas as pd

index = 'V1_2022year07-07'
csv_name2 = '../data_stocks/' + index + '_stock_data.csv'  # 股票数据
df_all = pd.read_csv(csv_name2)
columns = df_all.columns
df = df_all[columns[2:len(df_all.columns)-1]]
print(df)
df = df.fillna(value=0)
y = df[['open']]
X = df[df.columns[1:]]
print(X.columns)
X.copy().fillna(-1).reset_index(drop=True)
y = y.reset_index(drop=True)
rbx = StandardScaler()
rby = StandardScaler()
X = rbx.fit_transform(X)
y = rby.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(X, y)

sgdr = SGDRegressor()
sgdr.fit(x_train,y_train)
y_pred = sgdr.predict(x_test)
print(y_test)
print(y_pred)
#mean_squared_error
print('MSE为：',mean_squared_error(y_test,y_pred))
print('MSE为(直接计算)：',np.mean((y_test-y_pred)**2))

print('RMSE为：',np.sqrt(mean_squared_error(y_test,y_pred)))

#median_absolute_error
print(np.median(np.abs(y_test-y_pred)))
print(median_absolute_error(y_test,y_pred))

#mean_absolute_error
print(np.mean(np.abs(y_test-y_pred)))
print(mean_absolute_error(y_test,y_pred))

#mean_squared_log_error
print(mean_squared_log_error(y_test,y_pred))
print(np.mean((np.log(y_test+1)-np.log(y_pred+1))**2))

#explained_variance_score
print(explained_variance_score(y_test,y_pred))
print(1-np.var(y_test-y_pred)/np.var(y_test))

#r2_score
print(r2_score(y_test,y_pred))
print(1-(np.sum((y_test-y_pred)**2))/np.sum((y_test -np.mean(y_test))**2))



