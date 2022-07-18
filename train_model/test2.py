#导入数据读取器
from sklearn.datasets import load_boston
boston=load_boston()
#输出数据描述

print (boston.DESCR)

#数据分割
from sklearn.model_selection import train_test_split
import numpy as np
X=boston.data
y=boston.target

#随机25%的测试样本数据，其他为训练样本数据
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)
#分析回归目标值的差异
print ('差异')
print ('The max target value is:',np.max(boston.target))
print ('The min target value is:',np.min(boston.target))
print ('The avg target value is:',np.mean(boston.target))

#数据的标准化处理
from sklearn.preprocessing import StandardScaler
ss_X=StandardScaler()
ss_y=StandardScaler()

X_train=ss_X.fit_transform(X_train)
X_test=ss_X.fit_transform(X_test)
y_train=ss_y.fit_transform(y_train.reshape(-1,1))
y_test=ss_y.fit_transform(y_test.reshape(-1,1))

#LR模型的预测分析
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
#参数估计
lr.fit(X_train, y_train)
#预测分析
lr_y_predict=lr.predict(X_test)


#SGD模型
from sklearn.linear_model import SGDRegressor
sgdr=SGDRegressor(max_iter=5)
sgdr.fit(X_train, y_train.ravel())
sgdr_y_predict=sgdr.predict(X_test)

#性能评估
print ('LR:',lr.score(X_test,y_test))
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print ('R-squared:',r2_score(y_test, lr_y_predict))
print ('mean_squared:',mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print ('mean_absolute:',mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))

print ('SGDC:',sgdr.score(X_test,y_test))
print ('R-squared:',r2_score(y_test, sgdr_y_predict))
print ('mean_squared:',mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))
print ('mean_absolute:',mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))