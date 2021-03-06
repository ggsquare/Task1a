import csv
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
import pandas as pd
from math import exp
from sklearn.metrics import mean_squared_error

Id_train = pd.read_csv('train.csv',usecols=[0])
y = pd.read_csv('train.csv',usecols=[1])
X = pd.read_csv('train.csv',usecols=[2, 3, 4, 5, 6])

x1=pd.read_csv('train.csv',usecols=[2])
x2=pd.read_csv('train.csv',usecols=[3])
x3=pd.read_csv('train.csv',usecols=[4])
x4=pd.read_csv('train.csv',usecols=[5])
x5=pd.read_csv('train.csv',usecols=[6])

y=np.array(y)
x1=np.array(x1)
x2=np.array(x2)
x3=np.array(x3)
x4=np.array(x4)
x5=np.array(x5)

RMSE=[]
avgRMSE=[]

phi21=[1]*900
phi21=np.array(phi21)

X= np.column_stack((x1, x2, x3, x4, x5, x1**2, x2**2, x3**2, x4**2, x5**2, np.exp(x1), np.exp(x2), np.exp(x3), np.exp(x4), np.exp(x5),np.cos(x1),np.cos(x2),np.cos(x3),np.cos(x4),np.cos(x5), phi21))

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X):
    linReg = linear_model.Lasso()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Train the model using the training sets
    linReg.fit(X_train, y_train)
    # Make predictions using the testing set
    y_pred = linReg.predict(X_test)

    # RMSE.append(mean_squared_error(y_pred, y_test) ** 0.5)
    # bestRMSE=np.argmin(RMSE)
    weigh_factor=linReg.coef_

d = {'weighing factor': weigh_factor}
output = pd.DataFrame(d)
output.to_csv('output.csv', index=False, header=False)