import csv
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import mean_squared_error

Id_train = pd.read_csv('train.csv',usecols=[0])
y = pd.read_csv('train.csv',usecols=[1])
X = pd.read_csv('train.csv',usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10])

y=np.array(y)
X=np.array(X)

Ridge_coeff=[0.1, 1, 10, 100, 1000]
kf = KFold(n_splits=10)
RMSE=[]
avgRMSE=[]

for i in range(0,5):
    lambdas=Ridge_coeff[i]
    for train_index, test_index in kf.split(X):
        ridge = linear_model.Ridge(alpha=lambdas)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train the model using the training sets
        ridge.fit(X_train, y_train)
        # Make predictions using the testing set
        y_pred = ridge.predict(X_test)

        RMSE.append(mean_squared_error(y_pred, y_test)**0.5)
    avgRMSE.append(np.mean(RMSE))

d = {'error': avgRMSE}
output = pd.DataFrame(d)
output.to_csv('output.csv', index=False, header=False)