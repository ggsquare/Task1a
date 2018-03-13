import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

filename = '/Users/Anna/polybox/IntroductionML/Tasks/01/task1a_lm1d1z/train.csv'

file = pd.read_csv(filename, delimiter = ',')

X = file._drop_axis(['Id','y'], axis=1)
y = file['y']

lamda=[]
lamda.extend([0.1, 1, 10, 100, 1000])
avg = []
RMSE=[]

for l in range(0,5):

    clf = Ridge(lamda[l])
    kf = KFold(10)
    #i=1
    #kf.folds returns 10 folds each one of them containing two arrays -
    # one with the indices needed for the training set and one with the indices for the test set
    for train_index, test_index in kf.split(X): #trainindex: array of indexes of traindata,
        #print("Fold", i, ":")
        #print("TRAIN:", train_index,  "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        #i=i+1
        RMSE.append(mean_squared_error(y_pred, y_test)**0.5)
    avg.append(np.mean(RMSE))

# output results
d={'error': avg}
output=pd.DataFrame(d)
output.to_csv('task_1a_output.csv', index=False, header=False)
