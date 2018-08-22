import numpy as np
import pandas as pd

path = r'D:\文档\PycharmProjects\Kaggle\003'
data_train = pd.read_csv(path + r'HousePrices\train.csv')
data_test = pd.read_csv(path + r'HousePrices\test.csv')
col_len = len(data_train.columns) - 1
value_len_train = len(data_train)
value_len_test = len(data_test) 
col_left = []
for i in range(1, col_len):
    temp1 = data_train.iloc[:, i]
    temp2 = data_test.iloc[:, i]
    for (te, va) in zip((temp1, temp2), (value_len_train, value_len_test)):
        if te.count() != va or te.dtype == 'object':
            break
    else:
        col_left.append(i)
use_train = data_train.iloc[:, col_left].values.copy()
use_test = data_test.iloc[:, col_left].values.copy()
y_train = data_train.iloc[:, -1].values.copy()

