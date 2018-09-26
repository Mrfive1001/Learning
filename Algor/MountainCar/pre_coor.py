import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

from dnn import DNN

'''
构建神经网络来预测协态变量和时间
'''

# 读取数据
path = os.path.join(sys.path[0], 'Data')
path_data = os.path.join(path, 'all_samples_original.npy')
data_ori = np.load(path_data)
# 使用X和Y，将其归一化
X_ori = data_ori[:, :2].copy()
Y_ori = data_ori[:, 2:].copy()
s_dim = len(X_ori[0])
a_dim = len(Y_ori[0])
size = len(X_ori)
x_scale = preprocessing.MinMaxScaler()
y_scale = preprocessing.MinMaxScaler()
X = x_scale.fit_transform(X_ori)
Y = y_scale.fit_transform(Y_ori)
data = np.hstack((X,Y))

# 开始训练
train = 0

net = DNN(s_dim,a_dim,200,train,name='coor')
if train:
    net.learn_data(data,train_epi=10000,batch_size=4096)
    net.store_net()
else:
    Y_pre = y_scale.inverse_transform(net.predict(X))
    data_pre = np.hstack((X_ori,Y_pre))
