import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing, svm
import tensorflow as tf
from method import MountainCarIndirect
import gym

from dnn import DNN

'''
构建神经网络来预测协态变量和时间
'''


def load_data():
    path = os.path.join(sys.path[0], 'Data')
    path_data = os.path.join(path, 'all_samples_original.npy')
    return np.load(path_data)


def check_data(data):
    """
    分析数据的分布
    """
    # x数据的分布
    fig1 = plt.figure()
    sns.distplot(data[:, 0], kde=False)
    # 某一维与x,y的关系
    fig2 = plt.figure()
    ax = Axes3D(fig2)
    x, y, z = data[:, 0], data[:, 1], data[:, -1]
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #  将数据点分成三部分画，在颜色上有区分度
    ax.scatter(x, y, z, c='y')  # 绘制数据点
    plt.show()


if __name__ == '__main__':
    # 1 读取数据
    data_ori = load_data()
    # 2 检查数据分布是否正确
    # check_data(data_ori)
    # 3 数据预处理
    X_ori = data_ori[:, :2].copy()
    Y_ori = data_ori[:, 2:].copy()
    s_dim = len(X_ori[0])
    a_dim = len(Y_ori[0])
    size = len(X_ori)

    x_scale = preprocessing.MinMaxScaler()
    y_scale = preprocessing.MinMaxScaler()
    X = x_scale.fit_transform(X_ori)
    Y = y_scale.fit_transform(Y_ori)

    data = np.hstack((X, Y))
    # 4 训练或者进行测试网络
    train = 0
    g1 = tf.Graph()
    net = DNN(s_dim, a_dim, 100, train, name='Coor',graph = g1, out_activation='tanh')
    if train:
        net.learn_data(data,train_epi=10000,batch_size=4096)
        net.store_net()
    else:
        Y_pre = y_scale.inverse_transform(net.predict(X))
        data_pre = np.hstack((X_ori,Y_pre))
    env = MountainCarIndirect()
    # 5 使用预测得到的结果进行打靶 使用自己神经网络的系统
    for i in range(10):
        print('Epi: %d'%(i+1))
        observation = env.reset()
        coor = y_scale.inverse_transform(net.predict(x_scale.transform(observation.reshape((-1, s_dim))))).reshape((-1))
        env.verity_cor(observation,coor)
    plt.show()
    # 6 使用神经网络进行数据测试
    car = env.env.env
    for _ in range(10):
        observation = car.reset()
        coor = y_scale.inverse_transform(net.predict(x_scale.transform(observation.reshape((-1, s_dim))))).reshape((-1))
        while True:
            car.render()
            action,coor = env.choose_action(coor,observation)
            observation,_,done,info = car.step(action)
            print(observation)
            if done:
                break




