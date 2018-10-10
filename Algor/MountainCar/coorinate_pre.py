import os
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing, svm
from tensorflow import keras

from dnn import DNN
from method import MountainCarIndirect

'''
构建神经网络来预测协态变量和时间
'''
original = 1  # 是否使用原系统进行求解
data_name = 'all_samples_original.npy' if original else 'all_samples_net.npy'

def build_model(input_dim,out_dim,units_lists = None):
    """
    建立一个4层神经层的model
    """
    inputs = keras.Input(shape=(input_dim,))
    if units_lists is None:
        units_lists = [100,200,100,100]
    x = inputs
    for units in units_lists:
        x = keras.layers.Dense(units,activation='relu')(x)
    predictions = keras.layers.Dense(out_dim, activation='tanh')(x)
    # predictions_dot = tf.gradients(predictions,inputs)
    model = keras.Model(inputs,predictions)
    model.summary()
    return model

def load_data():
    path = os.path.join(sys.path[0], 'Data')
    path_data = os.path.join(path, data_name)
    return np.load(path_data)

def data_extract():
    """
    剔除原始数据不符合要求的序列
    :return:
    """
    data = load_data()
    time = data[:,-1]
    index_begin = 0
    new_data = None
    for index in range(1,len(data)):
        if abs(time[index]) <1e-10:
            data_temp = data[index_begin:index,:]
            index_begin = index
            if data_temp[:,0].min() < -1:
                continue
            if new_data is None:
                new_data = data_temp
            else:
                new_data = np.vstack((new_data,data_temp))
    path = os.path.join(sys.path[0], 'Data')
    path_data = os.path.join(path, data_name)
    np.save(path_data,new_data)
    return new_data

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

def plot_history(history):  
    plt.figure()  
    plt.xlabel('Epoch')  
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label = 'Val loss')
    plt.legend()


if __name__ == '__main__':
    # 1 读取数据
    data_ori = load_data()
    # 2 检查数据分布是否正确
    # check_data(data_ori)
    # 3 数据预处理
    X_ori = data_ori[:, :2].copy()
    Y_ori = data_ori[:, 2:].copy()

    x_scale = preprocessing.MinMaxScaler()
    y_scale = preprocessing.MinMaxScaler()
    X = x_scale.fit_transform(X_ori)
    Y = y_scale.fit_transform(Y_ori)
    env = MountainCarIndirect()
    # 4 建立模型
    train = 0
    if train:
        model = build_model(X.shape[1],Y.shape[1])
        model.summary()
        keras.utils.plot_model(model, to_file=os.path.join(sys.path[0],'DNN_Net/model.png'))
        # 设置训练方式
        model.compile(tf.train.RMSPropOptimizer(0.01),loss=tf.losses.mean_squared_error,metrics=['mae'])
    
        # 进行训练
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50) # 20代测试误差没有改进就退出
        history = model.fit(X,Y,batch_size=1024,epochs=1000,validation_split=0.1,callbacks = [early_stop])
        model.save(os.path.join(sys.path[0],'DNN_Net/model1.h5'))
        # 训练结果展示
        plot_history(history)
        plt.show()
    else:
        model = keras.models.load_model(os.path.join(sys.path[0],'DNN_Net/model1.h5'))    

    # # 5 使用预测得到的结果进行打靶 使用自己神经网络的系统
    # for i in range(10):
    #     print('Epi: %d'%(i+1))
    #     observation = env.reset()
    #     coor = y_scale.inverse_transform(model.predict(x_scale.transform(observation.reshape((-1, 2))))).reshape((-1))
    #     env.verity_cor(observation,coor)
    # plt.show()
    # 6 使用神经网络进行数据测试
    car = env.env.env
    for _ in range(10):
        observation = car.reset()
        time = 0
        observation_record = [observation]
        time_record = [time]
        coor = y_scale.inverse_transform(model.predict(x_scale.transform(observation.reshape((-1, 2))))).reshape((-1))
        while True:
            # car.render()

            action,_ = env.choose_action(coor,observation)
            observation,_,done,info = car.step(action)
            observation_record.append(observation)
            time_record.append(time)
            time += env.env.simulation_step
            coor = y_scale.inverse_transform(model.predict(x_scale.transform(observation.reshape((-1, 2))))).reshape((-1))
            print(observation)
            if done:
                break
    
        plt.figure(1)
        observation_record = np.array(observation_record)
        time_record = np.array(time_record)
        plt.plot(time_record, observation_record[:, 0], label='x_ture')
        plt.plot(time_record, 0.45 * np.ones(len(observation_record)), 'r')
        plt.xlabel('Time(s)')
        plt.ylabel('Xposition')
        plt.figure(2)
        plt.plot(time_record, observation_record[:, 1], label='v_ture')
        plt.xlabel('Time(s)')
        plt.ylabel('Vspeed')
    plt.show()
