import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn import preprocessing
from tensorflow import keras

sns.set()


def load_data():
    """
    返回读取的文件中的数据，并进行归一化
    """
    path = os.path.join(sys.path[0], 'Data')
    path_data = os.path.join(path, 'all_samples_net.npy')

    data_ori = np.load(path_data)
    X_ori = data_ori[:, :2].copy()
    Y_ori = data_ori[:, 2:].copy()
    x_scale = preprocessing.MinMaxScaler()
    y_scale = preprocessing.MinMaxScaler()
    X = x_scale.fit_transform(X_ori)
    Y = y_scale.fit_transform(Y_ori)

    data = {}
    data['X'] = X
    data['Y'] = Y

    return data

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
    model = keras.Model(inputs,predictions)
    model.summary()
    return model

def plot_history(history):
    """
    画出训练之后得到的结果
    """
    plt.figure()  
    plt.xlabel('Epoch')  
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label = 'Val loss')
    plt.legend()


def get_gradients(model,trainingExample):
    """
    得到模型在trainingExample对输入的梯度
    """
    outputTensor = model.output
    gradients = tf.gradients(outputTensor, model.input)
    sess = keras.backend.get_session()
    evaluated_gradients = sess.run(gradients,feed_dict={model.input:trainingExample})
    return evaluated_gradients

if __name__ == '__main__':
    data = load_data()
    X,Y = data['X'],data['Y']

    train = 1
    if train:
        model = build_model(X.shape[1], Y.shape[1])
        model.summary()
        keras.utils.plot_model(model, to_file=os.path.join(sys.path[0], 'Net/model2.png'))
        # 设置训练方式
        model.compile(tf.train.RMSPropOptimizer(0.01), loss=tf.losses.mean_squared_error, metrics=['mae'])

        # 进行训练
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)  # 20代测试误差没有改进就退出
        history = model.fit(X, Y, batch_size=1024, epochs=1000, validation_split=0.1, callbacks=[early_stop])
        model.save(os.path.join(sys.path[0], 'Net/model2.h5'))
        # 训练结果展示
        plot_history(history)
        plt.show()
    else:
        model = keras.models.load_model(os.path.join(sys.path[0], 'Net/model2.h5'))
        model.summary()
        for _ in range(100):
            print(get_gradients(model,X))