import os
import sys

import numpy as np
import tensorflow as tf
from sklearn import preprocessing


class DNN:
    """
    定义神经网络并且获得网络结构
    输入维度、输出维度、单元数、是否训练、名字
    """

    def __init__(self, s_dim, a_dim, units, batch_size=100, memory_size=1000, train=0, name=None, graph=None):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.units = units
        self.train = train
        self.name = name
        self.graph = graph
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.pointer = 0
        self.memory = np.zeros(
            (self.memory_size, s_dim + a_dim), dtype=np.float32)  # 存储s,a
        # 保存网络位置
        self.model_path0 = os.path.join(sys.path[0], 'DNN_Net')
        if not os.path.exists(self.model_path0):
            os.mkdir(self.model_path0)
        if self.name is None:
            self.name = 'temp'
        self.model_path0 = os.path.join(self.model_path0, self.name)
        if not os.path.exists(self.model_path0):
            os.mkdir(self.model_path0)
        self.model_path = os.path.join(self.model_path0, 'data.chkp')
        if self.graph is None:
            self.graph = tf.get_default_graph()
        with self.graph.as_default():
            # 输入向量
            my_actiivation = tf.nn.tanh
            self.s = tf.placeholder(tf.float32, [None, s_dim], name='s')
            self.areal = tf.placeholder(tf.float32, [None, a_dim], 'areal')
            # 网络和输出向量
            net0 = tf.layers.dense(
                self.s, self.units, activation=my_actiivation, name='l0')
            net1 = tf.layers.dense(
                net0, self.units, activation=my_actiivation, name='l1')
            net2 = tf.layers.dense(
                net1, self.units, name='l2', activation=my_actiivation)
            net3 = tf.layers.dense(
                net2, self.units, name='l3', activation=my_actiivation)
            net4 = tf.layers.dense(
                net3, self.units, name='l4', activation=my_actiivation)
            net5 = tf.layers.dense(
                net4, self.units, name='l5', activation=my_actiivation)

            self.apre = tf.layers.dense(net5, self.a_dim, name='apre')  # 输出线性
            self.get_dot = tf.gradients(self.apre,self.s)
            self.mae = tf.reduce_mean(tf.abs(self.areal - self.apre))
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.areal, self.apre))  # loss函数
            self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)  # 训练函数
            # 保存或者读取网络
            if self.graph is not None:
                self.sess = tf.Session(graph=self.graph)
            else:
                self.sess = tf.Session(graph=self.graph)
            self.actor_saver = tf.train.Saver()
        if self.train == 1:
            self.a_scale = None
            self.sess.run(tf.global_variables_initializer())
        else:
            self.a_scale = preprocessing.MinMaxScaler(feature_range=(-1,1))
            memory = np.load(os.path.join(self.model_path0,'memory_init.npy'))
            self.a_scale.fit(memory[:,self.s_dim:])
            self.actor_saver.restore(self.sess, self.model_path)
        

    def learn(self):
        """
        随机选取记忆池中batch_size样本进行学习
        """
        if self.pointer < self.memory_size:
            # 未存储够足够的记忆池的容量
            return
        else:
            if self.a_scale is None:
                self.a_scale = preprocessing.MinMaxScaler(feature_range=(-1,1))
                self.a_scale.fit(self.memory[:,self.s_dim:])
                np.save(os.path.join(self.model_path0,'memory_init.npy'),self.memory)
            # 随机选取样本进行训练
            indexs = np.random.choice(self.memory_size, size=self.batch_size)
            samples = self.memory[indexs, :]
            X_samples = samples[:, :self.s_dim]
            Y_samples = samples[:, self.s_dim:]
            Y_samples = self.a_scale.transform(Y_samples)
            _, loss = self.sess.run([self.train_op, self.loss],
                                         feed_dict={self.s: X_samples, self.areal: Y_samples})
            return loss

    def store_sample(self, X, Y):
        """
        存储X,Y到记忆池
        """
        transition = np.hstack((X, Y))
        index = self.pointer % self.memory_size
        self.memory[index, :] = transition
        self.pointer += 1

    def store_net(self):
        """
        将网络进行存储
        """
        self.actor_saver.save(self.sess, self.model_path)

    def predict(self, X):
        """
        X维度是(n,s_dim)或者(s_dim,)
        进行预测
        :return (n,a_dim)
        """
        try:
            # 判断是否是二维向量
            _ = X.shape[1]
        except Exception:
            X = np.array(X).reshape((-1,self.s_dim))
        y = self.sess.run(self.apre, feed_dict={self.s: X})
        if self.a_scale:
            return self.a_scale.inverse_transform(y)
        else:
            return y

    def predict_dot(self,X):
        """
        X维度是(n,s_dim)或者(s_dim,)
        预测输出对X的导数
        :return array(dot)
        """
        try:
            # 判断是否是二维向量
            _ = X.shape[1]
        except Exception:
            X = np.array(X).reshape((-1,self.s_dim))
        dot = self.sess.run(self.get_dot, feed_dict={self.s: X})[0]
        if self.a_scale:
            return self.a_scale.inverse_transform(dot)
        else:
            return dot

if __name__ == '__main__':
    pass
