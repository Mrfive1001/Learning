import math

import numpy as np
import tensorflow as tf
from rbm import RBM
from tensorflow.examples.tutorials.mnist import input_data

class NN(object):
    '''
    定义预测的神经网络
    '''
    def __init__(self, sizes, X, Y):
        # 初始化超参数
        self._sizes = sizes
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        self._learning_rate =  1.0
        self._momentum = 0.0
        self._epoches = 10
        self._batchsize = 100
        input_size = X.shape[1]

        # 初始循环
        for size in self._sizes + [Y.shape[1]]:
            # 定义上限
            max_range = 4 * math.sqrt(6. / (input_size + size))

            # 通过随机分布来初始化
            self.w_list.append(np.random.uniform( -max_range, max_range, [input_size, size]).astype(np.float32))

            # 初始化偏置
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size

    def load_from_rbms(self, dbn_sizes,rbm_list):
        # 冲rbms中读取数据
        for i in range(len(self._sizes)):
            self.w_list[i] = rbm_list[i].w
            self.b_list[i] = rbm_list[i].hb

    def train(self):
        # 开始训练
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y = tf.placeholder("float", [None, self._Y.shape[1]])

        # 定义变量和训练误差
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])

        # 定义损失函数
        cost = tf.reduce_mean(tf.square(_a[-1] - y))

        train_op = tf.train.MomentumOptimizer(self._learning_rate, self._momentum).minimize(cost)

        # 预测操作
        predict_op = tf.argmax(_a[-1], 1)

        # 训练循环
        with tf.Session() as sess:
            #Initialize Variables
            sess.run(tf.global_variables_initializer())

            # 开始训练
            for i in range(self._epoches):
                for start, end in zip(range(0, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):
                    sess.run(train_op, feed_dict={_a[0]: self._X[start:end], y: self._Y[start:end]})
                for j in range(len(self._sizes) + 1):
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])
                print("Accuracy rating for epoch " + str(i) + ": " + str(np.mean(np.argmax(self._Y, axis=1) ==
                              sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y}))))


# 读取数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# 创建3个RBM模型
RBM_hidden_sizes = [500, 200, 50]
inpX = trX
# 保存模型
rbm_list = []

# 输入数量
input_size = inpX.shape[1]

# 对RBM模型开始训练
print('Pre_train begins!')
for i, size in enumerate(RBM_hidden_sizes):
    print('RBM: ', i, ' ', input_size, '->', size)
    rbm_list.append(RBM(input_size, size))
    input_size = size
for rbm in rbm_list:
    print('New RBM:')
    rbm.train(inpX)
    inpX = rbm.rbm_outpt(inpX)
print('Train begins!')
nNet = NN(RBM_hidden_sizes, trX, trY)
nNet.load_from_rbms(RBM_hidden_sizes,rbm_list)
nNet.train()