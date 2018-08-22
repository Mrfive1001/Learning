import numpy as np
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt

# tensorboard使用

# 创建文件夹
log_dir = os.path.join(sys.path[0], 'log')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# 准备训练数据
n_train_samples = 200
X_train = np.linspace(-5, 5, n_train_samples)[:, np.newaxis]
Y_train = 1.2 * X_train + np.random.uniform(-1, 1, n_train_samples)[:, np.newaxis]

# 准备测试数据
n_test_samples = 200
X_test = np.linspace(-5, 5, n_test_samples)[:, np.newaxis]
Y_test = 1.2 * X_test + np.random.uniform(-1, 1, n_test_samples)[:, np.newaxis]

# 超参数设计
learning_rate = 0.01
batch_size = 20
summary_dir = log_dir
print('开始设计计算图')

with tf.name_scope('Input'):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Y')

with tf.name_scope('Net'):
    l1 = tf.layers.dense(X, 20, activation=tf.nn.relu, name='l1')
    l2 = tf.layers.dense(l1, 20, activation=tf.nn.relu, name='l2')
    Y_pre = tf.layers.dense(l2, 1, name='Y_pre')

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.square(Y_pre - Y), name='loss')
    tf.summary.scalar('loss', loss)

with tf.name_scope('Optimization'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
# 汇总记录节点
merge = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    # 将图写进文件
    summary_writer = tf.summary.FileWriter(logdir=summary_dir, graph=sess.graph)
    for i in range(201):
        # 开始训练
        j = np.random.randint(0, 10)  # 总共200训练数据，分十份，从中选一份出来训练
        X_batch = X_train[batch_size * j: batch_size * (j + 1)]
        Y_batch = Y_train[batch_size * j: batch_size * (j + 1)]
        # 进行一次训练
        _, summary, train_loss = sess.run([optimizer, merge, loss], feed_dict={X: X_batch, Y: Y_batch})
        test_loss = sess.run(loss, feed_dict={X: X_test, Y: Y_test})
        # 将所有日志写入文件
        summary_writer.add_summary(summary, global_step=i)
        if i == 200:
            # plot the results
            Y_train_pre = sess.run(Y_pre, feed_dict={X: X_train})
            Y_test_pre = sess.run(Y_pre, feed_dict={X: X_test})
            plt.plot(X_train, Y_train, 'bo', label='Train data')
            plt.plot(X_test, Y_test, 'gx', label='Test data')
            plt.plot(X_train, Y_train_pre, 'r', label='Predicted Train data')
            plt.plot(X_test, Y_test_pre + 1, 'y', label='Predicted Test data')
            plt.legend()
            plt.show()
    summary_writer.close()
