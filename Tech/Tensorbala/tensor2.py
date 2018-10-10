import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 制造数据
num = 2000
train_num = int(0.7 * num)
X = np.random.rand(num)[:, np.newaxis]  # 增加一个新的维度
Y = np.sin(X * 20) + 0.05 * np.random.randn(num)[:, np.newaxis]

# 构建网络和存储变量位置
x = tf.placeholder(dtype='float32', shape=[None, 1], name='x')
y = tf.placeholder(dtype='float32', shape=[None, 1], name='y')
l1 = tf.layers.dense(x, 50, activation=tf.nn.relu)
l2 = tf.layers.dense(l1, 50, activation=tf.nn.relu)
l3 = tf.layers.dense(l2, 50, activation=tf.nn.relu)
l4 = tf.layers.dense(l3, 50, activation=tf.nn.relu)

y_pre = tf.layers.dense(l4, 1)  # 定义输出变量
loss = tf.reduce_mean(tf.square(y - y_pre))  # 定义损失函数
train = tf.train.RMSPropOptimizer(0.01).minimize(loss)  # 定义训练过程
init_op = tf.global_variables_initializer()  # 一定记得初始化

# 画出一个图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X, Y)
plt.ion()  # 实时输出
plt.show()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(10000):
        _, loss1 = sess.run([train, loss], feed_dict={x: X[:train_num], y: Y[:train_num]})
        if i % 500 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            sort_x = np.sort(X, axis=0)
            Y_pre = sess.run(y_pre, feed_dict={x: sort_x})
            lines = ax.plot(sort_x, Y_pre, 'r-', lw=1)
            plt.pause(0.1)
        print(i, loss1)
