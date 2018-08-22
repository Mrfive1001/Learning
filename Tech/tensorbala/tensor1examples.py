import tensorflow as tf

# 常量的例子
mat1 = tf.constant([3., 3.], name="mat1")
mat2 = tf.constant([4., 4.], name="mat2")

s = mat1 + mat2
with tf.Session() as sess:
    print(sess.run(s))
# 变量的例子 一定要初始化
state = tf.Variable(initial_value=[1, 1])
update = tf.assign(state, state + tf.ones_like(state))
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(3):
        print(sess.run([state, update]))  # fetch
# placeholder占位符的例子 feed
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = input1 * input2
with tf.Session() as session:
    result_feed = session.run(output, feed_dict={input1: [2.,1], input2: [3.,3]})
    print("result:", result_feed)
