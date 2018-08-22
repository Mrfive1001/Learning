import numpy as np
import tensorflow as tf
# 常量 变量 占位符
# tensorflow Tensor常量不会改变，是个函数 constant(ones,zeros,linspace,range,random_uniform)类型是tensor
# 构建图的时候不会进行计算
print("build a graph")
a=tf.constant([[1,2],[3,4]])
b=tf.constant([[1,1],[0,1]])
print("a:",a)
print("b:",b)
print("type of a:",type(a))
c=tf.matmul(a,b)
print("c:",c)
# 创建会话
sess=tf.Session()
print("excuted in Session")
# 执行操作
result_c=sess.run(c)
print("result_c:\n",result_c)

# tensorflow Variable变量需要初始值，改变的时候使用assign(是个操作)
# 可以用来表示神经网络参数，存到变量里面
w=tf.Variable(initial_value=[[1,2],[3,4]],dtype=tf.float32)
x=tf.Variable(initial_value=[[1,1],[1,1]],dtype=tf.float32)
y=tf.matmul(w,x)
z=tf.sigmoid(y)
print(z)
# 必须有这个全局变量初始化的过程
init_op=tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init_op)
    z=session.run(z)
    print(z)
# tensorflow 占位符 placeholder(dtype,shape = ,)
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)
with tf.Session() as sess:
  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.