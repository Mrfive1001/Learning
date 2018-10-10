import tensorflow as tf
tf.enable_eager_execution()

a = tf.constant(1)
b = tf.constant(1)
c = a+b
print(c)

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)
print(C)

x = tf.get_variable('x', shape=[1], initializer=tf.constant_initializer(3.))
with tf.GradientTape() as tape: # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求
    y = tf.square(x)
y_grad = tape.gradient(y, x) # 计算 y 关于 x 的导数
print([y.numpy(), y_grad.numpy()])
