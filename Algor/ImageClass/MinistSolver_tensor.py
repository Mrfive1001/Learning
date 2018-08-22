from MinistSolve_keras import load_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
# 读取数据



def main():
    data = load_data()
    x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
    original_shape = data['original_shape']

    net_type = 2
    import tensorflow as tf
    if net_type == 1:
        # 利用tensorflow构建普通网络
        # tensorflow的输入shape是样本数量作为None的所有维度
        input = tf.placeholder(dtype = 'float32',shape=[None,x_train.shape[1]],name='x')
        out = tf.placeholder(dtype='float32',shape=[None,10],name='y')
        l1 = tf.layers.dense(input, 50, activation=tf.nn.relu)
        l2 = tf.layers.dense(l1, 50, activation=tf.nn.relu)
        l3 = tf.layers.dense(l2, 50, activation=tf.nn.relu)
        l4 = tf.layers.dense(l3, 50, activation=tf.nn.relu)
        out_pre = tf.layers.dense(l4,10,activation=tf.nn.softmax)
        

    else:
        # 利用tensorflow构建卷积网络
        x_train = x_train.reshape(len(x_train), *original_shape, 1)
        x_test = x_test.reshape(len(x_test), *original_shape, 1)

        input = tf.placeholder(dtype = 'float32',shape=[None,*original_shape,1],name='x')
        out = tf.placeholder(dtype='float32',shape=[None,10],name='y')
        conv1 = tf.layers.conv2d(input,16,(3,3),padding='same')
        max1 = tf.layers.max_pooling2d(conv1,(2,2),strides = 1)
        conv2 = tf.layers.conv2d(max1,16,(3,3),padding='same')
        max2 = tf.layers.max_pooling2d(conv2,(2,2),strides = 1)
        input1 = tf.layers.flatten(max2)
        l1 = tf.layers.dense(input1, 50, activation=tf.nn.relu)
        l2 = tf.layers.dense(l1, 50, activation=tf.nn.relu)
        l3 = tf.layers.dense(l2, 50, activation=tf.nn.relu)
        l4 = tf.layers.dense(l3, 50, activation=tf.nn.relu)
        out_pre = tf.layers.dense(l4,10,activation=tf.nn.softmax)

    loss = -tf.reduce_mean(out*tf.log(tf.clip_by_value(out_pre,1e-10,1)))
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction  = tf.equal(tf.argmax(out,axis = 1), tf.argmax(out_pre,axis=1))
    # cast转换数据格式
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化
    init_op = tf.global_variables_initializer() 
    sess = tf.Session()
    sess.run(init_op)
    batch_size = 1024
    epochs = 2000

    # 画图
    sns.set()
    sns.set_style('darkgrid')
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    losses = []
    accuracys = []
    print(sess.run(accuracy,feed_dict={out:y_test,input:x_test}))
    for epoch in range(epochs):
        batch_index = np.random.choice(range(len(x_train)),size=batch_size)
        _,loss_epoch = sess.run([train,loss],feed_dict={input:x_train[batch_index],out:y_train[batch_index]})
        accuracy_epoch = sess.run(accuracy,feed_dict={input:x_test,out:y_test})
        print(epoch+1,'loss:{0:.3},accuracy:{1:.3}'.format(loss_epoch,accuracy_epoch))
        losses.append(loss_epoch)
        accuracys.append(accuracy_epoch)
    ax1.plot(losses,label = 'loss')
    ax1.plot(accuracys,label = 'accuracy')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Rate')
    plt.legend()
    plt.savefig('Tensor_Conv.png')
    plt.show()



if __name__ == '__main__':
    main()
