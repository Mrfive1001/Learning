import numpy as np
import tensorflow as tf


class RBM:
    '''
    定义RBM模型并进行训练
    可以认为是降维
    或者用来做神经网络与训练
    '''
    def __init__(self, input_size, output_size):
        self._input_size = input_size  
        self._output_size = output_size  
        self.epochs = 5  
        self.learning_rate = 1.0  # 学习率
        self.batchsize = 100  

        self.w = np.zeros([input_size, output_size], np.float32)  
        self.hb = np.zeros([output_size], np.float32)   # 正向传播的权重
        self.vb = np.zeros([input_size], np.float32)    # 反向传播的权重
    
    def prob_h_given_v(self, visible, w, hb):
        # 利用sigmoid函数来进行预测概率
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)
    
    def prob_v_given_h(self, hidden, w, vb):
        # 反向预测概率
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)
    
    def sample_prob(self, probs):
        # 随机采样
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def train(self,X):
        # 非监督学习
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])

        # 梯度
        prv_w = np.zeros([self._input_size, self._output_size], np.float32)  
        prv_hb = np.zeros([self._output_size], np.float32) 
        prv_vb = np.zeros([self._input_size], np.float32)

        cur_w = np.zeros([self._input_size, self._output_size], np.float32)
        cur_hb = np.zeros([self._output_size], np.float32)
        cur_vb = np.zeros([self._input_size], np.float32)

        # 输入层
        v0 = tf.placeholder("float", [None, self._input_size])

        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        # 计算梯度来求解
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)

        # 更新参数
        update_w = _w + self.learning_rate * (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vb = _vb + self.learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self.learning_rate * tf.reduce_mean(h0 - h1, 0)

        # 方差
        err = tf.reduce_mean(tf.square(v0 - v1))

        with tf.Session() as sess:
            # 开始训练
            sess.run(tf.global_variables_initializer())
            # 每一轮
            for epoch in range(self.epochs):
                for start, end in zip(range(0, len(X), self.batchsize), range(self.batchsize, len(X), self.batchsize)):
                    batch = X[start:end]
                    # 更新参数
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error = sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                print('Epoch: %d' % epoch, 'reconstruction error: %f' % error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb
    
    def rbm_outpt(self, X):
        # 给DBN的期望输出
        # 也就是经过基础训练的输出
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)
    
