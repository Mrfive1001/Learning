import math
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

from dnn import DNN

sns.set()



'''
对MountainCar模型进行模型辨识
1 生成数据
2 学习数据进行数据验证
3 训练并保存网络
4 可以对模型预测
'''


class MountainCar:
    """
    定义预测出来的模型
    """

    def __init__(self, name='Goodone', net=None, train = 0):
        """
        初始化
        net: 训练的神经网络
        verity：使用还是验证阶段
        验证阶段，神经网络未训练
        使用阶段，神经网络已训练
        """
        self.env = gym.make("MountainCarContinuous-v0")
        self.name = name
        self.simulation_step = 0.1
        self.units = 50
        self.ratio = 200
        self.reset()
        if net:
            self.net = net
        else:
            self.net = DNN(1, 1, self.units,train = train, name=self.name)


    def save_samples(self, big_epis=100):
        """
        保存运行得到的数据
        得到的数据有big_epis*3000行
        """
        record = []
        for big_epi in range(big_epis):
            # 初始化
            # 为了能够达到目标点
            a = 0.0025
            change = 100
            observation = self.reset()
            for epi in range(10000):
                if epi % change == 0:
                    u = self.action_sample()*3
                    print(big_epi, int(20 * epi / 3000) * '=')
                observation_old = observation.copy()
                observation, _, done, _ = self.env.step(u)
                target = self._get_target(observation_old, observation, u)
                x = observation_old[0]
                # 保存真实值和计算得到的值，后期作为比较
                # record.append([x, target, -a * math.cos(3 * x)])
                record.append([x, target])
        data = np.array(record)
        np.save(os.path.join(self.net.model_path0, 'memory.npy'), data)
        return data

    def verity_data(self):
        """
        验证数据集的正确性，画出两个自己计算出来的值和真实值的区别
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        sns.set()

        self.data = self._load_data()
        data_size = len(self.data)
        indexs = np.random.choice(data_size, size=int(data_size / 10))
        df = pd.DataFrame(self.data[indexs, :], columns=['position', 'target_dot', 'real_dot'])
        plt.figure()
        plt.scatter(df['position'], df['target_dot']*1.1,s = 5,label = 'target')  # 为了显示出区别乘以1.1
        plt.scatter(df['position'], df['real_dot'],s = 5,label = 'real')
        plt.legend()
        plt.show()

    def train_model(self):
        """
        利用得到的数据对模型进行训练，首先对数据进行缩放，之后利用神经网络进行拟合
        """
        # 训练
        data = self._load_data()
        data[:,1:] = data[:,1:]*self.ratio
        self.net.learn_data(data)
        self.net.store_net()


    def verity_net(self):
        """
        验证神经网络的正确性
        """

        a = 0.0025
        x_ = np.arange(-1.1, 0.5, 0.001)
        y_tru = -a * np.cos(3 * x_)
        y_pre = self.net.predict(x_.reshape((-1, 1)))/self.ratio
        # 验证对所有的x的拟合情况
        fig = plt.figure()
        plt.plot(x_, y_tru, label='x_tru')
        plt.plot(x_, y_pre, label='x_pre')
        plt.legend()

        y_tru_dot = 3 * a * np.sin(3 * x_)
        y_pre_dot = self.net.predict_dot(x_.reshape((-1, 1)))[:, 0]/self.ratio
        # y_pre_dot = self.net.predict_dot(x_.reshape((-1, 1)))[:, 0]
        # 验证对所有的x_dot的拟合情况
        fig = plt.figure()
        plt.plot(x_, y_tru_dot, label='x_dot_tru')
        plt.plot(x_, y_pre_dot, label='x_dot_pre')
        plt.legend()

        plt.show()

    def _load_data(self):
        """
        将最开始得到的数据读取出来
        :return:
        """
        data = np.load(os.path.join(self.net.model_path0, 'memory.npy'))
        return data

    def action_sample(self):
        """
        随机选取符合环境的动作
        """
        return self.env.action_space.sample()

    def reset(self):
        """
        利用原始问题的初始化，随机初始化
        """
        self.state = self.env.reset()
        return self.state

    def step(self, action):
        """
        利用神经网络进行模型辨识
        """
        action = min(max(action, -1.0), 1.0)
        x, v = self.state
        # 神经网络得到的导数
        dot = self.get_dot(self.state)
        v_dot = 0.0015 * action + dot[0]
        v = v + v_dot * self.simulation_step
        v = min(max(v, -0.07), 0.07)

        # 通过v计算x
        x = x + self.simulation_step * v
        x = min(max(x, -1.2), 0.6)
        X = np.array([x, v])
        if X.ndim == 2:
            X = X.reshape((2,))
        self.state = X
        # 返回参数
        info = {}
        done = {}
        reward = {}
        if x >= 0.45:
            done = True
        return self.state, reward, done, info

    def get_dot(self, X):
        return self.net.predict(X[0:1])[0]/self.ratio

    def get_dot2(self, X):
        return self.net.predict_dot(X[0:1])[0]/self.ratio

    def _get_target(self, X, X_new, u):
        """
        得到神经网络需要的真实值
        首先求真实的导数，之后计算真实值
        """
        u = min(max(u, -1.0), 1.0)
        return (((X_new - X) / self.simulation_step)[1] - u * 0.0015)


if __name__ == '__main__':
    mc = MountainCar(train=0)
    # 1 生成数据
    # mc.save_samples()

    # 2 验证数据
    # mc.verity_data()

    # 3 进行网络训练
    # mc.train_model()

    # 4 验证网络
    mc.verity_net()
