import math

import gym
import numpy as np

from dnn import DNN
'''
对MountainCar模型进行模型辨识
'''

class MountainCar:
    """
    定义预测出来的模型
    """

    def __init__(self, train=0, name='Goodone'):
        """
        train: 是否进行训练，如果训练的话也等于训练次数
        """
        self.env = gym.make("MountainCarContinuous-v0")
        self.name = name
        self.simulation_step = 0.1
        self.ratio = 1
        if train:
            self.net = self.train_model(train)
        else:
            self.net = DNN(2, 1, 20, name=self.name, train=0,memory_size=1000, batch_size=200)


    def train_model(self, big_epis):
        """
        每一大轮循环20000代
        """
        net = DNN(2, 1, 20, name=self.name, train=1, memory_size=1000, batch_size=200)
        for big_epi in range(big_epis):
            # 初始化
            observation = self.reset()
            for epi in range(30000):
                u = self.action_sample()
                observation_old = observation.copy()
                observation, _, _, _ = self.env.step(u)
                target = self._get_target(observation_old, observation, u)
                net.store_sample(observation_old, target)
                if epi % 100 == 0:
                    result = net.learn()
                    if result:
                        print(big_epi, epi, result)
        net.store_net()
        return net

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
        dot = self._get_dot(self.state)
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
        return self.state, reward, done, info

    def _get_dot(self, X):
        return self.net.predict(X)[0]/self.ratio

    def _get_target(self, X, X_new, u):
        """
        得到神经网络需要的真实值
        首先求真实的导数，之后计算真实值
        """
        return (((X_new - X) / self.simulation_step)[1] - u * 0.0015)*self.ratio
