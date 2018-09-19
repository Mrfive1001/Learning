import math
import os
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

    def __init__(self, train=0,load = 1, name='Goodone',net = None):
        """
        train: 是否进行训练
        load: 是否读取以前的数据
        """
        self.env = gym.make("MountainCarContinuous-v0")
        self.name = name
        self.simulation_step = 0.1
        self.ratio = 1
        self.units = 50
        if net:
            self.net = net
        else:
            self.net = DNN(1, 1, self.units, name=self.name, train=train)
        if load:
            self.data = np.load(os.path.join(self.net.model_path0,'memory.npy'))
            if train:
                self.net.learn_data(self.data)
                self.net.store_net()
        else:
            self.data = self.get_samples()


    def get_samples(self,big_epis = 70):
        """
        保存运行得到的数据
        """
        record = []
        for big_epi in range(big_epis):
            # 初始化
            # 为了能够达到目标点
            a = 0.0025
            change = 100
            observation = self.reset()
            for epi in range(3000):
                if epi % change == 0:
                    u = self.action_sample()
                    print(big_epi, epi, u)
                observation_old = observation.copy()
                observation, _, done, _ = self.env.step(u)
                target = self._get_target(observation_old, observation, u)
                x = observation_old[0]
                # record.append([x, target,-a * math.cos(3 * x)])
                record.append([x, target])
        data = np.array(record)
        np.save(os.path.join(self.net.model_path0,'memory.npy'),data)
        return data


    def action_sample(self):
        """
        随机选取符合环境的动作
        """
        return self.env.action_space.sample()*3

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
        if x >= 0.45:
            done = True
        return self.state, reward, done, info

    def _get_dot(self, X):
        return self.net.predict(X[0:1])[0]/self.ratio

    def _get_target(self, X, X_new, u):
        """
        得到神经网络需要的真实值
        首先求真实的导数，之后计算真实值
        """
        u = min(max(u, -1.0), 1.0)
        return (((X_new - X) / self.simulation_step)[1] - u * 0.0015)*self.ratio


if __name__ == '__main__':
    mc = MountainCar(train=1,load=0)
    # mc.get_samples()