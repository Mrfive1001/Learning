import numpy as np
import gym
from dnn import DNN


def get_my_dot(X, u, net):
    '''
    :param X: 状态变量
    :param net:  神经网络，可以进行预测
    :return: 返回与X维度相同的导数,和神经网络预测结果
    '''
    B = np.array([0, 0.0015])
    d1 = net.predict(X)[0]
    d2 = B * u
    return d1 + d2, d1


def step_my(X, u, net):
    '''
    :param X: 状态变量
    :param net:  神经网络，可以进行预测
    :return: 返回下个状态，和神经网络预测结果
    '''
    simulation_step = 1  # 仿真步长
    u = min(max(u, -1.0), 1.0)
    x, v = X
    dot, d = get_my_dot(X, u, net)
    v = v + simulation_step * dot[0]
    v = min(max(v, -0.07), 0.07)
    x = x + simulation_step * dot[1]
    x = min(max(x, -1.2), 0.6)
    X = np.array([x, v])
    return X, d


def get_error(error, X1, X2, k=0.1):
    '''
    通过两个状态得到导数之间的误差
    '''
    dot = k * (X1 - X2)
    simulation_step = 1  # 仿真步长
    error = error + simulation_step * dot
    return error


env = gym.make("MountainCarContinuous-v0")
net = DNN(2, 2, 20, name='MyModel', train=1, memory_size=200)
for w in range(100000):
    error = 0
    observation = env.reset()
    for epi in range(100):
        u = env.action_space.sample()
        my_observation, d = step_my(observation, u, net)
        observation, reward, done, info = env.step(u)
        error = get_error(error, observation, my_observation)
        y = d + error
        net.store_sample(observation, y)
        result = net.learn()
        if result:
            print(w, result)