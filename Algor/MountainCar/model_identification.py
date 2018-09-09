import numpy as np
import gym


def get_my_dot(X, u, net):
    '''
    :param X: 状态变量
    :param net:  神经网络，可以进行预测
    :return: 返回与X维度相同的导数
    '''
    B = np.array([0, 0.0015])
    d1 = net.predict(X)
    d2 = B * u
    return d1 + d2


def step_my(X, u, net):
    '''
    :param X: 状态变量
    :param net:  神经网络，可以进行预测
    :return: 返回下个状态
    '''
    simulation_step = 1  # 仿真步长
    u = min(max(u, -1.0), 1.0)
    x, v = X
    dot = get_my_dot(X, u, net)
    v = v + simulation_step * dot[0]
    v = min(max(v, -0.07), 0.07)
    x = x + simulation_step * dot[1]
    x = min(max(x, -1.2), 0.6)
    X = np.array([x, v])
    return X

def get_error(error,X1,X2,k = 0.1):
    dot = k(X1-X2)
    simulation_step = 1  # 仿真步长
    error = error + simulation_step * dot

env = gym.make("MountainCarContinuous-v0")

observation = env.reset()
error = 0
for _ in range(1000):
    u = env.action_space.sample()
    my_observation = step_my(observation,u,net)
    observation, reward, done, info = env.step(u)
    error = get_error(error,observation,my_observation)
    net.store(observation,my_observation+error)
