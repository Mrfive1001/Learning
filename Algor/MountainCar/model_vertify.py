import numpy as np
import gym
from dnn import DNN
import matplotlib.pyplot as plt
import seaborn as sns
import math
np.random.seed(10)
sns.set()
'''
验证在不加神经网络的情况下eso的反馈情况
'''

def step_my(X, u, mu):
    '''
    输入状态量和控制量和误差
    返回下一个状态
    '''
    simulation_step = 0.1  # 仿真步长
    u = min(max(u, -1.0), 1.0)
    x, v = X

    # 预测导数
    v_dot = 0.0015 * u+mu
    v = v + v_dot * simulation_step
    v = min(max(v, -0.07), 0.07)

    x = x + simulation_step * v
    x = min(max(x, -1.2), 0.6)
    X = np.array([x, v])
    if X.ndim == 2:
        X = X.reshape((2,))
    return X


def get_mu(mu, X1, X2, k=0.001):
    '''
    通过两个状态得到导数之间的误差
    '''
    dot = (k * (X1 - X2))[1]
    simulation_step = 0.1  # 仿真步长
    mu = mu + simulation_step * dot
    return mu

# 定义环境和要保存画图的变量
env = gym.make("MountainCarContinuous-v0")
record_tru = []
record_pre = []
record_mu = []
record_mu_tru = []
# 初始化
observation = env.reset()
my_observation = observation
# 误差
mu = 0
for epi in range(10000):
    # 随机取动作
    # env.render()
    u = env.action_space.sample()
    # 自己环境和原始环境分别进行仿真
    my_observation = step_my(my_observation, u, mu)
    observation, reward, done, info = env.step(u)
    # 得到误差
    mu = get_mu(mu, observation, my_observation)
    # 保存得到的结果
    record_mu_tru.append(-math.cos(observation[0] * 3) * 0.0025)
    record_tru.append(observation)
    record_pre.append(my_observation)
    record_mu.append(mu)
    print(epi, observation, my_observation)
    if done:
        break
record_tru = np.array(record_tru)
record_pre = np.array(record_pre)
record_mu = np.array(record_mu)
record_mu_tru = np.array(record_mu_tru)
x_ = np.arange(len(record_pre))
fig = plt.figure()
axis = 0
plt.plot(x_, record_tru[:, axis], label='x_tru')
plt.plot(x_, record_pre[:, axis], label='x_pre')
plt.legend()
fig = plt.figure()
axis = 1
plt.plot(x_, record_tru[:, axis], label='v_tru')
plt.plot(x_, record_pre[:, axis], label='v_pre')
plt.legend()
fig = plt.figure()
plt.plot(x_, record_mu, label='mu_pre')
plt.plot(x_, record_mu_tru, label='mu_tru')
plt.legend()
plt.show()
