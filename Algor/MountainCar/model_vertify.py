import numpy as np
import gym
from dnn import DNN
import matplotlib.pyplot as plt
import seaborn as sns
import math
np.random.seed(10)
sns.set()


def step_my(X, u, mu):
    '''
    输入状态量和控制量和误差
    返回下一个状态
    '''
    simulation_step = 0.1  # 仿真步长
    u = min(max(u, -1.0), 1.0)
    x, v = X

    # 预测导数
    v_dot = 0.0015 * u + mu
    v = v + v_dot * simulation_step
    v = min(max(v, -0.07), 0.07)

    x = x + simulation_step * v
    x = min(max(x, -1.2), 0.6)
    X = np.array([x, v])
    if X.ndim == 2:
        X = X.reshape((2,))
    return X


def get_mu(mu, X1, X2, k=5):
    '''
    通过两个状态得到导数之间的误差
    '''
    dot = k * (X1 - X2)
    simulation_step = 0.1  # 仿真步长
    mu = mu + simulation_step * dot
    return mu


env = gym.make("MountainCarContinuous-v0")
record_tru = []
record_pre = []
record_mu = []
record_mu_tru = []
observation = env.reset()
mu = [0,0]
for epi in range(50000):
    u = env.action_space.sample()
    my_observation = step_my(observation, u, mu[1])
    observation, reward, done, info = env.step(u)
    mu = get_mu(mu, observation, my_observation)
    record_mu_tru.append(-math.cos(observation[0] * 3) * 0.0025)
    record_tru.append(observation)
    record_pre.append(my_observation)
    record_mu.append(list(mu))
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
plt.plot(x_, record_mu[:, 1], label='mu_pre')
plt.plot(x_, record_mu_tru, label='mu_tru')
plt.legend()
plt.show()
