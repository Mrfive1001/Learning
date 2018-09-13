import numpy as np
import gym
from dnn import DNN
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()
# 神经网络系数
ratio = 1e3
simulation_step = 0.1  # 仿真步长

def step_my(X, u, net, mu = 0):
    '''
    输入状态量、控制量、误差和网络
    返回下一个状态
    '''

    u = min(max(u, -1.0), 1.0)
    x, v = X
    dot = net.predict(X)[0]/ratio

    # 预测导数
    v_dot = 0.0015 * u + dot[0] + mu
    target = dot[0] + mu
    v = v + v_dot * simulation_step
    v = min(max(v, -0.07), 0.07)

    x = x + simulation_step * v
    x = min(max(x, -1.2), 0.6)
    X = np.array([x, v])
    if X.ndim == 2:
        X = X.reshape((2,))
    return X, target*ratio


def get_mu(mu, X1, X2, k=0.001):
    '''
    通过两个状态得到导数之间的误差
    '''
    dot = k * (X1 - X2)[1]
    mu = mu + simulation_step * dot
    return mu

# 定义环境和要保存画图的变量
env = gym.make("MountainCarContinuous-v0")
train = 1
net = DNN(2, 1, 20, name='MyModel', train=train, memory_size=200)
record_tru = []
record_pre = []
record_mu = []
record_mu_tru = []
# 初始化
observation = env.reset()
my_observation = observation
mu = 0
if train:
    for _ in range(5):
        observation = env.reset()
        my_observation = observation
        mu = 0
        for epi in range(20000):
            u = env.action_space.sample()
            my_observation, target = step_my(observation, u, net,mu)
            observation, reward, done, info = env.step(u)
            mu = get_mu(mu, observation, my_observation)
            net.store_sample(observation, target)
            if epi % 100 == 0:
                result = net.learn()
                if result:
                    print(epi, result)
    net.store_net()
observation = env.reset()
my_observation = observation
for epi in range(5000):
    u = env.action_space.sample()
    my_observation, d = step_my(observation, u, net)
    observation, reward, done, info = env.step(u)
    record_tru.append(observation)
    record_pre.append(my_observation)
record_tru = np.array(record_tru)
record_pre = np.array(record_pre)
# record_mu = np.array(record_mu)
# record_mu_tru = np.array(record_mu_tru)
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
plt.legend()
plt.show()
